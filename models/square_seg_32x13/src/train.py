from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import SquareSegDataset
from model import build_model


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * masks).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.bce_weight = float(bce_weight)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        loss = self.dice(logits, masks)
        if self.bce_weight <= 0.0:
            return loss
        bce = nn.functional.binary_cross_entropy_with_logits(logits, masks, pos_weight=self.pos_weight)
        return loss + self.bce_weight * bce


def dice_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> tuple[float, float]:
    predictions = (torch.sigmoid(logits) > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (predictions * masks).sum(dim=(1, 2, 3))
    prediction_sum = predictions.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = prediction_sum + mask_sum - intersection
    dice = ((2.0 * intersection + 1.0) / (prediction_sum + mask_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    return dice, iou


def save_prediction_samples(model: nn.Module, dataset: SquareSegDataset, out_dir: Path, device: torch.device, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_count = min(8, len(dataset))

    with torch.no_grad():
        for index in range(sample_count):
            image, mask = dataset[index]
            logits = model(image.unsqueeze(0).to(device)).cpu()
            prediction = (torch.sigmoid(logits)[0, 0].numpy() > threshold).astype(np.uint8) * 255
            Image.fromarray((image[0].numpy() * 255).astype(np.uint8)).save(out_dir / f"{index:03d}_image.png")
            Image.fromarray((mask[0].numpy() * 255).astype(np.uint8)).save(out_dir / f"{index:03d}_label.png")
            Image.fromarray(prediction).save(out_dir / f"{index:03d}_pred.png")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 0.0,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{stage}]", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_training else torch.no_grad()
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )

        with grad_context:
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, masks)

            if is_training:
                assert optimizer is not None
                assert scaler is not None
                scaler.scale(loss).backward()
                if max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        logits = logits.detach().float()
        dice, iou = dice_iou_from_logits(logits, masks, threshold)
        total_loss += float(loss.item())
        total_dice += dice
        total_iou += iou
        batch_count += 1

        progress.set_postfix(
            loss=f"{total_loss / batch_count:.4f}",
            dice=f"{total_dice / batch_count:.4f}",
            iou=f"{total_iou / batch_count:.4f}",
        )

    progress.close()
    return {
        "loss": total_loss / max(1, batch_count),
        "dice": total_dice / max(1, batch_count),
        "iou": total_iou / max(1, batch_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    artifacts_dir = Path(cfg["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    samples_dir = artifacts_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    expected_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    train_dataset = SquareSegDataset(cfg["train_dir"], expected_size=expected_size, augment=bool(cfg.get("augment", False)))
    test_dataset = SquareSegDataset(cfg["test_dir"], expected_size=expected_size, augment=False)

    print(f"Training data: {len(train_dataset)}  Testing data: {len(test_dataset)}")

    device = torch.device(args.device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
    )

    model = build_model(
        channels=int(cfg["channels"]),
        layers=int(cfg["layers"]),
        kernel_size=int(cfg["kernel_size"]),
    ).to(device)

    criterion = DiceBCELoss(
        bce_weight=float(cfg.get("bce_loss_weight", 1.0)),
        pos_weight=float(cfg.get("pos_weight", 1.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    threshold = float(cfg["threshold"])
    max_epochs = int(cfg["epochs"])
    patience = max(1, int(cfg.get("early_stopping_patience", 5)))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    best_iou = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=max_epochs,
            stage="train",
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=max_epochs,
            stage="test",
        )

        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
            "threshold": threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": {**cfg, "threshold": threshold},
            "epoch": epoch,
            "metrics": row,
            "history": history,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        if test_metrics["iou"] > best_iou:
            best_iou = float(test_metrics["iou"])
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            save_prediction_samples(model, test_dataset, samples_dir, device, threshold)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "patience": patience,
                }, sort_keys=True), flush=True)
                break

    (artifacts_dir / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
