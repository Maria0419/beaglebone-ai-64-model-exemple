from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SquareSegDataset, estimate_positive_weight
from model import build_model


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight: float, dice_weight: float, bce_weight: float = 1.0, logit_l2_weight: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.logit_l2_weight = float(logit_l2_weight)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=self.pos_weight.to(logits.device))
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * masks).sum(dim=dims)
        denom = probs.sum(dim=dims) + masks.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
        logit_l2 = logits.float().pow(2).mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss + self.logit_l2_weight * logit_l2


def dice_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (pred * masks).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = pred_sum + mask_sum - intersection
    dice = ((2.0 * intersection + 1.0) / (pred_sum + mask_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    return dice, iou


def save_prediction_samples(model: torch.nn.Module, dataset: SquareSegDataset, out_dir: Path, device: torch.device, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = min(8, len(dataset))
    with torch.no_grad():
        for idx in range(count):
            image, mask = dataset[idx]
            logits = model(image.unsqueeze(0).to(device)).cpu()
            pred = (torch.sigmoid(logits)[0, 0].numpy() > threshold).astype(np.uint8) * 255
            Image.fromarray((image[0].numpy() * 255).astype(np.uint8)).save(out_dir / f"{idx:03d}_image.png")
            Image.fromarray((mask[0].numpy() * 255).astype(np.uint8)).save(out_dir / f"{idx:03d}_label.png")
            Image.fromarray(pred).save(out_dir / f"{idx:03d}_pred.png")


def evaluate_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: list[float],
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    batches = 0
    stats = {float(t): {"dice": 0.0, "iou": 0.0} for t in thresholds}
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            logits = logits.float()
            total_loss += float(loss.item())
            for threshold in thresholds:
                dice, iou = dice_iou_from_logits(logits, masks, float(threshold))
                stats[float(threshold)]["dice"] += dice
                stats[float(threshold)]["iou"] += iou
            batches += 1

    results = []
    for threshold in thresholds:
        item = {
            "threshold": float(threshold),
            "dice": stats[float(threshold)]["dice"] / max(1, batches),
            "iou": stats[float(threshold)]["iou"] / max(1, batches),
        }
        results.append(item)
    best = max(results, key=lambda x: (x["iou"], x["dice"]))
    return {
        "loss": total_loss / max(1, batches),
        "best": best,
        "thresholds": results,
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
    train_ds = SquareSegDataset(cfg["train_dir"], expected_size=expected_size)
    test_ds = SquareSegDataset(cfg["test_dir"], expected_size=expected_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=int(cfg["num_workers"]) > 0,
    )

    device = torch.device(args.device)
    model = build_model(
        channels=int(cfg["channels"]),
        layers=int(cfg.get("layers", 15)),
        kernel_size=int(cfg.get("kernel_size", 5)),
    ).to(device)

    pos_weight_cfg = cfg.get("pos_weight", "auto")
    if pos_weight_cfg == "auto":
        pos_weight_value = estimate_positive_weight(train_ds.pairs)
    else:
        pos_weight_value = float(pos_weight_cfg)
    criterion = SegmentationLoss(
        pos_weight=pos_weight_value,
        dice_weight=float(cfg.get("dice_loss_weight", 1.0)),
        bce_weight=float(cfg.get("bce_loss_weight", 1.0)),
        logit_l2_weight=float(cfg.get("logit_l2_weight", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg.get("min_learning_rate", 1e-5)))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    thresholds = [float(x) for x in cfg.get("thresholds", [cfg.get("threshold", 0.5)])]
    best_iou = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            max_grad_norm = float(cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            batches += 1
        scheduler.step()

        eval_result = evaluate_sweep(model, test_loader, criterion, device, thresholds, use_amp)
        best = eval_result["best"]
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, batches),
            "test_loss": eval_result["loss"],
            "test_dice": best["dice"],
            "test_iou": best["iou"],
            "threshold": best["threshold"],
            "learning_rate": scheduler.get_last_lr()[0],
            "pos_weight": pos_weight_value,
            "dice_loss_weight": float(cfg.get("dice_loss_weight", 1.0)),
            "bce_loss_weight": float(cfg.get("bce_loss_weight", 1.0)),
            "logit_l2_weight": float(cfg.get("logit_l2_weight", 0.0)),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": {**cfg, "threshold": best["threshold"]},
            "epoch": epoch,
            "metrics": row,
            "threshold_sweep": eval_result["thresholds"],
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")
        if float(best["iou"]) > best_iou:
            best_iou = float(best["iou"])
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            save_prediction_samples(model, test_ds, samples_dir, device, float(best["threshold"]))
            (artifacts_dir / "threshold_sweep.json").write_text(json.dumps({"best": best, "results": eval_result["thresholds"]}, indent=2))

    (artifacts_dir / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
