import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import SquareSegDataset
from model import DiceBCELoss, build_model



def compute_segmentation_metrics(logits, masks, threshold):
    predictions = (torch.sigmoid(logits) > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (predictions * masks).sum(dim=(1, 2, 3))
    prediction_sum = predictions.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = prediction_sum + mask_sum - intersection
    dice = ((2.0 * intersection + 1.0) / (prediction_sum + mask_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    return dice, iou



def run_epoch(
    model,
    loader,
    criterion,
    device,
    threshold,
    epoch,
    total_epochs,
    stage,
    optimizer=None,
    max_grad_norm=0.0,
):
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{stage}]", leave=False)
    
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_training else torch.no_grad()

        with grad_context:
            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        logits = logits.detach().float()
        dice, iou = compute_segmentation_metrics(logits, masks, threshold)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    artifacts_dir = Path(cfg["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    image_height = cfg["image_height"]
    image_width = cfg["image_width"]
    train_dataset = SquareSegDataset(
        cfg["train_dir"],
        image_height=image_height,
        image_width=image_width,
        augment=bool(cfg.get("augment", False)),
    )
    test_dataset = SquareSegDataset(
        cfg["test_dir"],
        image_height=image_height,
        image_width=image_width,
        augment=False,
    )

    print(f"Training data: {len(train_dataset)}  Testing data: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]), shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(channels=cfg["channels"], layers=cfg["layers"], kernel_size=cfg["kernel_size"]).to(device)

    criterion = DiceBCELoss(
        bce_weight=float(cfg.get("bce_loss_weight", 1.0)),
        pos_weight=float(cfg.get("pos_weight", 1.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
    )

    threshold = 0.5
    max_epochs = cfg["epochs"]
    patience = max(1, cfg.get("early_stopping_patience", 5))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    best_iou = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            epoch=epoch,
            total_epochs=max_epochs,
            stage="train",
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            epoch=epoch,
            total_epochs=max_epochs,
            stage="test",
        )

        row = {
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
            "config": cfg,
            "epoch": epoch,
            "metrics": row,
            "history": history,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        if test_metrics["iou"] > best_iou:
            best_iou = float(test_metrics["iou"])
            epochs_without_improvement = 0
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
