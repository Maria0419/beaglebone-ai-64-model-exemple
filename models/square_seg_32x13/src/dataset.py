from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SquarePair:
    image: Path
    label: Path


def find_pairs(root: str | Path) -> list[SquarePair]:
    root = Path(root)
    images = sorted(root.rglob("*_image.tif"))
    pairs: list[SquarePair] = []
    missing: list[Path] = []
    for image in images:
        label = image.with_name(image.name.replace("_image.tif", "_label.tif"))
        if label.exists():
            pairs.append(SquarePair(image=image, label=label))
        else:
            missing.append(label)
    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"{len(missing)} label files are missing. First missing labels:\n{preview}")
    return pairs


def validate_pairs(root: str | Path) -> list[SquarePair]:
    pairs = find_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No '*_image.tif' files found under {root}")
    return pairs


def load_image_tensor(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def load_mask_tensor(path: str | Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class SquareSegDataset(Dataset):
    def __init__(self, root: str | Path, expected_size: tuple[int, int] = (128, 128)) -> None:
        self.root = Path(root)
        self.expected_size = expected_size
        self.pairs = validate_pairs(self.root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        image = load_image_tensor(pair.image)
        mask = load_mask_tensor(pair.label)
        if tuple(image.shape[-2:]) != self.expected_size:
            raise ValueError(f"{pair.image} has shape {tuple(image.shape[-2:])}, expected {self.expected_size}")
        if tuple(mask.shape[-2:]) != self.expected_size:
            raise ValueError(f"{pair.label} has shape {tuple(mask.shape[-2:])}, expected {self.expected_size}")
        return image, mask


def estimate_positive_weight(pairs: Iterable[SquarePair], max_samples: int = 512) -> float:
    positives = 0
    total = 0
    for idx, pair in enumerate(pairs):
        if idx >= max_samples:
            break
        mask = load_mask_tensor(pair.label)
        positives += int(mask.sum().item())
        total += int(mask.numel())
    if positives == 0:
        return 1.0
    negatives = total - positives
    return max(1.0, negatives / positives)
