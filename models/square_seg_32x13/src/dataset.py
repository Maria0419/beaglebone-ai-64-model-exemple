from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path))
    key: list[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.float32)


class SquareSegDataset(Dataset):
    def __init__(self, root: str | Path, expected_size: tuple[int, int] = (128, 128), augment: bool = False) -> None:
        self.root = Path(root)
        self.expected_size = tuple(expected_size)
        self.augment = augment

        self.image_files = sorted(self.root.rglob("*_image.tif"), key=_sort_key)
        if not self.image_files:
            raise FileNotFoundError(f"No '*_image.tif' files found under {self.root}")

        self.label_files: list[Path] = []
        missing: list[Path] = []
        for image_path in self.image_files:
            label_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
            if label_path.exists():
                self.label_files.append(label_path)
            else:
                missing.append(label_path)

        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(f"{len(missing)} label files are missing. First missing labels:\n{preview}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = _load_image(self.image_files[index])
        mask = _load_mask(self.label_files[index])

        if image.shape != self.expected_size:
            raise ValueError(f"{self.image_files[index]} has shape {image.shape}, expected {self.expected_size}")
        if mask.shape != self.expected_size:
            raise ValueError(f"{self.label_files[index]} has shape {mask.shape}, expected {self.expected_size}")

        if self.augment and torch.rand(1).item() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return image_tensor, mask_tensor
