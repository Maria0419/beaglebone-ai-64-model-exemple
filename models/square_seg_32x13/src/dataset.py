from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_image(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_mask(path):
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.float32)


class SquareSegDataset(Dataset):
    def __init__(self, root, image_height=128, image_width=128, augment=False):
        self.root = Path(root)
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment

        self.image_files = sorted(self.root.rglob("*_image.tif"))

        self.label_files = []

        for image_path in self.image_files:
            label_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
            self.label_files.append(label_path)



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = _load_image(self.image_files[index])
        mask = _load_mask(self.label_files[index])

        if self.augment and torch.rand(1).item() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return image_tensor, mask_tensor
