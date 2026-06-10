from __future__ import annotations

import torch
from torch import nn


class SquareSegModel(nn.Module):
    """Fully convolutional binary segmentation model."""

    def __init__(self, channels: int = 32, layers: int = 13, kernel_size: int = 3) -> None:
        super().__init__()

        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        in_channels = 1

        for _ in range(layers - 1):
            blocks.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding, bias=True))
            blocks.append(nn.ReLU(inplace=False))
            in_channels = channels

        blocks.append(nn.Conv2d(channels, 1, kernel_size=1, bias=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(channels: int = 32, layers: int = 13, kernel_size: int = 3) -> SquareSegModel:
    return SquareSegModel(channels=channels, layers=layers, kernel_size=kernel_size)
