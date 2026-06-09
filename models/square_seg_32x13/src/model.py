from __future__ import annotations

import torch
from torch import nn


class TinySquareSeg(nn.Module):
    """Fully convolutional binary segmentation model for TIDL.

    The exported inference graph intentionally uses only Conv and Relu ops.
    More layers increase receptive field without introducing pooling, resize,
    deconvolution, sigmoid, flatten, or gemm.
    """

    def __init__(self, channels: int = 48, layers: int = 15, kernel_size: int = 5) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be >= 2")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        blocks: list[nn.Module] = [
            nn.Conv2d(1, channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=False),
        ]
        for _ in range(layers - 2):
            blocks.extend([
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.ReLU(inplace=False),
            ])
        blocks.append(nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(channels: int = 48, layers: int = 15, kernel_size: int = 5) -> TinySquareSeg:
    return TinySquareSeg(channels=channels, layers=layers, kernel_size=kernel_size)
