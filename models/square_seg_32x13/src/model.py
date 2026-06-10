import torch
from torch import nn


class DiceLoss(nn.Module):
    def forward(self, logits, masks):
        probs = torch.sigmoid(logits)
        intersection = (probs * masks).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.bce_weight = float(bce_weight)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits, masks):
        loss = self.dice(logits, masks)
        if self.bce_weight <= 0.0:
            return loss
        bce = nn.functional.binary_cross_entropy_with_logits(logits, masks, pos_weight=self.pos_weight)
        return loss + self.bce_weight * bce


class SquareSegModel(nn.Module):
    """Fully convolutional binary segmentation model."""

    def __init__(self, channels=32, layers=13, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        blocks = []
        in_channels = 1

        for _ in range(layers - 1):
            blocks.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding, bias=True))
            blocks.append(nn.ReLU(inplace=False))
            in_channels = channels

        blocks.append(nn.Conv2d(channels, 1, kernel_size=1, bias=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def build_square_seg_model(params=None):
    params = dict(params or {})
    return SquareSegModel(
        channels=params.get("channels", 32),
        layers=params.get("layers", 13),
        kernel_size=params.get("kernel_size", 3),
    )


MODEL_BUILDERS = {
    "square_seg": build_square_seg_model,
}


def normalize_model_config(model_config=None, **legacy_params):
    if model_config is None:
        legacy_params = {key: value for key, value in legacy_params.items() if value is not None}
        return {
            "name": "square_seg",
            "params": legacy_params,
        }

    if not isinstance(model_config, dict):
        raise TypeError("model config must be a mapping")

    model_name = model_config.get("name")
    if not model_name:
        raise ValueError("model config must define a non-empty 'name'")

    params = model_config.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError("model.params must be a mapping")

    return {
        "name": model_name,
        "params": params,
    }


def build_model(model_config=None, **legacy_params):
    model_config = normalize_model_config(model_config, **legacy_params)
    model_name = model_config["name"]

    if model_name not in MODEL_BUILDERS:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"unsupported model '{model_name}'. Supported models: {supported}")

    return MODEL_BUILDERS[model_name](model_config["params"])
