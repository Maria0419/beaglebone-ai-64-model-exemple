import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from config_utils import load_config_with_base, resolve_path


def preprocess(path, height, width):
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, 1, height, width)


def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def get_benchmark(session):
    if not hasattr(session, "get_TI_benchmark_data"):
        return {}
    try:
        raw = session.get_TI_benchmark_data()
    except Exception:
        return {}
    return {str(key): int(value) if isinstance(value, (np.integer, int)) else float(value) for key, value in raw.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--provider", choices=["cpu", "tidl"], default=None)
    parser.add_argument("--artifacts-folder", default=None)
    parser.add_argument("--tidl-tools-path", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config) if args.config else None
    if config_path is not None:
        config_path, _, cfg = load_config_with_base(config_path, "run")

    onnx_default = cfg.get("onnx", "artifacts/model.onnx")
    image_default = cfg.get("image")
    output_default = cfg.get("output", "outputs/mask.png")
    provider = args.provider or cfg.get("provider", "tidl")
    artifacts_folder_default = cfg.get("artifacts_folder", "artifacts/tidl/model-artifacts/square_seg")
    tidl_tools_default = cfg.get("tidl_tools_path") or os.environ.get("TIDL_TOOLS_PATH", "/usr/lib")
    height = int(args.height or cfg.get("height") or cfg.get("image_height", 128))
    width = int(args.width or cfg.get("width") or cfg.get("image_width", 128))
    threshold = float(args.threshold if args.threshold is not None else cfg.get("threshold", 0.5))

    if config_path is not None:
        onnx_path = resolve_path(config_path, args.onnx or onnx_default)
        image_path = resolve_path(config_path, args.image or image_default)
        output_path = resolve_path(config_path, args.output or output_default)
        artifacts_folder = resolve_path(config_path, args.artifacts_folder or artifacts_folder_default)
        tidl_tools_path = resolve_path(config_path, args.tidl_tools_path or tidl_tools_default)
    else:
        onnx_path = Path(args.onnx or onnx_default)
        image_path = Path(args.image or image_default) if (args.image or image_default) else None
        output_path = Path(args.output or output_default)
        artifacts_folder = Path(args.artifacts_folder or artifacts_folder_default)
        tidl_tools_path = Path(args.tidl_tools_path or tidl_tools_default)

    if onnx_path is None or image_path is None or output_path is None or artifacts_folder is None or tidl_tools_path is None:
        raise ValueError("onnx, image, output, artifacts_folder, and tidl_tools_path must resolve to real paths")

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    else:
        os.environ.setdefault("TIDL_RT_PERFSTATS", "1")
        delegate_options = {
            "tidl_tools_path": str(tidl_tools_path),
            "artifacts_folder": str(Path(artifacts_folder)) + "/",
        }
        providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [delegate_options, {}]

    session = ort.InferenceSession(str(onnx_path), providers=providers, provider_options=provider_options)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: preprocess(Path(image_path), height, width)})[0]
    mask = (sigmoid(logits)[0, 0] > threshold).astype(np.uint8) * 255

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)

    report = {
        "output": str(output_path),
        "providers_requested": providers,
        "providers_active": session.get_providers(),
        "mask_shape": list(mask.shape),
        "threshold": threshold,
        "benchmark": get_benchmark(session),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
