import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from config_utils import load_config_with_base, resolve_path


def load_run_config(config_arg):
    config_path, _, cfg = load_config_with_base(config_arg, "run")
    onnx_path = resolve_path(config_path, cfg.get("onnx", "../artifacts/model.onnx"))
    image_path = resolve_path(config_path, cfg.get("image"))
    output_path = resolve_path(config_path, cfg.get("output", "../artifacts/mask.png"))
    artifacts_folder = resolve_path(config_path, cfg.get("artifacts_folder", "../artifacts/tidl/model-artifacts/square_seg"))
    tidl_tools_path = resolve_path(config_path, cfg.get("tidl_tools_path") or os.environ.get("TIDL_TOOLS_PATH", "/usr/lib"))

    if onnx_path is None or image_path is None or output_path is None:
        raise ValueError("onnx, image, and output must resolve to real paths")

    return {
        "provider": cfg.get("provider", "tidl"),
        "onnx_path": onnx_path,
        "image_path": image_path,
        "output_path": output_path,
        "artifacts_folder": artifacts_folder,
        "tidl_tools_path": tidl_tools_path,
        "height": cfg.get("height", cfg.get("image_height", 128)),
        "width": cfg.get("width", cfg.get("image_width", 128)),
        "threshold": float(cfg.get("threshold", 0.5)),
    }


def preprocess_image(path, height, width):
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, 1, height, width)


def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def build_session(cfg):
    if cfg["provider"] == "cpu":
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    else:
        os.environ.setdefault("TIDL_RT_PERFSTATS", "1")
        providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "tidl_tools_path": str(cfg["tidl_tools_path"]),
                "artifacts_folder": str(cfg["artifacts_folder"]) + "/",
            },
            {},
        ]

    session = ort.InferenceSession(
        str(cfg["onnx_path"]),
        providers=providers,
        provider_options=provider_options,
    )
    return session, providers


def get_benchmark(session):
    if not hasattr(session, "get_TI_benchmark_data"):
        return {}
    try:
        raw = session.get_TI_benchmark_data()
    except Exception:
        return {}
    return {
        str(key): int(value) if isinstance(value, (np.integer, int)) else float(value)
        for key, value in raw.items()
    }


def run_inference(session, cfg):
    input_tensor = preprocess_image(cfg["image_path"], cfg["height"], cfg["width"])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0]
    mask = (sigmoid(logits)[0, 0] > cfg["threshold"]).astype(np.uint8) * 255
    return mask


def save_mask(mask, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def build_report(session, providers, cfg, mask):
    return {
        "output": str(cfg["output_path"]),
        "providers_requested": providers,
        "providers_active": session.get_providers(),
        "mask_shape": list(mask.shape),
        "threshold": cfg["threshold"],
        "benchmark": get_benchmark(session),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_tidl.yaml")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    session, providers = build_session(cfg)
    mask = run_inference(session, cfg)
    save_mask(mask, cfg["output_path"])
    print(json.dumps(build_report(session, providers, cfg, mask), indent=2))


if __name__ == "__main__":
    main()
