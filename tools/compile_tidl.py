import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from config_utils import load_config_with_base, resolve_path


def preprocess(path, height, width, channels):
    if channels == 1:
        image = Image.open(path).convert("L")
    elif channels == 3:
        image = Image.open(path).convert("RGB")
    else:
        raise ValueError("image_channels must be 1 or 3")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        return array.reshape(1, 1, height, width)
    return array.transpose(2, 0, 1).reshape(1, 3, height, width)


def find_calibration_images(root, pattern):
    root = Path(root)
    images = sorted(root.rglob(pattern))
    if not images:
        raise FileNotFoundError(f"No calibration images matching '{pattern}' found under {root}")
    return images


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--artifacts-folder", default=None)
    parser.add_argument("--tidl-tools-path", default=None)
    parser.add_argument("--calibration-frames", type=int, default=None)
    parser.add_argument("--tensor-bits", type=int, default=None)
    parser.add_argument("--accuracy-level", type=int, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    config_path, _, cfg = load_config_with_base(args.config, "compile")
    height = int(cfg["image_height"])
    width = int(cfg["image_width"])
    channels = int(cfg.get("image_channels", 1))
    channels = int(cfg.get("image_channels", 1))
    default_model = cfg.get("onnx")
    if default_model is None and "artifacts_dir" in cfg:
        default_model = Path(cfg["artifacts_dir"]) / "model.onnx"
    default_artifacts = cfg.get("artifacts_folder")
    if default_artifacts is None and "artifacts_dir" in cfg:
        default_artifacts = Path(cfg["artifacts_dir"]) / "tidl" / "model-artifacts" / "square_seg"
    model_path = resolve_path(config_path, args.onnx or default_model)
    artifacts_folder = resolve_path(config_path, args.artifacts_folder or default_artifacts)
    tidl_tools_path = resolve_path(config_path, args.tidl_tools_path or cfg.get("tidl_tools_path") or os.environ.get("TIDL_TOOLS_PATH"))
    if model_path is None or artifacts_folder is None:
        raise ValueError("onnx and artifacts_folder must resolve to real paths")
    if not tidl_tools_path:
        raise EnvironmentError("Set TIDL_TOOLS_PATH, define tidl_tools_path in YAML, or pass --tidl-tools-path before compiling.")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not args.keep_artifacts:
        clean_dir(artifacts_folder)
    else:
        artifacts_folder.mkdir(parents=True, exist_ok=True)

    calibration_root = cfg.get("calibration_dir", cfg.get("train_dir"))
    if not calibration_root:
        raise ValueError("Set calibration_dir or train_dir in the compile YAML before compiling.")
    calibration_root = resolve_path(config_path, calibration_root)
    calibration_glob = str(cfg.get("calibration_glob", "*_image.tif"))
    calibration_frames = int(args.calibration_frames or cfg.get("calibration_frames", 64))
    calibration_images = find_calibration_images(calibration_root, calibration_glob)[:calibration_frames]

    delegate_options = {
        "tidl_tools_path": str(tidl_tools_path),
        "artifacts_folder": str(artifacts_folder) + "/",
        "platform": str(cfg.get("tidl_platform", "J7")),
        "version": str(cfg.get("tidl_version", "8.2")),
        "tensor_bits": int(args.tensor_bits or cfg.get("tidl_tensor_bits", 8)),
        "debug_level": int(cfg.get("tidl_debug_level", 1)),
        "max_num_subgraphs": int(cfg.get("tidl_max_num_subgraphs", 16)),
        "deny_list": str(cfg.get("tidl_deny_list", "")),
        "accuracy_level": int(args.accuracy_level or cfg.get("tidl_accuracy_level", 1)),
        "advanced_options:calibration_frames": calibration_frames,
        "advanced_options:calibration_iterations": int(cfg.get("tidl_calibration_iterations", 5)),
        "advanced_options:add_data_convert_ops": int(cfg.get("tidl_add_data_convert_ops", 3)),
        "advanced_options:quantization_scale_type": int(cfg.get("tidl_quantization_scale_type", 0)),
        "advanced_options:high_resolution_optimization": int(cfg.get("tidl_high_resolution_optimization", 0)),
        "advanced_options:pre_batchnorm_fold": int(cfg.get("tidl_pre_batchnorm_fold", 1)),
        "ti_internal_nc_flag": int(cfg.get("tidl_ti_internal_nc_flag", 1601)),
    }

    session = ort.InferenceSession(
        str(model_path),
        providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
        provider_options=[delegate_options, {}],
    )
    input_name = session.get_inputs()[0].name
    for idx, image_path in enumerate(calibration_images):
        session.run(None, {input_name: preprocess(image_path, height, width, channels)})
        print(f"compiled calibration frame {idx + 1}/{len(calibration_images)}: {image_path}")

    report = {
        "model": str(model_path),
        "artifacts_folder": str(artifacts_folder),
        "calibration_frames": len(calibration_images),
        "image_shape": [1, channels, height, width],
        "image_shape": [1, channels, height, width],
        "providers": session.get_providers(),
        "delegate_options": delegate_options,
    }
    (artifacts_folder / "compile_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
