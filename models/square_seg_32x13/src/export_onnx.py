import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config_with_base, resolve_path
from model import build_model


ALLOWED_OPS = {"Conv", "Relu"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--skip-ort-check", action="store_true")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()

    config_path, _, cfg = load_config_with_base(args.config, "export")
    outputs_dir = Path(cfg["artifacts_dir"])
    model_path = outputs_dir / "model.pt"
    default_output = cfg.get("output", outputs_dir / "model.onnx")
    output_path = resolve_path(config_path, args.output or default_output)
    opset = int(args.opset or cfg.get("opset", 11))
    tolerance = float(args.tolerance if args.tolerance is not None else cfg.get("tolerance", 1e-4))
    skip_ort_check = bool(cfg.get("skip_ort_check", False)) or args.skip_ort_check
    if output_path is None:
        raise ValueError("output path must resolve to a real path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_state = torch.load(model_path, map_location="cpu")
    ckpt_cfg = model_state.get("config", cfg)
    model = build_model(
        channels=int(ckpt_cfg.get("channels", cfg["channels"])),
        layers=int(ckpt_cfg.get("layers", cfg.get("layers", 15))),
        kernel_size=int(ckpt_cfg.get("kernel_size", cfg.get("kernel_size", 5))),
    )
    model.load_state_dict(model_state["model_state"])
    model.eval()

    dummy = torch.zeros(1, 1, int(cfg["image_height"]), int(cfg["image_width"]), dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
        external_data=False,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ops = sorted({node.op_type for node in onnx_model.graph.node})
    disallowed = sorted(set(ops) - ALLOWED_OPS)
    if disallowed:
        raise RuntimeError(f"ONNX contains unsupported ops for this project: {disallowed}. Full op list: {ops}")

    report = {"onnx": str(output_path), "ops": ops, "opset": opset}
    if not skip_ort_check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"input": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        report["max_abs_diff_torch_vs_ort"] = max_abs_diff
        if max_abs_diff > tolerance:
            raise RuntimeError(f"PyTorch and ONNX Runtime outputs differ: max_abs_diff={max_abs_diff}")

    (outputs_dir / "onnx_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
