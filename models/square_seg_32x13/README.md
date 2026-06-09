# Square Segmentation 32x13 Example

This folder is one concrete model example for the generic BeagleBone AI-64 TIDL 8.2 compiler wrapper.

Model summary:

- Input: `1x1x128x128` grayscale image, float32 `[0, 1]`.
- Output: `1x1x128x128` logits.
- ONNX ops: `Conv`, `Relu`.
- Opset: `11`.
- TIDL compile: INT8, TIDL 8.2.

Included artifacts:

- `artifacts/model.onnx`: exported ONNX model.
- `artifacts/checkpoints/best.pt`: PyTorch checkpoint used for export.
- `artifacts/tidl/model-artifacts/square_seg/`: compiled TIDL artifacts.
- `test_images/img_0001_image.tif`: one test image.
- `artifacts/outputs/`: example CPU/TIDL masks from previous board validation.

## Re-export ONNX

From this folder:

```bash
python3 src/export_onnx.py --config configs/export_final.yaml
```

## Recompile TIDL Artifacts

From the repository root:

```bash
./scripts/tidl82_onnx_compile.sh models/square_seg_32x13/configs/compile_final.yaml
```

## Run CPU on Host or Board

From this folder:

```bash
python3 src/run_tidl.py --config configs/run_cpu_board_final.yaml
```

## Run TIDL on BeagleBone AI-64

From this folder on the board:

```bash
sudo python3 src/run_tidl.py --config configs/run_tidl_final.yaml
```

Expected board behavior:

- `libtidl_onnxrt_EP loaded`
- `Final number of subgraphs created are : 1`
- `Offloaded Nodes - 25, Total Nodes - 25`
