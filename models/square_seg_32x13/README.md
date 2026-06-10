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
- `artifacts/model.pt`: PyTorch model state used for export in the standard example workflow.
- `artifacts/tidl/model-artifacts/square_seg/`: compiled TIDL outputs.
- `test_images/img_0001_image.tif`: one test image.
- `artifacts/`: exported PyTorch model state, ONNX, metrics, and example CPU/TIDL masks.

## Train the Example Model

From this folder:

```bash
python3 src/train.py --config configs/train.yaml
```

Edit `model.name` and `model.params` inside `configs/train.yaml` to change which model the example trains. If you add another architecture, add its builder in `src/model.py` and only keep the params that architecture uses.

## Export and Recompile in One Command

From the repository root:

```bash
./scripts/tidl82_export_compile.sh models/square_seg_32x13/configs/compile.yaml
```

This wrapper infers `models/square_seg_32x13/configs/export.yaml`, exports `artifacts/model.onnx` on the host, and then launches the Docker-based TIDL compile step.

## Manual Re-export ONNX

From this folder:

```bash
python3 src/export_onnx.py --config configs/export.yaml
```

## Manual Recompile TIDL Artifacts

From the repository root:

```bash
./scripts/tidl82_onnx_compile.sh models/square_seg_32x13/configs/compile.yaml
```

## Run CPU on Host or Board

From this folder:

```bash
python3 src/run_tidl.py --config configs/run_cpu_board.yaml
```

## Run TIDL on BeagleBone AI-64

From this folder on the board:

```bash
sudo python3 src/run_tidl.py --config configs/run_tidl.yaml
```

Expected board behavior:

- `libtidl_onnxrt_EP loaded`
- `Final number of subgraphs created are : 1`
- `Offloaded Nodes - 25, Total Nodes - 25`
