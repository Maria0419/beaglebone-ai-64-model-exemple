# BeagleBone AI-64 TIDL 8.2 ONNX Model Example

Generic host-side tooling for compiling ONNX models for the BeagleBone AI-64 TIDL runtime, plus one working square-segmentation example model.

The full report for this example is available at [docs/beaglebone_report.pdf](docs/beaglebone_report.pdf).

The generic part of this repository is the TIDL 8.2 compiler environment:

- `docker/tidl82/Dockerfile`: Ubuntu 18.04 / Python 3.6 / `onnxruntime_tidl 1.7.0` / `tidl_tools 08_02_00_01-rc1`.
- `scripts/tidl82_onnx_compile.sh`: generic Docker wrapper for TIDL 8.2 ONNX compilation.
- `scripts/tidl82_export_compile.sh`: one-command wrapper that exports ONNX on the host and then runs the TIDL compile wrapper.
- `tools/compile_tidl.py`: YAML-driven ONNX compile script.
- `configs/compile_tidl82_template.yaml`: starting point for compiling another model.

The model-specific part lives under `models/`:

- `models/square_seg_32x13/`: final square segmentation example.

## Board Runtime

This flow targets the BeagleBone AI-64 image:

[BeagleBone AI-64 Debian 11.7 2023-08-05 10GB Xfce eMMC Flasher](https://www.beagleboard.org/distros/beaglebone-ai-64-xfce-2023-08-05-emmc-flasher)

On the board, ONNX Runtime should expose:

```text
TIDLExecutionProvider
TIDLCompilationProvider
CPUExecutionProvider
```

The host compiler must match the board runtime. For this image, use TIDL tools `08_02_00_01-rc1`, not newer EdgeAI/TIDL releases.

## Host Setup

Install Docker on the host:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and log in again, then verify:

```bash
docker info
```

Build/test the compiler container:

```bash
./scripts/tidl82_onnx_compile.sh -- python3 -c 'import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())'
```

Expected ONNX Runtime version inside the container is `1.7.0`.

## Compile the Included Example

The square segmentation example already includes `model.onnx` and precompiled TIDL artifacts. To re-export the ONNX and then recompile it on the host with one command:

```bash
./scripts/tidl82_export_compile.sh models/square_seg_32x13/configs/compile.yaml
```

This wrapper infers the sibling `export.yaml`, runs the host-side ONNX export first, and then calls the Docker-based TIDL compile wrapper. The compile YAML still selects the calibration images, input size, and output artifact directory. If `calibration_dir` is an absolute host path, the wrapper mounts it read-only into the container automatically.

## Compile Another ONNX Model

Copy the template and edit it:

```bash
cp configs/compile_tidl82_template.yaml configs/my_model_compile.yaml
# edit paths, input size, calibration_dir, calibration_glob, artifacts_folder
./scripts/tidl82_onnx_compile.sh configs/my_model_compile.yaml
```

This compiler helper is generic for fixed-shape image ONNX models using 1-channel or 3-channel calibration images. If your model needs different preprocessing, edit `tools/compile_tidl.py`.

## Run on the Board

For the included example, copy `models/square_seg_32x13/` to the board and run from that folder:

```bash
python3 src/run_tidl.py --config configs/run_cpu_board.yaml
sudo python3 src/run_tidl.py --config configs/run_tidl.yaml
```

The inference script is YAML-driven and only takes `--config`. Before running it, edit the chosen YAML to point `image:` at the test image you want to run.

The TIDL run should show `TIDLExecutionProvider` and one offloaded subgraph.
