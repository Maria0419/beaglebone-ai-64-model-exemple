#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/tidl82_onnx_compile.sh [CONFIG_PATH]
  scripts/tidl82_onnx_compile.sh -- <custom command inside container>

Examples:
  scripts/tidl82_onnx_compile.sh models/square_seg_32x13/configs/compile_final.yaml
  scripts/tidl82_onnx_compile.sh -- python3 tools/compile_tidl.py --config models/square_seg_32x13/configs/compile_final.yaml

Environment variables:
  TIDL82_IMAGE       Docker image name. Default: tidl82-onnx-compiler:08_02_00_01_rc1
  DATA_DIR           Optional extra read-only data mount.
  TIDL82_NO_BUILD    Set to 1 to skip docker build.
EOF
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${TIDL82_IMAGE:-tidl82-onnx-compiler:08_02_00_01_rc1}"
DATA_DIR="${DATA_DIR:-}"
CONFIG_PATH="configs/compile_final.yaml"
CUSTOM_CMD=()

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "error: missing command after --" >&2
    usage >&2
    exit 2
  fi
  CUSTOM_CMD=("$@")
elif [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
  shift
  if [[ $# -gt 0 ]]; then
    echo "error: unexpected extra arguments: $*" >&2
    usage >&2
    exit 2
  fi
fi

DOCKER_CMD=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
  else
    cat >&2 <<'EOF'
error: Docker is not available for this user.

Fix one of these before running the TIDL compiler wrapper:
  sudo usermod -aG docker $USER
  # then log out and log in again

Or run the wrapper from an interactive shell where sudo docker works.
EOF
    exit 1
  fi
fi

if [[ "${TIDL82_NO_BUILD:-0}" != "1" ]]; then
  "${DOCKER_CMD[@]}" build -t "$IMAGE_NAME" "$PROJECT_DIR/docker/tidl82"
fi

RUN_ARGS=(
  run --rm
  -v "$PROJECT_DIR:/work"
  -w /work
  -e TIDL_TOOLS_PATH=/opt/tidl_tools
  -e LD_LIBRARY_PATH=/opt/tidl_tools
)

if [[ -n "$DATA_DIR" ]]; then
  RUN_ARGS+=( -v "$DATA_DIR:$DATA_DIR:ro" )
fi

# If the compile YAML uses an absolute calibration_dir outside the project,
# mount it read-only at the same path so the container can read calibration data.
if [[ ${#CUSTOM_CMD[@]} -eq 0 && -f "$PROJECT_DIR/$CONFIG_PATH" ]]; then
  CALIBRATION_DIR="$(python3 - "$PROJECT_DIR/$CONFIG_PATH" <<'PYHELPER'
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    sys.exit(0)
path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text()) or {}
compile_cfg = data.get("compile", {}) if isinstance(data, dict) else {}
value = compile_cfg.get("calibration_dir") if isinstance(compile_cfg, dict) else None
if value:
    print(value)
PYHELPER
)"
  if [[ -n "$CALIBRATION_DIR" && "$CALIBRATION_DIR" = /* && -d "$CALIBRATION_DIR" ]]; then
    RUN_ARGS+=( -v "$CALIBRATION_DIR:$CALIBRATION_DIR:ro" )
  fi
fi

if [[ ${#CUSTOM_CMD[@]} -eq 0 ]]; then
  CUSTOM_CMD=(python3 tools/compile_tidl.py --config "$CONFIG_PATH")
fi

set +e
"${DOCKER_CMD[@]}" "${RUN_ARGS[@]}" "$IMAGE_NAME" "${CUSTOM_CMD[@]}"
STATUS=$?
set -e

# TIDL tools are most reliable when run as root inside the container.
# Restore host ownership afterwards so the checkout stays editable.
"${DOCKER_CMD[@]}" run --rm -v "$PROJECT_DIR:/work" -w /work "$IMAGE_NAME" \
  chown -R "$(id -u):$(id -g)" /work >/dev/null 2>&1 || true

exit "$STATUS"
