#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/tidl82_export_compile.sh [COMPILE_CONFIG_PATH]

Example:
  scripts/tidl82_export_compile.sh models/square_seg_32x13/configs/compile.yaml
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "error: expected at most one argument" >&2
  usage >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILE_CONFIG="${1:-configs/compile.yaml}"
COMPILE_ABS="$PROJECT_DIR/$COMPILE_CONFIG"

if [[ ! -f "$COMPILE_ABS" ]]; then
  echo "error: compile config not found: $COMPILE_CONFIG" >&2
  exit 1
fi

CONFIG_DIR="$(cd "$(dirname "$COMPILE_ABS")" && pwd)"
EXPORT_CONFIG="$CONFIG_DIR/export.yaml"
EXPORT_SCRIPT="$(cd "$CONFIG_DIR/.." && pwd)/src/export_onnx.py"

if [[ ! -f "$EXPORT_CONFIG" ]]; then
  echo "error: export config not found: $EXPORT_CONFIG" >&2
  exit 1
fi

if [[ ! -f "$EXPORT_SCRIPT" ]]; then
  echo "error: export script not found: $EXPORT_SCRIPT" >&2
  exit 1
fi

echo "[1/2] Exporting ONNX"
python3 "$EXPORT_SCRIPT" --config "$EXPORT_CONFIG"

echo "[2/2] Compiling TIDL artifacts"
"$PROJECT_DIR/scripts/tidl82_onnx_compile.sh" "$COMPILE_CONFIG"
