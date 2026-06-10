#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/tidl82_export_compile.sh [COMPILE_CONFIG_PATH]
  scripts/tidl82_export_compile.sh --compile-config <path> [--export-config <path>]

Examples:
  scripts/tidl82_export_compile.sh models/square_seg_32x13/configs/compile.yaml
  scripts/tidl82_export_compile.sh     --export-config models/square_seg_32x13/configs/export.yaml     --compile-config models/square_seg_32x13/configs/compile.yaml

Environment variables:
  PYTHON_BIN         Host Python used for ONNX export. Default: python3
  TIDL82_IMAGE       Forwarded to tidl82_onnx_compile.sh
  DATA_DIR           Forwarded to tidl82_onnx_compile.sh
  TIDL82_NO_BUILD    Forwarded to tidl82_onnx_compile.sh
EOF
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMPILE_CONFIG=""
EXPORT_CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --compile-config)
      [[ $# -ge 2 ]] || { echo "error: missing value for --compile-config" >&2; exit 2; }
      COMPILE_CONFIG="$2"
      shift 2
      ;;
    --export-config)
      [[ $# -ge 2 ]] || { echo "error: missing value for --export-config" >&2; exit 2; }
      EXPORT_CONFIG="$2"
      shift 2
      ;;
    *)
      if [[ -n "$COMPILE_CONFIG" ]]; then
        echo "error: unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      COMPILE_CONFIG="$1"
      shift
      ;;
  esac
done

if [[ -z "$COMPILE_CONFIG" ]]; then
  COMPILE_CONFIG="configs/compile.yaml"
fi

COMPILE_ABS="$PROJECT_DIR/$COMPILE_CONFIG"
if [[ ! -f "$COMPILE_ABS" ]]; then
  echo "error: compile config not found: $COMPILE_CONFIG" >&2
  exit 1
fi

if [[ -z "$EXPORT_CONFIG" ]]; then
  EXPORT_CONFIG="$(python3 - "$COMPILE_ABS" "$PROJECT_DIR" <<'PYHELPER'
import sys
from pathlib import Path
compile_abs = Path(sys.argv[1]).resolve()
project_dir = Path(sys.argv[2]).resolve()
export_abs = compile_abs.with_name('export.yaml')
print(export_abs.relative_to(project_dir))
PYHELPER
)"
fi

EXPORT_ABS="$PROJECT_DIR/$EXPORT_CONFIG"
if [[ ! -f "$EXPORT_ABS" ]]; then
  echo "error: export config not found: $EXPORT_CONFIG" >&2
  exit 1
fi

EXPORT_SCRIPT="$(python3 - "$EXPORT_ABS" <<'PYHELPER'
import sys
from pathlib import Path
export_abs = Path(sys.argv[1]).resolve()
print(export_abs.parent.parent / 'src' / 'export_onnx.py')
PYHELPER
)"

if [[ ! -f "$EXPORT_SCRIPT" ]]; then
  echo "error: export script not found for config: $EXPORT_CONFIG" >&2
  exit 1
fi

echo "[1/2] Exporting ONNX with $EXPORT_CONFIG"
"$PYTHON_BIN" "$EXPORT_SCRIPT" --config "$EXPORT_ABS"

echo "[2/2] Compiling TIDL artifacts with $COMPILE_CONFIG"
"$PROJECT_DIR/scripts/tidl82_onnx_compile.sh" "$COMPILE_CONFIG"
