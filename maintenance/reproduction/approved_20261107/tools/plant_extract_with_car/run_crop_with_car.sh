#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ $# -lt 2 || "${1:-}" == --help ]]; then
    echo 'Usage: bash run_crop_with_car.sh INPUT.ply OUTPUT_DIRECTORY [additional options]'
    echo 'Uses envs/plant_semantics/bin/python. Override with CROP_PYTHON if needed.'
    if [[ "${1:-}" == --help ]]; then exit 0; else exit 2; fi
fi
INPUT_MESH="$1"
OUTPUT_DIRECTORY="$2"
shift 2
PY="${CROP_PYTHON:-$ROOT/envs/plant_semantics/bin/python}"
[[ -x "$PY" ]] || { echo "Missing Python: $PY"; exit 1; }
[[ -s "$INPUT_MESH" ]] || { echo "Missing mesh: $INPUT_MESH"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$SCRIPT_DIR/auto_crop_with_car.py" --input "$INPUT_MESH" --output-dir "$OUTPUT_DIRECTORY" "$@"
