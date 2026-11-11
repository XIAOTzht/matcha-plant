#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PY="$ROOT/envs/matcha/bin/python"

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash tools/plant_extract/run_crop.sh [INPUT_MESH.ply] [additional Python options]"
    echo "Default input: outputs/plant_test_camera/tetra_meshes/tetra_mesh_binary_search_7.ply"
    exit 0
fi

INPUT_MESH="${1:-$ROOT/outputs/plant_test_camera/tetra_meshes/tetra_mesh_binary_search_7.ply}"
if [[ $# -gt 0 ]]; then shift; fi
[[ -x "$PY" ]] || { echo "Missing Linux Python: $PY"; exit 1; }
[[ -s "$INPUT_MESH" ]] || { echo "Missing mesh: $INPUT_MESH"; exit 1; }

"$PY" "$SCRIPT_DIR/auto_crop_mesh.py" --input "$INPUT_MESH" "$@"
