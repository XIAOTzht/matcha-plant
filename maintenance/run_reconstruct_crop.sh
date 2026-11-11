#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

if [[ "${1:-}" == --help || $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: bash maintenance/run_reconstruct_crop.sh DATA_NAME RUN_NAME [N_IMAGES=12] [unposed|posed] [seed-mode=auto]"
  echo "Example: bash maintenance/run_reconstruct_crop.sh plant_new plant_new 12 unposed auto"
  [[ "${1:-}" == --help ]] && exit 0 || exit 2
fi

DATA_NAME="$1"
RUN_NAME="$2"
N_IMAGES="${3:-12}"
SFM_MODE="${4:-unposed}"
SEED_MODE="${5:-auto}"

[[ "$DATA_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid DATA_NAME or RUN_NAME" >&2; exit 2; }
[[ "$SEED_MODE" == auto || "$SEED_MODE" == green || "$SEED_MODE" == flowers ]] || { echo "Invalid seed mode" >&2; exit 2; }

bash "$SCRIPT_DIR/run_reconstruction.sh" "$DATA_NAME" "$RUN_NAME" "$N_IMAGES" "$SFM_MODE"

MATCHA_ENV="$ROOT/envs/matcha"
PY="$MATCHA_ENV/bin/python"
OUT="$ROOT/outputs/$RUN_NAME"
MESH="$OUT/tetra_meshes/tetra_mesh_binary_search_7.ply"
CROP_OUT="$OUT/plant_auto_crop"
[[ -x "$PY" ]] || { echo "Missing Linux Python environment: $PY" >&2; exit 1; }
[[ -s "$MESH" ]] || { echo "Missing reconstructed mesh: $MESH" >&2; exit 1; }
[[ ! -e "$CROP_OUT" ]] || { echo "Crop output exists; choose a new RUN_NAME or remove it manually: $CROP_OUT" >&2; exit 1; }

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${RUN_NAME}_crop_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
export PYTHONUNBUFFERED=1
echo "Cropping environment fragments from: $MESH"
"$PY" "$ROOT/tools/plant_extract/auto_crop_mesh.py" \
  --input "$MESH" \
  --output-dir "$CROP_OUT" \
  --seed-mode "$SEED_MODE"

[[ -s "$CROP_OUT/plant.ply" ]] || { echo "Automatic crop did not produce plant.ply" >&2; exit 1; }
echo "RECONSTRUCT_AND_CROP_SUCCESS"
echo "Cropped plant: $CROP_OUT/plant.ply"
