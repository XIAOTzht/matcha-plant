#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
if [[ "$#" -lt 2 || "$#" -gt 7 || "${1:-}" == --help ]]; then
  echo "Usage: bash maintenance/run_reconstruct_crop_semantics.sh DATA_NAME RUN_NAME [N_IMAGES=12] [unposed|posed] [seed-mode=auto] [extra_views=12] [GPU_ID=1]"
  exit 2
fi
DATA_NAME="$1"; RUN_NAME="$2"; N_IMAGES="${3:-12}"; SFM_MODE="${4:-unposed}"
SEED_MODE="${5:-auto}"; EXTRA_VIEWS="${6:-12}"; GPU_ID="${7:-1}"
bash "$SCRIPT_DIR/run_reconstruct_crop.sh" "$DATA_NAME" "$RUN_NAME" "$N_IMAGES" "$SFM_MODE" "$SEED_MODE"
MESH="$ROOT/outputs/$RUN_NAME/plant_auto_crop/plant.ply"
SEM_OUT="$ROOT/outputs/$RUN_NAME/semantic_v3"
[[ -s "$MESH" ]] || { echo "Missing cropped mesh: $MESH" >&2; exit 1; }
ROOT_OVERRIDE="$ROOT" PLANTSEM_ENV="${PLANTSEM_ENV:-$ROOT/envs/plant_semantics}" \
  bash "$ROOT/tools/plant_semantics/run_v3_server.sh" "$MESH" "$SEM_OUT" y "$EXTRA_VIEWS" "$GPU_ID"
echo "RECONSTRUCT_CROP_SEMANTICS_SUCCESS"
echo "Semantic mesh: $SEM_OUT/extra_views/complete/semantic_mesh.ply"
