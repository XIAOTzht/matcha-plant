#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
ENV="${PLANTSEM_ENV:-$ROOT/envs/plant_semantics}"
PY="$ENV/bin/python"
INPUT="${1:?Usage: bash run_v3_server.sh INPUT_MESH OUTPUT_DIR [UP_AXIS] [EXTRA_VIEWS] [GPU_ID]}"
OUTPUT="${2:?Usage: bash run_v3_server.sh INPUT_MESH OUTPUT_DIR [UP_AXIS] [EXTRA_VIEWS] [GPU_ID]}"
UP_AXIS="${3:-y}"
EXTRA_VIEWS="${4:-12}"
GPU_ID="${5:-1}"

[[ -x "$PY" ]] || { echo "Missing environment: $PY" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing input mesh: $INPUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/mplconfig_plant_semantics}"
export PLANTSEG_SERVER=1
mkdir -p "$OUTPUT" "$ROOT/logs" "$MPLCONFIGDIR"

"$PY" "$SCRIPT_DIR/check_install_server.py"
"$PY" "$SCRIPT_DIR/estimate_axis.py" --input "$INPUT" >/dev/null

BASE="$OUTPUT/segmentation"
"$PY" "$SCRIPT_DIR/plantseg.py" run --input "$INPUT" --output "$BASE/baseline" --up "$UP_AXIS" --device cuda --threads 8
for stage in prepare predict fuse; do
  "$PY" "$SCRIPT_DIR/refine_detail.py" "$stage" --source "$BASE/baseline" --output "$BASE/detail" --sam "$SCRIPT_DIR/models/mobile_sam_parallel.pt" --device cuda --threads 8
done
"$PY" "$SCRIPT_DIR/complete_five.py" run --base "$BASE/baseline" --detail "$BASE/detail" --output "$BASE/complete_before_boundary" --sam "$SCRIPT_DIR/models/mobile_sam_parallel.pt" --threads 8
"$PY" "$SCRIPT_DIR/refine_leaf_boundary.py" --source "$BASE/complete_before_boundary" --output "$BASE/complete_before_rim" --sam "$SCRIPT_DIR/models/mobile_sam_parallel.pt" --threads 8
"$PY" "$SCRIPT_DIR/refine_leaf_rim.py" --source "$BASE/complete_before_rim" --output "$BASE/complete_before_uncertainty" --threads 8
"$PY" "$SCRIPT_DIR/reassess_uncertainty.py" --source "$BASE/complete_before_uncertainty" --instances "$BASE/complete_before_boundary" --output "$BASE/complete"
"$PY" "$SCRIPT_DIR/validate_complete.py" --input "$INPUT" --output "$BASE/complete"

FINAL="$BASE/complete"
if [[ "$EXTRA_VIEWS" -gt 0 ]]; then
  "$PY" "$SCRIPT_DIR/extra_view_review.py" run --source "$FINAL" --output "$OUTPUT/extra_views" --views "$EXTRA_VIEWS" --size 896 --threads 8 --device cuda --sam "$SCRIPT_DIR/models/mobile_sam_parallel.pt"
  "$PY" "$SCRIPT_DIR/validate_complete.py" --input "$INPUT" --output "$OUTPUT/extra_views/complete"
  FINAL="$OUTPUT/extra_views/complete"
fi
printf 'semantic_mesh=%s\nlabels=%s\n' "$FINAL/semantic_mesh.ply" "$FINAL/labels.npz" | tee "$OUTPUT/result_paths.txt"
echo PLANTSEM_SERVER_SUCCESS
