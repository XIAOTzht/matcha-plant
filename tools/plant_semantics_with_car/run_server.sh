#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
if [[ $# -lt 4 || "${1:-}" == --help ]]; then
    echo 'Usage: bash run_server.sh ORIGINAL_CROP.ply METRIC.ply SCALE_REPORT.json OUTPUT_DIR [GPU_ID] [EXTRA_VIEWS]'
    exit 2
fi
INPUT="$1"
METRIC="$2"
SCALE_REPORT="$3"
OUTPUT="$4"
GPU_ID="${5:-1}"
EXTRA_VIEWS="${6:-12}"
ENV_DIR="${PLANTSEM_ENV:-$ROOT/envs/plant_semantics}"
PY="$ENV_DIR/bin/python"
[[ -x "$PY" ]] || { echo "Missing Python: $PY"; exit 1; }
[[ "$EXTRA_VIEWS" =~ ^[0-9]+$ ]] && (( EXTRA_VIEWS <= 24 )) || { echo 'Extra views must be 0..24'; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ROOT_OVERRIDE="$ROOT" PLANTSEM_ENV="$ENV_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID"
export TMPDIR="$ROOT/tmp/plant_semantics_with_car"
export XDG_CACHE_HOME="$ROOT/cache/plant_semantics_with_car"
export MPLCONFIGDIR="$ROOT/cache/plant_semantics_with_car/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$ROOT/logs"
trap 'echo "WITH_CAR_SEMANTICS_FAILED line=$LINENO exit=$?" >&2' ERR
[[ ! -e "$OUTPUT/complete" ]] || { echo 'Final directory already exists; inspect it or choose a fresh output root.'; exit 1; }
"$PY" "$SCRIPT_DIR/with_car_semantics.py" prepare --input "$INPUT" --metric "$METRIC" \
    --scale-report "$SCALE_REPORT" --output "$OUTPUT/prepared"
echo 'Stage 2/3: V3 plant-only segmentation on an upright model'
bash "$ROOT/tools/plant_semantics/run_v3_server.sh" "$OUTPUT/prepared/plant_upright.ply" \
    "$OUTPUT/plant_v3" y "$EXTRA_VIEWS" "$GPU_ID"
FINAL="$OUTPUT/plant_v3/segmentation/complete"
if (( EXTRA_VIEWS > 0 )); then FINAL="$OUTPUT/plant_v3/extra_views/complete"; fi
echo 'Stage 3/3: merge plant labels and cart rule into the original millimetre mesh'
"$PY" "$SCRIPT_DIR/with_car_semantics.py" merge --prepared "$OUTPUT/prepared" \
    --plant-result "$FINAL" --output "$OUTPUT/complete"
