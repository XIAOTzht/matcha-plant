#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
BASE="${1:-$ROOT/outputs/plant_turntable30_20261016_163741}"
TOOLS="$ROOT/tools/plant_extract"
DEST="$BASE/plus/plant_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/logs"
exec > >(tee "$ROOT/logs/turntable30_cleanup.log") 2>&1
echo "Cleanup output: $DEST"
"$ROOT/envs/matcha/bin/python" "$TOOLS/clean_turntable_mesh.py" \
    --run-dir "$BASE/plus" --input-dir "$BASE/input_plus" --output-dir "$DEST"
