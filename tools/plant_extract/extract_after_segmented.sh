#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/plant_segmented_auto_extract.log") 2>&1
echo "Waiting for matcha_plant_segmented to finish..."
while tmux has-session -t '=matcha_plant_segmented' 2>/dev/null; do
    sleep 20
done
# Session disappearance alone is not success; require the training success marker.
grep -q '^PLANT_SEGMENTED_MODEL_SUCCESS$' "$ROOT/logs/plant_smoke_segmented.log" || {
    echo "Training success marker missing. Inspect plant_smoke_segmented.log; extraction was not started."
    exit 1
}
bash "$SCRIPT_DIR/run_extract.sh" \
    "$ROOT/outputs/plant_smoke_segmented" \
    "$ROOT/data/plant_smoke_segmented/masks"
