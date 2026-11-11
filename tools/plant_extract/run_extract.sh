#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
MATCHA_ENV="$ROOT/envs/matcha"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_extract.sh RUN_DIR MASKS_DIR [NEW_OUTPUT_DIR]"
    echo "For new images without masks, use auto_extract_plant.py --images-dir instead."
    exit 2
fi

RUN_DIR="$1"
MASKS_DIR="$2"
DEST="${3:-$RUN_DIR/plant_auto_$(date +%Y%m%d_%H%M%S)}"
MESH="$RUN_DIR/tetra_meshes/tetra_mesh_binary_search_7.ply"
[[ -s "$MESH" ]] || { echo "Missing final mesh: $MESH"; exit 1; }
[[ -d "$RUN_DIR/mast3r_sfm" ]] || { echo "Missing matching mast3r_sfm"; exit 1; }
[[ -d "$MASKS_DIR" ]] || { echo "Missing masks directory: $MASKS_DIR"; exit 1; }
mkdir -p "$DEST"
exec > >(tee -a "$DEST/extract.log") 2>&1

"$MATCHA_ENV/bin/python" "$SCRIPT_DIR/auto_extract_plant.py" \
    --mesh "$MESH" \
    --sfm-dir "$RUN_DIR/mast3r_sfm" \
    --masks-dir "$MASKS_DIR" \
    --output-dir "$DEST"
