#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

if [[ "${1:-}" == --help || $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash maintenance/run_reconstruction.sh DATA_NAME RUN_NAME [N_IMAGES=12] [unposed|posed]"
  echo "Example: bash maintenance/run_reconstruction.sh plant_new plant_new 13 unposed"
  echo "Set GPU_ID=1 (default), ROOT_OVERRIDE=/project/root if needed."
  [[ "${1:-}" == --help ]] && exit 0 || exit 2
fi
DATA_NAME="$1"
RUN_NAME="$2"
N_IMAGES="${3:-12}"
SFM_MODE="${4:-unposed}"
[[ "$DATA_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || {
  echo "DATA_NAME and RUN_NAME must use letters, numbers, underscores or hyphens"; exit 2;
}
[[ "$N_IMAGES" =~ ^[0-9]+$ && "$N_IMAGES" -ge 3 ]] || { echo "N_IMAGES must be >=3"; exit 2; }
[[ "$SFM_MODE" == unposed || "$SFM_MODE" == posed ]] || { echo "Invalid SFM mode"; exit 2; }
MATCHA_ENV="$ROOT/envs/matcha"
PY="$MATCHA_ENV/bin/python"
DATA="$ROOT/data/$DATA_NAME"
OUT="$ROOT/outputs/$RUN_NAME"
[[ -x "$PY" ]] || { echo "Linux Python environment not available: $PY"; exit 1; }
[[ -d "$DATA/images" ]] || { echo "Missing images folder: $DATA/images"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists; choose a new RUN_NAME: $OUT"; exit 1; }
if [[ "$SFM_MODE" == posed ]]; then
  [[ -s "$DATA/sparse/0/cameras.bin" && -s "$DATA/sparse/0/images.bin" ]] || {
    echo "Posed input requires sparse/0/cameras.bin and images.bin"; exit 1;
  }
fi
mkdir -p "$ROOT/logs" "$ROOT/mplconfig" "$ROOT/tmp" "$OUT"
LOG="$ROOT/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
export PATH="$MATCHA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$ROOT/mplconfig"
export TMPDIR="$ROOT/tmp"
echo "Started: $(date -Is); source: $DATA; output: $OUT; physical GPU: $CUDA_VISIBLE_DEVICES"
cd "$ROOT/MAtCha"
git rev-parse HEAD > "$OUT/code_commit.txt" 2>/dev/null || true
git diff > "$OUT/code_changes.patch" 2>/dev/null || true
cp 2d-gaussian-splatting/scene/gaussian_model.py "$OUT/gaussian_model_used.py"
"$PY" train.py --source_path "$DATA" --output_path "$OUT" \
  --sfm_config "$SFM_MODE" --n_images "$N_IMAGES" \
  --free_gaussians_config default --tetra_downsample_ratio 0.5
if grep -Eq '^Traceback \(most recent call last\):|^RuntimeError:|^MemoryError:' "$LOG"; then
  echo "Child process error detected. Inspect $LOG"; exit 1
fi
MESH="$OUT/tetra_meshes/tetra_mesh_binary_search_7.ply"
"$PY" - "$MESH" <<'PY'
import sys
from plyfile import PlyData
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file() or not p.stat().st_size:raise RuntimeError(f"Missing mesh: {p}")
mesh=PlyData.read(str(p), known_list_len={'face':{'vertex_indices':3}})
if not len(mesh['vertex']) or not len(mesh['face']):raise RuntimeError('Empty mesh')
print('MESH_VERIFIED:',len(mesh['vertex']),'vertices',len(mesh['face']),'faces')
PY
echo "RECONSTRUCTION_SUCCESS: $MESH"
echo "Elapsed seconds: $SECONDS"
