#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ $# -lt 2 ]]; then
    echo 'Usage: bash run_server.sh INPUT_SEMANTIC_DIR NEW_OUTPUT_DIR [pipeline options]'
    echo 'Example options: --check-only --no-parts --no-previews --mm-per-unit 1'
    exit 2
fi
INPUT="$1"
OUTPUT="$2"
shift 2
PY="${PLANT_CONTACT_PYTHON:-$ROOT/envs/plant_semantics/bin/python}"
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/tmp/plant_contact" XDG_CACHE_HOME="$ROOT/cache/plant_contact"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROOT/logs"
exec "$PY" -u "$SCRIPT_DIR/run_pipeline.py" --input-dir "$INPUT" --output-dir "$OUTPUT" "$@"
