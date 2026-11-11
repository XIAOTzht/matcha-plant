#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != "--apply" ]]; then
    echo 'Usage: bash maintenance/finish_server.sh --apply'
    echo 'Build latest weights, verify backup, then remove retired content.'
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY="${PLANT_CONTACT_PYTHON:-$ROOT/envs/plant_semantics/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/backups"
LOCK_DIR="$ROOT/backups/.finish_server.lockdir"
if ! mkdir -- "$LOCK_DIR"; then
    if [[ -d "$LOCK_DIR" ]]; then
        echo "Final workflow lock directory exists: $LOCK_DIR" >&2
        [[ ! -f "$LOCK_DIR/owner.txt" ]] || cat "$LOCK_DIR/owner.txt" >&2
        echo 'Check the recorded task before removing a stale lock directory.' >&2
    else
        echo "Cannot create workflow lock directory: $LOCK_DIR; check permissions/storage." >&2
    fi
    exit 1
fi
release_workflow_lock() {
    rm -f -- "$LOCK_DIR/owner.txt"
    rmdir -- "$LOCK_DIR" || true
}
trap release_workflow_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'host=%s\npid=%s\nstarted=%s\n' "$(hostname)" "$$" "$(date -Is)" > "$LOCK_DIR/owner.txt"
LOG="$ROOT/backups/finish_server_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "FINAL_WORKFLOW_FAILED: inspect $LOG (line $LINENO)"' ERR
echo "Final workflow log: $LOG"
bash "$SCRIPT_DIR/check_project_layout.sh"
"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'maintenance'))
from server_project import active
jobs = active(root)
if jobs:
    raise SystemExit('Stop project jobs first: '+json.dumps(jobs))
PY
bash "$ROOT/scripts/plant_contact/run_latest_server.sh" --check-only
bash "$ROOT/scripts/plant_contact/run_latest_server.sh"
"$PY" -u "$SCRIPT_DIR/final_organize.py" --root "$ROOT" --apply
bash "$SCRIPT_DIR/check_project_layout.sh"
echo 'FINAL_SERVER_WORKFLOW_READY'
cat "$ROOT/outputs/with_car_20261021/latest_contact_paths.txt"
