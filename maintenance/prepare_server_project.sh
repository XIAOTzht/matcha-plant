#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${PLANT_CONTACT_PYTHON:-$ROOT/envs/plant_semantics/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$PY" -u "$SCRIPT_DIR/server_project.py" --root "$ROOT" cleanup "$@"
