#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
ENV="${PLANTSEM_ENV:-$ROOT/envs/plant_semantics}"
PIP_CACHE="${PIP_CACHE_DIR:-$ROOT/cache/pip_plant_semantics}"
TMP="${TMPDIR:-$ROOT/tmp/plant_semantics}"

command -v conda >/dev/null || { echo 'conda is required' >&2; exit 1; }
mkdir -p "$PIP_CACHE" "$TMP" "$ROOT/logs"
export PIP_CACHE_DIR="$PIP_CACHE" TMPDIR="$TMP"

if [[ -d "$ENV" && ! -x "$ENV/bin/python" ]]; then
  rm -rf -- "$ENV"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  conda create -y -p "$ENV" python=3.10 pip
fi
PY="$ENV/bin/python"

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.0+cu118 torchvision==0.22.0+cu118
"$PY" -m pip install -r "$SCRIPT_DIR/requirements_server.txt"
"$PY" "$SCRIPT_DIR/check_install_server.py"
echo "PLANTSEM_SERVER_SETUP_OK"
echo "Environment: $ENV"
