#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TMPDIR="$ROOT/tmp/plant_pipeline_setup" PIP_CACHE_DIR="$ROOT/cache/pip"
export XDG_CACHE_HOME="$ROOT/cache/plant_pipeline"
export CONDA_PKGS_DIRS="$ROOT/cache/conda_pkgs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$ROOT/logs"
command -v tmux >/dev/null || { echo 'tmux is required to run uploaded jobs in the background.' >&2; exit 1; }
SIM="$ROOT/envs/mujoco"
if [[ -z "${SIM_PYTHON:-}" && ! -x "$SIM/bin/python" ]]; then
    if [[ -e "$SIM" ]]; then
        echo "Existing non-Linux/incomplete environment preserved: $SIM. Set SIM_PYTHON to a valid Linux environment, or choose a fresh path." >&2
        exit 1
    fi
    BASE="${BASE_PYTHON:-}"
    if [[ -z "$BASE" ]] && command -v python3.12 >/dev/null; then BASE="$(command -v python3.12)"; fi
    if [[ -n "$BASE" ]]; then
        "$BASE" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.12 recommended"'
        "$BASE" -m venv "$SIM"
    elif command -v conda >/dev/null; then
        conda create -y -p "$SIM" python=3.12 pip
    else
        echo 'Need Python 3.12 or conda. Run: BASE_PYTHON=/absolute/path/to/python3.12 bash maintenance/setup_plant_pipeline.sh' >&2
        exit 1
    fi
fi
PY="${SIM_PYTHON:-$SIM/bin/python}"
"$PY" -m pip install -r "$ROOT/simulation/requirements-linux.txt"
if ! "$PY" "$ROOT/simulation/scripts/configure_server_render.py"; then
    if command -v conda >/dev/null && [[ "${INSTALL_RENDER_LIBS:-1}" == 1 ]]; then
        RENDER_LIBS="$ROOT/envs/mujoco_render_libs"
        echo "Installing CPU headless rendering libraries inside $RENDER_LIBS (no sudo or driver changes)."
        if [[ -f "$RENDER_LIBS/conda-meta/history" ]]; then
            conda install -y -p "$RENDER_LIBS" --override-channels -c conda-forge 'mesalib=24.3.4' libglvnd
        elif [[ ! -e "$RENDER_LIBS" ]]; then
            conda create -y -p "$RENDER_LIBS" --override-channels -c conda-forge 'mesalib=24.3.4' libglvnd
        else
            echo "Existing unrecognized directory preserved: $RENDER_LIBS" >&2
            exit 1
        fi
        "$PY" "$ROOT/simulation/scripts/configure_server_render.py"
    else
        echo "Headless renderer still unavailable. Inspect $ROOT/logs/render_setup/ and install project-local Mesa or repair EGL." >&2
        exit 1
    fi
fi
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/scripts/plant_pipeline/run_pipeline.py" --check-runtime --deep-check
echo 'PLANT_PIPELINE_SETUP_OK'
