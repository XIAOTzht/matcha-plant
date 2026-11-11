"""Persist a verified renderer without depending on the interactive SSH shell."""
from __future__ import annotations
import os
from pathlib import Path
from common import read_json

VARIABLES = {'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'MUJOCO_EGL_DEVICE_ID',
             '__EGL_VENDOR_LIBRARY_FILENAMES', 'LD_LIBRARY_PATH', 'LIBGL_ALWAYS_SOFTWARE'}
CONFIG = 'configs/server_render_runtime.json'


def render_environment(root, base=None):
    source = os.environ if base is None else base
    env = dict(source)
    path = Path(root)/CONFIG
    if path.is_file():
        saved = read_json(path)
        if saved.get('status') != 'OFFSCREEN_RENDER_VERIFIED':
            raise ValueError(f'Render configuration was not verified: {path}')
        values = saved.get('environment', {})
        if set(values)-VARIABLES or values.get('MUJOCO_GL') not in {'egl','osmesa','glfw'}:
            raise ValueError(f'Invalid renderer configuration: {path}')
        # Do not leak an incompatible platform/device from a tmux server shell.
        for key in VARIABLES:
            env.pop(key, None)
        for key,value in values.items():
            if not isinstance(value,str) or '\0' in value or '\n' in value:
                raise ValueError(f'Invalid renderer environment value: {key}')
        env.update(values)
        if 'LD_LIBRARY_PATH' not in values and source.get('LD_LIBRARY_PATH'):
            env['LD_LIBRARY_PATH'] = source['LD_LIBRARY_PATH']
    else:
        env.setdefault('MUJOCO_GL', 'egl' if os.name=='posix' else 'glfw')
    return env
