"""Check portable numerical dependencies, robot assets and offscreen rendering."""
from pathlib import Path
import argparse
from contextlib import contextmanager
import faulthandler
import importlib
import os
import sys
import time


@contextmanager
def step(name):
    started = time.monotonic()
    print('CHECK_START', name, flush=True)
    yield
    print('CHECK_OK', name, f'{time.monotonic()-started:.2f}s', flush=True)


def check(render_only=False, no_render=False):
    print('RUNTIME_CHECK_START', sys.version.split()[0],
          'backend='+os.environ.get('MUJOCO_GL', 'default'), flush=True)
    modules = {}
    names = ('numpy', 'mujoco') if render_only else (
        'numpy', 'scipy', 'mujoco', 'plyfile', 'fast_simplification', 'igl', 'PIL')
    for name in names:
        with step('import_'+name):
            modules[name] = importlib.import_module(name)
    mujoco = modules['mujoco']
    if render_only:
        # Backend selection must not wait for robot meshes or unrelated packages
        # on a slow network filesystem. Full asset checks still run in preflight.
        with step('compile_render_fixture'):
            model = mujoco.MjModel.from_xml_string('''<mujoco>
              <worldbody><light pos="0 -1 2"/>
                <geom type="plane" size="1 1 .1" rgba=".6 .6 .6 1"/>
                <geom type="sphere" size=".1" pos="0 0 .1" rgba="1 .1 .1 1"/>
              </worldbody></mujoco>''')
    else:
        root = Path(__file__).resolve().parents[1]
        with step('compile_so101_robot'):
            model = mujoco.MjModel.from_xml_path(str(root/'assets/robot/robotstudio_so101/so101.xml'))
    with step('forward_model'):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    if not no_render:
        with step('create_renderer'):
            renderer = mujoco.Renderer(model, width=160, height=120)
        try:
            with step('render_image'):
                renderer.update_scene(data)
                pixels = renderer.render()
                if pixels.shape != (120, 160, 3) or pixels.std() == 0:
                    raise RuntimeError('Renderer returned an empty or invalid image')
        finally:
            with step('close_renderer'):
                renderer.close()
        print('OFFSCREEN_RENDER_OK', flush=True)
    marker = 'RENDER_PROBE_OK' if render_only else 'SIMULATION_RUNTIME_OK'
    print(marker, sys.version.split()[0], 'mujoco', mujoco.__version__, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--no-render', action='store_true')
    mode.add_argument('--render-only', action='store_true', help='Render a tiny fixture; skip robot assets and unrelated imports')
    ap.add_argument('--traceback-after', type=float, default=0,
                    help='Print repeating Python stack traces after this many seconds (0 disables)')
    args = ap.parse_args()
    if args.traceback_after < 0:
        ap.error('--traceback-after must be nonnegative')
    if sys.version_info < (3, 11):
        raise RuntimeError('Planning requires Python >= 3.11; use Python 3.12')
    faulthandler.enable()
    if args.traceback_after:
        faulthandler.dump_traceback_later(args.traceback_after, repeat=True)
    try:
        check(args.render_only, args.no_render)
    finally:
        if args.traceback_after:
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
