"""A passing probe must survive a new SSH/tmux shell; failures cannot be saved."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import read_json, write_json
from render_runtime import CONFIG, render_environment

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'simulation/scripts'))
import configure_server_render as setup


class RenderRuntimeTests(unittest.TestCase):
    def test_saved_osmesa_overrides_stale_shell_egl_platform_and_device(self):
        with tempfile.TemporaryDirectory() as temp:
            write_json(Path(temp)/CONFIG,dict(status='OFFSCREEN_RENDER_VERIFIED',
                       environment=dict(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',
                                        LD_LIBRARY_PATH='/project/render/lib')))
            env=render_environment(temp,dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                MUJOCO_EGL_DEVICE_ID='7',__EGL_VENDOR_LIBRARY_FILENAMES='/wrong.json',
                LD_LIBRARY_PATH='/other/lib',CUDA_VISIBLE_DEVICES='1',PATH='/bin'))
            self.assertEqual(env['MUJOCO_GL'],'osmesa')
            self.assertEqual(env['PYOPENGL_PLATFORM'],'osmesa')
            self.assertEqual(env['LD_LIBRARY_PATH'],'/project/render/lib')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
            self.assertEqual(env['PATH'],'/bin')
            self.assertNotIn('MUJOCO_EGL_DEVICE_ID',env)
            self.assertNotIn('__EGL_VENDOR_LIBRARY_FILENAMES',env)

    def test_invalid_and_unverified_configuration_cannot_start(self):
        for saved in (dict(status='FAILED',environment=dict(MUJOCO_GL='osmesa')),
                      dict(status='OFFSCREEN_RENDER_VERIFIED',environment=dict(MUJOCO_GL='bad')),
                      dict(status='OFFSCREEN_RENDER_VERIFIED',environment=dict(MUJOCO_GL='egl',PATH='/unexpected'))):
            with self.subTest(saved=saved),tempfile.TemporaryDirectory() as temp:
                write_json(Path(temp)/CONFIG,saved)
                with self.assertRaises(ValueError):render_environment(temp,{})

    def test_probe_falls_back_and_saves_only_rendered_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            calls=[]
            def runner(argv,**kwargs):
                calls.append(kwargs['env'])
                self.assertIn('--render-only',argv)
                self.assertIn('-u',argv)
                if kwargs['env']['MUJOCO_GL']=='egl':
                    kwargs['stdout'].write('Cannot initialize EGL\n')
                    return SimpleNamespace(returncode=1)
                kwargs['stdout'].write('OFFSCREEN_RENDER_OK\nRENDER_PROBE_OK\n')
                return SimpleNamespace(returncode=0)
            with patch.object(setup,'candidates',return_value=iter([
                    ('egl',dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')),
                    ('osmesa',dict(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa'))])):
                self.assertTrue(setup.configure(temp,runner))
            self.assertEqual(len(calls),2)
            self.assertEqual(read_json(Path(temp)/CONFIG)['backend'],'osmesa')
            self.assertEqual(render_environment(temp,{})['MUJOCO_GL'],'osmesa')

    def test_probe_failure_never_creates_verified_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            def runner(argv,**kwargs):
                kwargs['stdout'].write('failure\n')
                return SimpleNamespace(returncode=1)
            with patch.object(setup,'candidates',return_value=iter([('egl',dict(MUJOCO_GL='egl'))])):
                self.assertFalse(setup.configure(temp,runner))
            self.assertFalse((Path(temp)/CONFIG).exists())
            self.assertEqual(read_json(Path(temp)/'logs/render_setup/failed_probes.json')['status'],'NO_WORKING_RENDERER')

    def test_zero_exit_without_render_marker_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            def runner(argv,**kwargs):
                kwargs['stdout'].write('SIMULATION_RUNTIME_OK\n')
                return SimpleNamespace(returncode=0)
            with patch.object(setup,'candidates',return_value=iter([('egl',dict(MUJOCO_GL='egl'))])):
                self.assertFalse(setup.configure(temp,runner))
            self.assertFalse((Path(temp)/CONFIG).exists())

    def test_timeouts_are_recorded_then_next_backend_is_tried(self):
        with tempfile.TemporaryDirectory() as temp:
            def runner(argv,**kwargs):
                self.assertEqual(kwargs['timeout'],300)
                if kwargs['env']['MUJOCO_GL']=='egl':
                    kwargs['stdout'].write('CHECK_START create_renderer\n');kwargs['stdout'].flush()
                    raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
                kwargs['stdout'].write('OFFSCREEN_RENDER_OK\nRENDER_PROBE_OK\n')
                return SimpleNamespace(returncode=0)
            with patch.object(setup,'candidates',return_value=iter([
                    ('egl',dict(MUJOCO_GL='egl')),('osmesa',dict(MUJOCO_GL='osmesa'))])):
                self.assertTrue(setup.configure(temp,runner))
            log=(Path(temp)/'logs/render_setup/01_egl.log').read_text(encoding='utf-8')
            self.assertIn('CHECK_START create_renderer',log)
            self.assertIn('RENDER_PROBE_TIMEOUT after 300s',log)

    def test_timeout_after_partial_success_cannot_save_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            def runner(argv,**kwargs):
                kwargs['stdout'].write('OFFSCREEN_RENDER_OK\nRENDER_PROBE_OK\n')
                raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
            with patch.object(setup,'candidates',return_value=iter([('osmesa',dict(MUJOCO_GL='osmesa'))])):
                self.assertFalse(setup.configure(temp,runner,timeout=600))
            self.assertFalse((Path(temp)/CONFIG).exists())

    def test_invalid_timeout_rejected_before_running(self):
        for timeout in (0,-1,float('inf'),float('nan')):
            with self.subTest(timeout=timeout),tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):setup.configure(temp,timeout=timeout)


if __name__=='__main__':unittest.main()
