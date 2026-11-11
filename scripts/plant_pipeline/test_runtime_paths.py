"""Regressions for NFS project paths, restart selection and verified reuse."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import Stages, code_signature, digest, read_json, write_json
from runtime_paths import (compatible_temporary_path_signatures, temporary_environment,
                           validate_unix_socket_directory)

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'maintenance'))
import plant_pipeline_bundle as bundle
import stop_plant_pipeline as stop


class TemporaryPathTests(unittest.TestCase):
    def test_real_server_root_old_temp_too_long_new_temp_fits(self):
        root = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
        with self.assertRaises(ValueError):
            validate_unix_socket_directory(root+'/tmp/plant_pipeline')
        validate_unix_socket_directory(root+'/t')

    def test_all_temp_variables_override_inherited_values_under_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = dict(TMPDIR='/wrong', TEMP='/wrong2', TMP='/wrong3', CUDA_VISIBLE_DEVICES='1')
            env = temporary_environment(root, source)
            for name in ('TMPDIR', 'TEMP', 'TMP'):
                self.assertEqual(env[name], str(root.resolve()/'t'))
            self.assertTrue((root/'t').is_dir())
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
            self.assertEqual(source['TMPDIR'], '/wrong')

    def test_reviewed_fix_reconstructs_exact_previous_deployed_code_signature(self):
        # A clean fixture uses exactly the LF payloads installed on Linux.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ('scripts/plant_pipeline', 'MAtCha/scripts'):
                (root/directory).mkdir(parents=True)
            names = ('common.py', 'run_pipeline.py', 'runtime_paths.py', 'check_ipc_runtime.py')
            for name in names:
                relative = 'scripts/plant_pipeline/'+name
                (root/relative).write_bytes(bundle.payload(PROJECT/relative))
            expected = code_signature(root, overrides={
                'scripts/plant_pipeline/common.py':'8bcc0699551b3edbc7ab6955bc43451552941081c46cd5fe21613b14322c3121',
                'scripts/plant_pipeline/run_pipeline.py':'eb695bb1a9f0af43f4d55c705b0151be8faf175e6dcbfbd1ee5c2af6d0b782e6'},
                omitted=('scripts/plant_pipeline/runtime_paths.py','scripts/plant_pipeline/check_ipc_runtime.py'))
            self.assertEqual(compatible_temporary_path_signatures(root), (expected,))
            # Any numerical-code edit changes the reconstructed predecessor,
            # so a saved pre-edit signature still cannot match.
            (root/'MAtCha/scripts/model.py').write_text('numerical change', encoding='utf-8')
            self.assertNotIn(expected, compatible_temporary_path_signatures(root))
            # A different orchestrator revision disables the exception outright.
            with (root/'scripts/plant_pipeline/run_pipeline.py').open('a', encoding='utf-8') as stream:
                stream.write('\n# unreviewed edit\n')
            self.assertEqual(compatible_temporary_path_signatures(root), ())

    def test_local_lock_checks_current_process_without_sending_signals(self):
        from common import JobLock
        self.assertTrue(JobLock._pid_alive(os.getpid()))
        self.assertFalse(JobLock._pid_alive(-1))
        with tempfile.TemporaryDirectory() as temp:
            with JobLock(Path(temp)/'test.lock'):
                with self.assertRaises(RuntimeError):
                    with JobLock(Path(temp)/'test.lock'):
                        self.fail('a second owner must be excluded')

    def test_dead_owner_reclaimed_without_removing_live_owner(self):
        from common import JobLock
        import socket
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.lock'
            directory = path.with_name('test.lock.d'); directory.mkdir()
            write_json(directory/'owner.json', dict(pid=123, hostname=socket.gethostname(),
                                                    acquired_epoch=0, token='old'))
            with patch.object(JobLock, '_pid_alive', return_value=False):
                with JobLock(path):
                    self.assertEqual(read_json(directory/'owner.json')['pid'], os.getpid())
            self.assertFalse(directory.exists())

    def test_runtime_only_resume_verifies_artifacts_and_keeps_previous_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = dict(input_sha256='photos', config_sha256='config', code_sha256='before')
            state = Stages(root, old)
            def action(directory):
                p = directory/'model.ply'; p.write_bytes(b'model')
                return dict(mesh=p)
            original = state.execute('01_sfm', action)
            new = dict(old, code_sha256='after')
            with self.assertRaises(ValueError):
                Stages(root, dict(new, input_sha256='different'), ('before',))
            updated = Stages(root, new, ('before',))
            self.assertEqual(updated.execute('01_sfm', lambda d: self.fail('must reuse')), original)
            backups = list(root.glob('pipeline_state.before_runtime_fix_*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(read_json(backups[0])['signature'], old)
            original['mesh'].write_bytes(b'tampered')
            saved = read_json(updated.path); saved['signature'] = old
            write_json(updated.path, saved)
            with self.assertRaisesRegex(ValueError, 'output changed'):
                Stages(root, new, ('before',))
            self.assertEqual(read_json(updated.path)['signature'], old)

    def test_stop_matches_exact_run_and_project_and_collects_only_descendants(self):
        root = Path('/project').resolve()
        script = str(root/'scripts/plant_pipeline/run_pipeline.py')
        rows = {
            10:dict(ppid=1, argv=['python', script, '--run', 'wanted']),
            11:dict(ppid=10, argv=['sh', 'worker']),
            12:dict(ppid=11, argv=['python', 'loader']),
            20:dict(ppid=1, argv=['python', script, '--run', 'wanted_other']),
            21:dict(ppid=20, argv=['python', 'other']),
            30:dict(ppid=1, argv=['python', '/other/run_pipeline.py', '--run', 'wanted']),
        }
        roots = stop.run_roots(rows, root, 'wanted')
        self.assertEqual(roots, {10})
        self.assertEqual(stop.descendants(rows, roots), {10, 11, 12})

    @unittest.skipUnless(os.name == 'posix', 'Real AF_UNIX sockets require Linux')
    def test_actual_reconstruction_ipc_primitive(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(PROJECT/'scripts/plant_pipeline/check_ipc_runtime.py')],
                                    env=temporary_environment(temp), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PIPELINE_IPC_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
