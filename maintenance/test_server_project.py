"""Maintenance tests use tiny isolated projects; never alter actual datasets."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('server_project', HERE/'server_project.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class MaintenanceTests(unittest.TestCase):
    def fixture(self, root):
        for rel in m.REQUIRED:
            p = root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('keep', encoding='utf-8')
        for rel in ('tools/contact_weights/__pycache__/test.pyc', 'logs/job.log',
                    'tmp/pending.part', 'cache/model.pth',
                    'envs/plant_semantics/lib/__pycache__/runtime.pyc',
                    'outputs/scene/training.pt', 'incoming/resume.partial',
                    'MAtCha/checkpoints/large.pth', 'MAtCha/train_extension.cu',
                    'tools/contact_weights/README.md'):
            p = root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'keep')

    def test_preview_then_cleanup_preserves_dependencies_and_results(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            m.cleanup(root)
            after={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            m.cleanup(root,True)
            self.assertFalse((root/'tools/contact_weights/__pycache__').exists())
            for rel in ('envs/plant_semantics/lib/__pycache__/runtime.pyc',
                        'cache/model.pth','outputs/scene/training.pt','incoming/resume.partial'):
                self.assertEqual((root/rel).read_bytes(),b'keep')
            backup=next((root/'backups').glob('organization_*'))
            self.assertEqual((backup/'archived/logs/job.log').read_bytes(),b'keep')
            self.assertEqual((backup/'archived/tmp/pending.part').read_bytes(),b'keep')
            self.assertTrue((root/'logs').is_dir())
            self.assertTrue((backup/'CLEANUP_SUCCESS').is_file())

    def test_export_excludes_runtime_data_and_keeps_vendor_source(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            out=m.export(root)
            self.assertTrue((out/'MAtCha/train.py').is_file())
            self.assertTrue((out/'MAtCha/train_extension.cu').is_file())
            self.assertTrue((out/'tools/contact_weights/README.md').is_file())
            for rel in ('envs','outputs','cache','incoming','MAtCha/checkpoints','tmp'):
                self.assertFalse((out/rel).exists(),rel)
            self.assertTrue(json.loads((out/'SOURCE_EXPORT_MANIFEST.json').read_text())['files'])

    def test_missing_required_input_blocks_mutation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            (root/m.REQUIRED[-1]).unlink()
            with self.assertRaises(RuntimeError):
                m.cleanup(root,True)
            self.assertTrue((root/'logs/job.log').is_file())
            self.assertFalse((root/'backups').exists())

    def test_cannot_mutate_outside_project(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                m.safe(root,'../outside')

if __name__ == '__main__':
    unittest.main()
