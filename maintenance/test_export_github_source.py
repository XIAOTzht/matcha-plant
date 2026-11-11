import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import export_github_source as tool


class ExportTests(unittest.TestCase):
    def test_runtime_excluded_and_native_sources_retained(self):
        for name in [
            'tools/plant_semantics/models/a/model.safetensors',
            'simulation/outputs/a.json', 'MAtCha/mast3r/.gradio/certificate.pem',
            'MAtCha/native/_C.so', 'MAtCha/a.egg-info/PKG-INFO',
            'configs/server_render_runtime.json', 'simulation/configs/latest_path.json',
        ]:
            self.assertIsNotNone(tool.exclusion(name,10),name)
        for name in [
            'simulation/assets/robot/robotstudio_so101/assets/arm.stl',
            'MAtCha/lib/glm/detail/a.inl','MAtCha/lib/CMakeLists.txt',
            'MAtCha/lib/LICENSE.md','MAtCha/x/cuda.Dockerfile',
            'maintenance/reproduction/approved_20261107/snapshot_manifest.json',
        ]:
            self.assertIsNone(tool.exclusion(name,10),name)

    def populate(self,root):
        for name in tool.REQUIRED:
            p=root/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('fixture\n')
        (root/'README.md').write_text('old root description')
        (root/'tools/plant_semantics/models').mkdir()
        (root/'tools/plant_semantics/models/model.safetensors').write_bytes(b'not source')

    def test_snapshot_and_archive_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'project';dest=Path(temp)/'snapshot'
            self.populate(root)
            with contextlib.redirect_stdout(io.StringIO()):
                tool.export(root,dest)
            manifest=json.loads((dest/'SOURCE_EXPORT_MANIFEST.json').read_text())
            self.assertTrue(manifest['source_stability_rechecked'])
            self.assertEqual((root/'README.md').read_text(),'old root description')
            self.assertEqual((dest/'docs/LEGACY_SERVER_README.md').read_text(),'old root description')
            self.assertFalse((dest/'tools/plant_semantics/models').exists())
            self.assertTrue((dest/'simulation/scripts/coordinated_motion.py').exists())
            with zipfile.ZipFile(str(dest)+'.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertIn('snapshot/configs/plant_pipeline.json',z.namelist())
            with self.assertRaises(FileExistsError):
                tool.export(root,dest)

    def test_missing_source_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'project';root.mkdir()
            dest=Path(temp)/'snapshot'
            with self.assertRaises(ValueError):
                tool.export(root,dest)
            self.assertFalse(dest.exists())

    def test_credential_pattern_stops_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'project';dest=Path(temp)/'snapshot'
            self.populate(root)
            (root/'tools/credentials.txt').write_bytes(
                b'-----BEGIN OPENSSH ' + b'PRIVATE KEY-----\n')
            with self.assertRaises(RuntimeError):
                tool.export(root,dest)
            self.assertTrue((dest/'EXPORT_FAILED').exists())
            self.assertFalse((dest/'SOURCE_EXPORT_READY').exists())


if __name__=='__main__':
    unittest.main()
