"""Distribution contracts: complete source, no local assets, verifiable contents."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_package


class DistributionTests(unittest.TestCase):
    def test_archive_is_complete_reproducible_and_source_only(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / 'first.zip'
            second = Path(folder) / 'second.zip'
            profile = 'full' if (build_package.ROOT / 'run_sam3.ps1').is_file() else 'v3'
            build_package.build_package(first, profile=profile)
            build_package.build_package(second, profile=profile)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = set(archive.namelist())
                self.assertTrue({
                    'run_auto.ps1', 'segment.cmd', 'setup_local.ps1',
                    'run_v3.ps1', 'segment_v3.cmd', 'START_HERE_V3.md',
                    'run_extra_views.ps1', 'extra_view_review.py',
                    'tests/test_geometry.py', 'docs/DEVELOPMENT.md',
                    'third_party/MobileSAM/LICENSE', 'PACKAGE_MANIFEST.json',
                } <= names)
                self.assertFalse(any(
                    name.startswith('vendor/') or name.endswith(('.pt', '.safetensors', '.pyc'))
                    or name == 'runtime.local.json' for name in names))
                manifest = json.loads(archive.read('PACKAGE_MANIFEST.json'))
                self.assertEqual({item['path'] for item in manifest['files']},
                                 names - {'PACKAGE_MANIFEST.json'})
                for item in manifest['files']:
                    data = archive.read(item['path'])
                    self.assertEqual(len(data), item['bytes'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
            digest = first.with_suffix('.zip.sha256').read_text(encoding='ascii').split()[0]
            self.assertEqual(digest, hashlib.sha256(first.read_bytes()).hexdigest())

    def test_v3_archive_excludes_sam3_and_historical_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'v3.zip'
            build_package.build_package(target, profile='v3')
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
                self.assertTrue({'run_v3.ps1', 'segment_v3.cmd', 'START_HERE_V3.md',
                                 'extra_view_review.py', 'docs/UNCERTAINTY.md',
                                 'third_party/MobileSAM/LICENSE'} <= names)
                self.assertFalse(any('sam3' in name.lower() or name.startswith('experiments/')
                                     or '/archive/' in name for name in names))
                self.assertFalse(any(name.endswith(('.pt', '.safetensors', '.pyc'))
                                     or name == 'runtime.local.json' for name in names))
                manifest = json.loads(archive.read('PACKAGE_MANIFEST.json'))
                self.assertEqual(manifest['profile'], 'v3')
                self.assertEqual(manifest['entrypoint'], 'run_v3.ps1')
                self.assertEqual({item['path'] for item in manifest['files']},
                                 names - {'PACKAGE_MANIFEST.json'})
                self.assertEqual(archive.read('README.md'), archive.read('START_HERE_V3.md'))
                for item in manifest['files']:
                    self.assertEqual(hashlib.sha256(archive.read(item['path'])).hexdigest(), item['sha256'])
                extracted = Path(folder) / 'extracted'
                archive.extractall(extracted)
            rebuilt = Path(folder) / 'rebuilt.zip'
            build_package.build_package(rebuilt, root=extracted, profile='v3')
            self.assertEqual(target.read_bytes(), rebuilt.read_bytes())


if __name__ == '__main__':
    unittest.main()
