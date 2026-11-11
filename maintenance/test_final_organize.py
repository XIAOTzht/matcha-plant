"""Archive, verification-failure and retention tests on isolated tiny fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import final_organize as m

class FinalOrganizerTests(unittest.TestCase):
    def test_shell_lock_releases_after_stage_failure_and_preserves_competing_lock(self):
        bash=shutil.which('bash')
        if os.name == 'nt':
            bash='D:/Git/bin/bash.exe' if Path('D:/Git/bin/bash.exe').is_file() else None
        if not bash:
            self.skipTest('Bash unavailable')
        with TemporaryDirectory() as tmp:
            root=Path(tmp); maintenance=root/'maintenance'; maintenance.mkdir()
            source=Path(__file__).with_name('finish_server.sh')
            target=maintenance/source.name
            target.write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
            (maintenance/'check_project_layout.sh').write_text('#!/bin/bash\nexit 23\n',newline='\n')
            command=[bash,target.as_posix(),'--apply']
            result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(result.returncode,23,result.stdout+result.stderr)
            self.assertIn('FINAL_WORKFLOW_FAILED',result.stdout)
            lock=root/'backups/.finish_server.lockdir'
            self.assertFalse(lock.exists())
            lock.mkdir(); (lock/'owner.txt').write_text('existing task')
            result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(result.returncode,1)
            self.assertIn('existing task',result.stderr)
            self.assertEqual((lock/'owner.txt').read_text(),'existing task')

    def test_directory_lock_excludes_competitors_and_releases_after_error(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'.lockdir'
            with self.assertRaisesRegex(ValueError,'test failure'):
                with m.directory_lock(path):
                    owner=json.loads((path/'owner.json').read_text())
                    self.assertEqual(owner['pid'],os.getpid())
                    with self.assertRaisesRegex(RuntimeError,'lock directory exists'):
                        with m.directory_lock(path):
                            self.fail('Concurrent lock was allowed')
                    self.assertTrue((path/'owner.json').is_file())
                    raise ValueError('test failure')
            self.assertFalse(path.exists())
            with m.directory_lock(path):
                self.assertTrue(path.is_dir())
            self.assertFalse(path.exists())

    def test_directory_lock_preserves_unidentified_existing_locks(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'.lockdir'; path.mkdir()
            with self.assertRaisesRegex(RuntimeError,'Owner record unavailable'):
                with m.directory_lock(path):
                    self.fail('Existing lock was removed')
            self.assertTrue(path.is_dir())

    def fixture(self, root):
        keep = ('envs/runtime.bin', 'cache/download.pth', 'weights/model.pth',
            'third_party/extension.so', 'tools/plant_extract/helper.py',
            'tools/plant_extract_with_car/crop.py', 'tools/plant_semantics/models/model.pth',
            'MAtCha/build/extension.so', 'maintenance/final_organize.py',
            'scripts/plant_contact/run_server.sh', 'docs/PLANT_CONTACT_GUIDE.md',
            'data/with_car_20261021/images/001.jpg',
            m.SCENE+'/free_gaussians/checkpoint.pt',
            m.SCENE+'/plant_car_semantics_v2/labels.npz',
            m.SCENE+'/contact_latest_new/weights/model.ply', 'backups/previous/archive.tar')
        old = ('data/old/images/001.jpg', 'outputs/old/result.ply',
            m.SCENE+'/contact_test_old/weights/model.ply', m.SCENE+'/plant_car_semantics_v1/mesh.ply',
            'incoming/upload.zip', 'logs/old.log', 'tmp/unpack/file',
            'exports/github_old/source.py', 'maintenance/organize_matcha.py',
            'maintenance/packages/old.zip', 'scripts/with_car_stage1/run.sh',
            'README_PROJECT_LAYOUT.md', 'tools/contact_weights/__pycache__/old.pyc')
        for rel in keep+old:
            p=root/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(rel.encode())
        return keep, old

    def test_preview_then_verified_backup_removes_only_retired_files(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); keep,old=self.fixture(root)
            with patch.object(m,'prerequisites',return_value='contact_latest_new'), patch.object(m,'active',return_value=[]):
                m.organize(root)
                self.assertFalse((root/'backups/.final_organize.lockdir').exists())
                m.organize(root,True)
            backup=next((root/'backups').glob('final_cleanup_*'))
            self.assertTrue((backup/'BACKUP_VERIFIED').exists())
            self.assertTrue((backup/'FINAL_ORGANIZE_SUCCESS').exists())
            self.assertFalse((root/'backups/.final_organize.lockdir').exists())
            for rel in keep:
                self.assertEqual((root/rel).read_bytes(),rel.encode())
            with tarfile.open(backup/'old_content.tar') as tf:
                for rel in old:
                    self.assertFalse((root/rel).exists(),rel)
                    self.assertEqual(tf.extractfile(rel).read(),rel.encode())
            self.assertEqual(json.loads((backup/'manifest.json').read_text())['status'],'SUCCESS')

    def test_failed_verification_never_removes_originals(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); _,old=self.fixture(root)
            with patch.object(m,'prerequisites',return_value='contact_latest_new'), patch.object(m,'active',return_value=[]), patch.object(m,'verify_archive',side_effect=RuntimeError('corrupt archive')):
                with self.assertRaisesRegex(RuntimeError,'corrupt'):
                    m.organize(root,True)
            for rel in old:
                self.assertEqual((root/rel).read_bytes(),rel.encode())
            backup=next((root/'backups').glob('final_cleanup_*'))
            self.assertFalse((backup/'BACKUP_VERIFIED').exists())
            self.assertFalse((root/'backups/.final_organize.lockdir').exists())

    def test_missing_real_dependencies_blocks_any_backup_or_delete(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); _,old=self.fixture(root)
            with self.assertRaises(RuntimeError):
                m.organize(root,True)
            self.assertTrue(all((root/p).exists() for p in old))
            self.assertFalse((root/'backups/.final_organize.lockdir').exists())

    def test_real_prerequisites_accept_current_export_and_reject_old_policy(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            extra=('maintenance/run_reconstruction.sh',
                'tools/plant_extract_with_car/run_crop_with_car.sh',
                'tools/plant_semantics_with_car/run_server.sh',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl',
                'MAtCha/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth')
            stages=('plant_car_auto_v1/plant_car.ply', 'plant_car_auto_v1/report.json',
                'plant_car_metric_v1/plant_car_mm.ply', 'plant_car_metric_v1/scale_report.json',
                'tetra_meshes_gpu1_ds05/tetra_mesh_binary_search_7.ply')
            latest='contact_latest_fixture'
            outputs=('PLANT_CONTACT_EXPORT_SUCCESS','result_paths.json','organs/organ_instances.npz',
                'weights/contact_weights.npz','weights/contact_risk_heatmap.ply','weights/contact_weighted_mesh.ply')
            for rel in (*m.REQUIRED,*extra,*(m.SCENE+'/'+p for p in stages),*(m.SCENE+'/'+latest+'/'+p for p in outputs)):
                p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
            for rel in ('tools/plant_extract','tools/plant_semantics',m.SCENE+'/free_gaussians',m.SCENE+'/mast3r_sfm'):
                (root/rel).mkdir(parents=True,exist_ok=True)
            m.save(root/m.SCENE/'latest_contact.json',{'relative_output':latest})
            report=dict(data_export='PASS',mm_per_unit=1.0,geometry_policy={},
                visualization={'version':'weight-colors-0.4.2-smooth-green-yellow'})
            m.save(root/m.SCENE/latest/'weights/contact_weights_report.json',report)
            for policy in ('contact_policy.json','organ_weight_policy.json'):
                m.save(root/m.SCENE/latest/policy,{})
            self.assertEqual(m.prerequisites(root),latest)
            m.save(root/m.SCENE/latest/'contact_policy.json',{'old':True})
            with self.assertRaisesRegex(RuntimeError,'policy mismatch'):
                m.prerequisites(root)

    def test_tamper_detected_and_symlink_never_followed(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            (root/'outside.txt').write_bytes(b'outside')
            target=root/'incoming/link'
            try:
                target.symlink_to(root/'outside.txt')
            except OSError:
                target=None
            records=m.inventory(root,['incoming'])
            archive=root/'test.tar'
            m.write_archive(root,archive,records)
            m.verify_archive(archive,records)
            if target:
                self.assertEqual(records['incoming/link']['kind'],'symlink')
                with tarfile.open(archive) as tf:
                    self.assertTrue(tf.getmember('incoming/link').issym())
            file_key=next(k for k,v in records.items() if v['kind']=='file')
            records[file_key]['sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                m.verify_archive(archive,records)

    def test_retained_symlink_dependency_blocks_retirement(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            link=root/'data/with_car_20261021/images/dependency.jpg'
            try:
                link.symlink_to(root/'data/old/images/001.jpg')
            except OSError:
                self.skipTest('Symlink permission unavailable')
            paths=m.plan(root,'contact_latest_new')
            with self.assertRaisesRegex(RuntimeError,'depends on retired'):
                m.check_retained_links(root,paths,'contact_latest_new')

    def test_changed_originals_or_running_jobs_block_removal(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); _,old=self.fixture(root)
            original=m.inventory
            calls=0
            def changed(*args):
                nonlocal calls
                calls+=1
                if calls==2:
                    (root/old[0]).write_bytes(b'changed')
                return original(*args)
            with patch.object(m,'prerequisites',return_value='contact_latest_new'), patch.object(m,'active',return_value=[]), patch.object(m,'inventory',side_effect=changed):
                with self.assertRaisesRegex(RuntimeError,'Original content changed'):
                    m.organize(root,True)
            self.assertTrue(all((root/p).exists() for p in old))
            with patch.object(m,'prerequisites',return_value='contact_latest_new'), patch.object(m,'active',return_value=[{'pid':123}]):
                with self.assertRaisesRegex(RuntimeError,'Stop project jobs'):
                    m.organize(root,True)

if __name__=='__main__':
    unittest.main()
