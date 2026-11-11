"""Transport integrity, resumability and reversible deployment regression checks."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from common import JobLock, Stages, digest, read_json, run_name, verify_images, write_json
from photos import export, ingest, prepare

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'maintenance'))
import plant_pipeline_bundle as bundle


class PipelineTests(unittest.TestCase):
    def test_run_name_rejects_paths_and_shell(self):
        self.assertEqual(run_name('plant_01-a'),'plant_01-a')
        for value in ('../outside', 'abc;date', '-x', 'with space', '', 'a'*65):
            with self.assertRaises(ValueError):run_name(value)

    def test_success_reuses_only_verified_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            stages=Stages(Path(temp),dict(input='a'))
            def action(d):
                (d/'result').write_text('intact');return dict(result=d/'result')
            original=stages.execute('01',action)
            reused=Stages(temp,dict(input='a')).execute('01',lambda d:self.fail('must reuse'))
            self.assertEqual(original,reused)
            original['result'].write_text('tampered')
            with self.assertRaises(ValueError):Stages(temp,dict(input='a')).execute('01',action)
            with self.assertRaises(ValueError):Stages(temp,dict(input='b'))

    def test_failure_retries_fresh_and_retains_history(self):
        with tempfile.TemporaryDirectory() as temp:
            stages=Stages(temp,dict(input='a'))
            def fail(d):
                (d/'partial').write_text('failed attempt');raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):stages.execute('01',fail)
            self.assertEqual(read_json(stages.path)['status'],'FAILED')
            def success(d):
                (d/'result').write_text('ok');return dict(result=d/'result')
            result=stages.execute('01',success)
            self.assertIn('attempt_002',str(result['result']))
            self.assertTrue((Path(temp)/'01/attempt_001/partial').is_file())

    def test_empty_output_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            def empty(d):
                (d/'result').touch();return dict(result=d/'result')
            with self.assertRaises(ValueError):Stages(temp,{}).execute('01',empty)

    def test_lock_excludes_another_job(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'job.lock'
            with JobLock(path):
                with self.assertRaises(RuntimeError):
                    with JobLock(path):pass
            with JobLock(path):pass

    def test_photos_roundtrip_idempotent_and_tampering(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);images=root/'photos';images.mkdir()
            for i in range(3):Image.new('RGB',(10,10),(i,0,0)).save(images/f'{i}.jpg')
            archive=root/'photos.zip';prepare(images,'plant_1',archive)
            target=ingest(archive,digest(archive),root/'server','plant_1')
            self.assertEqual(len(verify_images(target)['images']),3)
            self.assertEqual(ingest(archive,digest(archive),root/'server','plant_1'),target)
            (target/'images/0001.jpg').write_bytes(b'wrong')
            with self.assertRaises(ValueError):verify_images(target)

    def test_bad_image_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i in range(3):(root/f'{i}.jpg').write_bytes(b'not a photo')
            with self.assertRaises(Exception):prepare(root,'test',root/'photos.zip')

    def test_photo_archive_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape','bad')
            with self.assertRaises(ValueError):ingest(archive,digest(archive),root/'server','plant_1')
            self.assertFalse((root/'escape').exists())

    def test_result_export_requires_success_and_includes_trajectory_and_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'simulation/outputs/jobs/plant_1';job.mkdir(parents=True)
            semantic=root/'outputs/semantic';semantic.mkdir(parents=True)
            weights=root/'outputs/contact/weights';weights.mkdir(parents=True)
            report=root/'outputs/pipeline/plant_1/result_paths.json'
            sim=dict(path_report=str(job/'path_report.json'), scene_report=str(job/'scene_report.json'),
                     validation_report=str(job/'validation_report.json'),timed_trajectory=str(job/'timed_trajectory.npz'),
                     line_csv=str(job/'nozzle_path_line.csv'),media=dict(line=str(job/'nozzle_path_line.png')),
                     natural_selection=str(job/'natural_selection.json'),natural_search_report=str(job/'natural_search_report.json'),
                     demonstration_selection=str(job/'demonstration_selection.json'),
                     demonstration_search_report=str(job/'demonstration_search_report.json'))
            paths=[*map(Path,[sim['path_report'],sim['scene_report'],sim['validation_report'],sim['timed_trajectory'],
                              sim['line_csv'],sim['media']['line']]),job/'job_report.json',job/'timed_trajectory.csv',
                   semantic/'semantic_mesh_mm.ply',semantic/'labels.npz',semantic/'coordinate_system.json',
                   root/'scale.json',weights/'contact_weights.npz',weights/'contact_risk_heatmap.ply',
                   weights/'contact_weighted_mesh.ply',weights/'contact_weights_report.json',weights/'contact_weight_colorbar.png',
                   semantic/'report.json',semantic/'cart_template.json',semantic/'source_mapping.npz',
                   semantic/'validation.json',semantic/'labeled_original_rgb_mm.ply',
                   job/'natural_selection.json',job/'natural_search_report.json',job/'timing_plan.json',job/'trajectory_q.npz',
                   job/'demonstration_selection.json',job/'demonstration_search_report.json',job/'demonstration_verification.json']
            for p in paths:p.write_bytes(b'verified fixture')
            write_json(report,dict(status='PHOTO_TO_PATH_SUCCESS',simulation=sim,simulation_report=str(job/'job_report.json'),
                       scale_report=str(root/'scale.json'),semantic_mesh=str(semantic/'semantic_mesh_mm.ply'),
                       contact_weights=str(weights/'contact_weights.npz'),cart_underside_report=str(semantic/'report.json')))
            with self.assertRaises(ValueError):export(root,'plant_1')
            (report.parent/'PHOTO_TO_PATH_SUCCESS').write_text('success')
            export(root,'plant_1')
            archive=root/'exports/plant_1/results.zip'
            with zipfile.ZipFile(archive) as z:
                self.assertIn('simulation/outputs/jobs/plant_1/timed_trajectory.csv',z.namelist())
                self.assertIn('outputs/contact/weights/contact_risk_heatmap.ply',z.namelist())
                self.assertIn('outputs/semantic/cart_template.json',z.namelist())
                self.assertIn('outputs/semantic/labeled_original_rgb_mm.ply',z.namelist())
                for name in ('natural_selection.json','natural_search_report.json','timing_plan.json','trajectory_q.npz',
                             'demonstration_selection.json','demonstration_search_report.json','demonstration_verification.json'):
                    self.assertIn('simulation/outputs/jobs/plant_1/'+name,z.namelist())
                self.assertIsNone(z.testzip())
            self.assertEqual(Path(str(archive)+'.sha256').read_text().split()[0],digest(archive))


class BundleTests(unittest.TestCase):
    def archive(self,root,name,data):
        archive=root/'bundle.zip'
        manifest={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))}
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(name,data);z.writestr('bundle_manifest.json',json.dumps(dict(files=manifest)))
        return archive

    def test_code_install_backup_and_model_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);server=root/'server';(server/'MAtCha').mkdir(parents=True)
            target=server/'configs/plant_pipeline.json';target.parent.mkdir();target.write_text('old')
            model=server/'MAtCha/model.pth';model.write_bytes(b'weights')
            archive=self.archive(root,'configs/plant_pipeline.json',b'new')
            bundle.install(archive,digest(archive),server)
            self.assertEqual(target.read_text(),'new');self.assertEqual(model.read_bytes(),b'weights')
            copies=list((server/'backups/code_updates').glob('*/configs/plant_pipeline.json'))
            self.assertEqual(len(copies),1);self.assertEqual(copies[0].read_text(),'old')
            with self.assertRaises(ValueError):bundle.install(archive,'0'*64,server)

    def test_bundle_path_escape_and_model_updates_rejected(self):
        for name in ('../escape','MAtCha/model.pth','outputs/model.ply','envs/bin/python','simulation/configs/latest_path.json',
                     'configs/server_render_runtime.json','simulation/configs/server_render_runtime.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);archive=self.archive(root,name,b'bad')
                with self.assertRaises(ValueError):bundle.inspect(archive,digest(archive))


if __name__=='__main__':unittest.main()
