"""Deployment tests that exercise real package files, backups and pipeline CLI."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('contact_bundle',ROOT/'maintenance/plant_contact_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class DeploymentTests(unittest.TestCase):
    def test_bundle_install_backup_idempotent_and_reject_tamper(self):
        with TemporaryDirectory(prefix='plant_contact_bundle_') as tmp:
            base = Path(tmp)
            archive = base/'bundle.zip'
            manifest = bundle.build(ROOT,archive)
            self.assertGreater(len(manifest['files']),20)
            target = base/'project'
            target.mkdir()
            preserve = target/'outputs'/'model.ply'
            preserve.parent.mkdir()
            preserve.write_bytes(b'KEEP MODEL')
            old = target/'tools'/'contact_weights'/'weight_palette.py'
            old.parent.mkdir(parents=True)
            old.write_text('# previous version\n',encoding='utf-8')
            bundle.install(archive,target,bundle.digest(archive))
            backups=list((target/'backups/code_updates').glob('plant_contact_*'))
            self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'tools/contact_weights/weight_palette.py').read_text(),'# previous version\n')
            self.assertEqual(preserve.read_bytes(),b'KEEP MODEL')
            bundle.install(archive,target,bundle.digest(archive))
            self.assertEqual(len(list((target/'backups/code_updates').glob('plant_contact_*'))),1)
            self.assertFalse((target/'maintenance/.plant_contact_install.lock').exists())
            with self.assertRaises(ValueError):
                bundle.install(archive,target,'0'*64)
            for name in ('../outside.py','/outside.py','tools/contact_weights/../../bad.py','outputs/model.ply'):
                self.assertFalse(bundle.allowed(name))
            self.assertEqual(preserve.read_bytes(),b'KEEP MODEL')

    def test_full_cli_pipeline_preserves_source_and_refuses_overwrite(self):
        with TemporaryDirectory(prefix='plant contact unicode_') as tmp:
            base=Path(tmp)/'模型 with spaces'
            source=base/'input';source.mkdir(parents=True)
            v=np.zeros(6,dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('label','u1'),('uncertain','u1')])
            v['x']=[0,20,0,50,70,50];v['y']=[0,0,20,0,0,20]
            v['label']=[2,2,2,3,3,3];v['red']=100;v['green']=180;v['blue']=80
            f=np.zeros(2,dtype=[('vertex_indices','i4',(3,)),('label','u1'),('uncertain','u1')])
            f['vertex_indices']=[[0,1,2],[3,4,5]];f['label']=[2,3]
            mesh=source/'semantic_mesh_mm.ply'
            PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False,byte_order='<').write(str(mesh))
            np.savez_compressed(source/'labels.npz',vertex_labels=v['label'],face_labels=f['label'],
                                uncertain=np.zeros(6,bool),face_uncertain=np.zeros(2,bool))
            before=hashlib.sha256(mesh.read_bytes()).hexdigest()
            output=base/'result'
            cmd=[sys.executable,str(ROOT/'scripts/plant_contact/run_pipeline.py'),
                 '--input-dir',str(source),'--output-dir',str(output),'--max-pixels','100']
            checked=subprocess.run(cmd+['--check-only'],capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            self.assertFalse(output.exists())
            result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((output/'PLANT_CONTACT_EXPORT_SUCCESS').is_file())
            self.assertFalse((output/'PIPELINE_RUNNING').exists())
            for policy in ('contact_policy.json', 'organ_weight_policy.json'):
                self.assertEqual(json.loads((output/policy).read_text(encoding='utf-8')),
                    json.loads((ROOT/'tools/contact_weights'/policy).read_text(encoding='utf-8-sig')))
            self.assertTrue((output/'weights/organ_weights_detail.png').is_file())
            self.assertTrue((output/'weights/contact_weight_colorbar.png').is_file())
            self.assertEqual(len(list((output/'organs/parts').glob('*.ply'))),2)
            self.assertEqual(hashlib.sha256(mesh.read_bytes()).hexdigest(),before)
            paths=json.loads((output/'result_paths.json').read_text(encoding='utf-8'))
            self.assertTrue(Path(paths['heatmap']).is_file())
            refused=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertNotEqual(refused.returncode,0)
            self.assertIn('Output is not empty',refused.stderr)


if __name__=='__main__':
    unittest.main()
