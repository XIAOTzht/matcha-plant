import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
from plant_scene import fingerprint, load_inputs


class VirtualInputTests(unittest.TestCase):
    def fixture(self, root):
        xyz=np.array([[60,0,60],[61,0,60],[60,1,60],[-1,63,-1],[1,63,-1],[0,63,1]],dtype='f4')
        faces=np.array([[0,1,2],[3,5,4]],dtype='i4');labels=np.array([4,4,4,5,5,5],dtype='i4')
        v=np.zeros(6,dtype=[('x','f4'),('y','f4'),('z','f4'),('label','i4')])
        for i,k in enumerate('xyz'):v[k]=xyz[:,i]
        v['label']=labels
        f=np.zeros(2,dtype=[('vertex_indices','i4',(3,)),('label','i4')]);f['vertex_indices']=faces;f['label']=[4,5]
        PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False).write(str(root/'mesh.ply'))
        self.arrays=dict(vertex_labels=labels,face_labels=np.array([4,5]),virtual_substrate=np.arange(6)>=3,face_virtual_substrate=np.arange(2)>=1)
        np.savez(root/'labels.npz',**self.arrays)
        fp=np.array(fingerprint(xyz,faces,labels))
        shared=dict(vertex_labels=labels,vertex_instance_id=np.zeros(6,'i4'),mesh_fingerprint=fp,mm_per_unit=np.array(1.))
        np.savez(root/'organs.npz',**shared)
        np.savez(root/'weights.npz',**shared,face_labels=np.array([4,5]),vertex_hard_collision=np.ones(6,bool),
            face_hard_collision=np.ones(2,bool),vertex_effective_weight=np.zeros(6),face_effective_weight=np.zeros(2))
        (root/'coordinate_system.json').write_text(json.dumps(dict(units='mm',board_center_mm=[0,0,0],board_frame_axes=np.eye(3).tolist())))
        (root/'policy.json').write_text(json.dumps(dict(hard_collision=dict(classes=[4,5,6]),watering_target=dict(class_id=5,approach_height_range_mm=[5,15],default_height_mm=10))))
        self.assumption=dict(surface_source='assumed_virtual_substrate',measured=False,vertex_start=3,vertex_count=3,
            face_start=1,face_count=1,assumed_height_mm=63,footprint_xz_mm=[[-1,-1],[1,-1],[0,1]])
        (root/'virtual_substrate.json').write_text(json.dumps(self.assumption))
        return dict(source_root='.',semantic_mesh='mesh.ply',semantic_labels='labels.npz',weights='weights.npz',
            organs='organs.npz',coordinate_system='coordinate_system.json',policy='policy.json')

    def test_provenance_shared_by_scene_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.fixture(root)
            result=load_inputs(root,cfg);report=result[-1]
            self.assertEqual(report['watering_surface_source'],'assumed_virtual_substrate')
            self.assertFalse(report['measured_soil_available'])
            self.assertFalse(report['real_robot_ready'])
            del result

    def test_missing_sidecar_cannot_silently_become_real_soil(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.fixture(root);(root/'virtual_substrate.json').unlink()
            with self.assertRaisesRegex(ValueError,'missing its assumption'):load_inputs(root,cfg)

    def test_false_height_or_label_claim_rejected(self):
        for field,value in [('assumed_height_mm',64),('measured',True),('vertex_count',2)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=self.fixture(root);self.assumption[field]=value
                (root/'virtual_substrate.json').write_text(json.dumps(self.assumption))
                with self.assertRaises(ValueError):load_inputs(root,cfg)


if __name__ == '__main__':unittest.main()
