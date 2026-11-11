from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

import prepare_turntable30 as turn
import auto_extract_plant as extract
import run_fixed_pose_sfm as runner


class TurntableTests(unittest.TestCase):
    def test_known_angles_and_equivalent_camera_transform(self):
        params=np.array([.1,-.25,.05])
        axis,center=turn.axis_geometry(params)
        xyz=np.array([.2,.4,1.1])
        for sign in (1,-1):
            poses=turn.make_poses(params,sign)
            np.testing.assert_allclose(poses[0],np.eye(4),atol=1e-12)
            for i,E in enumerate(poses):
                expected=Rotation.from_rotvec(axis*np.deg2rad(sign*i*30)).apply(xyz-center)+center
                np.testing.assert_allclose(E[:3,:3]@xyz+E[:3,3],expected,atol=1e-12)
                np.testing.assert_allclose(np.linalg.det(E[:3,:3]),1,atol=1e-12)
            angle=Rotation.from_matrix(poses[1,:3,:3]@poses[0,:3,:3].T).magnitude()
            self.assertAlmostEqual(np.rad2deg(angle),30)
            centers=np.linalg.inv(poses)[:,:3,3]
            np.testing.assert_allclose(np.linalg.norm(centers-center,axis=1),1,atol=1e-12)

    def test_binary_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/'src'; src.mkdir()
            from PIL import Image
            images=[]; masks=[]
            for i in range(12):
                im=src/f'img_{i:03d}.jpg'; mask=src/f'img_{i:03d}_mask.png'
                Image.new('RGB',(96,128),'white').save(im)
                Image.new('L',(96,128),255).save(mask)
                images.append(im); masks.append(mask)
            params=np.array([0.,-.2,0.]); dest=root/'dataset'
            turn.export_dataset(dest,images,masks,96,128,100,params,1,{})
            views,_=extract.read_colmap(dest)
            poses=turn.make_poses(params,1)
            for v,E in zip(views,poses):
                np.testing.assert_allclose(v.R,E[:3,:3],atol=1e-10)
                np.testing.assert_allclose(v.t,E[:3,3],atol=1e-10)
            self.assertTrue((dest/'diagnostics'/'camera_orbit.png').is_file())
            runner.validate_scene_layout(dest)
            # Reproduce MAtCha's dataset-root heuristic from run_mast3r.py.
            root_photos=[p for p in dest.iterdir() if p.name.endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG'))]
            self.assertEqual(root_photos,[])
            # The old diagnostic location must be rejected before model loading.
            Image.new('RGB',(8,8),'white').save(dest/'camera_orbit.png')
            with self.assertRaisesRegex(ValueError,'skip calibration'):
                runner.validate_scene_layout(dest)

    def test_fixed_pose_patch_changes_actual_make_camera_function(self):
        path=Path(__file__).resolve().parents[2]/'MAtCha/mast3r/mast3r/cloud_opt/sparse_ga.py'
        import ast
        source=path.read_text(encoding='utf-8'); tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sparse_scene_optimizer')
        block=ast.get_source_segment(source,node)+'\n'
        patched=runner.patch_optimizer_source(block)
        self.assertIn('cam2w = TURN_FIXED_POSES.to(device=device, dtype=dtype)',patched)
        self.assertLess(patched.index('cam2w = TURN_FIXED_POSES'),patched.index('        depthmaps = []'))
        self.assertIn('return K, (inv(cam2w), cam2w), depthmaps',patched)


if __name__=='__main__': unittest.main()
