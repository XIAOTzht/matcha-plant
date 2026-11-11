"""CPU tests: real CLI, binary/text COLMAP, disconnected leaves and environment."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement

import auto_extract_plant as app


def box(center, half):
    offsets = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
    v = np.asarray(center) + offsets * np.asarray(half)
    f = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                  [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return v, f


def rotation_to_q(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def fixture(root, binary=False):
    sparse = root / "sfm" / "sparse" / "0"
    sparse.mkdir(parents=True)
    masks = root / "masks"
    masks.mkdir()
    xyz, tri, colors = [], [], []
    # Pot, disconnected leaf, nongreen flower. Environment is bigger than pot.
    items = [([0,0,0],[.2,.3,.2],[120,65,35]),
             ([.34,.48,0],[.16,.07,.06],[20,150,30]),
             ([-.2,.72,.03],[.1,.1,.08],[220,20,140]),
             ([1.5,0,0],[.35,.5,.35],[180,180,180]),
             ([0,0,1.8],[.4,.4,.3],[180,180,180]),
             ([0,-1,0],[2,.05,2],[180,180,180])]
    for center, half, color in items:
        v, f = box(center, half)
        tri.append(f + len(xyz)*8)
        xyz.append(v)
        colors.extend([color]*8)
    xyz = np.concatenate(xyz)
    tri = np.concatenate(tri)
    rec = np.empty(len(xyz), dtype=[("x","f4"),("y","f4"),("z","f4"),
                                    ("red","u1"),("green","u1"),("blue","u1")])
    for i,k in enumerate(("x","y","z")):
        rec[k] = xyz[:,i]
    for i,k in enumerate(("red","green","blue")):
        rec[k] = np.array(colors)[:,i]
    ff = np.empty(len(tri), dtype=[("vertex_indices","i4",(3,))])
    ff["vertex_indices"] = tri
    mesh = root / "mesh.ply"
    PlyData([PlyElement.describe(rec,"vertex"),PlyElement.describe(ff,"face")]).write(str(mesh))
    views = []
    pose_lines = []
    for i,angle in enumerate(np.linspace(0,2*np.pi,8,endpoint=False)):
        center = np.array([4*np.cos(angle), 1.5, 4*np.sin(angle)])
        forward = np.array([0,.2,0]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
        down = np.cross(forward,right)
        R = np.stack([right,down,forward]); t = -R@center
        name = f"img_{i:03d}.jpg"
        v = app.View(name,256,256,200,200,128,128,R,t,np.zeros((256,256),bool))
        im = Image.new("L",(256,256)); draw = ImageDraw.Draw(im)
        u,y,inside = app.project(xyz[:24],v)
        assert inside.all()
        for f in tri[:36]:
            draw.polygon([(float(u[j]),float(y[j])) for j in f],fill=255)
        im.save(masks/(Path(name).stem+"_mask.png"))
        q = rotation_to_q(R)
        views.append((v,q,t))
        vals = [str(i+1), *map(str,q), *map(str,t), "1", name]
        pose_lines.append(" ".join(vals)+"\n\n")
    if binary:
        with (sparse/"cameras.bin").open("wb") as f:
            f.write(struct.pack("<QiiQQdddd",1,1,1,256,256,200,200,128,128))
        with (sparse/"images.bin").open("wb") as f:
            f.write(struct.pack("<Q",len(views)))
            for i,(v,q,t) in enumerate(views):
                f.write(struct.pack("<idddddddi",i+1,*q,*t,1))
                f.write(v.name.encode()+b"\0"+struct.pack("<Q",0))
    else:
        (sparse/"cameras.txt").write_text("# cameras\n1 PINHOLE 256 256 200 200 128 128\n")
        (sparse/"images.txt").write_text("# images\n"+"".join(pose_lines))
    return mesh, masks


class ExtractionTests(unittest.TestCase):
    def cli(self, root, mesh, masks, out):
        return subprocess.run([sys.executable,str(Path(app.__file__)),"--mesh",str(mesh),
                               "--sfm-dir",str(root/"sfm"),"--masks-dir",str(masks),
                               "--output-dir",str(out),"--dilate","2"],
                              text=True,capture_output=True)

    def test_end_to_end_text_and_binary(self):
        for binary in (False,True):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                mesh,masks = fixture(root,binary)
                original = mesh.read_bytes()
                out = root/"result"
                result = self.cli(root,mesh,masks,out)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                report = json.loads((out/"report.json").read_text())
                self.assertEqual(report["output_faces"],36)  # all three foreground components
                self.assertEqual(report["output_vertices"],24)
                output = PlyData.read(str(out/"plant.ply"), mmap=False)
                source = PlyData.read(str(mesh), mmap=False)
                np.testing.assert_array_equal(output["vertex"].data,source["vertex"].data[:24])
                self.assertTrue((out/"plant_preview.jpg").stat().st_size>1000)
                self.assertEqual(mesh.read_bytes(),original)
                # No accidental overwriting of an earlier extraction.
                again = self.cli(root,mesh,masks,out)
                self.assertNotEqual(again.returncode,0)

    def test_missing_mask_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mesh,masks = fixture(root)
            (masks/"img_002_mask.png").unlink()
            result = self.cli(root,mesh,masks,root/"result")
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Cannot match",result.stderr)
            self.assertFalse((root/"result"/"plant.ply").exists())

    def test_behind_camera_does_not_vote(self):
        v = app.View("x",100,100,50,50,50,50,np.eye(3),np.zeros(3),np.ones((100,100),bool))
        points = np.array([[0,0,1],[0,0,-1],[100,0,1]])
        got = app.score_points(points,[v,v,v],3,.7)
        np.testing.assert_array_equal(got,[True,False,False])


if __name__ == "__main__":
    unittest.main()
