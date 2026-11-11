import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull

from add_virtual_substrate import append_arrays, estimate_surface, surface_mesh, run


def pot_points():
    angle, height = np.meshgrid(np.linspace(0, 2*np.pi, 100, endpoint=False), np.linspace(5, 75, 90))
    return np.column_stack([55*np.cos(angle.ravel()), height.ravel(), 45*np.sin(angle.ravel())])


def source(directory):
    q = pot_points()
    v = np.zeros(len(q), dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),
        ('label','u1'),('uncertain','u1'),('inferred','u1'),('label_source','u1'),('geometry_inferred','u1')])
    for i,k in enumerate('xyz'):v[k] = q[:,i]
    v['label']=4;v['red']=190;v['green']=180;v['blue']=170;v['geometry_inferred'][0]=1
    t = np.arange((len(q)//3)*3).reshape(-1,3)
    f = np.zeros(len(t),dtype=[('vertex_indices','i4',(3,)),('label','u1'),('geometry_inferred','u1')])
    f['vertex_indices']=t;f['label']=4;f['geometry_inferred'][0]=1
    for name in ('labeled_original_rgb_mm.ply','semantic_mesh_mm.ply'):
        PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False).write(str(directory/name))
    nv,nf=len(v),len(f)
    data=dict(vertex_labels=v['label'].copy(),face_labels=f['label'].copy(),
        uncertain=np.zeros(nv,bool),inferred=np.zeros(nv,bool),label_source=np.zeros(nv,'u1'),
        class_scores=np.full((nv,6),np.nan,'f4'),class_score_valid=np.zeros(nv,bool),
        class_score_ids=np.arange(1,7,dtype='u1'),face_uncertain=np.zeros(nf,bool),face_boundary=np.zeros(nf,bool),
        plant_vertex_indices=np.arange(nv,dtype='i8'),cart_geometry_inferred=v['geometry_inferred'].astype(bool),
        cart_face_geometry_inferred=f['geometry_inferred'].astype(bool))
    np.savez_compressed(directory/'labels.npz',**data)
    (directory/'coordinate_system.json').write_text(json.dumps(dict(units='mm',board_frame_axes=np.eye(3).tolist(),board_center_mm=[0,0,0])))
    (directory/'classes.json').write_text('[]')
    return data


class VirtualSubstrateTests(unittest.TestCase):
    def test_inset_height_normals_and_area(self):
        q=pot_points();t=estimate_surface(q,np.full(len(q),4))
        p,f=surface_mesh(t)
        self.assertAlmostEqual(t['estimated_rim_height_mm']-t['assumed_height_mm'],10)
        self.assertLess(p[:,0].max(),50);self.assertGreater(p[:,0].min(),-50)
        self.assertTrue(np.all(p[:,1] == t['assumed_height_mm']))
        n=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
        self.assertTrue(np.all(n[:,1]>0))
        self.assertAlmostEqual(n[:,1].sum()/2,ConvexHull(t['footprint_xz_mm']).volume,places=4)

    def test_unsupported_or_invalid_assumptions_rejected(self):
        q=pot_points();labels=np.full(len(q),4)
        for kwargs in (dict(rim_offset_mm=np.nan),dict(height_mm=85),dict(height_mm=-1),dict(wall_inset_mm=1)):
            with self.assertRaises(ValueError):estimate_surface(q,labels,**kwargs)
        with self.assertRaises(ValueError):estimate_surface(q,np.zeros(len(q)))
        with self.assertRaises(ValueError):estimate_surface(q[q[:,0]>0],labels[q[:,0]>0])

    def test_append_only_and_provenance_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();old=source(src)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
            out=root/'new';args=argparse.Namespace(source=src,output=out,assume_pot_interior_substrate=True,
                rim_offset_mm=10.,wall_inset_mm=6.,height_mm=None,spacing_mm=2.)
            run(args)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()})
            n=len(old['vertex_labels']);f=len(old['face_labels'])
            with np.load(out/'labels.npz') as z:
                for k,a in old.items():np.testing.assert_array_equal(z[k][:len(a)],a)
                self.assertTrue((z['vertex_labels'][n:]==5).all())
                self.assertTrue((z['face_labels'][f:]==5).all())
                self.assertFalse(z['class_score_valid'][n:].any())
                self.assertTrue(np.isnan(z['class_scores'][n:]).all())
                self.assertFalse(z['cart_geometry_inferred'][n:].any())
                self.assertTrue(z['virtual_substrate'][n:].all())
            m=PlyData.read(str(out/'labeled_original_rgb_mm.ply'),known_list_len={'face':{'vertex_indices':3}})
            self.assertEqual(m['vertex'].data['geometry_inferred'][0],1)
            self.assertTrue(m['vertex'].data['virtual_substrate'][n:].all())
            del m
            with self.assertRaises(ValueError):run(args)

    def test_existing_substrate_or_missing_opt_in_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();d=source(src)
            args=argparse.Namespace(source=src,output=root/'new',assume_pot_interior_substrate=False,
                rim_offset_mm=10.,wall_inset_mm=6.,height_mm=None,spacing_mm=2.)
            with self.assertRaises(ValueError):run(args)
            args.assume_pot_interior_substrate=True;d['face_labels'][0]=5;np.savez_compressed(src/'labels.npz',**d)
            with self.assertRaises(ValueError):run(args)


if __name__ == '__main__':unittest.main()
