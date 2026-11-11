"""Regression tests for occlusion, abstention and index-preserving export."""
import tempfile
import unittest
from pathlib import Path
import sys

# Support discovery from either the project folder or an unrelated working folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plantseg as ps
import numpy as np


class GeometryTests(unittest.TestCase):
    def test_hidden_surface_receives_no_front_label(self):
        import open3d as o3d
        # Two identical projected triangles; triangle 0 is one unit in front of 1.
        xyz=np.array([[-1,-1,1],[1,-1,1],[0,1,1],[-1,-1,0],[1,-1,0],[0,1,0]],np.float32)
        tri=np.array([[0,1,2],[3,4,5]],np.int32)
        scene=ps.scene_from_arrays(xyz,tri)
        hits=scene.cast_rays(o3d.core.Tensor(np.array([[[0,0,3,0,0,-1]]],np.float32)))
        fid=hits['primitive_ids'].numpy().astype(np.int32)
        self.assertEqual(int(fid[0,0]),0)
        uv=hits['primitive_uvs'].numpy()
        bary=np.concatenate([1-uv.sum(-1,keepdims=True),uv],axis=-1)
        scores=np.zeros((1,1,6),np.float32); scores[0,0,2]=.9
        total,seen=ps.accumulate_vertex_view(tri,fid,bary,scores,6)
        self.assertTrue(seen[:3].all())
        self.assertFalse(seen[3:].any())
        self.assertEqual(float(total[3:].sum()),0)

    def test_conflicting_or_weak_evidence_abstains(self):
        scores=np.array([[0,.51,.50,0,0,0],[0,0,.15,0,0,0],[0,0,0,.8,0,0]],np.float32)
        labels,_,_=ps.evidence_labels(scores)
        np.testing.assert_array_equal(labels,[0,0,3])

    def test_per_vertex_visibility_ignores_occluded_vertices(self):
        xyz=np.array([[-1,-1,1],[1,-1,1],[0,1,1],[-1,-1,0],[1,-1,0],[0,1,0]],np.float32)
        tri=np.array([[0,1,2],[3,4,5]],np.int32)
        camera={'origin':[0,0,3],'direction':[0,0,-1],'right':[1,0,0],
                'screen_up':[0,1,0],'extent':4}
        scores=np.zeros((16,16,6),np.float32);scores[...,2]=.8
        values,seen=ps.sample_visible_vertices(ps.scene_from_arrays(xyz,tri),xyz,camera,scores)
        self.assertFalse(seen[3:].any())
        self.assertEqual(float(values[3:].sum()),0)
        # Translating/scaling world coordinates leaves projection and occlusion invariant.
        moved=xyz*7+np.array([10,-4,2],np.float32)
        camera2=dict(camera,origin=(np.array(camera['origin'])*7+[10,-4,2]).tolist(),extent=28)
        values2,seen2=ps.sample_visible_vertices(ps.scene_from_arrays(moved,tri),moved,camera2,scores)
        np.testing.assert_array_equal(seen,seen2)
        np.testing.assert_allclose(values,values2,atol=1e-6)

    def test_export_preserves_geometry_and_indices(self):
        from plyfile import PlyData
        xyz=np.array([[4,1,0],[2,0,0],[2,1,0]],np.float32)
        tri=np.array([[2,0,1]],np.int32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'labeled.ply'
            ps.write_labeled_ply(path,xyz,tri,np.ones_like(xyz)*.5,np.array([1,2,3],np.uint8),np.ones(3,np.float32))
            ply=PlyData.read(path,mmap=False)
            v=ply['vertex'].data
            np.testing.assert_array_equal(np.column_stack([v['x'],v['y'],v['z']]),xyz)
            np.testing.assert_array_equal(np.vstack(ply['face']['vertex_indices']),tri)
            np.testing.assert_array_equal(v['label'],[1,2,3])

    def test_detail_camera_matches_original_crop_projection(self):
        from refine_detail import camera_crop
        camera={'origin':[0,0,3], 'direction':[0,0,-1], 'right':[1,0,0],
                'screen_up':[0,1,0], 'extent':4}
        # Pixel-space boundary coordinates, so (4,8) is the crop center.
        crop=camera_crop(camera, [2,6,6,10], 16)
        xyz=np.array([[-1,0,1],[-.75,.25,1]],np.float32)
        for point in xyz:
            original=(point-np.array(camera['origin']))@np.array(camera['right'])/4*16+8
            detail=(point-np.array(crop['origin']))@np.array(crop['right'])/crop['extent']*16+8
            self.assertAlmostEqual(float(detail), float((original-2)*4), places=6)
        self.assertEqual(crop['extent'],1)
        with self.assertRaises(ValueError):
            camera_crop(camera, [0,0,4,8],16)

    def test_pot_envelope_rejects_exterior_substrate(self):
        from refine_detail import pot_envelope
        angle=np.linspace(0,2*np.pi,100,endpoint=False)
        height=np.linspace(0,2,20)
        a,h=np.meshgrid(angle,height)
        side=np.column_stack([np.cos(a).ravel(),h.ravel(),np.sin(a).ravel()]).astype(np.float32)
        probes=np.array([[0,1.9,0],[0,0.5,0],[1.5,1.9,0],[0,3,0]],np.float32)
        xyz=np.vstack([side,probes])
        labels=np.r_[np.full(len(side),4,np.uint8),np.zeros(len(probes),np.uint8)]
        result=pot_envelope(xyz,labels,'y')
        np.testing.assert_array_equal(result['substrate'][-4:],[True,False,False,False])
        moved=xyz*5+np.array([10,-3,8],np.float32)
        result2=pot_envelope(moved,labels,'y')
        np.testing.assert_array_equal(result['substrate'],result2['substrate'])

    def test_complete_labels_keep_metadata_and_faces(self):
        from complete_five import closed_face_labels
        from plyfile import PlyData
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
        tri=np.array([[0,1,2],[1,2,3]],np.int32)
        labels=np.array([1,2,2,3],np.uint8)
        scores=np.eye(5,dtype=np.float32)[labels-1]
        faces,boundary=closed_face_labels(tri,labels,scores)
        np.testing.assert_array_equal(faces,[2,2])
        self.assertTrue(boundary.all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'full.ply'
            ps.write_labeled_ply(path,xyz,tri,np.ones_like(xyz)*.5,labels,np.ones(4),True,
                                 uncertain=[0,1,0,1],inferred=[0,0,0,1],face_labels=faces)
            exported=PlyData.read(path,mmap=False)
            np.testing.assert_array_equal(exported['vertex']['label'],labels)
            np.testing.assert_array_equal(exported['vertex']['uncertain'],[0,1,0,1])
            np.testing.assert_array_equal(exported['face']['label'],[2,2])
            np.testing.assert_array_equal(np.vstack(exported['face']['vertex_indices']),tri)

    def test_graph_refinement_does_not_jump_disconnected_surfaces(self):
        from complete_five import mesh_graph,diffuse
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.0001],[1,0,.0001],[0,1,.0001]],np.float32)
        tri=np.array([[0,1,2],[3,4,5]],np.int32)
        graph=mesh_graph(xyz,tri,np.zeros_like(xyz))
        unary=np.zeros((6,5),np.float32)
        unary[:3,0]=1;unary[3:,1]=1
        result=diffuse(graph,unary,np.full(6,.1),np.zeros(6,bool),iterations=5)
        np.testing.assert_array_equal(result.argmax(1)+1,[1,1,1,2,2,2])

    def test_unobserved_vertices_follow_surface_seed_not_default_class(self):
        from complete_five import fill_unobserved_components
        from scipy import sparse
        graph=sparse.csr_matrix(np.array([[0,1,0,0],[.5,0,.5,0],[0,1,0,0],[0,0,0,0]],np.float32))
        unary=np.zeros((4,5),np.float32);unary[:,0]=1;unary[0]=[0,1,0,0,0]
        observed=np.array([True,False,False,False])
        refined,propagated=fill_unobserved_components(graph,unary,observed,iterations=30)
        np.testing.assert_array_equal(refined.argmax(1)+1,[2,2,2,1])
        np.testing.assert_array_equal(propagated,[False,True,True,False])

    def test_preview_handles_missing_pot_and_missing_detail_views(self):
        from argparse import Namespace
        from PIL import Image
        from complete_five import preview
        tri=np.array([[0,1,2]],np.int32)
        labels=np.array([2,2,2],np.uint8)
        rays=dict(face_id=np.zeros((2,2),np.int32),
                  barycentric=np.broadcast_to([1.,0.,0.],(2,2,3)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base,detail,out=(root/name for name in ['base','detail','out'])
            for path in (base/'views',detail/'views',out):
                path.mkdir(parents=True)
            cameras=[{'name':f'view_{i:03d}'} for i in range(14)]
            for camera in cameras:
                name=camera['name']
                Image.new('RGB',(2,2),'green').save(base/'views'/f'{name}.png')
                np.savez(base/'views'/f'{name}_rays.npz',**rays)
            np.savez(detail/'labels.npz',vertex_labels=labels)
            args=Namespace(base=base,detail=detail,output=out)
            ps.save_json(detail/'manifest.json',{'views':[]})
            preview(args,{'views':cameras},tri,labels)
            self.assertTrue((out/'comparison_preview.jpg').exists())
            self.assertFalse((out/'detail_preview.jpg').exists())
            crop={'name':'detail_000','kind':'linear_parts','elevation':12}
            Image.new('RGB',(2,2),'green').save(detail/'views'/'detail_000.png')
            np.savez(detail/'views'/'detail_000_rays.npz',**rays)
            ps.save_json(detail/'manifest.json',{'views':[crop]})
            preview(args,{'views':cameras},tri,labels)
            self.assertTrue((out/'detail_preview.jpg').exists())

    def test_leaf_boundary_does_not_cross_disconnected_surface(self):
        from scipy.sparse.csgraph import dijkstra
        from refine_leaf_boundary import surface_graph
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],
                      [0,0,.0001],[1,0,.0001],[0,1,.0001]],np.float32)
        tri=np.array([[0,1,2],[3,4,5]],np.int32)
        distance=dijkstra(surface_graph(xyz,tri),indices=[0],directed=False,min_only=True)
        self.assertTrue(np.isfinite(distance[:3]).all())
        self.assertTrue(np.isinf(distance[3:]).all())

    def test_leaf_boundary_requires_multiview_and_protects_other_organs(self):
        from refine_leaf_boundary import multiview_accept,select_mask
        labels=np.array([1,3,1,1,4,5,3,1],np.uint8)
        bud=np.array([0,0,0,0,0,0,1,0],bool)
        positive=np.array([3,4,2,3,5,5,5,5])
        observed=np.array([4,4,2,8,5,5,5,5])
        distance=np.array([.01,.01,.01,.01,.01,.01,.01,.1])
        allowed=multiview_accept(labels,bud,positive,observed,distance,1.)
        np.testing.assert_array_equal(allowed,[True,True,False,False,False,False,False,False])
        masks=np.ones((3,10,10),bool);core=np.zeros((10,10),bool);core[4:6,4:6]=True
        result,_=select_mask(masks,np.ones(3),core,np.ones_like(core))
        self.assertIsNone(result)  # Overbroad mask cannot replace the whole image.


if __name__=='__main__': unittest.main()
