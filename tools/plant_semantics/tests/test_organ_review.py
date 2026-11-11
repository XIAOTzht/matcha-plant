"""Shape/consensus/export checks without loading neural models."""
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from organ_shape_rules import mask_shape, shape_factor, decide_relabels, extend_surface_consensus
from organ_surface_consistency import (alpha_expansion, energy, regularize_surface,
    stem_shape_cost, refine_stem_rims)
from refine_organ_semantics import export
from with_car_semantics import mesh
from plyfile import PlyData, PlyElement


class OrganReviewTests(unittest.TestCase):
    def test_rim_review_follows_blade_preserves_stalk_and_disconnected_surface(self):
        # A flat blade with a narrow falsely tubular rim, plus an actual tube
        # joined at one point and a separate nearby ribbon with no mesh link.
        side=61
        xx,yy=np.meshgrid(np.linspace(0,1,side),np.linspace(0,1,side))
        blade=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        tri=[]
        for y in range(side-1):
            for x in range(side-1):
                a=y*side+x;tri.extend([[a,a+1,a+side],[a+1,a+side+1,a+side]])
        height,theta=np.meshgrid(np.linspace(1.02,1.7,25),np.arange(12)*2*np.pi/12,indexing='ij')
        tube=np.column_stack([(.5+.012*np.cos(theta)).ravel(),height.ravel(),(.012*np.sin(theta)).ravel()])
        start=len(blade)
        for j in range(24):
            for k in range(12):
                a=start+j*12+k;b=start+j*12+(k+1)%12
                tri.extend([[a,b,a+12],[b,b+12,a+12]])
        tri.append([(side-1)*side+side//2,start,start+1])
        detached=blade[:side].copy();detached[:,1]=.99;detached[:,2]=.0001
        # Separate thin ribbon only; close spatial proximity cannot connect it.
        detached=np.vstack([detached,detached+[0,.005,0]])
        off=start+len(tube)
        for k in range(side-1):tri.extend([[off+k,off+k+1,off+k+side],[off+k+1,off+k+side+1,off+k+side]])
        xyz=np.vstack([blade,tube,detached,[[0,0,-1],[1,0,-1],[0,1,-1]]])
        tri.append([len(xyz)-3,len(xyz)-2,len(xyz)-1]);tri=np.asarray(tri,np.int32)
        labels=np.ones(len(xyz),np.uint8);labels[:len(blade)]=2;labels[-3:]=[4,5,6]
        rim=np.flatnonzero((np.arange(len(xyz))<len(blade))&(xyz[:,1]>.97))
        labels[rim]=1
        features=np.tile([.85,.08,.07,.85,.08,.07],(len(xyz),1))
        features[labels==2]=[.05,.9,.05,.05,.9,.05]
        total=np.zeros((len(xyz),6));votes=np.zeros_like(total)
        total[labels==2,2]=.5;votes[labels==2,2]=2
        rgb=np.full_like(xyz,.4)
        out,report=refine_stem_rims(xyz,tri,rgb,labels,total,votes,features,2.5)
        self.assertGreater(np.mean(out[rim]==2),.70)
        np.testing.assert_array_equal(out[labels!=1],labels[labels!=1])
        np.testing.assert_array_equal(out[off:off+len(detached)],1)
        # Most of a true stalk extends away from the blade and stays a stalk.
        self.assertGreater(np.mean(out[start:off]==1),.95)
        self.assertTrue(report['nonstem_exact'])
        # A strong independent stem observation overrides the attachment rule.
        total[rim,1]=1.;votes[rim,1]=3
        protected,_=refine_stem_rims(xyz,tri,rgb,labels,total,votes,features,2.5)
        np.testing.assert_array_equal(protected[rim],1)
        # Scale and translation do not change the interpretation.
        transformed,_=refine_stem_rims(xyz*7+10000,tri,rgb,labels,total,votes,features,17.5)
        np.testing.assert_array_equal(protected,transformed)

    def test_sheet_shape_resolves_old_stem_using_organ_evidence(self):
        # The old stem prior exceeds a weak flower observation. Sheet geometry
        # should remove that default, without altering immutable support classes.
        xx,yy=np.meshgrid(np.linspace(-1,1,11),np.linspace(-1,1,11))
        xyz=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        tri=[]
        for y in range(10):
            for x in range(10):
                a=y*11+x;tri.extend([[a,a+1,a+11],[a+1,a+12,a+11]])
        xyz=np.vstack([xyz,[[0,0,1],[1,0,1],[0,1,1]]])
        tri=np.asarray(tri+[[121,122,123]],np.int32)
        labels=np.ones(len(xyz),np.uint8);labels[-3:]=[4,5,6]
        total=np.zeros((len(xyz),6));total[:121,3]=.16
        votes=np.zeros_like(total);votes[:121,3]=1
        features=np.tile([.1,.8,.1,.1,.8,.1],(len(xyz),1))
        out,report=regularize_surface(xyz,tri,np.full_like(xyz,.4),labels,
            total,votes,features,3.,cell_ratio=.01)
        np.testing.assert_array_equal(out[:121],3)
        np.testing.assert_array_equal(out[-3:],[4,5,6])
        self.assertTrue(report['hard_labels_preserved'])
        # Geometry has no preference for flower over leaf: reversing independent
        # observations reverses the answer, with all positions/RGB unchanged.
        total[:121,2]=total[:121,3];total[:121,3]=0
        votes[:121,2]=1;votes[:121,3]=0
        out,_=regularize_surface(xyz,tri,np.full_like(xyz,.4),labels,
            total,votes,features,3.,cell_ratio=.01)
        np.testing.assert_array_equal(out[:121],2)
        # Shape alone cannot invent an organ identity on an unobserved component.
        out,_=regularize_surface(xyz,tri,np.full_like(xyz,.4),labels,
            total*0,votes*0,features,3.,cell_ratio=.01)
        np.testing.assert_array_equal(out,labels)

    def test_curved_slender_stem_kept_despite_neighbouring_mask_spill(self):
        # Curvature reduces large-scale linearity. Use a short bent tube with
        # leaf evidence from an overlapping camera mask, but no sheet shape.
        t,angle=np.meshgrid(np.linspace(-.65,.65,31),np.linspace(0,2*np.pi,13)[:-1])
        xyz=np.column_stack([np.sin(t).ravel(),np.cos(t).ravel(),(.02*np.sin(angle)).ravel()])
        tri=[]
        for j in range(12):
            for i in range(30):
                a=j*31+i;b=((j+1)%12)*31+i
                tri.extend([[a,a+1,b],[a+1,b+1,b]])
        tri=np.asarray(tri,np.int32)
        labels=np.ones(len(xyz),np.uint8)
        total=np.zeros((len(xyz),6));total[:,2]=.5
        features=np.tile([.68,.18,.14,.60,.24,.16],(len(xyz),1))
        out,_=regularize_surface(xyz,tri,np.full_like(xyz,.4),labels,
            total,np.zeros_like(total),features,2.)
        np.testing.assert_array_equal(out,1)

    def test_sheet_cost_is_continuous_and_requires_two_scales(self):
        planar=np.linspace(.22,.6,1001)
        f=np.zeros((len(planar),6));f[:,[0,3]]=.1;f[:,1]=planar;f[:,4]=planar
        cost=stem_shape_cost(f)
        self.assertTrue((np.diff(cost)>=0).all())
        self.assertLess(np.max(np.diff(cost)),.002)
        # A narrow blade may be flat but also long: planarity cannot suffice.
        f[:,3]=.8
        np.testing.assert_array_equal(stem_shape_cost(f),0)

    def test_surface_optimizer_recovers_consistent_organ_keeps_isolated_stem(self):
        # One broad grid has confident leaf votes, but a peppered old label.
        xx,yy=np.meshgrid(np.linspace(-1,1,21),np.linspace(-1,1,21))
        xyz=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        tri=[]
        for y in range(20):
            for x in range(20):
                a=y*21+x;tri.extend([[a,a+1,a+21],[a+1,a+22,a+21]])
        xyz=np.vstack([xyz,[[0,0,.001],[.1,0,.001],[0,.1,.001]]])
        tri=np.asarray(tri+[[441,442,443]],np.int32)
        labels=np.full(len(xyz),2,np.uint8);labels[::13]=1;labels[-3:]=1
        evidence=np.zeros((len(xyz),6));evidence[:441,2]=1.;evidence[-3:,1]=2
        features=np.tile([.05,.9,.01,.05,.9,.01],(len(xyz),1));features[-3:]=[.95,.02,.01,.95,.02,.01]
        out,report=regularize_surface(xyz,tri,np.full_like(xyz,.4),labels,
            evidence,np.zeros_like(evidence),features,3.,cell_ratio=.01)
        np.testing.assert_array_equal(out[:441],2)
        np.testing.assert_array_equal(out[-3:],1)
        self.assertTrue(np.all(np.diff(report['energy_history'])<0))

    def test_expansion_matches_bruteforce_binary_potts(self):
        import itertools
        unary=np.array([[0.,2.],[1.1,0.],[1.4,0.],[0.,1.1]])
        edges=np.array([[0,1],[1,2],[2,3]]);weights=np.array([1.,.6,.7])
        found,_=alpha_expansion(unary,edges,weights)
        best=min(energy(np.array(choice),unary,edges,weights)for choice in itertools.product(range(2),repeat=4))
        self.assertAlmostEqual(energy(found,unary,edges,weights),best)

    def test_broad_planar_mask_cannot_be_stem_but_leaf_flower_remain_possible(self):
        xx, yy = np.meshgrid(np.linspace(-2,2,30),np.linspace(-1,1,20))
        points = np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        features = np.tile([.1,.8,.01,.1,.8,.01],(len(points),1))
        mask = np.ones(xx.shape,bool)
        s = mask_shape(mask,points,features,15.)
        self.assertEqual(shape_factor(1,s),0.)
        self.assertGreater(shape_factor(2,s),0.)
        self.assertGreater(shape_factor(3,s),0.)
        self.assertLess(shape_factor(3,s)/shape_factor(2,s),1.5)
        rotation,_ = np.linalg.qr(np.random.default_rng(2).normal(size=(3,3)))
        other = mask_shape(mask,points@rotation*7+12,features,105.)
        for k in ('elongation','thickness_ratio','width_fraction','length_fraction'):
            self.assertAlmostEqual(s[k],other[k],places=6)

    def test_thin_tubular_mask_can_be_stem(self):
        x, theta = np.meshgrid(np.linspace(-3,3,30),np.linspace(0,2*np.pi,20))
        points = np.column_stack([x.ravel(),.08*np.cos(theta.ravel()),.08*np.sin(theta.ravel())])
        features = np.tile([.9,.05,.01,.9,.05,.01],(len(points),1))
        s = mask_shape(np.ones(x.shape,bool),points,features,7.)
        self.assertGreater(shape_factor(1,s),.7)

    def test_whole_arrangement_cannot_supply_single_flower_vote(self):
        shape=dict(elongation=1.6,width_fraction=.4,length_fraction=.45,
                   linear_fraction=.02,planar_fraction=.8)
        self.assertEqual(shape_factor(3,shape),0.)
        self.assertEqual(shape_factor(2,shape),0.)

    def test_missing_detection_is_not_opposition_but_conflicting_identity_is(self):
        labels=np.array([1,1],np.uint8)
        scores=np.full((2,6),.1)
        total=np.zeros((2,6));total[:,2]=1.5
        votes=np.zeros((2,6),np.uint16);votes[:,2]=3;votes[1,3]=2
        observed=np.full(2,20,np.uint16)
        new,*_=decide_relabels(labels,scores,total,votes,observed)
        np.testing.assert_array_equal(new,[2,1])

    def test_consensus_and_hard_class_preservation(self):
        labels = np.array([1,1,1,4,5,6,0,2],np.uint8)
        scores = np.full((8,6),.1,np.float32)
        scores[2,0] = .99
        total = np.zeros((8,6)); total[:,3]=2
        votes = np.zeros((8,6),np.uint16); votes[:,3]=4
        observed = np.full(8,5,np.uint16)
        votes[1,3]=2; votes[2,3]=3
        new,changed,_,_ = decide_relabels(labels,scores,total,votes,observed)
        np.testing.assert_array_equal(new,[3,1,1,4,5,6,0,3])
        self.assertEqual(int(changed.sum()),2)
        # No evidence must not invent a new class.
        new,*_ = decide_relabels(labels,scores,total*0,votes*0,observed)
        np.testing.assert_array_equal(new,labels)

    def test_surface_extension_does_not_jump_disconnected_surfaces_or_tubes(self):
        xyz=np.array([[0,0,0],[.001,0,0],[.002,0,0],
                      [.001,.0001,0],[.002,.0001,0],[.003,.0001,0]],np.float32)
        tri=np.array([[0,1,2],[3,4,5]],np.int32)
        labels=np.ones(6,np.uint8);reviewed=labels.copy();reviewed[0]=2
        votes=np.zeros((6,6),np.uint16);votes[:,2]=1;votes[0,2]=4
        evidence=np.zeros((6,6),np.float32);evidence[:,2]=.8
        features=np.tile([.1,.8,.1,.1,.8,.1],(6,1))
        features[2,3:]=[.9,.05,.05]
        result,changed=extend_surface_consensus(xyz,tri,np.zeros((6,3)),labels,evidence,votes,
            np.ones(6)*5,features,1.,reviewed)
        np.testing.assert_array_equal(result,[2,2,1,1,1,1])
        self.assertEqual(int(changed.sum()),1)

    def test_export_preserves_geometry_rgb_source_and_label_alignment(self):
        xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],np.float32)
        tri = np.array([[0,1,2],[3,4,5],[0,6,7]],np.int32)
        old = np.array([1,1,2,4,5,6,2,3],np.uint8)
        v = np.empty(8,dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue','label','uncertain','inferred','label_source')])
        for k,a in zip(('x','y','z'),xyz.T): v[k]=a
        v['red']=50;v['green']=60;v['blue']=70;v['label']=old
        for k in ('uncertain','inferred','label_source'):v[k]=0
        f=np.empty(3,dtype=[('vertex_indices','i4',(3,)),('label','u1'),('uncertain','u1')])
        f['vertex_indices']=tri;f['label']=[1,0,4];f['uncertain']=0
        data=dict(vertex_labels=old.copy(),face_labels=f['label'].copy(),
            uncertain=np.zeros(8,bool),inferred=np.zeros(8,bool),label_source=np.zeros(8,np.uint8),
            class_scores=np.full((8,6),1/6,np.float32),class_score_ids=np.arange(1,7),
            class_score_valid=np.ones(8,bool),face_boundary=np.ones(3,bool),
            face_uncertain=np.zeros(3,bool),contact_uncertain=np.zeros(8,bool))
        data['uncertain'][7]=True;v['uncertain'][7]=1
        new=old.copy();new[:2]=2
        means=np.zeros((8,6),np.float32);means[:,2]=.5
        coord=dict(board_frame_axes=np.eye(3).tolist(),board_center_mm=[0,0,0])
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'new';out.mkdir()
            inp=source/'labeled_original_rgb_mm.ply'
            PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False).write(str(inp))
            np.savez_compressed(source/'labels.npz',**data)
            before=inp.read_bytes();npz_before=(source/'labels.npz').read_bytes()
            ply,positions,faces=mesh(inp)
            export(source,out,data,old,new,means,np.ones(8),[],positions,faces,ply,coord)
            self.assertEqual(before,inp.read_bytes())
            self.assertEqual(npz_before,(source/'labels.npz').read_bytes())
            result,px,pt=mesh(out/'labeled_original_rgb_mm.ply')
            np.testing.assert_array_equal(px,xyz)
            np.testing.assert_array_equal(pt,tri)
            np.testing.assert_array_equal(result['vertex'].data['red'],v['red'])
            with np.load(out/'labels.npz') as arrays:
                np.testing.assert_array_equal(result['vertex'].data['label'],arrays['vertex_labels'])
                np.testing.assert_array_equal(result['face'].data['label'],arrays['face_labels'])
                self.assertTrue(arrays['uncertain'][7])
                self.assertTrue(arrays['uncertain'][:2].all())
                self.assertTrue(arrays['contact_uncertain'][:2].all())
                self.assertEqual(arrays['face_labels'][0],2)
                np.testing.assert_array_equal(arrays['face_labels'][1:],f['label'][1:])
            self.assertTrue((out/'ORGAN_SEMANTIC_REVIEW_EXPORTED').is_file())
            # plyfile uses mmap; close views before Windows removes temp files.
            del result, ply
            import gc
            gc.collect()


if __name__ == '__main__':
    unittest.main()
