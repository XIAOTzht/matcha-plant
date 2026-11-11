import unittest
import numpy as np
from repair_cart_underside import repair_geometry,plane_pieces,tessellate_cap,validate_template,remap_data


def template():
    return dict(units='mm',bottom_y_mm=-10.,chassis_top_y_mm=-5.,
        footprint_xz_mm=[[-20,-20],[20,-20],[20,20],[-20,20]],
        wheels=[dict(center_xy_mm=[x,-10],radius_mm=3.,z_range_mm=[z-3,z+3])
                for x in (-10,10)for z in (-20,20)])


class UndersideTests(unittest.TestCase):
    def test_cap_area_and_wheel_slots(self):
        t=template();p=validate_template(t);area=0.
        for part in plane_pieces(p,t):
            points,tri=tessellate_cap(part)
            if not len(tri):continue
            a=points[tri[:,1]]-points[tri[:,0]];b=points[tri[:,2]]-points[tri[:,0]]
            area+=np.sum(abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]))/2
            centers=points[tri].mean(1)
            for w in t['wheels']:
                inside=(abs(centers[:,0]-w['center_xy_mm'][0])<3-1e-8)&(centers[:,1]>w['z_range_mm'][0]+1e-8)&(centers[:,1]<w['z_range_mm'][1]-1e-8)
                self.assertFalse(inside.any())
        self.assertAlmostEqual(area,1600-4*6*3,places=5)

    def test_preserve_non_cart_and_wheel_cut_spike(self):
        q=np.array([[0,-5,0],[4,-5,0],[2,-22,1], # cart spike
                    [0,-20,4],[4,-20,4],[2,-22,5], # plant below assumed bottom
                    [-10,-11,-20],[-9,-11,-20],[-10,-12,-20]],float)
        tri=np.arange(9).reshape(-1,3);labels=np.array([6]*3+[2]*3+[6]*3);fl=np.array([6,2,6])
        r,f,origin,fs,cap,stats=repair_geometry(q,tri,labels,fl,template())
        self.assertNotIn(2,origin)
        for i in range(3,9):self.assertIn(i,origin)
        for k in (1,2):
            retained=f[fs==k];self.assertEqual(len(retained),1)
            np.testing.assert_array_equal(origin[retained[0]],tri[k])
        self.assertTrue(np.all(r[cap,1]==-10))
        new=(origin<0)&(~cap)
        self.assertTrue(np.allclose(r[new,1],-10))

    def test_non_cart_face_is_protected(self):
        q=np.array([[0,-5,0],[4,-5,0],[2,-22,1]],float)
        r,f,ori,fs,_,_=repair_geometry(q,np.array([[0,1,2]]),np.array([6,6,6]),np.array([4]),template())
        np.testing.assert_array_equal(ori[f[fs==0][0]],[0,1,2])

    def test_arrays_and_index_remap(self):
        n=9;m=4;origin=np.array([0,2,3,4,5,6,7,8,-1]);fo=np.array([1,2,3,-1])
        data=dict(vertex_labels=np.array([6,6,2,3,4,5,6,6,6],np.uint8),face_labels=np.full(m,6,np.uint8),
            uncertain=np.zeros(n,bool),inferred=np.zeros(n,bool),label_source=np.ones(n,np.uint8),
            class_score_valid=np.ones(n,bool),class_scores=np.ones((n,6),np.float32),
            face_uncertain=np.zeros(m,bool),plant_vertex_indices=np.array([2,3,4,5]),class_score_ids=np.arange(6))
        d=remap_data(data,origin,fo,n,m)
        np.testing.assert_array_equal(d['plant_vertex_indices'],[1,2,3,4])
        self.assertEqual(d['vertex_labels'][-1],6);self.assertTrue(d['inferred'][-1])
        self.assertFalse(d['class_score_valid'][-1]);self.assertTrue(np.isnan(d['class_scores'][-1]).all())
        self.assertEqual(d['face_labels'][-1],6)

    def test_invalid_template_rejected(self):
        t=template();t['bottom_y_mm']=float('nan')
        with self.assertRaises(ValueError):validate_template(t)


if __name__=='__main__':unittest.main()
