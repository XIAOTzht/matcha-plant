import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from build_watering_scene import _build_stepped_nozzle
from plant_scene import read_json
from robot_volume_query import AdaptiveVolumeQuery


class SteppedNozzleTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(Path(__file__).resolve().parents[1]/'configs/weighted_stepped_nozzle_scene.json')['nozzle']
        self.xml=ET.Element('mujoco')
        asset=ET.SubElement(self.xml,'asset'); world=ET.SubElement(self.xml,'worldbody')
        mount=ET.SubElement(world,'body',name='placeholder_nozzle')
        _build_stepped_nozzle(asset,mount,self.config,np.array([0.,0.,-1.]))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(self.xml,encoding='unicode'))
        self.data=mujoco.MjData(self.model);mujoco.mj_forward(self.model,self.data)

    def test_visual_vertices_stay_in_segment_envelopes(self):
        sections=[self.config['rear_grip'],*self.config['segments']]
        for mesh in self.xml.findall('./asset/mesh'):
            xyz=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3)
            axial=-xyz[:,2]*1000; radius=np.linalg.norm(xyz[:,:2],axis=1)*1000
            covered=np.zeros(len(xyz),dtype=bool)
            for section in sections:
                covered|=((axial>=section['start_mm']-1e-5)&(axial<=section['end_mm']+1e-5)
                          &(radius<=section['diameter_mm']/2+1e-5))
            self.assertTrue(covered.all(),mesh.get('name'))
        for i in range(self.model.ngeom):
            if self.model.geom(i).name.endswith('_visual'):
                self.assertEqual(int(self.model.geom_contype[i]),0)
                self.assertEqual(int(self.model.geom_conaffinity[i]),0)

    def test_front_cap_is_checked_when_handle_is_clear(self):
        ids=[i for i in range(self.model.ngeom) if self.model.geom_contype[i]]
        self.assertEqual(len(ids),7)
        query=AdaptiveVolumeQuery(self.model,ids,dict(cell_tolerance_mm=.1))
        distance=lambda points: abs(points[:,2]+.115)
        handle=self.model.geom('nozzle_grasp_handle').id
        cap=self.model.geom('nozzle_front_cap').id
        self.assertGreater(query.clearance(self.data,handle,distance,.0005),.0005)
        self.assertLessEqual(query.clearance(self.data,cap,distance,.0005),.0005)
        np.testing.assert_allclose(self.data.site_xpos[self.model.site('nozzle_tip').id],[0,0,-.120],atol=1e-10)


if __name__=='__main__':unittest.main()
