import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from run_server_job import copy_scene


class PortableSceneTests(unittest.TestCase):
    def test_copy_rebases_robot_resources_but_keeps_local_display_and_inline_nozzle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);robot=root/'robot.stl';robot.write_bytes(b'resource')
            source=root/'original/scene';source.mkdir(parents=True)
            display=source/'display_meshes/leaf.stl';display.parent.mkdir();display.write_bytes(b'leaf')
            tree=ET.Element('mujoco');asset=ET.SubElement(tree,'asset')
            ET.SubElement(asset,'mesh',file=os.path.relpath(robot,source))
            ET.SubElement(asset,'mesh',file='display_meshes/leaf.stl')
            ET.SubElement(asset,'mesh',vertex='0 0 0 1 1 1')
            ET.ElementTree(tree).write(source/'scene.xml')
            destination=root/'different/deeper/output/scene'
            copy_scene(source,destination)
            meshes=ET.parse(destination/'scene.xml').findall('asset/mesh')
            self.assertEqual((destination/meshes[0].get('file')).resolve(),robot)
            self.assertEqual((destination/meshes[1].get('file')).read_bytes(),b'leaf')
            self.assertIsNone(meshes[2].get('file'))


if __name__=='__main__':unittest.main()
