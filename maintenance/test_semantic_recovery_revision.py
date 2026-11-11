"""Exercise actual revision routing with file-producing command substitutes."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import refine_plant_semantics as revision


class RevisionIntegrationTests(unittest.TestCase):
    def test_virtual_source_is_not_repaired_again_and_old_results_are_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = 'virtual_source'; rev = 'new_review'
            source = root/'outputs/pipeline'/run/'08d_virtual_substrate/result'
            source.mkdir(parents=True)
            for name in ('semantic_mesh_mm.ply', 'labeled_original_rgb_mm.ply',
                         'labels.npz', 'coordinate_system.json'):
                (source/name).write_bytes(b'original source bytes')
            (source/'virtual_substrate.json').write_text('{"measured":false}')
            scale = root/'scale.json'; scale.write_text('{}')
            report = dict(status='PHOTO_TO_PATH_SUCCESS',semantic_mesh=str(source/'semantic_mesh_mm.ply'),
                          scale_report=str(scale),watering_surface_source='assumed_virtual_substrate')
            index = source.parents[1]/'result_paths.json'
            index.write_text(json.dumps(report))
            (index.parent/'PHOTO_TO_PATH_SUCCESS').write_text('source success')
            for rel in ('configs/plant_pipeline.json','tools/contact_weights/contact_policy.json',
                        'tools/contact_weights/organ_weight_policy.json'):
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(dict(cart_underside=dict(enabled=True))))
            before={p:hashlib.sha256(p.read_bytes()).hexdigest()for p in source.rglob('*')if p.is_file()}
            calls=[]
            def fake_command(argv,directory,*args):
                argv=list(map(str,argv));calls.append(argv)
                result=Path(directory)/'result';result.mkdir(parents=True)
                if 'refine_organ_semantics.py' in ' '.join(argv):
                    names=['semantic_mesh_mm.ply','labels.npz','report.json','semantic_surface_comparison.jpg',
                           'ORGAN_SEMANTIC_REVIEW_EXPORTED','virtual_substrate.json']
                    for name in names:(result/name).write_text('{}')
                    (result/'validation.json').write_text('{"status":"PASS"}')
                else:
                    for rel in ['weights/contact_weights.npz','weights/contact_risk_heatmap.ply','PLANT_CONTACT_EXPORT_SUCCESS']:
                        p=result/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'new result')
            argv=['refine_plant_semantics.py','--root',str(root),'--run',run,'--revision',rev]
            with patch.object(sys,'argv',argv),patch.object(revision,'command',side_effect=fake_command), \
                    patch.object(revision,'code_signature',return_value='new source hash'), \
                    patch.object(revision,'runtime',return_value=dict(semantics=Path('python'),planning=Path('python'))), \
                    patch.object(revision,'JobLock',return_value=nullcontext()):
                revision.main()
            self.assertEqual(len(calls),2)
            self.assertFalse(any('repair_cart_underside' in ' '.join(c)for c in calls))
            result=json.loads((root/'outputs/semantic_refinement'/rev/'result_paths.json').read_text())
            self.assertFalse(result['path_regenerated'])
            self.assertTrue(result['existing_cart_repair_preserved'])
            self.assertEqual(result['watering_surface_source'],'assumed_virtual_substrate')
            for p,digest in before.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)


if __name__=='__main__':
    unittest.main()
