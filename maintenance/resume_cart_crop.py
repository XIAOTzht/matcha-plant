"""Authorize the reviewed cart-geometry crop upgrade after stage 05 failed.

Only the crop implementation may differ. Verify inputs/config/policies and all
successful reconstruction artifacts, back up state, then allow normal --resume.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts/plant_pipeline'))
from common import JobLock, code_signature, digest, now, read_json, run_name, verify_images, write_json

OLD_CROP_SHA256 = '0b8bcd1a9a9bc14eac7165daeccf8cc019ac646aa3eea51ac7fc4d299660e8ff'
REVIEWED_CROP_SHA256 = '49fd2844dc779b5147dea88f24110def3965d52ea05fa92c1848fbc06992dd2c'
REVIEWED_GEOMETRY_SHA256 = '65bc45ce2dd6f2a9044e784061ed447578c663b1c812ab2b5596726065aa20a6'


def migrate(root, run):
    root = Path(root).resolve()
    crop = 'tools/plant_extract_with_car/auto_crop_with_car.py'
    geometry = 'tools/plant_extract_with_car/cart_geometry.py'
    for name, expected in ((crop, REVIEWED_CROP_SHA256), (geometry, REVIEWED_GEOMETRY_SHA256)):
        if digest(root/name) != expected:
            raise ValueError('Unreviewed crop code; refusing to reuse reconstruction: '+name)
    with JobLock(root/'logs/plant_pipeline.lock'):
        path = root/'outputs/pipeline'/run/'pipeline_state.json'
        state = read_json(path)
        if state.get('current_stage') != '05_crop' or state.get('status') != 'FAILED':
            raise ValueError('Upgrade requires this run to be stopped after failed stage 05_crop')
        stages = state['stages']
        if any(stages.get(name, {}).get('status') != 'SUCCESS'
               for name in ('01_sfm','02_alignment','03_gaussians','04_mesh')):
            raise ValueError('All four reconstruction stages must have succeeded')
        if stages.get('05_crop', {}).get('status') != 'FAILED' or any(
                name > '05_crop' and entry.get('status') == 'SUCCESS' for name, entry in stages.items()):
            raise ValueError('Cannot migrate a run with successful crop or downstream outputs')
        old = state['signature']
        current_code = code_signature(root)
        previous_code = code_signature(root, overrides={crop:OLD_CROP_SHA256}, omitted=(geometry,))
        if old['code_sha256'] not in (previous_code, current_code):
            raise ValueError('Code outside the reviewed crop upgrade changed; refusing signature migration')
        data = root/'data/pipeline'/run
        verify_images(data)
        if digest(data/'input_manifest.json') != old['input_sha256']:
            raise ValueError('Input photo manifest changed')
        if digest(root/'configs/plant_pipeline.json') != old['config_sha256']:
            raise ValueError('Pipeline config changed; this helper supports the default pipeline configuration')
        for name, expected in old['policies'].items():
            p = (root/name).resolve(); p.relative_to(root)
            if digest(p) != expected:
                raise ValueError('Numerical policy changed: '+name)
        for name, entry in stages.items():
            if entry.get('status') != 'SUCCESS':
                continue
            print('VERIFY_RECONSTRUCTION_STAGE', name, flush=True)
            for artifact in entry['artifacts'].values():
                p = Path(artifact['path']).resolve(); p.relative_to(root)
                if not p.is_file() or digest(p) != artifact['sha256']:
                    raise ValueError('Completed stage output changed/missing: '+str(p))
        if old['code_sha256'] == current_code:
            print('CART_CROP_UPGRADE_ALREADY_VERIFIED', run, flush=True)
            return
        backup = path.with_name('pipeline_state.before_cart_crop_upgrade_'+uuid.uuid4().hex[:8]+'.json')
        shutil.copy2(path, backup)
        state.setdefault('crop_algorithm_updates', []).append(dict(
            previous_code_sha256=old['code_sha256'], code_sha256=current_code, updated_utc=now(),
            previous_state=str(backup), reason='fixed-cart upper-board anchored crop without plant color',
            reuse_stages=['01_sfm','02_alignment','03_gaussians','04_mesh'], restart_stage='05_crop'))
        state['signature'] = dict(old, code_sha256=current_code)
        write_json(path, state)
        print('CART_CROP_UPGRADE_VERIFIED', run, 'backup='+str(backup), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=PROJECT)
    ap.add_argument('--run', type=run_name, required=True)
    args = ap.parse_args()
    migrate(args.root, args.run)
