#!/usr/bin/env python3
"""Permit only the reviewed single-triangle display fix on a stopped revision.

Verify both job signatures and every completed artifact, then back up the state
files before updating their code signature. All numeric inputs remain unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/plant_pipeline'))
from common import JobLock, code_signature, confined, digest, now, read_json, run_name, write_json
from continue_with_virtual_substrate import select_source

BUILDER = 'simulation/scripts/build_watering_scene.py'
OLD_BUILDER = 'df5850d5182f357542cb01318f536bd4c2e0fbff551ecba8a8b1c322e4ddb5c3'
FIXED_BUILDER = 'ac325e5e3185e3c9caa4d2d35c4a05f80a04f26af3a6bfbc80b723182d543e5a'


def verify_artifacts(root, state):
    for name, stage in state['stages'].items():
        if stage.get('status') != 'SUCCESS':continue
        print('VERIFY_COMPLETED_STAGE', name, flush=True)
        for entry in stage['artifacts'].values():
            p = confined(root, entry['path'])
            if not p.is_file() or digest(p) != entry['sha256']:
                raise ValueError('Completed output changed/missing: '+str(p))


def migrate(root, run):
    root = Path(root).resolve()
    if digest(root/BUILDER) != FIXED_BUILDER:
        raise ValueError('Unreviewed display builder; refusing signature migration')
    with JobLock(root/'logs/plant_pipeline.lock'):
        path = root/'outputs/pipeline'/run/'pipeline_state.json'
        job = root/'simulation/outputs/jobs'/run/'pipeline_state.json'
        outer, inner = read_json(path), read_json(job)
        if (outer.get('status') != 'FAILED' or outer.get('current_stage') != '10_planning'
                or outer['stages'].get('10_planning', {}).get('status') != 'FAILED'
                or inner.get('status') != 'FAILED' or inner.get('current_stage') != '02_scene'
                or inner['stages'].get('02_scene', {}).get('status') != 'FAILED'):
            raise ValueError('Recovery requires stopped failures at 10_planning / 02_scene')
        if any(e.get('status') == 'SUCCESS' for n, e in inner['stages'].items() if n != '01_inputs'):
            raise ValueError('Cannot migrate a job with a completed scene or path')
        if any((p.parent/marker).exists() for p, marker in ((path,'PHOTO_TO_PATH_SUCCESS'),(job,'SIMULATION_JOB_SUCCESS'))):
            raise ValueError('Preserving existing success markers')
        for name in ('08d_virtual_substrate','09_weights'):
            if outer['stages'].get(name,{}).get('status') != 'SUCCESS':
                raise ValueError('Missing successful virtual substrate/weights: '+name)
        if inner['stages'].get('01_inputs',{}).get('status') != 'SUCCESS':
            raise ValueError('Input import did not succeed')
        current = code_signature(root)
        previous = code_signature(root, overrides={BUILDER:OLD_BUILDER})
        for state in (outer, inner):
            if state['signature']['code_sha256'] not in (previous, current):
                raise ValueError('Code outside the reviewed display fix changed; use a new revision')
        signature = outer['signature']
        if (signature.get('version') != 'virtual-substrate-revision-1'
                or signature.get('explicit_assumption_authorized') is not True
                or signature.get('prepare_only') is not False):
            raise ValueError('Expected an explicitly authorized virtual-substrate planning revision')
        _, reused, lineage, _ = select_source(root, run_name(signature['source']['source_run']))
        if signature['source'] != lineage:
            raise ValueError('Upstream source identity changed')
        cfg = root/'configs/plant_pipeline.json'
        expected_fields = dict(settings=digest(cfg),
            launcher_sha256=digest(root/'maintenance/continue_with_virtual_substrate.py'),
            contact_policy=digest(root/'tools/contact_weights/contact_policy.json'),
            geometry_policy=digest(root/'tools/contact_weights/organ_weight_policy.json'))
        if any(signature.get(k) != v for k,v in expected_fields.items()):
            raise ValueError('Launcher, settings or weight policy changed')
        semantic = confined(root, outer['stages']['08d_virtual_substrate']['artifacts']['mesh']['path']).parent
        contact = confined(root, outer['stages']['09_weights']['artifacts']['weights']['path']).parent.parent
        inputs = dict(semantic_mesh=semantic/'semantic_mesh_mm.ply',semantic_labels=semantic/'labels.npz',
            coordinate_system=semantic/'coordinate_system.json', weights=contact/'weights/contact_weights.npz',
            organs=contact/'organs/organ_instances.npz',policy=root/'tools/contact_weights/contact_policy.json',
            palette=root/'tools/contact_weights/weight_palette.py',virtual_substrate=semantic/'virtual_substrate.json')
        settings = read_json(cfg)
        expected_sim = {k:digest(p) for k,p in inputs.items()}
        expected_sim.update(settings=digest(cfg),template=digest(root/settings['simulation_scene_template']),
            timing=digest(root/settings['simulation_timing_config']), scale=digest(reused['06_metric']['report']),
            version='photo-to-path-1',code_sha256=inner['signature']['code_sha256'])
        for key, name in [('coordination','simulation_coordination_config'),('natural','simulation_natural_config'),
                          ('demonstration','simulation_demonstration_config')]:
            if settings.get(name):expected_sim[key] = digest(root/settings[name])
        if inner['signature'] != expected_sim:
            raise ValueError('Simulation inputs/settings changed; refusing to reuse the imported scene inputs')
        # Validate BOTH states completely before either is changed.
        verify_artifacts(root, outer)
        verify_artifacts(root, inner)
        tag = uuid.uuid4().hex[:8]
        for p, state in ((path, outer), (job, inner)):
            if state['signature']['code_sha256'] == current:continue
            backup = p.with_name('pipeline_state.before_display_fix_'+tag+'.json')
            shutil.copyfile(p, backup)
            state.setdefault('display_code_updates', []).append(dict(previous_state=str(backup),
                previous_code_sha256=state['signature']['code_sha256'],code_sha256=current,time=now(),
                reason='Subdivide three-vertex visual color bins without altering source geometry or weights'))
            state['signature']['code_sha256'] = current
            write_json(p, state)
        for p, name in ((path,'PIPELINE_FAILED.json'),(job,'JOB_FAILED.json')):
            failure = p.parent/name
            if failure.exists():
                failure.rename(p.parent/(failure.stem+'.before_display_fix_'+tag+'.json'))
        print('DISPLAY_FIX_RECOVERY_VERIFIED', run, 'resume_stage=02_scene', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--run', type=run_name, required=True)
    args = ap.parse_args()
    migrate(args.root, args.run)
