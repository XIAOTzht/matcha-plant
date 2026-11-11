#!/usr/bin/env python3
"""Reuse a photo run's models in a new, explicitly assumed-substrate revision.

Never edits the original run/state or reruns photo reconstruction. Rebuilds
organ/weight indices, plans, then requires the normal dynamic replay checks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/plant_pipeline'))
from common import JobLock, Stages, code_signature, command, confined, digest, now, read_json, run_name, write_json
from run_pipeline import runtime
from render_runtime import render_environment
from runtime_paths import temporary_environment


def select_source(root, run):
    path = root/'outputs/pipeline'/run/'pipeline_state.json'
    state = read_json(path)
    if state.get('status') == 'RUNNING':
        raise ValueError('Source run is still marked RUNNING; inspect it before creating a revision')
    stages = state['stages']
    selected = next((name for name in ('08c_cart_underside', '08b_organ_review', '08_cart_hardware')
                     if stages.get(name, {}).get('status') == 'SUCCESS'), None)
    if selected is None:
        raise ValueError('No completed metric semantic stage to reuse')
    verified = {}
    for name in ('04_mesh', '05_crop', '06_metric', selected):
        if stages.get(name, {}).get('status') != 'SUCCESS':
            raise ValueError('Missing successful source stage: '+name)
        for key, artifact in stages[name]['artifacts'].items():
            p = confined(root, artifact['path'])
            if not p.is_file() or digest(p) != artifact['sha256']:
                raise ValueError('Source artifact changed/missing: '+str(p))
            verified[name+'/'+key] = dict(path=str(p), sha256=artifact['sha256'])
    source = confined(root, stages[selected]['artifacts']['mesh']['path']).parent
    for name in ('labeled_original_rgb_mm.ply', 'semantic_mesh_mm.ply', 'labels.npz', 'coordinate_system.json', 'classes.json'):
        p = confined(root, source/name)
        if not p.is_file():
            raise ValueError('Missing semantic input: '+str(p))
        verified['semantic/'+name] = dict(path=str(p), sha256=digest(p))
    artifacts = {name: {k: confined(root, a['path']) for k, a in stages[name]['artifacts'].items()}
                 for name in ('04_mesh', '05_crop', '06_metric')}
    return source, artifacts, dict(source_run=run, semantic_stage=selected, artifacts=verified,
        source_state_sha256=digest(path), source_code_sha256=state['signature']['code_sha256']), path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--run', type=run_name, required=True, help='Original photo run, including a failed planning run')
    ap.add_argument('--revision', type=run_name, required=True, help='New result/job name')
    ap.add_argument('--assume-pot-interior-substrate', action='store_true')
    ap.add_argument('--rim-offset-mm', type=float, default=10.)
    ap.add_argument('--wall-inset-mm', type=float, default=6.)
    ap.add_argument('--height-mm', type=float)
    ap.add_argument('--submit', action='store_true', help='Start in server tmux')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prepare-only', action='store_true', help='Build matching weights and MuJoCo scene; no full success marker')
    args = ap.parse_args()
    if not args.assume_pot_interior_substrate:
        ap.error('Explicit --assume-pot-interior-substrate is required')
    if args.run == args.revision:
        ap.error('Use a different revision name to protect all original outputs')
    root = args.root.resolve()
    out = root/'outputs/pipeline'/args.revision
    job = root/'simulation/outputs/jobs'/args.revision
    if not args.resume and (out.exists() or job.exists()):
        raise ValueError('Revision exists. Use --resume or select a new revision name')
    py = runtime(root)
    if args.submit:
        if os.name != 'posix':
            raise ValueError('--submit requires Linux tmux')
        session = 'plant_'+args.revision
        if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
            print('VIRTUAL_SUBSTRATE_ALREADY_RUNNING', session); return
        with JobLock(root/'logs/plant_pipeline.lock'):
            pass
        log = root/'logs'/('pipeline_'+args.revision+'.log')
        argv = [py['semantics'], '-u', Path(__file__).resolve(), '--root', root, '--run', args.run,
            '--revision', args.revision, '--assume-pot-interior-substrate',
            '--rim-offset-mm', str(args.rim_offset_mm), '--wall-inset-mm', str(args.wall_inset_mm)]
        if args.height_mm is not None:argv += ['--height-mm', str(args.height_mm)]
        if args.resume:argv.append('--resume')
        if args.prepare_only:argv.append('--prepare-only')
        shell = shlex.join(list(map(str, argv)))+' >> '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
        print('VIRTUAL_SUBSTRATE_SUBMITTED', session, 'log='+str(log), flush=True)
        return
    env = temporary_environment(root, dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    cfg = root/'configs/plant_pipeline.json'
    with JobLock(root/'logs/plant_pipeline.lock'):
        source, reused, lineage, original_state = select_source(root, args.run)
        signature = dict(version='virtual-substrate-revision-1', source=lineage, code_sha256=code_signature(root),
            launcher_sha256=digest(Path(__file__)), settings=digest(cfg),
            contact_policy=digest(root/'tools/contact_weights/contact_policy.json'),
            geometry_policy=digest(root/'tools/contact_weights/organ_weight_policy.json'),
            rim_offset_mm=args.rim_offset_mm, wall_inset_mm=args.wall_inset_mm, height_mm=args.height_mm,
            prepare_only=args.prepare_only, explicit_assumption_authorized=True)
        stages = Stages(out, signature)
        if stages.state.get('status') == 'PHOTO_TO_PATH_SUCCESS':
            print('PHOTO_TO_PATH_SUCCESS', out); return
        if not (out/'upstream_pipeline_state.json').exists():
            shutil.copyfile(original_state, out/'upstream_pipeline_state.json')
        write_json(out/'source_reuse.json', lineage)
        try:
            def substrate(d):
                argv = [py['semantics'], root/'tools/plant_extract_with_car/add_virtual_substrate.py',
                    '--source', source, '--output', d/'result', '--assume-pot-interior-substrate',
                    '--rim-offset-mm', args.rim_offset_mm, '--wall-inset-mm', args.wall_inset_mm]
                if args.height_mm is not None:argv += ['--height-mm', args.height_mm]
                command(argv, d, root, env)
                if read_json(d/'result/validation.json')['status'] != 'PASS':
                    raise ValueError('Virtual surface append-only validation failed')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz',
                    report=d/'result/report.json', assumption=d/'result/virtual_substrate.json',
                    validation=d/'result/validation.json', marker=d/'result/VIRTUAL_SUBSTRATE_EXPORTED')
            se = stages.execute('08d_virtual_substrate', substrate)
            def weights(d):
                command([py['semantics'], root/'scripts/plant_contact/run_pipeline.py', '--input-dir', se['mesh'].parent,
                    '--output-dir', d/'result', '--no-parts'], d, root, env)
                return dict(weights=d/'result/weights/contact_weights.npz', organs=d/'result/organs/organ_instances.npz',
                    marker=d/'result/PLANT_CONTACT_EXPORT_SUCCESS')
            we = stages.execute('09_weights', weights)
            def planning(d):
                argv = [py['planning'], root/'simulation/scripts/run_server_job.py', '--run', args.revision,
                    '--semantic-dir', se['mesh'].parent, '--contact-dir', we['weights'].parent.parent,
                    '--scale-report', reused['06_metric']['report'], '--settings', cfg, '--resume']
                if args.prepare_only:argv.append('--prepare-only')
                command(argv, d, root, render_environment(root, env))
                if args.prepare_only:
                    return dict(state=job/'pipeline_state.json')
                report = read_json(job/'job_report.json')
                if (report['status'] != 'SIMULATION_JOB_SUCCESS'
                        or report.get('watering_surface_source') != 'assumed_virtual_substrate'
                        or not all(report['validation']['checks'].values())):
                    raise ValueError('Path/replay verification or assumed-surface provenance failed')
                return dict(report=job/'job_report.json', marker=job/'SIMULATION_JOB_SUCCESS')
            pl = stages.execute('10_planning', planning)
            if args.prepare_only:
                write_json(out/'PREPARED_ONLY.json', dict(status='SCENE_PREPARED_ONLY',
                    note='No path planned or verified; no PHOTO_TO_PATH_SUCCESS marker', job=str(job)))
                stages.state.update(status='SCENE_PREPARED_ONLY');write_json(stages.path, stages.state)
                print('SCENE_PREPARED_ONLY', out);return
            simulation = read_json(pl['report'])
            result = dict(status='PHOTO_TO_PATH_SUCCESS', run=args.revision, source_run=args.run,
                finished_utc=now(), watering_surface_source='assumed_virtual_substrate', measured_soil_available=False,
                virtual_substrate_report=str(se['report']), virtual_substrate=str(se['assumption']),
                semantic_mesh=str(se['mesh']), contact_weights=str(we['weights']),
                raw_mesh=str(reused['04_mesh']['mesh']), crop_mesh=str(reused['05_crop']['mesh']),
                metric_mesh=str(reused['06_metric']['mesh']), scale_report=str(reused['06_metric']['report']),
                input=str(root/'data/pipeline'/args.run),
                source_reuse=str(out/'source_reuse.json'), simulation_report=str(pl['report']), simulation=simulation,
                real_robot_ready=False, reconstruction_reused=True,
                review_required=['Assumed pot-interior surface height and footprint', 'Plant/organ semantics',
                    'Virtual robot placement and placeholder nozzle'])
            write_json(out/'result_paths.json', result)
            stages.state.update(status='PHOTO_TO_PATH_SUCCESS', finished_utc=now());write_json(stages.path, stages.state)
            (out/'PHOTO_TO_PATH_SUCCESS').write_text(now()+'\nASSUMED_VIRTUAL_SUBSTRATE\n', encoding='ascii')
            (out/'PIPELINE_FAILED.json').unlink(missing_ok=True)
            print('PHOTO_TO_PATH_SUCCESS (ASSUMED_VIRTUAL_SUBSTRATE)', out, flush=True)
        except BaseException as exc:
            write_json(out/'PIPELINE_FAILED.json', dict(error=str(exc), stage=stages.state.get('current_stage'), time=now()))
            raise


if __name__ == '__main__':
    main()
