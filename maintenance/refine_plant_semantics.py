#!/usr/bin/env python3
"""Revise semantics/weights of an existing completed run, without reconstruction.

Creates a separate revision. Existing pipeline results and stage state are read
only. Planning is optional, always with a new job name and new label/weight pair.
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/plant_pipeline'))
from common import (JobLock, Stages, code_signature, confined, command, digest,
                    now, read_json, run_name, write_json)
from run_pipeline import runtime
from render_runtime import render_environment
from runtime_paths import temporary_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run', type=run_name, required=True, help='Existing completed photo run')
    p.add_argument('--revision', type=run_name, required=True, help='New result/job name')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--submit', action='store_true', help='Start this operation in tmux')
    p.add_argument('--plan', action='store_true', help='Also regenerate and validate a path; never reuse the old path')
    args = p.parse_args()
    root = args.root.resolve()
    if args.gpu < 0 or args.revision == args.run:
        p.error('Use a nonnegative GPU and a new revision name')
    report_path = root/'outputs/pipeline'/args.run/'result_paths.json'
    report = read_json(report_path)
    if report['status'] != 'PHOTO_TO_PATH_SUCCESS' or not (report_path.parent/'PHOTO_TO_PATH_SUCCESS').is_file():
        raise ValueError('Select an existing completed run')
    source = confined(root, report['semantic_mesh']).parent
    scale = confined(root, report['scale_report'])
    for name in ('labeled_original_rgb_mm.ply','labels.npz','coordinate_system.json'):
        if not (source/name).is_file():
            raise ValueError('Missing semantic source: '+str(source/name))
    out = root/'outputs/semantic_refinement'/args.revision
    py = runtime(root)
    if args.submit:
        if os.name != 'posix':
            raise ValueError('--submit is for Linux tmux; run this on the server')
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise ValueError('Revision exists; use --resume or select a new revision')
        session = 'review_'+args.revision
        if subprocess.run(['tmux','has-session','-t',session], capture_output=True).returncode == 0:
            print('SEMANTIC_REVIEW_ALREADY_RUNNING', session); return
        with JobLock(root/'logs/plant_pipeline.lock'):
            pass
        log = root/'logs'/f'semantic_{args.revision}.log'
        argv = [py['semantics'], '-u', root/'maintenance/refine_plant_semantics.py',
                '--root', root, '--run', args.run, '--revision', args.revision, '--gpu', str(args.gpu)]
        if args.resume: argv.append('--resume')
        if args.plan: argv.append('--plan')
        shell = shlex.join(list(map(str,argv)))+' >> '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,shell], check=True)
        print('SEMANTIC_REVIEW_SUBMITTED', session, 'log='+str(log), flush=True)
        return
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Revision exists; use --resume or select a new revision')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PLANTSEG_SERVER='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env = temporary_environment(root, env)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    if args.plan and not args.resume and (root/'simulation/outputs/jobs'/args.revision).exists():
        raise ValueError('Simulation revision already exists; choose a new name')
    with JobLock(root/'logs/plant_pipeline.lock'):
        stages = Stages(out, dict(source_mesh=digest(source/'labeled_original_rgb_mm.ply'),
            source_labels=digest(source/'labels.npz'), coordinate_system=digest(source/'coordinate_system.json'),
            scale=digest(scale), settings=digest(root/'configs/plant_pipeline.json'),
            contact_policy=digest(root/'tools/contact_weights/contact_policy.json'),
            geometry_policy=digest(root/'tools/contact_weights/organ_weight_policy.json'),
            code_sha256=code_signature(root), launcher_sha256=digest(Path(__file__)), gpu=args.gpu, plan=args.plan))
        try:
            def review(d):
                command([py['semantics'],root/'tools/plant_semantics/refine_organ_semantics.py',
                    '--source',source,'--output',d/'result','--device','cuda'], d,root,env)
                # Provenance transport does not change the approved classifier.
                # Its unchanged export keeps all numeric fields in labels.npz.
                for name in ('virtual_substrate.json','cart_template.json','source_mapping.npz'):
                    if (source/name).is_file():
                        shutil.copy2(source/name,d/'result'/name)
                if read_json(d/'result/validation.json')['status'] != 'PASS':
                    raise ValueError('Semantic export consistency check failed')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz',
                    report=d/'result/report.json', preview=d/'result/semantic_surface_comparison.jpg',
                    validation=d/'result/validation.json', marker=d/'result/ORGAN_SEMANTIC_REVIEW_EXPORTED')
            se = stages.execute('01_semantics',review)
            scene_semantics = se
            underside = read_json(root/'configs/plant_pipeline.json').get('cart_underside',{})
            # Existing repaired/virtual geometry already has a checked cart
            # underside. Rebuilding it here could discard appended provenance.
            existing_cart_repair = (source/'cart_template.json').is_file() or (source/'virtual_substrate.json').is_file()
            if underside.get('enabled',False) and not existing_cart_repair:
                def repair_underside(d):
                    command([py['semantics'],root/'tools/plant_extract_with_car/repair_cart_underside.py',
                        '--source',se['mesh'].parent,'--output',d/'result',
                        '--chassis-thickness-mm',underside.get('assumed_chassis_thickness_mm',6.)],d,root,env)
                    if read_json(d/'result/validation.json')['status'] != 'PASS':
                        raise ValueError('Cart underside export validation failed')
                    return dict(mesh=d/'result/semantic_mesh_mm.ply',labels=d/'result/labels.npz',
                        report=d/'result/report.json',template=d/'result/cart_template.json',
                        marker=d/'result/CART_UNDERSIDE_REPAIR_SUCCESS')
                scene_semantics=stages.execute('01b_cart_underside',repair_underside)
            def weights(d):
                command([py['semantics'],root/'scripts/plant_contact/run_pipeline.py',
                    '--input-dir',scene_semantics['mesh'].parent,'--output-dir',d/'result','--no-parts'],d,root,env)
                return dict(weights=d/'result/weights/contact_weights.npz',
                    heatmap=d/'result/weights/contact_risk_heatmap.ply',
                    marker=d/'result/PLANT_CONTACT_EXPORT_SUCCESS')
            we = stages.execute('02_weights',weights)
            result = dict(status='SEMANTIC_REVISION_EXPORTED', source_run=args.run, revision=args.revision,
                semantic_mesh=str(scene_semantics['mesh']), contact_weights=str(we['weights']), heatmap=str(we['heatmap']),
                cart_underside_report=str(scene_semantics['report']) if underside.get('enabled',False) else None,
                existing_cart_repair_preserved=existing_cart_repair,
                watering_surface_source=report.get('watering_surface_source',
                    'assumed_virtual_substrate' if (source/'virtual_substrate.json').is_file() else 'semantic_mesh_substrate'),
                comparison=str(se['preview']), report=str(se['report']),
                semantic_accuracy_measured=False, visual_review_required=True,
                previous_path_is_obsolete=True, path_regenerated=False, finished_utc=now())
            if args.plan:
                def plan(d):
                    command([py['planning'],root/'simulation/scripts/run_server_job.py','--run',args.revision,
                        '--semantic-dir',scene_semantics['mesh'].parent,'--contact-dir',we['weights'].parent.parent,
                        '--scale-report',scale,'--settings',root/'configs/plant_pipeline.json','--resume'],
                        d,root,render_environment(root,env))
                    job = root/'simulation/outputs/jobs'/args.revision
                    if read_json(job/'job_report.json')['status'] != 'SIMULATION_JOB_SUCCESS':
                        raise ValueError('Replanned trajectory failed validation')
                    return dict(report=job/'job_report.json',marker=job/'SIMULATION_JOB_SUCCESS')
                pl = stages.execute('03_planning',plan)
                result.update(path_regenerated=True, simulation_report=str(pl['report']))
            write_json(out/'result_paths.json',result)
            (out/'SEMANTIC_REVISION_EXPORTED').write_text(now()+'\n',encoding='ascii')
            (out/'REVISION_FAILED.json').unlink(missing_ok=True)
            print('SEMANTIC_REVISION_EXPORTED',out,flush=True)
        except BaseException as exc:
            write_json(out/'REVISION_FAILED.json',dict(error=str(exc),stage=stages.state.get('current_stage'),time=now()))
            raise


if __name__ == '__main__':
    main()
