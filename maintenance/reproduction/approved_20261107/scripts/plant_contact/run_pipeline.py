#!/usr/bin/env python3
"""Portable semantic mesh -> organ proposals -> spatial contact weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
ORGANS = PROJECT / 'tools' / 'plant_organ_instances'
WEIGHTS = PROJECT / 'tools' / 'contact_weights'


def check_runtime():
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10+ is required; use envs/plant_semantics, not envs/matcha.')
    versions = {}
    for name in ('numpy', 'scipy', 'PIL', 'plyfile'):
        module = importlib.import_module(name)
        versions[name] = getattr(module, '__version__', 'available')
    from scipy.sparse.csgraph import dijkstra
    return versions


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_stage(name, command, output):
    print(f'\nSTAGE: {name}', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    log = output / 'logs' / f'{name}.log'
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen([sys.executable, '-u', *map(str, command)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
        for line in proc.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        result = proc.wait()
    if result:
        raise RuntimeError(f'{name} failed (exit {result}). Inspect {log}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True, help='semantic_mesh_mm.ply + matching labels.npz')
    ap.add_argument('--output-dir', type=Path, required=True, help='New empty output root; never overwrites old runs.')
    ap.add_argument('--mesh-name', default='semantic_mesh_mm.ply')
    ap.add_argument('--mm-per-unit', type=float, default=1.)
    ap.add_argument('--policy', type=Path, default=WEIGHTS / 'contact_policy.json')
    ap.add_argument('--geometry-policy', type=Path, default=WEIGHTS / 'organ_weight_policy.json')
    ap.add_argument('--seed-file', type=Path)
    ap.add_argument('--coordinate-system', type=Path, help='Optional preview orientation; auto-detects input coordinate_system.json.')
    ap.add_argument('--min-area-mm2', type=float, default=10.)
    ap.add_argument('--marker-separation', type=float, default=1.)
    ap.add_argument('--max-pixels', type=int, default=600)
    ap.add_argument('--no-parts', action='store_true', help='Skip individual-part PLY exports to reduce file count on NFS.')
    ap.add_argument('--no-previews', action='store_true')
    ap.add_argument('--check-only', action='store_true', help='Check runtime/files/parameters without producing model outputs.')
    args = ap.parse_args()
    versions = check_runtime()
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    if Path(args.mesh_name).name != args.mesh_name:
        ap.error('--mesh-name must be a file name, without directories.')
    mesh, labels = source / args.mesh_name, source / 'labels.npz'
    for path in (mesh, labels, args.policy, args.geometry_policy,
                 ORGANS/'split_plant_organs.py', WEIGHTS/'build_organ_contact_weights.py'):
        if not path.is_file():
            ap.error(f'Missing input or tool: {path}')
    if args.seed_file and not args.seed_file.is_file():
        ap.error(f'Missing seed file: {args.seed_file}')
    if source == out or out in source.parents:
        ap.error('Output must differ from and must not contain the input directory.')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error('Output is not empty. Choose a new directory to preserve previous results.')
    import math
    if any(not math.isfinite(v) or v <= 0 for v in (args.mm_per_unit,args.min_area_mm2,args.marker_separation)) or args.max_pixels < 64:
        ap.error('Scale/area/separation must be positive and finite; max-pixels >=64.')
    policy = json.loads(args.policy.read_text(encoding='utf-8-sig'))
    gp = json.loads(args.geometry_policy.read_text(encoding='utf-8-sig'))
    for k in ('leaf_class_id', 'flower_class_id'):
        if str(gp[k]) not in policy['classes']:
            ap.error('Leaf/flower IDs must exist in the semantic policy.')
    if gp['leaf_class_id'] == gp['flower_class_id']:
        ap.error('Leaf/flower IDs must differ.')
    coord = args.coordinate_system
    if coord is None and (source/'coordinate_system.json').is_file():
        coord = source/'coordinate_system.json'
    if coord and not coord.is_file():
        ap.error(f'Missing coordinate-system file: {coord}')
    print(json.dumps(dict(python=sys.executable, dependencies=versions, input=str(source),
        output=str(out), palette='shared tools/contact_weights/weight_palette.py'), ensure_ascii=False), flush=True)
    if args.check_only:
        print('PLANT_CONTACT_PREFLIGHT_OK (no model outputs written)', flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out/'logs').mkdir()
    write_json(out/'contact_policy.json', policy)
    write_json(out/'organ_weight_policy.json', gp)
    running = out/'PIPELINE_RUNNING'
    running.write_text(datetime.now(timezone.utc).isoformat(), encoding='ascii')
    organs, weights = out/'organs', out/'weights'
    try:
        split = [ORGANS/'split_plant_organs.py', '--mesh', mesh, '--labels', labels,
            '--output-dir', organs, '--leaf-id', gp['leaf_class_id'], '--flower-id', gp['flower_class_id'],
            '--mm-per-unit', args.mm_per_unit, '--min-area-mm2', args.min_area_mm2,
            '--marker-separation', args.marker_separation, '--max-pixels', args.max_pixels]
        if not args.no_parts:
            split.append('--export-parts')
        if args.seed_file:
            split += ['--seed-file', args.seed_file]
        run_stage('01_split_organs', split, out)
        run_stage('02_build_weights', [WEIGHTS/'build_organ_contact_weights.py', '--mesh', mesh,
            '--labels', labels, '--instances-dir', organs, '--output-dir', weights,
            '--policy', args.policy, '--geometry-policy', args.geometry_policy], out)
        if not args.no_previews:
            organ_preview = [ORGANS/'render_organ_preview.py', '--instances-dir', organs, '--weights-dir', weights]
            whole_preview = [WEIGHTS/'render_weight_preview.py', '--input', weights/'contact_risk_heatmap.ply',
                '--output', weights/'contact_risk_heatmap_preview.jpg', '--single-view']
            if coord:
                organ_preview += ['--coordinate-system', coord]
                whole_preview += ['--coordinate-system', coord]
            run_stage('03_organ_previews', organ_preview, out)
            run_stage('04_weight_preview', whole_preview, out)
            run_stage('05_colorbar', [WEIGHTS/'render_weight_colorbar.py', '--output', weights/'contact_weight_colorbar.png'], out)
        paths = dict(source_mesh=str(mesh), source_labels=str(labels),
            organ_mesh=str(organs/'organ_instances.ply'), organ_arrays=str(organs/'organ_instances.npz'),
            organ_report=str(organs/'organ_report.json'), part_meshes=str(organs/'parts') if not args.no_parts else None,
            weighted_mesh=str(weights/'contact_weighted_mesh.ply'), heatmap=str(weights/'contact_risk_heatmap.ply'),
            weight_arrays=str(weights/'contact_weights.npz'), report=str(weights/'contact_weights_report.json'))
        if not args.no_previews:
            paths.update(organ_preview=str(organs/'organ_instances_preview.png'),
                weight_preview=str(weights/'contact_risk_heatmap_preview.jpg'),
                weight_closeup=str(weights/'organ_weights_detail.png'), colorbar=str(weights/'contact_weight_colorbar.png'))
        write_json(out/'result_paths.json', paths)
        (out/'result_paths.txt').write_text('\n'.join(f'{k}: {v}' for k,v in paths.items() if v) + '\n', encoding='utf-8')
        write_json(out/'pipeline_report.json', dict(version='plant-contact-pipeline-0.1.0',
            export_status='PASS', organ_accuracy='REVIEW_REQUIRED', python=sys.executable,
            dependencies=versions, source=str(source), output=str(out), mm_per_unit=args.mm_per_unit,
            note='No robot simulation or path planning was run. Original inputs are unchanged.'))
        (out/'PLANT_CONTACT_EXPORT_SUCCESS').write_text('Export complete; organ proposals require review.\n', encoding='utf-8')
        print(f'PLANT_CONTACT_EXPORT_SUCCESS: {out}; organ proposals still require review', flush=True)
        return 0
    except BaseException as exc:
        write_json(out/'PIPELINE_FAILED.json', dict(error=str(exc), status='FAILED',
            note='Partial outputs are preserved for diagnostics. Use a fresh output directory for retry.'))
        raise
    finally:
        running.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
