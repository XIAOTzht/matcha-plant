#!/usr/bin/env python3
"""Conservative server maintenance. Preview first; preserves environments and results."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

EXCLUDE = {'.git', '__pycache__', 'envs', '.venv', 'venv', 'node_modules',
    'cache', '.cache', 'tmp', 'logs', 'incoming', 'backups', 'exports',
    'outputs', 'data', 'weights', 'checkpoints', 'pretrained',
    '.pytest_cache', 'mplconfig', 'mplconfig_plant_semantics', 'packages'}
TOPS = ('MAtCha', 'tools', 'scripts', 'maintenance', 'docs', 'third_party')
EXTS = {'.py', '.sh', '.ps1', '.md', '.txt', '.json', '.yaml', '.yml', '.toml',
    '.cfg', '.ini', '.cu', '.cuh', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    '.cmake', '.in', '.rst', '.bat', '.xml', '.html', '.css', '.js', '.ts',
    '.tsx', '.jsx', '.vert', '.frag', '.glsl', '.svg', '.png', '.jpg', '.ipynb'}
REQUIRED = (
    'MAtCha/train.py',
    'tools/plant_extract_with_car/calibrate_upper_plate.py',
    'tools/plant_semantics_with_car/fix_cart_hardware.py',
    'tools/contact_weights/build_organ_contact_weights.py',
    'tools/contact_weights/contact_policy.json',
    'tools/contact_weights/organ_weight_policy.json',
    'tools/plant_organ_instances/split_plant_organs.py',
    'scripts/plant_contact/run_server.sh',
    'outputs/with_car_20261021/plant_car_semantics_v2/semantic_mesh_mm.ply',
    'outputs/with_car_20261021/plant_car_semantics_v2/labels.npz')

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

def safe(root, rel):
    p = root/rel
    p.resolve().relative_to(root)
    for q in (p, *p.parents):
        if q == root:
            break
        if q.is_symlink() or (hasattr(q, 'is_junction') and q.is_junction()):
            raise ValueError('Refusing linked path: '+str(q))
    return p

def walk(root):
    for top in TOPS:
        base = safe(root, top)
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDE and not (Path(folder)/d).is_symlink()
                and not (hasattr(Path(folder)/d, 'is_junction') and (Path(folder)/d).is_junction())]
            yield Path(folder), files

def active(root):
    if not Path('/proc').is_dir():
        return []
    jobs = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (os.getpid(), os.getppid()):
            continue
        try:
            words = (p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not Path(words[0]).name.lower().startswith(('python', 'pip', 'conda', 'mamba', 'tar', 'gzip', 'nvcc', 'ninja')):
                continue
            cwd = (p/'cwd').resolve()
            if str(root) in ' '.join(words) or cwd == root or root in cwd.parents:
                jobs.append(dict(pid=int(p.name), command=' '.join(words)[:300]))
        except (OSError, ValueError):
            pass
    return jobs

def check(root, runtime=False):
    missing = []
    for rel in REQUIRED:
        ok = (root/rel).exists()
        print(('OK   ' if ok else 'MISS ')+rel, flush=True)
        if not ok:
            missing.append(rel)
    if runtime:
        py = root/'envs/plant_semantics/bin/python'
        if not py.is_file():
            missing.append(str(py))
        else:
            subprocess.run([str(py), str(root/'scripts/plant_contact/check_environment.py')], check=True)
    if missing:
        raise RuntimeError('Missing required files: '+', '.join(missing))
    print('PROJECT_LAYOUT_CHECK_OK')

def cleanup(root, apply=False):
    check(root)
    actions = []
    for folder, _ in walk(root):
        p = folder/'__pycache__'
        if p.is_dir():
            safe(root, p.relative_to(root))
            actions.append(dict(action='delete_bytecode', path=p.relative_to(root).as_posix()))
    for rel in ('logs', 'tmp', 'mplconfig', 'mplconfig_plant_semantics'):
        p = safe(root, rel)
        if p.is_dir() and any(p.iterdir()):
            actions.append(dict(action='archive', path=rel))
    print(json.dumps(dict(mode='APPLY' if apply else 'PREVIEW', actions=actions,
        preserved=['envs','data','outputs','weights','cache','incoming','.git','all .part/.partial files']),
        ensure_ascii=False, indent=2), flush=True)
    if not apply:
        print('PREVIEW_ONLY: no files changed')
        return
    jobs = active(root)
    if jobs:
        raise RuntimeError('Stop project jobs first: '+json.dumps(jobs))
    backup = safe(root, 'backups/organization_'+stamp())
    backup.mkdir(parents=True)
    save(backup/'cleanup_manifest.json', dict(status='STARTED', actions=actions))
    try:
        for a in actions:
            p = safe(root, a['path'])
            if a['action'] == 'archive':
                dst = backup/'archived'/a['path']
                dst.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dst)
                p.mkdir()
            else:
                if p.name != '__pycache__':
                    raise ValueError('Unexpected deletion target')
                shutil.rmtree(p)
            a['status'] = 'DONE'
            save(backup/'cleanup_manifest.json', dict(status='RUNNING', actions=actions))
        docs = safe(root, 'docs')
        docs.mkdir(exist_ok=True)
        scene = root/'outputs/with_car_20261021'
        save(docs/'SERVER_RESULT_INDEX.json', dict(
            relative_to_project=True,
            semantic='outputs/with_car_20261021/plant_car_semantics_v2',
            metric='outputs/with_car_20261021/plant_car_metric_v1',
            mesh='outputs/with_car_20261021/tetra_meshes_gpu1_ds05/tetra_mesh_binary_search_7.ply',
            weight_runs=[p.relative_to(root).as_posix() for p in sorted(scene.glob('contact*')) if p.is_dir()],
            latest='outputs/with_car_20261021/latest_contact.json'))
        save(backup/'cleanup_manifest.json', dict(status='SUCCESS', actions=actions))
        (backup/'CLEANUP_SUCCESS').write_text('Bytecode deleted; logs/scratch archived.\n')
        print('SERVER_PROJECT_CLEANUP_SUCCESS:', backup, flush=True)
    except BaseException:
        save(backup/'cleanup_manifest.json', dict(status='FAILED', actions=actions))
        raise

def export(root):
    target = safe(root, 'exports/github_source_'+stamp())
    candidates = [p for p in root.iterdir() if p.is_file() and
        (p.name in ('.gitignore','.gitattributes','.gitmodules') or p.suffix == '.md')]
    for folder, files in walk(root):
        for name in files:
            p = folder/name
            if p.is_symlink() or 'history' in p.relative_to(root).parts:
                continue
            if p.suffix.lower() in EXTS or name in ('LICENSE','COPYING','NOTICE','Makefile','Dockerfile','.gitmodules','.gitignore','.gitattributes'):
                if '.backup_' not in name and not name.endswith(('.bak','.backup')):
                    candidates.append(p)
    records, skipped = [], []
    target.mkdir(parents=True)
    for p in sorted(set(candidates)):
        safe(root, p.relative_to(root))
        rel = p.relative_to(root)
        if p.stat().st_size > 10*1024**2:
            skipped.append(rel.as_posix())
            continue
        data = p.read_bytes()
        dst = target/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        records.append(dict(path=rel.as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    save(target/'SOURCE_EXPORT_MANIFEST.json', dict(files=records, skipped=skipped,
        note='Source snapshot includes vendored upstream code; retain licenses. Review before publishing. No runtime data or environments.'))
    print('GITHUB_SOURCE_EXPORT_SUCCESS:', target, flush=True)
    return target

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    sub = ap.add_subparsers(dest='action', required=True)
    p = sub.add_parser('cleanup'); p.add_argument('--apply', action='store_true')
    p = sub.add_parser('check'); p.add_argument('--runtime', action='store_true')
    sub.add_parser('export-github')
    args = ap.parse_args()
    root = args.root.resolve()
    if root == root.parent or not (root/'tools').is_dir() or not (root/'MAtCha').is_dir():
        ap.error('Not a recognized matcha_plant project root')
    if args.action == 'cleanup':
        cleanup(root, args.apply)
    elif args.action == 'check':
        check(root, args.runtime)
    else:
        export(root)

if __name__ == '__main__':
    main()
