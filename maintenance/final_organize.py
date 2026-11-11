#!/usr/bin/env python3
"""Archive and verify retired project content before removing original copies."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import tarfile
import time

@contextmanager
def directory_lock(path):
    """Atomic mkdir works on NFS without the network file-lock service.

    Never automatically steal locks: a task on another host may still be alive.
    A killed task can leave an owner record for manual inspection.
    """
    try:
        path.mkdir()
    except FileExistsError as exc:
        owner = path/'owner.json'
        try:
            detail = owner.read_text(encoding='utf-8')
        except OSError:
            detail = 'Owner record unavailable; inspect the existing directory.'
        raise RuntimeError('Organizer lock directory exists: '+str(path)+'; '+detail) from exc
    try:
        save(path/'owner.json', dict(host=socket.gethostname(), pid=os.getpid(),
            started=datetime.now().isoformat()))
        yield
    finally:
        (path/'owner.json.tmp').unlink(missing_ok=True)
        (path/'owner.json').unlink(missing_ok=True)
        path.rmdir()

from server_project import REQUIRED, active, safe

SCENE = 'outputs/with_car_20261021'
STAGES = {'free_gaussians', 'mast3r_sfm', 'tetra_meshes_gpu1_ds05',
          'plant_car_auto_v1', 'plant_car_metric_v1', 'plant_car_semantics_v2'}
MAINTENANCE_OLD = {'backup_before_car.sh', 'organize_matcha_server.sh',
    'organize_matcha.py', 'test_plantsem_v3_server.sh', 'history', 'packages'}

def save(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4*1024*1024), b''):
        h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def prerequisites(root):
    for rel in (*REQUIRED, 'maintenance/run_reconstruction.sh',
                'tools/plant_extract_with_car/run_crop_with_car.sh',
                'tools/plant_semantics_with_car/run_server.sh',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
                'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl',
                'MAtCha/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'):
        if not safe(root, rel).is_file():
            raise RuntimeError('Missing final-flow dependency: '+rel)
    for rel in ('tools/plant_extract', 'tools/plant_extract_with_car',
                'tools/plant_semantics', 'tools/plant_semantics_with_car'):
        if not safe(root, rel).is_dir():
            raise RuntimeError('Missing tool directory: '+rel)
    for rel in ('plant_car_auto_v1/plant_car.ply', 'plant_car_auto_v1/report.json',
                'plant_car_metric_v1/plant_car_mm.ply', 'plant_car_metric_v1/scale_report.json',
                'tetra_meshes_gpu1_ds05/tetra_mesh_binary_search_7.ply'):
        if not safe(root, SCENE+'/'+rel).is_file():
            raise RuntimeError('Missing retained stage: '+rel)
    for name in ('free_gaussians', 'mast3r_sfm'):
        if not safe(root, SCENE+'/'+name).is_dir():
            raise RuntimeError('Missing retained reconstruction directory: '+name)
    scene = safe(root, SCENE)
    name = load(scene/'latest_contact.json')['relative_output']
    if not isinstance(name, str) or '/' in name or '\\' in name or not name.startswith('contact_latest_'):
        raise ValueError('Invalid latest weight directory')
    output = safe(root, SCENE+'/'+name)
    for policy in ('contact_policy.json', 'organ_weight_policy.json'):
        if load(output/policy) != load(root/'tools/contact_weights'/policy):
            raise RuntimeError('Latest export policy mismatch: '+policy)
    for rel in ('PLANT_CONTACT_EXPORT_SUCCESS', 'result_paths.json',
                'organs/organ_instances.npz', 'weights/contact_weights.npz',
                'weights/contact_risk_heatmap.ply', 'weights/contact_weighted_mesh.ply'):
        if not safe(root, SCENE+'/'+name+'/'+rel).is_file():
            raise RuntimeError('Latest export incomplete: '+rel)
    report = load(output/'weights/contact_weights_report.json')
    if report.get('data_export') != 'PASS' or report.get('mm_per_unit') != 1.0:
        raise RuntimeError('Latest export has not passed or is not in millimeters')
    if report.get('geometry_policy') != load(root/'tools/contact_weights/organ_weight_policy.json'):
        raise RuntimeError('Latest weights do not match the current organ policy')
    if report.get('visualization', {}).get('version') != 'weight-colors-0.4.2-smooth-green-yellow':
        raise RuntimeError('Latest palette is outdated')
    return name

def plan(root, latest):
    selected = []
    def children(rel, keep):
        parent = safe(root, rel)
        if parent.is_dir():
            selected.extend(p.relative_to(root).as_posix() for p in parent.iterdir() if p.name not in keep)
    children('data', {'with_car_20261021'})
    children('outputs', {'with_car_20261021'})
    children(SCENE, STAGES | {latest, 'latest_contact.json', 'latest_contact_paths.txt'})
    children('scripts', {'plant_contact'})
    for folder in ('logs', 'tmp', 'incoming', 'exports', 'mplconfig', 'mplconfig_plant_semantics'):
        children(folder, set())
    for rel in ('README_项目导航.md', 'README_PROJECT_LAYOUT.md', 'cleanup_manifest.tsv',
                'docs/conda_installed_packages.json', 'docs/EXPERIMENTS.md', 'docs/SCRIPT_MOVES.md',
                *('maintenance/'+n for n in MAINTENANCE_OLD)):
        p = root/rel
        if p.exists() or p.is_symlink():
            selected.append(rel)
    # Remove generated bytecode only in our own tools; leave compiled extensions intact.
    for base in ('tools', 'scripts/plant_contact', 'maintenance'):
        for folder, dirs, _ in os.walk(safe(root, base), followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(folder)/d).is_symlink()]
            if '__pycache__' in dirs:
                selected.append((Path(folder)/'__pycache__').relative_to(root).as_posix())
                dirs.remove('__pycache__')
    result = []
    for rel in sorted(set(selected), key=lambda x: (len(Path(x).parts), x)):
        if not any(Path(p) in Path(rel).parents for p in result):
            result.append(rel)
    for rel in result:
        # Ancestors cannot be links; selected leaf symlinks are backed up as links.
        safe(root, str(Path(rel).parent))
    return result

def inventory(root, paths):
    records = {}
    started = time.monotonic()
    def visit(p):
        nonlocal started
        if hasattr(p, 'is_junction') and p.is_junction():
            raise ValueError('Junction not supported: '+str(p))
        rel = p.relative_to(root).as_posix()
        s = p.lstat()
        if stat.S_ISLNK(s.st_mode):
            row = dict(kind='symlink', target=os.readlink(p))
        elif stat.S_ISDIR(s.st_mode):
            row = dict(kind='directory')
        elif stat.S_ISREG(s.st_mode):
            with p.open('rb') as f:
                sha = digest(f)
            after = p.stat()
            if (s.st_size, s.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('File changed during read: '+rel)
            row = dict(kind='file', size=s.st_size, sha256=sha, mtime_ns=s.st_mtime_ns)
        else:
            raise ValueError('Special file cannot be archived: '+rel)
        records[rel] = row
        if time.monotonic()-started > 20:
            print('Scanning:', len(records), 'entries;', rel, flush=True)
            started = time.monotonic()
        if row['kind'] == 'directory':
            for child in sorted(p.iterdir()):
                visit(child)
    for rel in paths:
        visit(root/rel)
    return records

def check_retained_links(root, paths, latest):
    """Do not retire a directory supplying symlink targets of retained stages."""
    targets = [root/p for p in paths]
    bases = ['data/with_car_20261021', 'MAtCha', 'tools', 'third_party', 'weights',
             'scripts/plant_contact', *(SCENE+'/'+p for p in STAGES | {latest})]
    for base in bases:
        p = root/base
        if not p.exists():
            continue
        for folder, dirs, files in os.walk(p, followlinks=False):
            for name in dirs+files:
                link = Path(folder)/name
                if not link.is_symlink():
                    continue
                resolved = link.resolve()
                if any(resolved == t or t in resolved.parents for t in targets):
                    raise RuntimeError('Retained symlink depends on retired content: '+str(link))
            dirs[:] = [d for d in dirs if d != '__pycache__']

def write_archive(root, archive, records):
    with tarfile.open(archive, 'w', dereference=False) as tf:
        tick = time.monotonic()
        for i, (rel, row) in enumerate(records.items(), 1):
            p = root/rel
            info = tf.gettarinfo(str(p), arcname=rel)
            if row['kind'] == 'file':
                # Store hardlinked files independently, avoiding references outside backup.
                info.type, info.linkname, info.size = tarfile.REGTYPE, '', row['size']
                with p.open('rb') as stream:
                    tf.addfile(info, stream)
            else:
                tf.addfile(info)
            if time.monotonic()-tick > 20:
                print('Backing up:', i, '/', len(records), rel, flush=True)
                tick = time.monotonic()

def verify_archive(archive, records):
    found = set()
    tick = time.monotonic()
    with tarfile.open(archive, 'r:') as tf:
        for member in tf:
            rel = member.name.rstrip('/')
            if rel not in records or rel in found:
                raise RuntimeError('Unexpected/duplicate archive entry: '+rel)
            row = records[rel]
            if row['kind'] == 'file':
                if not member.isfile() or member.size != row['size']:
                    raise RuntimeError('Archive file mismatch: '+rel)
                with tf.extractfile(member) as f:
                    if digest(f) != row['sha256']:
                        raise RuntimeError('Archive hash mismatch: '+rel)
            elif row['kind'] == 'directory':
                if not member.isdir():
                    raise RuntimeError('Archive directory mismatch: '+rel)
            elif not member.issym() or member.linkname != row['target']:
                raise RuntimeError('Archive link mismatch: '+rel)
            found.add(rel)
            if time.monotonic()-tick > 20:
                print('Verifying backup:', len(found), '/', len(records), flush=True)
                tick = time.monotonic()
    if found != set(records):
        raise RuntimeError('Archive is incomplete')

def organize(root, apply=False):
    latest = prerequisites(root)
    paths = plan(root, latest)
    check_retained_links(root, paths, latest)
    print(json.dumps(dict(mode='APPLY' if apply else 'PREVIEW', retire=paths,
        retained_weight=SCENE+'/'+latest,
        retained_dependencies=['envs', 'cache', 'MAtCha', 'third_party', 'weights', 'tools', 'backups']),
        ensure_ascii=False, indent=2), flush=True)
    if not apply:
        print('PREVIEW_ONLY: no changes')
        return
    jobs = active(root)
    if jobs:
        raise RuntimeError('Stop project jobs before organizing: '+json.dumps(jobs))
    backups = safe(root, 'backups')
    backups.mkdir(exist_ok=True)
    with directory_lock(backups/'.final_organize.lockdir'):
        dest = backups/('final_cleanup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        dest.mkdir()
        manifest = dict(status='STARTED', paths=paths, removed=[], retained_weight=SCENE+'/'+latest)
        save(dest/'manifest.json', manifest)
        try:
            print('Inventory and SHA256 of old content...', flush=True)
            records = inventory(root, paths)
            save(dest/'files.json', records)
            size = sum(v.get('size', 0) for v in records.values()) + 2048*len(records)
            if shutil.disk_usage(backups).free < size + 256*1024**2:
                raise RuntimeError('Insufficient backup space; originals retained')
            print('Backup input bytes:', size, '; entries:', len(records), flush=True)
            partial = dest/'old_content.tar.partial'
            write_archive(root, partial, records)
            print('Checking every archived file against SHA256...', flush=True)
            verify_archive(partial, records)
            print('Checking that original files have not changed...', flush=True)
            if records != inventory(root, paths):
                raise RuntimeError('Original content changed; no removal performed')
            if prerequisites(root) != latest or plan(root, latest) != paths or active(root):
                raise RuntimeError('Project state changed; no removal performed')
            check_retained_links(root, paths, latest)
            archive = dest/'old_content.tar'
            partial.rename(archive)
            with archive.open('rb') as stream:
                sha = digest(stream)
            (dest/'old_content.tar.sha256').write_text(sha+'  old_content.tar\n', encoding='ascii')
            with archive.open('r+b') as stream:
                os.fsync(stream.fileno())
            (dest/'BACKUP_VERIFIED').write_text('Every archived file verified before removal.\n', encoding='ascii')
            manifest['status'] = 'BACKUP_VERIFIED'
            save(dest/'manifest.json', manifest)
            print('BACKUP_VERIFIED:', dest, flush=True)
            for rel in paths:
                p = root/rel
                safe(root, str(Path(rel).parent))
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                manifest['removed'].append(rel)
                save(dest/'manifest.json', manifest)
            index = dict(relative_to_project=True, semantic=SCENE+'/plant_car_semantics_v2',
                metric=SCENE+'/plant_car_metric_v1', mesh=SCENE+'/tetra_meshes_gpu1_ds05',
                weights=SCENE+'/'+latest+'/weights', organs=SCENE+'/'+latest+'/organs',
                backup=dest.relative_to(root).as_posix(), final_command='bash maintenance/finish_server.sh --apply')
            save(root/'docs/SERVER_RESULT_INDEX.json', index)
            (root/'docs/CURRENT_FILES.md').write_text('# Current project files\n\n'+
                '\n'.join('- '+k+': `'+str(v)+'`' for k,v in index.items())+'\n', encoding='utf-8')
            manifest['status'] = 'SUCCESS'
            save(dest/'manifest.json', manifest)
            (dest/'FINAL_ORGANIZE_SUCCESS').write_text('Archive verified; retired originals removed.\n')
            print('FINAL_ORGANIZE_SUCCESS:', dest, flush=True)
        except BaseException as exc:
            manifest.update(status='FAILED', error=str(exc))
            save(dest/'manifest.json', manifest)
            raise

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    if root == root.parent or not (root/'MAtCha').is_dir():
        ap.error('Not a matcha_plant project')
    organize(root, args.apply)

if __name__ == '__main__':
    main()
