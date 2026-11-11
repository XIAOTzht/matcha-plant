"""Build, verify and install the source/robot integration update; never package models or results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import uuid
import zipfile

DIRECTORIES = ['scripts/plant_pipeline', 'scripts/plant_contact', 'tools/contact_weights',
               'maintenance/reproduction/approved_20261107',
               'tools/plant_semantics/tests',
               'tools/plant_organ_instances', 'tools/plant_extract_with_car', 'tools/plant_semantics_with_car',
               'simulation/scripts', 'simulation/configs', 'simulation/docs', 'simulation/assets/robot/robotstudio_so101']
FILES = ['configs/plant_pipeline.json', 'simulation/requirements-linux.txt',
         'MAtCha/scripts/run_sfm.py', 'MAtCha/scripts/align_charts.py',
         'MAtCha/scripts/refine_free_gaussians.py', 'MAtCha/scripts/extract_tetra_mesh.py',
         'MAtCha/matcha/dm_extractors/adaptive_tsdf.py',
         'MAtCha/configs/mast3r/unposed.yaml', 'MAtCha/configs/charts_alignment/default.yaml',
         'MAtCha/configs/free_gaussians_refinement/default.yaml', 'MAtCha/configs/adaptive_tetrahedralization/default.yaml',
         'maintenance/plant_pipeline_bundle.py', 'maintenance/plant_pipeline_client.ps1',
         'maintenance/upload_plant_pipeline.ps1', 'maintenance/submit_plant_photos.ps1',
         'maintenance/fetch_plant_results.ps1', 'maintenance/setup_plant_pipeline.sh',
         'maintenance/stop_plant_pipeline.py',
         'maintenance/export_crop_diagnostics.py', 'maintenance/diagnose_plant_crop.ps1',
         'maintenance/resume_cart_crop.py',
         'maintenance/continue_with_virtual_substrate.py', 'maintenance/start_virtual_substrate.ps1',
         'maintenance/resume_display_mesh.py',
         'docs/VIRTUAL_SUBSTRATE.md',
         'docs/SEMANTIC_RECOVERY.md',
         'docs/APPROVED_REPRODUCTION.md',
         'maintenance/reproduce_approved_semantics.py',
         'maintenance/approved_reproduction.py', 'maintenance/start_approved_reproduction.ps1',
         'maintenance/test_approved_reproduction.py',
         'maintenance/refine_plant_semantics.py', 'maintenance/start_semantic_review.ps1',
         'maintenance/test_semantic_recovery_revision.py',
         'maintenance/fetch_semantic_review_inputs.ps1', 'docs/SEMANTIC_SHAPE_REVIEW.md',
         'docs/PHOTO_TO_PATH_GUIDE.md', 'docs/PLANT_CONTACT_GUIDE.md']
SUFFIXES = {'.py','.json','.sh','.ps1','.md','.txt','.xml','.stl','.obj','.yaml','.yml'}
RETIRED_EXPERIMENTS = {
    'tools/plant_semantics/organ_recovery.py': '9b5f967c71b8b8643bb5fec0351935cc03d00b647ba5d7b6eb886b60d5698600',
    'tools/plant_semantics/tests/test_organ_recovery.py': '6377816c5878c413ca321713c4ddbcc5f68547cb0615427ca075cba736056061',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def allowed(name):
    p=PurePosixPath(name)
    if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name:
        return False
    if name in {'configs/server_render_runtime.json','simulation/configs/server_render_runtime.json'}:return False
    if name in FILES:return True
    if name.startswith('tools/plant_semantics/') and len(p.parts)==3:
        return p.suffix.lower() in SUFFIXES or p.name=='VERSION'
    if name=='simulation/assets/robot/robotstudio_so101/so101.png':return True
    return (any(name.startswith(d+'/') for d in DIRECTORIES) and '__pycache__' not in p.parts
            and not p.name.startswith('latest_') and (p.suffix.lower() in SUFFIXES or p.name in {'LICENSE','NOTICE'}))


def payload(path):
    data=path.read_bytes()
    if path.suffix.lower() in {'.py','.json','.sh','.ps1','.md','.txt','.xml','.yaml','.yml'}:
        try:
            text=data.decode('utf-8-sig').replace('\r\n','\n')
            data=text.encode('utf-8-sig' if path.suffix.lower()=='.ps1' else 'utf-8')
        except UnicodeDecodeError:
            raise ValueError(f'Expected UTF-8 source: {path}') from None
    return data


def build(root, output):
    root=Path(root).resolve();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    candidates={root/p for p in FILES}
    for directory in DIRECTORIES:
        candidates.update(p for p in (root/directory).rglob('*') if p.is_file())
    candidates.update(p for p in (root/'tools/plant_semantics').iterdir() if p.is_file())
    manifest={}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(candidates):
            name=p.relative_to(root).as_posix()
            if not allowed(name):continue
            if p.is_symlink():raise ValueError(f'Symlink source unsupported: {p}')
            data=payload(p)
            z.writestr(name,data)
            manifest[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        missing=set(FILES)-set(manifest)
        if missing:raise ValueError(f'Missing required deployment files: {sorted(missing)}')
        z.writestr('bundle_manifest.json',json.dumps(dict(version=1,files=manifest),indent=2))
    digest=sha(output)
    Path(str(output)+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='ascii')
    print(json.dumps(dict(status='PIPELINE_BUNDLE_CREATED',files=len(manifest),bytes=output.stat().st_size,sha256=digest)))
    return manifest


def inspect(archive, sha256):
    if sha(archive)!=sha256:raise ValueError('Bundle SHA256 mismatch')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();names=[p.filename for p in entries]
        if len(set(names))!=len(names) or sum(p.file_size for p in entries)>300*1024**2:
            raise ValueError('Duplicate entries or oversized source bundle')
        manifest=json.loads(z.read('bundle_manifest.json'))['files']
        if set(names)!=(set(manifest)|{'bundle_manifest.json'}):raise ValueError('Archive/manifest file list differs')
        for name,metadata in manifest.items():
            if not allowed(name):raise ValueError(f'Unexpected deployment path: {name}')
            info=z.getinfo(name)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink entry forbidden')
            data=z.read(name)
            if len(data)!=metadata['bytes'] or hashlib.sha256(data).hexdigest()!=metadata['sha256']:
                raise ValueError(f'Corrupt deployment entry: {name}')
    return manifest


def install(archive, sha256, root):
    manifest=inspect(archive,sha256);root=Path(root).resolve()
    if not (root/'MAtCha').is_dir():raise ValueError('Select the existing server project root (MAtCha directory required)')
    updates=[]
    for name,info in manifest.items():
        destination=root/name
        destination.resolve().relative_to(root)
        if destination.is_symlink():raise ValueError(f'Preserving symbolic link destination: {destination}')
        if destination.exists() and not destination.is_file():raise ValueError(f'Destination is not a file: {destination}')
        if not destination.is_file() or sha(destination)!=info['sha256']:updates.append(name)
    backup=root/'backups/code_updates'/('plant_pipeline_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True,exist_ok=False)
    journal=[]
    retired=[]
    try:
        with zipfile.ZipFile(archive) as z:
            for name in updates:
                dst=root/name;prior=dst.is_file()
                if prior:
                    old=backup/name;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,old)
                journal.append((name,prior))
                dst.parent.mkdir(parents=True,exist_ok=True)
                tmp=dst.with_name(dst.name+'.deploy_tmp')
                tmp.write_bytes(z.read(name));os.replace(tmp,dst)
        approved_classifier = manifest.get('tools/plant_semantics/refine_organ_semantics.py',{}).get('sha256')
        if approved_classifier == '1309be073f05c94f1148ad771209a887f3f56cf7d3b4f6a6957adcf33144a366':
            for name,expected in RETIRED_EXPERIMENTS.items():
                dst=root/name
                if name not in manifest and dst.is_file() and not dst.is_symlink() and sha(dst)==expected:
                    dst.resolve().relative_to(root)
                    old=backup/name;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,old)
                    journal.append((name,True));dst.unlink();retired.append(name)
        (backup/'update_manifest.json').write_text(json.dumps(dict(changed=updates,retired_experiments=retired,sha256=sha256),indent=2),encoding='utf-8')
    except BaseException:
        for name,prior in reversed(journal):
            dst=root/name
            if prior:shutil.copy2(backup/name,dst)
            else:dst.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(status='PLANT_PIPELINE_CODE_INSTALLED',changed=len(updates),retired_experiments=retired,backup=str(backup),
                          note='Existing models, data, outputs and Python environments preserved. No job started.')))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='action',required=True)
    p=sp.add_parser('build');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sp.add_parser('inspect');p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p=sp.add_parser('install');p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--root',type=Path,required=True)
    args=vars(ap.parse_args());action=args.pop('action');result=globals()[action](**args)
    if action=='inspect':print('PIPELINE_BUNDLE_CHECK_OK',len(result))


if __name__=='__main__':main()
