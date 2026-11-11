#!/usr/bin/env python3
"""Build/check/install a small source-only plant-contact bundle using stdlib.

No environments, model weights, meshes or historical outputs are packaged.
Install backs up changed files, adds/updates only manifest-listed code files,
and never removes unrelated project content.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
import zipfile

PREFIXES = ('tools/plant_organ_instances/', 'tools/contact_weights/', 'scripts/plant_contact/')
EXACT = {'maintenance/plant_contact_bundle.py', 'maintenance/upload_plant_contact.ps1',
         'maintenance/prepare_server_project.sh', 'maintenance/check_project_layout.sh',
         'maintenance/server_project.py', '.gitignore', '.gitattributes',
         'maintenance/test_server_project.py', 'README.md',
         'docs/PLANT_CONTACT_GUIDE.md', 'docs/SERVER_MAINTENANCE.md',
         'maintenance/final_organize.py', 'maintenance/test_final_organize.py',
         'maintenance/finish_server.sh', 'docs/FINAL_LAYOUT.md'}
SUFFIXES = {'.py', '.json', '.md', '.txt', '.sh', '.ps1'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def allowed(name):
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:
        return False
    if part.suffix not in SUFFIXES and name not in EXACT:
        return False
    return name in EXACT or any(name.startswith(p) and '/' not in name[len(p):] for p in PREFIXES)


def contained(root, name):
    if not allowed(name):
        raise ValueError(f'Unexpected source path: {name}')
    target = root.joinpath(*PurePosixPath(name).parts)
    target.resolve().relative_to(root.resolve())
    # Do not update files through project symlinks.
    for p in (target, *target.parents):
        if p == root:
            break
        if p.is_symlink():
            raise ValueError(f'Linked target is not supported: {p}')
    return target


def build(root, output):
    root, output = root.resolve(), output.resolve()
    names = set(EXACT)
    for prefix in PREFIXES:
        directory = root / prefix
        if not directory.is_dir():
            raise ValueError(f'Missing tool directory: {directory}')
        names.update(p.relative_to(root).as_posix() for p in directory.iterdir()
                     if p.is_file() and p.suffix in SUFFIXES)
    files = []
    payload = {}
    for name in sorted(names):
        source = contained(root, name)
        if not source.is_file():
            raise ValueError(f'Missing required source: {source}')
        data = source.read_bytes()
        # Windows PowerShell gets BOM; Bash must have LF and no BOM.
        if source.suffix == '.sh':
            data = data.decode('utf-8-sig').replace('\r\n','\n').encode('utf-8')
        elif source.suffix == '.ps1':
            data = data.decode('utf-8-sig').replace('\r\n','\n').encode('utf-8-sig')
        elif source.suffix in ('.py','.json','.md','.txt'):
            data = data.decode('utf-8-sig').replace('\r\n','\n').encode('utf-8')
        payload[name] = data
        files.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(format='plant-contact-source-1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_only=True, palette='weight-colors-0.4.2-smooth-green-yellow', files=files,
        note='No models, environments, datasets, outputs, deletion or package installation.')
    if output.exists():
        raise ValueError('Archive already exists; choose a new archive name.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in payload.items():
            z.writestr(name, data)
        z.writestr('bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    sha = digest(output)
    output.with_suffix(output.suffix+'.sha256').write_text(f'{sha}  {output.name}\n',encoding='ascii')
    print(json.dumps(dict(status='BUNDLE_CREATED', archive=str(output), sha256=sha,
                          files=len(files), bytes=output.stat().st_size)))
    return manifest


def load_bundle(archive, expected=None):
    if expected and digest(archive).lower() != expected.lower():
        raise ValueError('Archive SHA256 mismatch. Upload again; no project files were changed.')
    with zipfile.ZipFile(archive) as z:
        listed = z.namelist()
        if len(listed) != len(set(listed)) or len(listed) > 150:
            raise ValueError('Duplicate or excessive archive entries.')
        if sum(i.file_size for i in z.infolist()) > 20*1024*1024:
            raise ValueError('Source bundle is unexpectedly large.')
        manifest = json.loads(z.read('bundle_manifest.json'))
        if manifest.get('format') != 'plant-contact-source-1':
            raise ValueError('Unsupported source bundle format.')
        entries = manifest['files']
        names = [r['path'] for r in entries]
        if len(names) != len(set(names)) or set(listed) != set(names)|{'bundle_manifest.json'}:
            raise ValueError('Archive file list does not match manifest.')
        payload = {}
        for record in entries:
            name = record['path']
            if not allowed(name):
                raise ValueError(f'Unexpected file in archive: {name}')
            data = z.read(name)
            if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'File checksum mismatch: {name}')
            payload[name] = data
    return manifest, payload


def install(archive, root, expected):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError('Project root must already exist.')
    manifest, payload = load_bundle(archive, expected)
    plan = []
    for name, data in payload.items():
        target = contained(root, name)
        if target.exists() and not target.is_file():
            raise ValueError(f'Target is not a file: {target}')
        if target.is_file() and digest(target) == hashlib.sha256(data).hexdigest():
            continue
        plan.append((name,target,data))
    if not plan:
        print('PLANT_CONTACT_CODE_ALREADY_CURRENT; no files changed.')
        return
    for p in (root/'maintenance',root/'backups'):
        if p.is_symlink():
            raise ValueError(f'Linked maintenance/backup directory is unsupported: {p}')
    lock = root/'maintenance'/'.plant_contact_install.lock'
    lock.parent.mkdir(exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Install lock exists: {lock}. Inspect any running installer before retrying.') from exc
    os.close(fd)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    backup = root/'backups'/'code_updates'/('plant_contact_'+stamp)
    backup.resolve().relative_to(root)
    if any(p.is_symlink() for p in (backup.parent,backup.parent.parent)):
        lock.unlink()
        raise ValueError('Backup location must be inside the real project tree.')
    applied, created = [], []
    try:
        backup.mkdir(parents=True)
        records = []
        for name,target,data in plan:
            previous = backup/name
            exists = target.is_file()
            if exists:
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, previous)
            records.append(dict(path=name,previously_existed=exists,new_sha256=hashlib.sha256(data).hexdigest()))
        (backup/'update_manifest.json').write_text(json.dumps(dict(status='PENDING', files=records),indent=2),encoding='utf-8')
        for name,target,data in plan:
            target.parent.mkdir(parents=True,exist_ok=True)
            # Atomic replacement in the same destination directory.
            temp = target.with_name(target.name+'.upload_'+uuid.uuid4().hex)
            created.append(temp)
            temp.write_bytes(data)
            if target.exists():
                shutil.copymode(target,temp)
            os.replace(temp,target)
            applied.append((name,target))
        for record in manifest['files']:
            if digest(contained(root,record['path'])) != record['sha256']:
                raise RuntimeError(f'Installed file checksum mismatch: {record["path"]}')
        (backup/'update_manifest.json').write_text(json.dumps(dict(status='SUCCESS',files=records),indent=2),encoding='utf-8')
        (root/'maintenance'/'plant_contact_installed.json').write_text(json.dumps(dict(status='SUCCESS',
            installed_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=digest(archive),
            changed_files=len(plan), previous_code_backup=str(backup), manifest=manifest),indent=2),encoding='utf-8')
        print('PLANT_CONTACT_CODE_INSTALLED')
        print('Project:',root)
        print('Changed files:',len(plan))
        print('Previous code backup:',backup)
        print('Models, inputs, outputs and Python environments were not changed.')
    except BaseException:
        for name,target in reversed(applied):
            previous = backup/name
            if previous.is_file():
                shutil.copy2(previous,target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        for temp in created:
            temp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action',required=True)
    b = sub.add_parser('build')
    b.add_argument('--root',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True)
    for name in ('inspect','install'):
        s = sub.add_parser(name)
        s.add_argument('--archive',type=Path,required=True)
        s.add_argument('--sha256',required=name=='install')
        if name=='install':
            s.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    if args.action=='build':
        build(args.root,args.output)
    elif args.action=='inspect':
        m,p=load_bundle(args.archive,args.sha256)
        print(json.dumps(dict(status='BUNDLE_CHECK_OK',files=len(p),source_only=m['source_only'],
                              palette=m['palette']),ensure_ascii=False))
    else:
        install(args.archive,args.root,args.sha256)


if __name__=='__main__':
    main()
