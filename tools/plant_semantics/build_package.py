"""Build a reproducible source archive with a per-file integrity manifest.

Only explicitly selected project sources are distributed. Model weights, local
runtime settings, compiled vendor packages and generated results stay outside.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
ROOT_FILES = (
    '.gitignore', 'VERSION', 'README.md', 'START_HERE_V3.md', 'CHANGELOG.md', 'requirements.txt',
    'plantseg.py', 'refine_detail.py', 'complete_five.py', 'refine_leaf_boundary.py', 'refine_leaf_rim.py',
    'organ_shape_rules.py', 'organ_surface_consistency.py', 'refine_organ_semantics.py',
    'reassess_uncertainty.py', 'extra_view_review.py', 'run_extra_views.ps1',
    'validate_complete.py', 'estimate_axis.py', 'check_install.py', 'fetch_models.py',
    'build_package.py', 'run_v3.ps1', 'segment_v3.cmd', 'run_auto.ps1', 'run_all_five.ps1', 'segment.cmd',
    'runtime.ps1', 'setup_local.ps1', 'check_code.ps1',
    'sam3_predict.py', 'sam3_fuse.py', 'sam3_prepare_detail.py', 'run_sam3.ps1',
    'run.ps1', 'run_detail.ps1', 'run_complete.ps1',
)
TREE_TYPES = {'docs': {'.md'}, 'tests': {'.py'}, 'experiments': {'.py', '.md'}}
ARCHIVE_TIME = (2026, 10, 20, 0, 0, 0)


def collect_sources(root=ROOT, profile='full'):
    if profile not in ('full', 'v3'):
        raise ValueError(f'Unknown distribution profile: {profile}')
    files = [root / name for name in ROOT_FILES]
    for folder, suffixes in TREE_TYPES.items():
        files.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and path.suffix in suffixes
                     and '__pycache__' not in path.parts)
    mobile_sam = root / 'third_party' / 'MobileSAM'
    files.extend(path for path in mobile_sam.rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts
                 and (path.suffix == '.py' or path.name == 'LICENSE'))
    files.append(root / 'models' / 'weight_hashes.json')
    files.extend((root / 'models' / 'sam3').glob('*.md'))
    files.append(mobile_sam / 'LICENSE')
    if profile == 'v3':
        files = [path for path in files
                 if not any(part.lower() in ('sam3', 'experiments', 'archive')
                            for part in path.relative_to(root).parts)
                 and not path.name.lower().startswith(('sam3_', 'run_sam3.'))
                 and path.name.lower() != 'sam3.md']
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing distribution sources: {missing}')
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def write_member(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_package(target, root=ROOT, profile='full'):
    files = collect_sources(root, profile)
    payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    if profile == 'v3':
        payload['README.md'] = (root / 'START_HERE_V3.md').read_bytes()
        # Shared maintenance docs mention an optional historical component.
        # Its documentation is excluded from this profile; avoid dead links.
        for name, data in list(payload.items()):
            if name.endswith('.md'):
                text = data.decode('utf-8')
                text = re.sub(r'\[([^\]]+)\]\((?:docs/)?SAM3\.md\)',
                              r'\1（仅完整维护源码包附带）', text)
                payload[name] = text.encode('utf-8')
    manifest = {
        'version': (root / 'VERSION').read_text(encoding='utf-8').strip(),
        'kind': 'source_only',
        'profile': profile,
        'entrypoint': 'run_v3.ps1',
        'excluded': ['model weights', 'vendor', 'runtime.local.json', 'results', 'Python runtime'],
        'files': [{'path': name, 'bytes': len(data),
                   'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in payload.items()],
    }
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w') as archive:
        for name, data in payload.items():
            write_member(archive, name, data)
        write_member(archive, 'PACKAGE_MANIFEST.json', json_bytes(manifest))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
        for record in manifest['files']:
            data = archive.read(record['path'])
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise RuntimeError(f"Archive integrity check failed: {record['path']}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + '.sha256').write_text(
        f'{digest}  {target.name}\n', encoding='ascii')
    return {'path': str(target), 'version': manifest['version'],
            'files': len(payload) + 1, 'bytes': target.stat().st_size, 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['full', 'v3'], default='full')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or ROOT / ('plant_semantics_v3_source.zip' if args.profile == 'v3' else 'plant_semantics_source.zip')
    print(json.dumps(build_package(target, profile=args.profile), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
