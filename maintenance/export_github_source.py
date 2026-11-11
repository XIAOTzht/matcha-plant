"""Export source and robot assets to a separate folder. Never push or modify source."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

TOPS = {'MAtCha','tools','scripts','configs','simulation','maintenance','docs','third_party'}
PRUNE = {'.git','envs','.venv','venv','__pycache__','.cache','cache','tmp','t',
         'logs','backups','incoming','outputs','exports','data','weights','checkpoints',
         'pretrained','packages','node_modules','build','dist','CMakeFiles',
         '.pytest_cache','.mypy_cache','.ruff_cache','.gradio','.ssh','.aws'}
EXTS = {'.py','.pyx','.pxd','.pxi','.sh','.ps1','.bat','.md','.rst','.txt','.json',
        '.yaml','.yml','.toml','.cfg','.ini','.xml','.c','.cc','.cpp','.cxx','.h',
        '.hpp','.cu','.cuh','.cmake','.in','.stl','.obj','.mtl','.png','.jpg',
        '.jpeg','.svg','.vert','.frag','.glsl','.html','.css','.js','.ts',
        '.inl','.pyi','.typed','.cmd','.dockerfile'}
SPECIAL = {'Makefile','Dockerfile','VERSION','.gitmodules','.gitattributes','.gitkeep'}
RUNTIME = {'server_render_runtime.json','SERVER_RESULT_INDEX.json',
           'plant_contact_installed.json','cmake_install.cmake','CMakeCache.txt',
           'compile_commands.json','downloads.json'}
SECRETS = [
    rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',
    rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
    rb'\bAKIA[A-Z0-9]{16}\b',
]
REQUIRED = [
    'MAtCha/train.py','MAtCha/environment.yml','MAtCha/LICENSE',
    'MAtCha/scripts/run_sfm.py','MAtCha/scripts/extract_tetra_mesh.py',
    'scripts/plant_pipeline/photos.py','scripts/plant_pipeline/run_pipeline.py',
    'configs/plant_pipeline.json','tools/plant_semantics/plantseg.py',
    'tools/contact_weights/build_organ_contact_weights.py',
    'tools/contact_weights/contact_policy.json','tools/contact_weights/organ_weight_policy.json',
    'simulation/scripts/run_server_job.py','simulation/scripts/coordinated_motion.py',
    'simulation/scripts/trajectory_tools.py','simulation/scripts/validate_watering_trajectory.py',
    'simulation/requirements-linux.txt','simulation/configs/coordinated_motion.json',
    'simulation/assets/robot/robotstudio_so101/so101.xml',
    'simulation/assets/robot/robotstudio_so101/LICENSE',
    'maintenance/reproduction/approved_20261107/snapshot_manifest.json',
]


def exclusion(name, size):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        return 'unsafe_path'
    if len(p.parts) > 1 and p.parts[0] not in TOPS:
        return 'outside_source_roots'
    if any(x in PRUNE or x.endswith('.egg-info') or x.startswith('mplconfig') for x in p.parts[:-1]):
        return 'runtime_or_generated_directory'
    if name.startswith(('tools/plant_semantics/models/','MAtCha/media/')):
        return 'downloaded_models_or_upstream_showcase'
    if p.name in RUNTIME or p.name.startswith('latest_'):
        return 'runtime_or_result_pointer'
    if name=='MAtCha/2d-gaussian-splatting/submodules/tetra-triangulation/Makefile':
        return 'generated_cmake_makefile'
    if p.name.startswith('.env') or p.name in {'id_rsa','id_ed25519','.netrc','.pypirc'}:
        return 'credential_file'
    if p.name == '.gitignore':
        return 'replaced_export_ignore'
    if p.name.endswith(('.bak','.backup')) or '.backup_' in p.name:
        return 'backup_file'
    if p.suffix.lower() not in EXTS and p.name not in SPECIAL and not p.name.lower().startswith(('license','copying','notice')):
        return 'binary_archive_or_non_source'
    if size > 20*1024**2:
        return 'exceeds_source_size_limit'
    return None


def files(root):
    for folder, dirs, names in os.walk(root, followlinks=False):
        base = Path(folder)
        dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.endswith('.egg-info')
                         and not d.startswith('mplconfig') and not (base/d).is_symlink()
                         and (base != root or d in TOPS))
        for name in sorted(names):
            if not (base/name).is_symlink():
                yield base/name


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


README = """# Plant reconstruction, semantic risk and robot planning

Multi-view RGB photos → MAtCha reconstruction → fixed-cart crop and metric scale
→ plant semantics → organ proposals → continuous contact risk → SO-101 planning
→ MuJoCo trajectory checks and replay.

## Source map

- MAtCha/: reconstruction, vendored dependencies and local modifications.
- tools/plant_extract_with_car/: crop, scale calibration and cart repair.
- tools/plant_semantics/ and tools/plant_semantics_with_car/: semantic evidence,
  shape/surface refinement and cart/hardware correction.
- tools/plant_organ_instances/: organ proposals and surface geometry.
- tools/contact_weights/: organ, proximity and persistent-exposure costs.
- scripts/plant_pipeline/: photo-to-path orchestration and stage state.
- configs/plant_pipeline.json: integrated configuration.
- simulation/: robot meshes, weighted planning, coordinated motion and validation.
- maintenance/: setup, submission, transfer, diagnosis and reproduction utilities.
- maintenance/reproduction/approved_20261107/: frozen source needed by reproduction.

## Running

Start with docs/PHOTO_TO_PATH_GUIDE.md. This is source and robot assets, not an
installed environment. Reconstruction needs compatible CUDA/PyTorch and native
extensions; consult MAtCha/environment.yml, MAtCha/install.py and MAtCha/README.md.
Semantic dependencies: tools/plant_semantics/requirements_server.txt.
Simulation dependencies: simulation/requirements-linux.txt.
Adapt environment files and CUDA paths for a new host. A clean-clone installation
has not yet been tested.

For an existing prepared server, maintenance/setup_plant_pipeline.sh reuses/checks
reconstruction and semantic environments and sets up simulation. It does not
replace all upstream installation steps. scripts/plant_pipeline/photos.py --help
describes the photo job interface. Windows clients are in maintenance/*.ps1;
check host/user/root arguments before use. Credentials do not belong in source.

## External assets

Photographs, pretrained checkpoints, plant meshes, installed environments and
results are excluded. Acquisition helpers include MAtCha/download_checkpoints.py
and tools/plant_semantics/fetch_models.py. Verify all model versions and required
files against installation checks, including CLIPSeg: one helper may not download
every model. Machine render configuration is regenerated by setup.

Approved-result reproduction differs from processing new photos. It requires
the saved input/evidence capsule and approved path, stored separately; see
docs/APPROVED_REPRODUCTION.md. Frozen source remains included for that reason.

Capture rules assume the same 135 x 110 mm blue cart plate. Cart-bottom repair
and virtual substrate are inferred geometry. Robot registration and nozzle are
simulation settings. Risk is geometric exposure, not measured force, tissue
damage or proof of globally optimal motion.

## Provenance

SOURCE_EXPORT_MANIFEST.json records file SHA256 from the server working tree,
including uncommitted changes. SOURCE_EXPORT_EXCLUSIONS.json records omissions.
No Git history or nested repositories are copied. Source stability is rechecked
after copying. The limited credential-pattern scan is not a complete security audit.

Preserve every upstream LICENSE/NOTICE/COPYING file. Components have different
terms, including research/noncommercial restrictions. No blanket MIT license is
assigned. See THIRD_PARTY_NOTICES.md.

This README supersedes older root layout notes, retained as
docs/LEGACY_SERVER_README.md if present.
"""

NOTICES = """# Third-party source and assets

Retain all upstream LICENSE, COPYING and NOTICE files. File hashes establish
snapshot contents, not upstream commit identities (not yet verified).

- MAtCha: MAtCha/LICENSE (MIT in locally reviewed copy).
- MASt3R / DUSt3R / CroCo: respective LICENSE and NOTICE; noncommercial
  CC BY-NC-SA conditions appear in the reviewed tree.
- 2D Gaussian Splatting, rasterizer and simple-knn: LICENSE.md files;
  research/noncommercial conditions appear in the reviewed tree.
- Depth Anything V2: MAtCha/Depth-Anything-V2/LICENSE.
- MobileSAM: tools/plant_semantics/third_party/MobileSAM/LICENSE.
- SO-101 assets: simulation/assets/robot/robotstudio_so101/LICENSE (Apache-2.0).
- GLM, tetra-triangulation, ASMK and other nested components: consult their
  individual license and notice files.

The copied server license texts govern, rather than this summary. Checkpoint
licenses must also be checked when models are downloaded separately.
"""

# Directory-only rules avoid hiding legitimate files with matching names.
IGNORE = ''.join(x+'/\n' for x in sorted(PRUNE - {'.git'})) + """
*.egg-info/
mplconfig*/
tools/plant_semantics/models/
MAtCha/media/
*.so
*.dll
*.dylib
*.o
*.a
*.pyc
*.pyd
*.pt
*.pth
*.ckpt
*.safetensors
*.zip
*.tar
*.tar.gz
*.whl
*.partial
*.part
*.bak
.env
.env.*
id_rsa
id_ed25519
.netrc
.pypirc
.DS_Store
Thumbs.db
configs/server_render_runtime.json
simulation/configs/latest_*.json
"""


def export(root, destination):
    root, destination = root.resolve(), destination.resolve()
    archive = Path(str(destination)+'.zip')
    if root == destination or root in destination.parents:
        raise ValueError('Choose a separate destination outside the active project')
    if destination.exists() or archive.exists():
        raise FileExistsError('Choose a new export destination')
    selected, excluded = [], []
    for p in files(root):
        rel = p.relative_to(root).as_posix()
        why = exclusion(rel,p.stat().st_size)
        (excluded if why else selected).append((p,rel,why))
    missing = sorted(set(REQUIRED)-{rel for _,rel,_ in selected})
    if missing:
        raise ValueError('Missing required source: '+', '.join(missing))
    destination.mkdir(parents=True)
    copied = []
    try:
        for p,rel,_ in selected:
            before = p.stat()
            data = p.read_bytes()
            after = p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Source changed during export: '+rel)
            if p.suffix.lower() not in {'.png','.jpg','.jpeg','.stl'}:
                if any(re.search(pattern,data) for pattern in SECRETS):
                    raise RuntimeError('Potential credential; inspect file: '+rel)
            out = 'docs/LEGACY_SERVER_README.md' if rel=='README.md' else rel
            dst = destination/out
            if dst.exists():
                raise RuntimeError('Destination collision: '+out)
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(data)
            dst.chmod(stat.S_IMODE(before.st_mode))
            copied.append(dict(source=rel,path=out,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        for row in copied:
            if hashlib.sha256((root/row['source']).read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('Source changed across snapshot: '+row['source'])
        for name,content in {'README.md':README,'.gitignore':IGNORE,'THIRD_PARTY_NOTICES.md':NOTICES}.items():
            (destination/name).write_text(content,encoding='utf-8')
        save(destination/'SOURCE_EXPORT_MANIFEST.json',
             dict(version=1,time_utc=datetime.now(timezone.utc).isoformat(),files=copied,
                  source_stability_rechecked=True,fresh_install_tested=False))
        save(destination/'SOURCE_EXPORT_EXCLUSIONS.json',
             dict(pruned_directory_names=sorted(PRUNE),
                  files=[dict(path=r,reason=w) for _,r,w in excluded]))
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(destination.rglob('*')):
                if p.is_file():
                    z.write(p,destination.name+'/'+p.relative_to(destination).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise RuntimeError('Archive verification failed')
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
        (destination/'SOURCE_EXPORT_READY').write_text('Exported; not published.\n',encoding='utf-8')
        print(json.dumps(dict(status='GITHUB_SOURCE_EXPORTED',files=len(copied),
                              source_bytes=sum(r['bytes'] for r in copied),
                              destination=str(destination),archive=str(archive),sha256=digest)))
    except BaseException:
        (destination/'EXPORT_FAILED').write_text('Incomplete snapshot; do not publish.\n',encoding='utf-8')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    export(a.root,a.destination)


if __name__=='__main__':
    main()
