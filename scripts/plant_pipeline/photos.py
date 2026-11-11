"""Verified photo transport and tmux submission. All server paths stay below project."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from common import JobLock, confined, digest, read_json, run_name, verify_images, write_json
from render_runtime import render_environment

ROOT = Path(__file__).resolve().parents[2]


def prepare(images, run, output):
    from PIL import Image
    files = sorted(p for p in Path(images).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(files) < 3:
        raise ValueError('At least three JPG/PNG photos required (normally use a complete set around the plant)')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_STORED) as z:
        for i,p in enumerate(files,1):
            with Image.open(p) as image:
                size = image.size; image.verify()
            name = f'images/{i:04d}{p.suffix.lower()}'
            z.write(p, name)
            rows.append(dict(path=name, original_name=p.name, sha256=digest(p), width=size[0], height=size[1]))
        z.writestr('input_manifest.json', json.dumps(dict(run=run_name(run), images=rows), ensure_ascii=False, indent=2))
    sha = digest(output)
    Path(str(output)+'.sha256').write_text(sha+'  '+output.name+'\n', encoding='ascii')
    print(json.dumps(dict(status='PHOTO_BUNDLE_OK', images=len(rows), archive=str(output), sha256=sha)))


def ingest(archive, sha256, root, run):
    if digest(archive) != sha256:
        raise ValueError('Photo archive SHA256 mismatch')
    parent = Path(root)/'data/pipeline'; parent.mkdir(parents=True, exist_ok=True)
    target = parent/run_name(run)
    with tempfile.TemporaryDirectory(prefix=run+'_', dir=parent) as temp:
        temp = Path(temp)
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 30*1024**3:
                raise ValueError('Duplicate or oversized photo archive')
            for item in z.infolist():
                name = item.filename
                p = PurePosixPath(name)
                if '\\' in name or p.is_absolute() or '..' in p.parts:
                    raise ValueError('Unsafe photo archive path')
                if name != 'input_manifest.json' and (len(p.parts) != 2 or p.parts[0] != 'images' or p.suffix.lower() not in {'.jpg','.jpeg','.png'}):
                    raise ValueError(f'Unexpected photo archive entry: {name}')
                dst = confined(temp, name); dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, dst.open('wb') as stream:
                    shutil.copyfileobj(source, stream)
        manifest = verify_images(temp)
        if manifest['run'] != run:
            raise ValueError('Run name differs from photo manifest')
        if target.exists():
            verify_images(target)
            if digest(target/'input_manifest.json') != digest(temp/'input_manifest.json'):
                raise ValueError('Run already contains other photos; choose another run name')
            print('PHOTO_INPUT_ALREADY_VERIFIED', target)
            return target
        # Temporary directory and destination share a filesystem.
        temp.rename(target)
        # TemporaryDirectory cleanup tolerates its old path being moved away.
    print('PHOTO_INPUT_INSTALLED', target)
    return target


def start(root, run, gpu, resume):
    root = Path(root).resolve(); verify_images(root/'data/pipeline'/run)
    if gpu < 0:raise ValueError('GPU index must be nonnegative')
    py = root/'envs/plant_semantics/bin/python'
    pipeline = root/'scripts/plant_pipeline/run_pipeline.py'
    subprocess.run([str(py),str(pipeline),'--run',run,'--gpu',str(gpu),'--check-only'],check=True)
    session = 'plant_'+run
    if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0:
        print('PIPELINE_ALREADY_RUNNING',session); return
    if (root/'outputs/pipeline'/run).exists() and not resume:
        raise ValueError('Run has previous outputs. Use --resume to continue, or select a new run name.')
    with JobLock(root/'logs/plant_pipeline.lock'):
        pass  # Reject a busy server before reporting submission.
    log = root/'logs'/f'pipeline_{run}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    argv = [str(py),'-u',str(pipeline),'--run',run,'--gpu',str(gpu)] + (['--resume'] if resume else [])
    # Validate saved configuration now. Apply it only to planning subprocesses;
    # Mesa's library search path must not change the old CUDA environments.
    render_environment(root)
    shell = shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    print('PIPELINE_SUBMITTED',session,'log='+str(log),flush=True)


def export(root, run):
    root = Path(root).resolve()
    result = root/'outputs/pipeline'/run/'result_paths.json'
    report = read_json(result)
    if report['status'] != 'PHOTO_TO_PATH_SUCCESS' or not (result.parent/'PHOTO_TO_PATH_SUCCESS').is_file():
        raise ValueError('Run has not completed successfully')
    simulation = report['simulation']
    wanted = [result, Path(report['simulation_report']), Path(simulation['path_report']),
              Path(report['scale_report']), Path(simulation['scene_report']),
              Path(simulation['validation_report']), Path(simulation['timed_trajectory']), Path(simulation['line_csv']),
              Path(simulation['timed_trajectory']).parent/'timed_trajectory.csv']
    wanted += list(map(Path,simulation['media'].values()))
    # Keep the search decision even when no candidate improves on the baseline.
    for key in ('natural_selection','natural_search_report','demonstration_selection','demonstration_search_report'):
        if simulation.get(key):wanted.append(Path(simulation[key]))
    path_folder=Path(simulation['path_report']).parent
    for name in ('timing_plan.json','trajectory_q.npz','trajectory.csv','natural_search_report.json',
                 'demonstration_search_report.json','demonstration_verification.json',
                 'coordination_report.json', 'coordination_config.json'):
        if (path_folder/name).is_file():wanted.append(path_folder/name)
    # Include metric weighted/semantic models and geometry needed to inspect them.
    mesh = Path(report['semantic_mesh']); weights = Path(report['contact_weights'])
    wanted += [mesh, mesh.parent/'labels.npz', mesh.parent/'coordinate_system.json',weights,
               weights.parent/'contact_risk_heatmap.ply',weights.parent/'contact_weighted_mesh.ply',
               weights.parent/'contact_weights_report.json',
               weights.parent/'contact_weight_colorbar.png']
    if report.get('virtual_substrate_report'):
        wanted += [Path(report['virtual_substrate_report']), mesh.parent/'virtual_substrate.json',
                   mesh.parent/'validation.json', mesh.parent/'source_mapping.npz',
                   mesh.parent/'labeled_original_rgb_mm.ply']
        for name in ('upstream_report.json', 'upstream_source_mapping.npz'):
            if (mesh.parent/name).is_file():wanted.append(mesh.parent/name)
        if report.get('source_reuse'):wanted.append(Path(report['source_reuse']))
    if report.get('cart_underside_report'):
        # Ship synthetic-geometry provenance with the repaired mesh, not just colors.
        wanted += [Path(report['cart_underside_report']),mesh.parent/'cart_template.json',
                   mesh.parent/'source_mapping.npz',mesh.parent/'validation.json',
                   mesh.parent/'labeled_original_rgb_mm.ply']
    destination = root/'exports'/run
    destination.mkdir(parents=True,exist_ok=True)
    archive = destination/'results.zip'
    with zipfile.ZipFile(archive.with_suffix('.zip.partial'),'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in dict.fromkeys(wanted):
            p = confined(root,p)
            if not p.is_file(): raise ValueError(f'Missing result artifact: {p}')
            z.write(p,p.relative_to(root).as_posix())
    os.replace(archive.with_suffix('.zip.partial'),archive)
    Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('RESULTS_EXPORTED',archive)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='action',required=True)
    p=subs.add_parser('prepare');p.add_argument('--images',type=Path,required=True);p.add_argument('--run',type=run_name,required=True);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('ingest');p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',type=run_name,required=True)
    p=subs.add_parser('start');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',type=run_name,required=True);p.add_argument('--gpu',type=int,default=1);p.add_argument('--resume',action='store_true')
    p=subs.add_parser('export');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',type=run_name,required=True)
    args=ap.parse_args();values=vars(args);action=values.pop('action')
    globals()[action](**values)


if __name__=='__main__':main()
