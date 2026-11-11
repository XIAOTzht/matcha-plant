"""Package and recompute the accepted model/weights, then revalidate its path.

Saved upstream RGB geometry and inference evidence are explicit inputs. This
does not claim to rerun photo reconstruction or to search for a new path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/plant_pipeline'))
from common import JobLock, Stages, digest, read_json, run_name, write_json
from render_runtime import render_environment


def array_digest(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError('Object arrays cannot be fingerprinted')
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def npz_digest(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return {key: array_digest(data[key]) for key in data.files}


def mesh_digest(path):
    from plyfile import PlyData
    data = PlyData.read(str(path), known_list_len={'face': {'vertex_indices': 3}})
    return {element.name+'.'+field: array_digest(element.data[field])
            for element in data.elements for field in element.data.dtype.names}


def build(output):
    """An input/code bundle, explicitly including the old RGB/evidence/scene."""
    source = ROOT/'outputs/semantic_review_inputs/plant_20261106_new_20261106_205823'
    evidence = ROOT/'outputs/semantic_refinement/plant_20261106_new_rim_shape_v2'
    cart = ROOT/'outputs/semantic_refinement/plant_20261106_new_cart_flat_v2'
    contact = ROOT/'outputs/semantic_refinement/plant_20261106_new_cart_flat_v2_contact'
    simulation = ROOT/'simulation/outputs/coordinated_20261107'
    from reproduce_approved_semantics import SNAPSHOT, verify_inputs
    verify_inputs(source, evidence)
    references = dict(semantics=npz_digest(evidence/'labels.npz'),
        cart=npz_digest(cart/'labels.npz'), mapping=npz_digest(cart/'source_mapping.npz'),
        organs=npz_digest(contact/'organs/organ_instances.npz'),
        weights=npz_digest(contact/'weights/contact_weights.npz'),
        cart_mesh=mesh_digest(cart/'labeled_original_rgb_mm.ply'))
    files = {}
    def add(directory, names, prefix):
        for name in names:
            files[prefix+'/'+name] = directory/name
    add(source, ['labeled_original_rgb_mm.ply','labels.npz','coordinate_system.json',
                 'classes.json','report.json','pipeline_result_paths.json'], 'source')
    add(evidence, ['review_config.json','report.json','cameras.json',
                  'evidence_provenance.json','review_evidence_checkpoint.npz'], 'evidence')
    add(cart, ['cart_template.json'], 'reference')
    add(simulation/'coordinated_v2', ['trajectory_q.npz','path_report.json'], 'simulation/approved_path')
    add(simulation/'replay_v2', ['config_snapshot.json','validation_report.json'], 'reference/replay')
    add(simulation, ['palette.json','result_index.json'], 'reference')
    for prefix, directory in [('frozen', SNAPSHOT), ('simulation/scripts', ROOT/'simulation/scripts'),
                              ('simulation/assets/robot/robotstudio_so101', ROOT/'simulation/assets/robot/robotstudio_so101'),
                              ('simulation/scene', simulation/'scene')]:
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                if prefix == 'simulation/scripts' and path.suffix != '.py':
                    continue
                files[prefix+'/'+path.relative_to(directory).as_posix()] = path
    # Rebase the saved scene's external meshes to the matching bundled assets.
    import xml.etree.ElementTree as ET
    tree = ET.parse(simulation/'scene/scene.xml')
    for mesh in tree.findall('asset/mesh'):
        name = mesh.get('file')
        if name and 'robotstudio_so101/' in name:
            mesh.set('file', '../assets/robot/robotstudio_so101/'+name.split('robotstudio_so101/',1)[1])
    from io import BytesIO
    xml = BytesIO(); tree.write(xml, encoding='utf-8', xml_declaration=True)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version='approved-20261107-reproduction-1', files={}, references=references,
        scope='Saved RGB geometry + saved inference evidence -> old semantics/cart/organs/weights; approved path replay',
        photo_reconstruction_rerun=False, neural_inference_rerun=False, path_search_rerun=False)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            data = xml.getvalue() if name == 'simulation/scene/scene.xml' else path.read_bytes()
            archive.writestr(name, data)
            manifest['files'][name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        archive.writestr('reproduction_manifest.json', json.dumps(manifest, indent=2))
    checksum = digest(output)
    Path(str(output)+'.sha256').write_text(checksum+'  '+output.name+'\n', encoding='ascii')
    print(json.dumps(dict(status='APPROVED_INPUT_BUNDLE_CREATED', archive=str(output),
                         sha256=checksum, bytes=output.stat().st_size, files=len(files))))


def unpack(archive, sha256, destination):
    if digest(archive) != sha256:
        raise ValueError('Reproduction archive checksum mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = json.loads(z.read('reproduction_manifest.json'))
        if (len(names) != len(set(names)) or set(names) != set(manifest['files'])|{'reproduction_manifest.json'}):
            raise ValueError('Reproduction file manifest mismatch')
        for name, info in manifest['files'].items():
            p = PurePosixPath(name)
            if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
                raise ValueError('Invalid reproduction member: '+name)
            data = z.read(name)
            if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('Corrupt reproduction input: '+name)
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        write_json(destination/'reproduction_manifest.json', manifest)
    return manifest


def verified_stage(source, name):
    """Read a completed upstream stage without editing its state or artifacts."""
    state = read_json(Path(source)/'pipeline_state.json')
    entry = state['stages'].get(name,{})
    if entry.get('status') != 'SUCCESS':
        raise ValueError('Source stage was not verified: '+name)
    artifacts = entry.get('artifacts',{})
    if not artifacts:
        raise ValueError('Empty source stage: '+name)
    result = {}
    for key, info in artifacts.items():
        path = Path(info['path']).resolve()
        path.relative_to(Path(source).resolve())
        if not path.is_file() or digest(path) != info['sha256']:
            raise ValueError('Source stage file changed: '+str(path))
        result[key] = path
    return result


def run(archive, sha256, revision, sim_python, resume=False, submit=False,
        weights_python=None, source_revision=None):
    out = ROOT/'outputs/reproduction'/revision
    if digest(archive) != sha256:
        raise ValueError('Reproduction archive checksum mismatch')
    if out.exists() and not resume:
        raise ValueError('Revision exists; choose a new revision or explicit --resume')
    if submit:
        if os.name != 'posix':
            raise ValueError('--submit requires Linux tmux')
        import shlex
        session = 'repro_'+revision
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode == 0:
            raise ValueError('Reproduction session already exists: '+session)
        if digest(archive) != sha256:
            raise ValueError('Reproduction archive checksum mismatch')
        with JobLock(ROOT/'logs/plant_pipeline.lock'):
            pass
        log = ROOT/'logs'/('reproduction_'+revision+'.log')
        argv = [sys.executable,'-u',Path(__file__),'run','--archive',archive,
                '--sha256',sha256,'--revision',revision,'--sim-python',sim_python]
        if resume:
            argv.append('--resume')
        if weights_python:
            argv.extend(['--weights-python',weights_python])
        if source_revision:
            argv.extend(['--source-revision',source_revision])
        shell = shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
        print('APPROVED_REPRODUCTION_SUBMITTED',revision,'log='+str(log),flush=True)
        return
    sim_root = ROOT/'simulation/outputs/reproduction'/revision
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    weights_python = weights_python or Path(sys.executable)
    source = ROOT/'outputs/reproduction'/source_revision if source_revision else None
    if source_revision == revision:
        raise ValueError('Continuation requires a new revision')
    signature = dict(archive=sha256, launcher=digest(__file__),
        semantics_launcher=digest(ROOT/'maintenance/reproduce_approved_semantics.py'),
        semantic_python=sys.executable, simulation_python=str(sim_python))
    if source is not None or str(weights_python) != sys.executable:
        signature.update(weights_python=str(weights_python),source_revision=source_revision,
            source_state_sha256=digest(source/'pipeline_state.json') if source else None)
    with JobLock(ROOT/'logs/plant_pipeline.lock'):
        stages = Stages(out, signature)
        try:
            def call(argv, folder, environment=env):
                from common import command
                command(argv, folder, ROOT, environment)
            def inputs(d):
                if source is not None:
                    if read_json(source/'pipeline_state.json')['signature']['archive'] != sha256:
                        raise ValueError('Source revision belongs to a different input bundle')
                    return verified_stage(source,'01_inputs')
                unpack(archive, sha256, d/'capsule')
                return dict(manifest=d/'capsule/reproduction_manifest.json')
            imported = stages.execute('01_inputs', inputs)
            capsule = imported['manifest'].parent
            manifest = read_json(imported['manifest'])
            for name, info in manifest['files'].items():
                if digest(capsule/name) != info['sha256']:
                    raise ValueError('Installed reproduction input changed: '+name)
            def check(actual, key):
                if actual != manifest['references'][key]:
                    expected = manifest['references'][key]
                    fields = sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
                    write_json(out/(key+'_reference_mismatch.json'),dict(fields=fields,
                        actual=actual,expected=expected,comparison='Exact array shape, dtype and value bytes'))
                    raise ValueError('Accepted reference differs: '+key+'; fields='+', '.join(fields))
            def semantics(d):
                if source is not None:
                    result = verified_stage(source,'02_semantics')
                    check(npz_digest(result['labels']),'semantics')
                    return result
                call([sys.executable, ROOT/'maintenance/reproduce_approved_semantics.py',
                    '--source',capsule/'source','--evidence',capsule/'evidence',
                    '--snapshot',capsule/'frozen','--output',d/'result'],d)
                check(npz_digest(d/'result/labels.npz'),'semantics')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz')
            semantic = stages.execute('02_semantics', semantics)
            def repair(d):
                if source is not None:
                    result = verified_stage(source,'03_cart');directory=result['mesh'].parent
                    check(npz_digest(result['labels']),'cart')
                    check(npz_digest(directory/'source_mapping.npz'),'mapping')
                    check(mesh_digest(directory/'labeled_original_rgb_mm.ply'),'cart_mesh')
                    return result
                call([sys.executable,capsule/'frozen/tools/plant_extract_with_car/repair_cart_underside.py',
                    '--source',semantic['mesh'].parent,'--output',d/'result',
                    '--template',capsule/'reference/cart_template.json'],d)
                check(npz_digest(d/'result/labels.npz'),'cart')
                check(npz_digest(d/'result/source_mapping.npz'),'mapping')
                check(mesh_digest(d/'result/labeled_original_rgb_mm.ply'),'cart_mesh')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz')
            cart = stages.execute('03_cart', repair)
            def weights(d):
                call([weights_python,capsule/'frozen/scripts/plant_contact/run_pipeline.py',
                    '--input-dir',cart['mesh'].parent,'--output-dir',d/'result','--no-parts'],d)
                check(npz_digest(d/'result/organs/organ_instances.npz'),'organs')
                check(npz_digest(d/'result/weights/contact_weights.npz'),'weights')
                return dict(weights=d/'result/weights/contact_weights.npz',organs=d/'result/organs/organ_instances.npz',
                            preview=d/'result/weights/contact_risk_heatmap_preview.jpg')
            contact = stages.execute('04_weights',weights)
            def replay(d):
                if sim_root.exists():
                    raise ValueError('Replay trial exists; choose a new revision')
                shutil.copytree(capsule/'simulation/scene',sim_root/'scene')
                shutil.copytree(capsule/'simulation/approved_path',sim_root/'approved_path')
                shutil.copytree(capsule/'simulation/scripts',sim_root/'scripts')
                c = read_json(sim_root/'scene/config_snapshot.json')
                c['source_root'] = os.path.relpath(ROOT,sim_root).replace(os.sep,'/')
                c.update(semantic_mesh=cart['mesh'].relative_to(ROOT).as_posix(),
                    semantic_labels=cart['labels'].relative_to(ROOT).as_posix(),
                    coordinate_system=(cart['mesh'].parent/'coordinate_system.json').relative_to(ROOT).as_posix(),
                    weights=contact['weights'].relative_to(ROOT).as_posix(),
                    organs=contact['organs'].relative_to(ROOT).as_posix(),
                    policy=(capsule/'frozen/tools/contact_weights/contact_policy.json').relative_to(ROOT).as_posix(),
                    palette=(capsule/'reference/palette.json').relative_to(ROOT).as_posix())
                write_json(sim_root/'scene/config_snapshot.json',c)
                # Mesh paths are relative to this new scene directory.
                import xml.etree.ElementTree as ET
                tree = ET.parse(sim_root/'scene/scene.xml')
                for mesh in tree.findall('asset/mesh'):
                    name = mesh.get('file')
                    if name and not name.startswith('display_meshes/'):
                        original = (capsule/'simulation/scene'/name).resolve()
                        mesh.set('file',os.path.relpath(original,sim_root/'scene').replace(os.sep,'/'))
                tree.write(sim_root/'scene/scene.xml',encoding='utf-8',xml_declaration=True)
                p = read_json(sim_root/'approved_path/path_report.json')
                p['scene_relative'] = 'scene/scene.xml'
                p['timing_config_relative'] = os.path.relpath(capsule/'reference/replay/config_snapshot.json',sim_root).replace(os.sep,'/')
                write_json(sim_root/'approved_path/path_report.json',p)
                call([sim_python,sim_root/'scripts/validate_watering_trajectory.py',
                    '--path',sim_root/'approved_path','--config',capsule/'reference/replay/config_snapshot.json',
                    '--output-dir',sim_root/'replay','--no-promote'],d/'validation',render_environment(ROOT,env))
                report = read_json(sim_root/'replay/validation_report.json')
                baseline = read_json(capsule/'reference/replay/validation_report.json')
                cost = report['contact_risk']['simulated']['persistence_cost']
                expected = baseline['contact_risk']['simulated']['persistence_cost']
                if (report['status']!='SAMPLED_REPLAY_CHECKS_PASSED' or len(report['checks'])!=18
                        or not all(report['checks'].values()) or abs(cost-expected)>1e-9
                        or abs(report['duration_s']-baseline['duration_s'])>1e-9):
                    raise ValueError('Approved replay metrics differ; inspect validation report')
                write_json(sim_root/'configs/latest_replay.json',dict(output='replay',
                    report='replay/validation_report.json',status=report['status']))
                call([sim_python,sim_root/'scripts/render_trajectory_replay.py',
                    '--replay',sim_root/'replay','--fps','10'],d/'render',render_environment(ROOT,env))
                return dict(report=sim_root/'replay/validation_report.json',animation=sim_root/'replay/trajectory_replay.gif')
            replayed = stages.execute('05_approved_path_replay', replay)
            write_json(out/'result_paths.json',dict(status='APPROVED_REPRODUCTION_SUCCESS',
                semantic_mesh=str(cart['mesh']),weights=str(contact['weights']),weight_preview=str(contact['preview']),
                animation=str(replayed['animation']),validation=str(replayed['report']),
                photo_reconstruction_rerun=False,neural_inference_rerun=False,path_search_rerun=False))
            (out/'APPROVED_REPRODUCTION_SUCCESS').write_text('Approved inputs/method reproduced; approved path replayed\n')
            (out/'REPRODUCTION_FAILED.json').unlink(missing_ok=True)
            print('APPROVED_REPRODUCTION_SUCCESS',out,flush=True)
        except BaseException as error:
            write_json(out/'REPRODUCTION_FAILED.json',dict(error=str(error),stage=stages.state.get('current_stage')))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='action',required=True)
    b = s.add_parser('build'); b.add_argument('--output',type=Path,required=True)
    r = s.add_parser('run')
    r.add_argument('--archive',type=Path,required=True);r.add_argument('--sha256',required=True)
    r.add_argument('--revision',type=run_name,required=True);r.add_argument('--sim-python',type=Path,required=True)
    r.add_argument('--resume',action='store_true')
    r.add_argument('--submit',action='store_true')
    r.add_argument('--weights-python',type=Path)
    r.add_argument('--source-revision',type=run_name)
    a = vars(p.parse_args());action = a.pop('action');globals()[action](**a)


if __name__ == '__main__':
    main()
