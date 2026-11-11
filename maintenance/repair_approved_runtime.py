"""Audit a failed exact reproduction and retry weights in a pinned runtime.

The model/classification and old revision stay read-only. Exact comparisons
remain authoritative; this never substitutes saved reference weights as output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/plant_pipeline'))
from common import JobLock, digest, read_json, run_name, write_json
from approved_reproduction import verified_stage

VERSIONS = {'numpy':'2.5.4','scipy':'1.18.1','Pillow':'12.3.0','plyfile':'1.1.5'}


def compare_arrays(actual_path, expected_path):
    """Report semantic/instance changes separately from float field differences."""
    import numpy as np
    fields = {}
    with np.load(actual_path,allow_pickle=False) as a, np.load(expected_path,allow_pickle=False) as b:
        for name in sorted(set(a.files)|set(b.files)):
            if name not in a.files or name not in b.files:
                fields[name] = dict(exact=False,reason='missing field');continue
            x,y = a[name],b[name]
            if x.dtype != y.dtype or x.shape != y.shape:
                fields[name] = dict(exact=False,reason='dtype or shape differs',actual_dtype=str(x.dtype),
                    expected_dtype=str(y.dtype),actual_shape=list(x.shape),expected_shape=list(y.shape));continue
            exact = bool(np.array_equal(x,y,equal_nan=True)) if x.dtype.kind in 'fc' else bool(np.array_equal(x,y))
            row = dict(exact=exact,dtype=str(x.dtype),shape=list(x.shape))
            if not exact and x.dtype.kind in 'biuf':
                same = (x==y)
                if x.dtype.kind=='f':same |= np.isnan(x)&np.isnan(y)
                row['changed_entries']=int(np.sum(~same))
                if x.dtype.kind=='f':
                    finite = np.isfinite(x)&np.isfinite(y)
                    row['finite_pattern_equal']=bool(np.array_equal(np.isfinite(x),np.isfinite(y)))
                    row['max_abs_difference']=float(np.max(abs(x[finite].astype('f8')-y[finite].astype('f8')))) if np.any(finite) else None
            fields[name]=row
    return dict(all_fields_equal=all(v['exact']for v in fields.values()),fields=fields,
                actual=str(actual_path),expected=str(expected_path))


def build(output):
    import numpy as np
    import hashlib
    contact=ROOT/'outputs/semantic_refinement/plant_20261106_new_cart_flat_v2_contact'
    # A calculation fixture catches the Pillow polygon-rasterization difference
    # independently of plant anatomy or handpicked source coordinates.
    sys.path.insert(0,str(ROOT/'maintenance/reproduction/approved_20261107/tools/plant_organ_instances'))
    from organ_geometry import smooth_center_field
    x,y=np.meshgrid(np.linspace(0,1,19),np.linspace(0,.7,17))
    q=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    grid=np.arange(len(q)).reshape(x.shape)
    a,b,c,d=[v.ravel()for v in (grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:])]
    faces=np.vstack([np.column_stack([a,b,c]),np.column_stack([b,d,c])]).astype('i4')
    q[:,:2]=q[:,:2]@np.array([[.83,.31],[-.31,.83]])
    expected,*_=smooth_center_field(q,faces)
    from io import BytesIO
    fixture=BytesIO();np.savez_compressed(fixture,projected=q,faces=faces,expected_center=expected)
    payload={'organs.npz':(contact/'organs/organ_instances.npz').read_bytes(),
             'weights.npz':(contact/'weights/contact_weights.npz').read_bytes(),'raster_fixture.npz':fixture.getvalue()}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_STORED)as z:
        for name,data in payload.items():z.writestr(name,data)
        z.writestr('manifest.json',json.dumps(dict(files={n:hashlib.sha256(d).hexdigest()for n,d in payload.items()},versions=VERSIONS)))
    checksum=digest(output)
    Path(str(output)+'.sha256').write_text(checksum+'  '+output.name+'\n',encoding='ascii')
    print('APPROVED_RUNTIME_REFERENCE_READY',str(output),checksum,output.stat().st_size)


def unpack_reference(archive,sha256,output):
    import hashlib
    if digest(archive)!=sha256:raise ValueError('Runtime reference archive checksum mismatch')
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive)as z:
        manifest=json.loads(z.read('manifest.json'))
        allowed={'organs.npz','weights.npz','raster_fixture.npz'}
        if (set(manifest['files'])!=allowed or set(z.namelist())!=allowed|{'manifest.json'}
                or len(z.namelist())!=4 or manifest['versions']!=VERSIONS):
            raise ValueError('Unexpected runtime reference members')
        for name,h in manifest['files'].items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Corrupt runtime reference: '+name)
            (output/name).write_bytes(data)
    return output


def probe(python):
    code="import json,sys,importlib.metadata as m;print(json.dumps({'python':sys.version.split()[0],'versions':{k:m.version(k) for k in ['numpy','scipy','Pillow','plyfile']}}))"
    return json.loads(subprocess.check_output([str(python),'-c',code],text=True,encoding='utf-8',stderr=subprocess.PIPE))


def prepare(source_revision,revision,reference_archive,sha256,base_python,submit=False):
    if source_revision==revision:raise ValueError('Use a separate new revision')
    source=ROOT/'outputs/reproduction'/source_revision
    if (ROOT/'outputs/reproduction'/revision).exists():raise ValueError('New revision already exists')
    audit=ROOT/'outputs/diagnostics'/('reproduction_runtime_'+revision)
    audit.mkdir(parents=True,exist_ok=False)
    with JobLock(ROOT/'logs/plant_pipeline.lock'):
        inputs=verified_stage(source,'01_inputs');capsule=inputs['manifest'].parent
        verified_stage(source,'02_semantics');verified_stage(source,'03_cart')
        state=read_json(source/'pipeline_state.json')
        weights=state['stages'].get('04_weights',{})
        if weights.get('status')!='FAILED':raise ValueError('Expected a failed weights stage')
        directory=Path(weights['output'])/'result'
        reference=unpack_reference(reference_archive,sha256,audit/'reference')
        manifest=read_json(inputs['manifest'])
        from approved_reproduction import npz_digest
        for name in ('organs','weights'):
            if npz_digest(reference/(name+'.npz'))!=manifest['references'][name]:
                raise ValueError('Reference file does not match original input bundle: '+name)
        report=dict(source_revision=source_revision,reference_only=True,source_files_modified=False,
            original_runtime=probe(sys.executable),organs=compare_arrays(directory/'organs/organ_instances.npz',reference/'organs.npz'),
            weights=compare_arrays(directory/'weights/contact_weights.npz',reference/'weights.npz'))
        write_json(audit/'runtime_difference_report.json',report)
        for name in ('organs','weights'):
            print('EXACT_DIFFERENCES',name,json.dumps({k:v for k,v in report[name]['fields'].items()if not v['exact']}),flush=True)
    # Isolate newer CPU libraries from the existing CUDA semantic environment.
    environment=ROOT/'envs/approved_weights_20261107'
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not environment.exists():
        version=subprocess.check_output([str(base_python),'-c','import sys;print("%d.%d"%sys.version_info[:2])'],text=True).strip()
        if tuple(map(int,version.split('.'))) < (3,12):
            raise ValueError('Pinned CPU packages require Python 3.12 or newer, got '+version)
        subprocess.run([str(base_python),'-m','venv',str(environment)],check=True)
    if not python.is_file():raise ValueError('Existing pinned environment is incomplete: '+str(environment))
    try:runtime=probe(python)
    except subprocess.CalledProcessError:runtime=None
    if runtime is None or runtime['versions']!=VERSIONS:
        subprocess.run([str(python),'-m','pip','install',*[k+'=='+v for k,v in VERSIONS.items()]],check=True)
    runtime=probe(python)
    if runtime['versions']!=VERSIONS:raise ValueError('Pinned runtime versions do not match')
    fixture_code="""import sys,json,numpy as np
sys.path.insert(0,sys.argv[1])
from organ_geometry import smooth_center_field
with np.load(sys.argv[2],allow_pickle=False)as z:
    result,*_=smooth_center_field(z['projected'],z['faces'])
    if not np.array_equal(result,z['expected_center']):
        raise ValueError('Pinned raster fixture differs; cannot certify old center computation')
print('PINNED_RASTER_FIXTURE_EXACT')
"""
    subprocess.run([str(python),'-c',fixture_code,str(capsule/'frozen/tools/plant_organ_instances'),str(reference/'raster_fixture.npz')],check=True)
    write_json(audit/'pinned_runtime.json',runtime)
    # Match the existing upstream interpreter identity for a traceable new job.
    archive_signature=state['signature']['archive']
    candidates=sorted((ROOT/'incoming').glob('approved_reproduction_*/*.zip'))
    archive=next((p for p in candidates if digest(p)==archive_signature),None)
    if archive is None:raise ValueError('Original verified input ZIP not found under incoming/')
    argv=[sys.executable,'-u',str(ROOT/'maintenance/approved_reproduction.py'),'run',
        '--archive',str(archive),'--sha256',archive_signature,'--revision',revision,
        '--source-revision',source_revision,'--weights-python',str(python),
        '--sim-python',state['signature']['simulation_python']]
    if submit:argv.append('--submit')
    subprocess.run(argv,check=True)
    print('APPROVED_RUNTIME_REPAIR_PREPARED',revision,'audit='+str(audit),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    b=s.add_parser('build');b.add_argument('--output',type=Path,required=True)
    r=s.add_parser('prepare');r.add_argument('--source-revision',type=run_name,required=True)
    r.add_argument('--revision',type=run_name,required=True);r.add_argument('--reference-archive',type=Path,required=True)
    r.add_argument('--sha256',required=True);r.add_argument('--base-python',type=Path,required=True)
    r.add_argument('--submit',action='store_true')
    args=vars(p.parse_args());action=args.pop('action');globals()[action](**args)


if __name__=='__main__':main()
