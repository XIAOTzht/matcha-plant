"""Probe real headless rendering in fresh processes; save only a passing backend."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'scripts/plant_pipeline'))
from common import now, write_json
from render_runtime import CONFIG, VARIABLES


def candidates(root, base):
    """Try GPU EGL, known NVIDIA vendor, then CPU Mesa, without changing drivers."""
    root=Path(root)
    preferred=base.get('MUJOCO_GL','egl')
    seen=set()
    def add(name,values):
        key=tuple(sorted(values.items()))
        if key in seen:return None
        seen.add(key)
        return name,values
    if preferred in {'egl','osmesa'}:
        values={k:v for k,v in base.items() if k in VARIABLES}
        values.update(MUJOCO_GL=preferred,PYOPENGL_PLATFORM=preferred)
        if preferred=='osmesa':
            values.pop('MUJOCO_EGL_DEVICE_ID',None);values.pop('__EGL_VENDOR_LIBRARY_FILENAMES',None)
        row=add('requested_'+preferred,values)
        if row:yield row
    for vendor in (None, Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
                   Path('/etc/glvnd/egl_vendor.d/10_nvidia.json')):
        if vendor is not None and not vendor.is_file():continue
        values=dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        if base.get('LD_LIBRARY_PATH'):values['LD_LIBRARY_PATH']=base['LD_LIBRARY_PATH']
        if vendor is not None:values['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
        row=add('egl_all_devices' if vendor is None else 'egl_nvidia_vendor',values)
        if row:yield row
    values=dict(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1')
    if base.get('LD_LIBRARY_PATH'):values['LD_LIBRARY_PATH']=base['LD_LIBRARY_PATH']
    row=add('osmesa_system',values)
    if row:yield row
    for directory in (root/'envs/mujoco_render_libs/lib',root/'envs/mujoco/lib'):
        if not any(directory.glob('libOSMesa.so*')):continue
        values=dict(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1',
                    LD_LIBRARY_PATH=str(directory)+(os.pathsep+base['LD_LIBRARY_PATH'] if base.get('LD_LIBRARY_PATH') else ''))
        row=add('osmesa_project_libraries',values)
        if row:yield row


def configure(root, runner=subprocess.run, timeout=300):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Renderer probe timeout must be a positive finite number')
    root=Path(root).resolve()
    logs=root/'logs/render_setup';logs.mkdir(parents=True,exist_ok=True)
    base=dict(os.environ)
    failures=[]
    for i,(name,values) in enumerate(candidates(root,base),1):
        env={k:v for k,v in base.items() if k not in VARIABLES};env.update(values)
        path=logs/f'{i:02d}_{name}.log'
        print('RENDER_PROBE',name,f'timeout={timeout:g}s','log='+str(path),flush=True)
        argv=[sys.executable,'-u',str(root/'simulation/scripts/check_server_runtime.py'),
              '--render-only','--traceback-after',str(min(60,timeout))]
        started=time.monotonic()
        # Direct-to-file output survives timeouts and is readable via tail while
        # the child is still running. Never discard a slow import's diagnostics.
        with path.open('w',encoding='utf-8') as stream:
            stream.write(f'RENDER_PROBE_START {name} timeout={timeout:g}s\n');stream.flush()
            try:
                result=runner(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',timeout=timeout)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=-1
                stream.write(f'\nRENDER_PROBE_TIMEOUT after {timeout:g}s; partial output retained above.\n')
            stream.write(f'\nRENDER_PROBE_EXIT code={code} elapsed={time.monotonic()-started:.2f}s\n')
        output=path.read_text(encoding='utf-8',errors='replace')
        if code==0 and 'OFFSCREEN_RENDER_OK' in output and 'RENDER_PROBE_OK' in output:
            write_json(root/CONFIG,dict(status='OFFSCREEN_RENDER_VERIFIED',backend=values['MUJOCO_GL'],
                                      checked_utc=now(),environment=values,probe_log=str(path)))
            print(output,end='',flush=True)
            print('SERVER_RENDER_CONFIG_SAVED',root/CONFIG,flush=True)
            return True
        failures.append(dict(candidate=name,exit_code=code,log=str(path)))
        print('RENDER_PROBE_FAILED',name,'log='+str(path),flush=True)
        print('\n'.join(output.splitlines()[-20:]),flush=True)
    write_json(logs/'failed_probes.json',dict(status='NO_WORKING_RENDERER',attempts=failures))
    print('NO_WORKING_RENDERER: inspect the stage output and traceback in the probe logs.',flush=True)
    return False


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=PROJECT)
    ap.add_argument('--probe-timeout',type=float,default=float(os.environ.get('RENDER_PROBE_TIMEOUT','300')),
                    help='Seconds per backend (default: 300; env: RENDER_PROBE_TIMEOUT)')
    args=ap.parse_args()
    if os.name!='posix':ap.error('Server headless renderer setup is Linux-only')
    if not math.isfinite(args.probe_timeout) or args.probe_timeout <= 0:
        ap.error('--probe-timeout must be a positive finite number')
    raise SystemExit(0 if configure(args.root,timeout=args.probe_timeout) else 1)
