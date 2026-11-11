#!/usr/bin/env python3
"""Run existing MAtCha MASt3R with strictly fixed full camera poses.

Runtime-only adaptation: no repository files are modified. Depth/scale optimization
stays active. Official post-hoc camera alignment is disabled because all dense
points must use the same poses as the exported cameras throughout optimization.
"""
import ast
import inspect
from pathlib import Path
import sys
import numpy as np


def patch_optimizer_source(source):
    marker='        depthmaps = []\n'
    if source.count(marker)!=1:
        raise RuntimeError('Unsupported sparse_scene_optimizer layout; no changes made')
    source=source.replace(marker,'        # Exact input poses, also used to generate all 3D points.\n'
                          '        cam2w = TURN_FIXED_POSES.to(device=device, dtype=dtype)\n'+marker,1)
    ast.parse(source)
    return source


def validate_scene_layout(scene):
    scene=Path(scene)
    root_images=[p.name for p in scene.iterdir() if p.is_file()
                 and p.suffix.lower() in ('.jpg','.jpeg','.png')]
    if root_images:
        raise ValueError('Images in the calibrated dataset root make MAtCha skip calibration loading: '
                         + ', '.join(root_images) + '. Move diagnostics into a subdirectory.')
    sparse=scene/'sparse'/'0'
    for name in ('cameras.bin','images.bin'):
        if not (sparse/name).is_file():
            raise FileNotFoundError(sparse/name)
    names=sorted(p.name for p in (scene/'images').iterdir() if p.is_file()
                 and p.suffix.lower() in ('.jpg','.jpeg','.png'))
    expected=[f'img_{i:03d}.jpg' for i in range(12)]
    if names!=expected:
        raise ValueError(f'Expected exactly 12 turntable photos in images/: {names}')


def main():
    args=sys.argv[1:]
    if '--matcha-dir' not in args:
        raise ValueError('--matcha-dir is required')
    i=args.index('--matcha-dir'); root=Path(args[i+1]).resolve(); del args[i:i+2]
    required=['--use_calibrated_poses','--fix_rotation','--fix_translation','--fix_focal','--fix_principal_point']
    for flag in required:
        if flag not in args: raise ValueError(f'Missing required flag {flag}')
    if '--align_camera_locations' in args:
        raise ValueError('Post-hoc alignment forbidden for strict fixed-pose experiment')
    if '--output_dir' not in args: raise ValueError('--output_dir required')
    output=Path(args[args.index('--output_dir')+1])
    if '--scene_path' not in args: raise ValueError('--scene_path required')
    validate_scene_layout(args[args.index('--scene_path')+1])
    import torch
    sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'mast3r'))
    import mast3r.utils.path_to_dust3r
    import mast3r.cloud_opt.sparse_ga as ga
    original=ga.sparse_scene_optimizer
    namespace=ga.__dict__
    source=patch_optimizer_source(inspect.getsource(original))
    exec(compile(source,str(root/'mast3r/mast3r/cloud_opt/sparse_ga.py')+'[fixed poses]','exec'),namespace)
    adapted=namespace['sparse_scene_optimizer']

    def fixed_optimizer(*pos,**kw):
        names=pos[0] if pos else kw['imgs']
        init=kw.get('init',{})
        poses=torch.stack([init[name]['cam2w'].detach().clone() for name in names])
        if not torch.isfinite(poses).all(): raise ValueError('Nonfinite input camera pose')
        ga.TURN_FIXED_POSES=poses
        kw.update(opt_quat=False,opt_tran=False,opt_pp=False,opt_focal=False)
        # log_sizes controls per-view depth scaling. It remains trainable; it can
        # no longer move cameras because make_K_cam_depth uses TURN_FIXED_POSES.
        result=adapted(*pos,**kw)
        for stage in result[1:]:
            if stage is not None and not torch.allclose(stage['cam2w'],poses,atol=1e-6):
                raise RuntimeError('Camera poses changed during optimization')
        return result

    ga.sparse_scene_optimizer=fixed_optimizer
    script=root/'mast3r'/'run_mast3r.py'
    sys.argv=[str(script),*args]
    ns={'__name__':'__main__','__file__':str(script),'__package__':None}
    exec(compile(script.read_text(encoding='utf-8'),str(script),'exec'),ns)
    import json
    cams=json.loads((output/'cameras.json').read_text())
    expected=ga.TURN_FIXED_POSES.detach().cpu().numpy()
    if not np.allclose(np.array(cams['cams2world']),expected,atol=1e-5):
        raise RuntimeError('Exported camera poses do not match exact input poses')
    if not (output/'points.ply').is_file() or not list((output/'pointmaps').glob('*.json')):
        raise RuntimeError('Missing SfM points/pointmaps')
    (output/'FIXED_POSES_VERIFIED.txt').write_text('Fixed 30-degree camera poses verified during optimization and after export.\n')
    print('TURNTABLE30_SFM_FIXED_POSES_VERIFIED',flush=True)


if __name__=='__main__':
    main()
