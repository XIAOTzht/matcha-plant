"""Linux photo -> reconstruction -> metric semantics -> contact weights -> SO-101 path."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import Stages, JobLock, code_signature, command, digest, now, read_json, run_name, verify_images, write_json
from render_runtime import render_environment
from runtime_paths import compatible_temporary_path_signatures, temporary_environment

ROOT = Path(__file__).resolve().parents[2]


def runtime(root):
    return dict(reconstruction=Path(os.environ.get('MATCHA_PYTHON', root/'envs/matcha/bin/python')),
                semantics=Path(os.environ.get('PLANT_CONTACT_PYTHON', root/'envs/plant_semantics/bin/python')),
                planning=Path(os.environ.get('SIM_PYTHON', root/'envs/mujoco/bin/python')))


def preflight(root, cfg, gpu, deep=False):
    py = runtime(root)
    for kind, path in py.items():
        if not path.is_file():
            raise RuntimeError(f'Missing {kind} Python: {path}')
    required = [
        'MAtCha/train.py', 'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
        'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
        'MAtCha/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
        'tools/plant_semantics/models/grounding-dino-tiny/model.safetensors',
        'tools/plant_semantics/models/clipseg-rd64-refined/model.safetensors',
        'tools/plant_semantics/models/mobile_sam_parallel.pt',
        'MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl',
        'MAtCha/scripts/run_sfm.py', 'MAtCha/scripts/align_charts.py',
        'MAtCha/scripts/refine_free_gaussians.py', 'MAtCha/scripts/extract_tetra_mesh.py',
        'tools/plant_semantics/run_v3_server.sh', 'tools/plant_semantics/third_party/MobileSAM/mobile_sam/__init__.py',
        'tools/plant_semantics/refine_organ_semantics.py', 'tools/plant_semantics/organ_shape_rules.py',
        'tools/plant_semantics/organ_surface_consistency.py',
        'tools/plant_extract_with_car/auto_crop_with_car.py',
        'tools/plant_extract_with_car/calibrate_upper_plate.py',
        'tools/plant_semantics_with_car/run_server.sh',
        'tools/plant_semantics_with_car/fix_cart_hardware.py',
        'scripts/plant_contact/run_pipeline.py',
        'simulation/scripts/run_server_job.py', 'simulation/assets/robot/robotstudio_so101/so101.xml',
        cfg['simulation_scene_template'], cfg['simulation_timing_config']]
    if cfg.get('simulation_coordination_config'):
        required.extend([cfg['simulation_coordination_config'], 'simulation/scripts/coordinated_motion.py',
                         'simulation/scripts/render_joint_coordination.py'])
    if cfg.get('simulation_natural_config'):
        required.extend([cfg['simulation_natural_config'],'simulation/scripts/natural_motion.py'])
    if cfg.get('simulation_demonstration_config'):
        required.extend([cfg['simulation_demonstration_config'],'simulation/scripts/demonstration_motion.py'])
    if cfg.get('cart_underside',{}).get('enabled',False):
        thickness=cfg['cart_underside'].get('assumed_chassis_thickness_mm',6.)
        if not isinstance(thickness,(int,float)) or not 1<=thickness<=20:
            raise ValueError('Cart underside thickness must be 1..20 mm')
        required.append('tools/plant_extract_with_car/repair_cart_underside.py')
    for name in required:
        if not (root/name).is_file() or not (root/name).stat().st_size:
            raise RuntimeError(f'Missing server code/model: {root/name}')
    if os.name == 'posix':
        subprocess.run([str(py['reconstruction']), '-u', str(root/'scripts/plant_pipeline/check_ipc_runtime.py')],
                       check=True, env=temporary_environment(root), timeout=120)
    if deep:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PLANTSEG_SERVER='1')
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        subprocess.run([str(py['reconstruction']), '-c',
            'import torch,numpy,plyfile,yaml; import diff_surfel_rasterization,simple_knn._C; assert torch.cuda.is_available(), "CUDA unavailable"; print("RECONSTRUCTION_CUDA_OK",torch.cuda.get_device_name(0))'],
            check=True, env=env, cwd=root/'MAtCha')
        subprocess.run([str(py['semantics']), str(root/'tools/plant_semantics/check_install_server.py')], check=True, env=env)
        subprocess.run([str(py['planning']), str(root/'simulation/scripts/check_server_runtime.py')], check=True,
                       env=render_environment(root,env))
    return py


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=run_name)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--gpu', type=int)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--deep-check', action='store_true')
    ap.add_argument('--check-runtime', action='store_true', help='Check environments/models before uploading any photos')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve(); cfg_path = args.config or root/'configs/plant_pipeline.json'
    if any(c.isspace() for c in str(root)):
        raise ValueError('MAtCha subprocesses require a project path without whitespace')
    cfg = read_json(cfg_path); gpu = cfg['gpu_id'] if args.gpu is None else args.gpu
    if not isinstance(gpu, int) or gpu < 0:
        raise ValueError('GPU index must be a nonnegative integer')
    if cfg['plate_length_mm'] <= 0 or cfg['plate_width_mm'] <= 0 or not 0 < cfg['tetra_downsample_ratio'] <= 1:
        raise ValueError('Invalid dimensions or sampling ratio')
    py = preflight(root, cfg, gpu, args.deep_check)
    if args.check_runtime:
        print('PIPELINE_RUNTIME_OK', flush=True)
        return
    if not args.run:
        ap.error('--run is required except with --check-runtime')
    data = root/'data/pipeline'/args.run; manifest = verify_images(data)
    print(f'PIPELINE_PREFLIGHT_OK images={len(manifest["images"])} gpu={gpu}', flush=True)
    if args.check_only:
        return
    if os.name != 'posix':
        raise RuntimeError('Reconstruction and segmentation entry is Linux-only; deploy and run on the server')
    out = root/'outputs/pipeline'/args.run
    if out.exists() and not args.resume:
        raise ValueError('Run exists; use --resume to verify/reuse completed stages or choose a new run')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ROOT_OVERRIDE=str(root),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PLANTSEG_SERVER='1',
               MPLCONFIGDIR=str(root/'cache/plant_pipeline/matplotlib'),
               PATH=str(py['reconstruction'].parent)+os.pathsep+os.environ['PATH'])
    env.update(XDG_CACHE_HOME=str(root/'cache/plant_pipeline'), HF_HOME=str(root/'cache/huggingface'),
               TORCH_HOME=str(root/'cache/torch'), PLANTSEM_ENV=str(py['semantics'].parent.parent))
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)  # matcha torch does not support expandable_segments
    env = temporary_environment(root, env)
    print('PIPELINE_TEMP', env['TMPDIR'], flush=True)
    for key in ('MPLCONFIGDIR', 'TMPDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    with JobLock(root/'logs/plant_pipeline.lock'):
        dependent_configs = ['tools/contact_weights/contact_policy.json', 'tools/contact_weights/organ_weight_policy.json',
                             cfg['simulation_scene_template'], cfg['simulation_timing_config']]
        if cfg.get('simulation_coordination_config'):
            dependent_configs.append(cfg['simulation_coordination_config'])
        if cfg.get('simulation_natural_config'):dependent_configs.append(cfg['simulation_natural_config'])
        if cfg.get('simulation_demonstration_config'):dependent_configs.append(cfg['simulation_demonstration_config'])
        stages = Stages(out, dict(input_sha256=digest(data/'input_manifest.json'), config_sha256=digest(cfg_path), gpu=gpu,
                                 policies={p:digest(root/p) for p in dependent_configs}, code_sha256=code_signature(root)),
                        compatible_code_signatures=compatible_temporary_path_signatures(root))
        write_json(out/'config_snapshot.json', cfg)
        write_json(out/'runtime.json', {k: str(v) for k,v in py.items()})
        (out/'PHOTO_TO_PATH_SUCCESS').unlink(missing_ok=True)
        try:
            # Separate reconstruction stages allow mesh extraction to retry without retraining.
            def sfm(d):
                command([py['reconstruction'], 'scripts/run_sfm.py', '--source_path', data,
                         '--output_path', d/'result', '--config', 'unposed', '--n_images', len(manifest['images'])], d, root/'MAtCha', env)
                result = dict(cameras=d/'result/cameras.json')
                for category in ('images', 'sparse', 'pointmaps'):
                    paths = sorted(p for p in (d/'result'/category).rglob('*') if p.is_file())
                    if not paths:
                        raise ValueError(f'SfM produced no {category} files')
                    result.update({f'{category}_{i}':p for i,p in enumerate(paths)})
                return result
            sf = stages.execute('01_sfm', sfm); scene = sf['cameras'].parent
            def align(d):
                # Alignment writes charts into its source scene; copy the compact
                # SfM tree so failed retries cannot corrupt a completed stage.
                target = d/'scene'; shutil.copytree(scene, target)
                command([py['reconstruction'], 'scripts/align_charts.py', '--source_path', target,
                         '--mast3r_scene', target, '--output_path', target, '--config', 'default',
                         '--depth_model', 'depthanythingv2'], d, root/'MAtCha', env)
                return dict(charts=target/'charts_data.npz', cameras=target/'cameras.json')
            al = stages.execute('02_alignment', align); aligned = al['charts'].parent
            def train(d):
                command([py['reconstruction'], 'scripts/refine_free_gaussians.py', '--mast3r_scene', aligned,
                         '--output_path', d/'model', '--config', 'default'], d, root/'MAtCha', env)
                files = sorted((d/'model/point_cloud').glob('iteration_*/point_cloud.ply'),
                               key=lambda p: int(p.parent.name.split('_')[-1]))
                if not files:
                    raise ValueError('Training produced no Gaussian checkpoint')
                return dict(checkpoint=files[-1], config=d/'model/cfg_args')
            tr = stages.execute('03_gaussians', train); model = tr['config'].parent
            def mesh(d):
                command([py['reconstruction'], 'scripts/extract_tetra_mesh.py', '--mast3r_scene', aligned,
                         '--model_path', model, '--output_path', d/'mesh', '--config', 'default',
                         '--downsample_ratio', cfg['tetra_downsample_ratio'], '--interpolate_views'], d, root/'MAtCha', env)
                p = d/'mesh/tetra_mesh_binary_search_7.ply'
                command([py['semantics'], '-c',
                    'from plyfile import PlyData;import sys; p=PlyData.read(sys.argv[1]);assert len(p["vertex"]) and len(p["face"]);print("MESH_VERIFIED")', p], d/'verify', root, env)
                return dict(mesh=p)
            me = stages.execute('04_mesh', mesh)
            def crop(d):
                command([py['semantics'], root/'tools/plant_extract_with_car/auto_crop_with_car.py',
                         '--input', me['mesh'], '--output-dir', d/'result'], d, root, env)
                return dict(mesh=d/'result/plant_car.ply', report=d/'result/report.json')
            cr = stages.execute('05_crop', crop)
            def scale(d):
                command([py['semantics'], root/'tools/plant_extract_with_car/calibrate_upper_plate.py',
                         '--input', cr['mesh'], '--crop-report', cr['report'], '--output-dir', d/'result',
                         '--length-mm', cfg['plate_length_mm'], '--width-mm', cfg['plate_width_mm']], d, root, env)
                report = read_json(d/'result/scale_report.json')
                if report['scale_disagreement_fraction'] > cfg['max_scale_disagreement']:
                    raise ValueError('Reference plate scale check failed')
                return dict(mesh=d/'result/plant_car_mm.ply', report=d/'result/scale_report.json')
            sc = stages.execute('06_metric', scale)
            def semantics(d):
                command(['bash', root/'tools/plant_semantics_with_car/run_server.sh', cr['mesh'], sc['mesh'],
                         sc['report'], d/'result', gpu, cfg['extra_semantic_views']], d, root, env)
                return dict(mesh=d/'result/complete/semantic_mesh_mm.ply', labels=d/'result/complete/labels.npz',
                            marker=d/'result/complete/WITH_CAR_SEMANTICS_SUCCESS')
            se = stages.execute('07_semantics', semantics)
            def hardware(d):
                command([py['semantics'], root/'tools/plant_semantics_with_car/fix_cart_hardware.py',
                         '--source', se['mesh'].parent, '--output', d/'result',
                         '--scale-report', sc['report'], '--metric', sc['mesh']], d, root, env)
                if read_json(d/'result/validation.json')['status'] != 'PASS':
                    raise ValueError('Semantic hardware validation failed')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz',
                            coordinates=d/'result/coordinate_system.json', validation=d/'result/validation.json')
            hw = stages.execute('08_cart_hardware', hardware)
            # New photo runs review whole leaves/flower heads independently of
            # old labels. Existing completed runs use maintenance/refine_plant_semantics.py.
            def organ_review(d):
                command([py['semantics'],root/'tools/plant_semantics/refine_organ_semantics.py',
                         '--source',hw['mesh'].parent,'--output',d/'result','--device','cuda'],d,root,env)
                if read_json(d/'result/validation.json')['status'] != 'PASS':
                    raise ValueError('Organ review export validation failed')
                return dict(mesh=d/'result/semantic_mesh_mm.ply', labels=d/'result/labels.npz',
                            validation=d/'result/validation.json', report=d/'result/report.json',
                            comparison=d/'result/semantic_surface_comparison.jpg')
            reviewed = stages.execute('08b_organ_review',organ_review)
            scene_semantics = reviewed
            underside = cfg.get('cart_underside', {})
            if underside.get('enabled', False):
                def repair_underside(d):
                    command([py['semantics'],root/'tools/plant_extract_with_car/repair_cart_underside.py',
                        '--source',reviewed['mesh'].parent,'--output',d/'result',
                        '--chassis-thickness-mm',underside.get('assumed_chassis_thickness_mm',6.)],d,root,env)
                    if read_json(d/'result/validation.json')['status'] != 'PASS':
                        raise ValueError('Cart underside export validation failed')
                    return dict(mesh=d/'result/semantic_mesh_mm.ply',labels=d/'result/labels.npz',
                        report=d/'result/report.json',template=d/'result/cart_template.json',
                        validation=d/'result/validation.json',marker=d/'result/CART_UNDERSIDE_REPAIR_SUCCESS')
                scene_semantics = stages.execute('08c_cart_underside',repair_underside)
            def weights(d):
                command([py['semantics'], root/'scripts/plant_contact/run_pipeline.py',
                         '--input-dir', scene_semantics['mesh'].parent, '--output-dir', d/'result', '--no-parts'], d, root, env)
                return dict(weights=d/'result/weights/contact_weights.npz', organs=d/'result/organs/organ_instances.npz',
                            marker=d/'result/PLANT_CONTACT_EXPORT_SUCCESS')
            we = stages.execute('09_weights', weights)
            def planning(d):
                sim_env = render_environment(root,env)
                command([py['planning'], root/'simulation/scripts/run_server_job.py', '--run', args.run,
                         '--semantic-dir', scene_semantics['mesh'].parent, '--contact-dir', we['weights'].parent.parent,
                         '--scale-report', sc['report'], '--settings', cfg_path, '--resume'], d, root, sim_env)
                job = root/'simulation/outputs/jobs'/args.run
                report = read_json(job/'job_report.json')
                if report['status'] != 'SIMULATION_JOB_SUCCESS':
                    raise ValueError('Planner/trajectory validation did not pass')
                return dict(report=job/'job_report.json', marker=job/'SIMULATION_JOB_SUCCESS')
            pl = stages.execute('10_planning', planning)
            result = dict(status='PHOTO_TO_PATH_SUCCESS', run=args.run, finished_utc=now(),
                          input=str(data), semantic_mesh=str(scene_semantics['mesh']), contact_weights=str(we['weights']),
                          cart_underside_report=str(scene_semantics['report']) if underside.get('enabled',False) else None,
                          semantic_comparison=str(reviewed['comparison']), semantic_review_report=str(reviewed['report']),
                          raw_mesh=str(me['mesh']), crop_mesh=str(cr['mesh']), metric_mesh=str(sc['mesh']),
                          scale_report=str(sc['report']),
                          simulation_report=str(pl['report']), simulation=read_json(pl['report']),
                          real_robot_ready=False,
                          review_required=['scale selection', 'semantic/organ labels', 'inferred cart underside when enabled', 'virtual robot layout and nozzle dimensions'])
            write_json(out/'result_paths.json', result)
            stages.state.update(status='PHOTO_TO_PATH_SUCCESS', finished_utc=now()); write_json(stages.path, stages.state)
            (out/'PHOTO_TO_PATH_SUCCESS').write_text(now()+'\n', encoding='ascii')
            (out/'PIPELINE_FAILED.json').unlink(missing_ok=True)
            write_json(root/'outputs/pipeline/latest_success.json', dict(run=args.run, report=str(out/'result_paths.json')))
            print(f'PHOTO_TO_PATH_SUCCESS {out}', flush=True)
        except BaseException as exc:
            write_json(out/'PIPELINE_FAILED.json', dict(error=str(exc), stage=stages.state.get('current_stage'), time=now()))
            raise


if __name__ == '__main__':
    main()
