"""Import this run's metric plant, search all arm joints and validate a timed route."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'scripts/plant_pipeline'))
from common import Stages, code_signature, command, digest, now, read_json, run_name, write_json
from render_runtime import render_environment


def copy_scene(source, destination):
    """Copy mutable targets and rebase external robot meshes at any directory depth."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    shutil.copytree(source, destination)
    tree = ET.parse(destination/'scene.xml')
    for mesh in tree.getroot().findall('asset/mesh'):
        name = mesh.get('file')
        if not name:continue  # Procedural nozzle meshes are embedded in XML.
        original = (source/name).resolve()
        try:
            new = destination/original.relative_to(source)
        except ValueError:
            new = original
        mesh.set('file', os.path.relpath(new, destination).replace(os.sep, '/'))
    tree.write(destination/'scene.xml', encoding='utf-8', xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=run_name)
    ap.add_argument('--semantic-dir', type=Path, required=True)
    ap.add_argument('--contact-dir', type=Path, required=True)
    ap.add_argument('--scale-report', type=Path, required=True)
    ap.add_argument('--settings', type=Path, default=PROJECT/'configs/plant_pipeline.json')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prepare-only', action='store_true', help='Validate inputs and build scene; no path success marker')
    args = ap.parse_args()
    semantic, contact = args.semantic_dir.resolve(), args.contact_dir.resolve()
    settings = read_json(args.settings)
    template = (PROJECT/settings['simulation_scene_template']).resolve()
    timing = (PROJECT/settings['simulation_timing_config']).resolve()
    coordination = (PROJECT/settings['simulation_coordination_config']).resolve() if settings.get('simulation_coordination_config') else None
    natural = (PROJECT/settings['simulation_natural_config']).resolve() if settings.get('simulation_natural_config') else None
    demonstration = (PROJECT/settings['simulation_demonstration_config']).resolve() if settings.get('simulation_demonstration_config') else None
    paths = dict(semantic_mesh=semantic/'semantic_mesh_mm.ply', semantic_labels=semantic/'labels.npz',
                 coordinate_system=semantic/'coordinate_system.json', weights=contact/'weights/contact_weights.npz',
                 organs=contact/'organs/organ_instances.npz', policy=PROJECT/'tools/contact_weights/contact_policy.json',
                 palette=PROJECT/'tools/contact_weights/weight_palette.py')
    if (semantic/'virtual_substrate.json').is_file():
        paths['virtual_substrate'] = semantic/'virtual_substrate.json'
    for path in [*paths.values(), args.scale_report, template, timing]:
        if not path.is_file():
            raise ValueError(f'Missing input: {path}')
    if coordination is not None and not coordination.is_file():
        raise ValueError(f'Missing coordination settings: {coordination}')
    if natural is not None and not natural.is_file():raise ValueError(f'Missing natural settings: {natural}')
    if demonstration is not None and not demonstration.is_file():raise ValueError(f'Missing demonstration settings: {demonstration}')
    out = ROOT/'outputs/jobs'/args.run
    if out.exists() and not args.resume:
        raise ValueError('Run exists; use --resume or a new --run')
    signature = {k: digest(v) for k,v in paths.items()}
    signature.update(settings=digest(args.settings), template=digest(template), timing=digest(timing),
                     scale=digest(args.scale_report), version='photo-to-path-1', code_sha256=code_signature(PROJECT))
    if coordination is not None:
        signature['coordination'] = digest(coordination)
    if natural is not None:signature['natural']=digest(natural)
    if demonstration is not None:signature['demonstration']=digest(demonstration)
    stages = Stages(out, signature)
    (out/'SIMULATION_JOB_SUCCESS').unlink(missing_ok=True)
    env = render_environment(PROJECT,dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    def call(script, argv, directory):
        command([sys.executable, ROOT/'scripts'/script, *argv], directory, ROOT, env)
    try:
        def inputs(d):
            config = read_json(template)
            config['source_root'] = '..'
            config.update({k: p.relative_to(PROJECT).as_posix() for k,p in paths.items()})
            sys.path.insert(0, str(PROJECT/'tools/contact_weights'))
            from weight_palette import palette_description
            write_json(d/'palette.json', dict(palette=palette_description()))
            config['palette'] = (d/'palette.json').relative_to(PROJECT).as_posix()
            config['watering_target_override'] = dict(approach_height_range_mm=[5, 15], default_height_mm=10)
            config['target_candidates']['height_above_soil_mm'] = 10
            write_json(d/'scene_config.json', config)
            write_json(d/'scale_report.json', read_json(args.scale_report))
            # Full geometry/order/unit/policy checks happen before display generation.
            from plant_scene import load_inputs
            *_, report = load_inputs(ROOT, config)
            write_json(d/'input_validation.json', report)
            return dict(config=d/'scene_config.json', validation=d/'input_validation.json', palette=d/'palette.json')
        imported = stages.execute('01_inputs', inputs)
        def scene(d):
            call('build_watering_scene.py', ['--config', imported['config'], '--output-dir', d/'scene'], d)
            artifacts = dict(scene=d/'scene/scene.xml', report=d/'scene/scene_report.json',
                             config=d/'scene/config_snapshot.json', targets=d/'scene/watering_candidates.json',
                             geometry=d/'scene/full_resolution_world_geometry.npz')
            artifacts.update({f'display_{i}':p for i,p in enumerate(sorted((d/'scene/display_meshes').glob('*.stl')))})
            return artifacts
        built = stages.execute('02_scene', scene)
        if args.prepare_only:
            print('SCENE_PREPARED_ONLY', built['scene'], flush=True)
            return
        budget = settings['planner']
        def poses(d):
            call('explore_watering_poses.py', ['--scene', built['scene'], '--output-dir', d/'result', '--seeds', budget['pose_seeds']], d)
            result = d/'result/pose_candidates.json'
            if not read_json(result)['entries']:
                raise ValueError('No feasible watering pose; inspect scene layout and segmentation')
            return dict(poses=result)
        po = stages.execute('03_poses', poses)
        # Neighboring real soil faces can permit a much gentler final posture.
        # Refine a copy: the completed scene's candidates must stay immutable.
        if budget.get('pose_refine_rounds', 0) > 0:
            def refine_goals(d):
                target = d/'scene'
                copy_scene(built['scene'].parent, target)
                call('explore_watering_poses.py', ['--scene', target/'scene.xml', '--output-dir', d/'result',
                     '--refine-from', po['poses'], '--refine-rounds', budget['pose_refine_rounds']], d)
                return dict(poses=d/'result/refined_pose_candidates.json', scene=target/'scene.xml',
                            report=target/'scene_report.json', targets=target/'watering_candidates.json',
                            geometry=target/'full_resolution_world_geometry.npz')
            refined_goals = stages.execute('03b_refined_goals', refine_goals)
            po = dict(poses=refined_goals['poses'])
            built = {k:refined_goals[k] for k in ('scene','report','geometry')}
        def corridors(d):
            call('build_weighted_corridors.py', ['--poses', po['poses'], '--output-dir', d/'result',
                 '--postures', budget['postures'], '--corridors', budget['corridors'], '--config', timing], d)
            result = d/'result/goal_screening_report.json'
            if not read_json(result)['entries']:
                raise ValueError('No accepted soil target/posture; inspect goal screening')
            return dict(screening=result, scores=d/'result/candidate_scores.json', seeds=d/'result/candidate_paths.npz')
        co = stages.execute('04_corridors', corridors)
        def plan(d):
            call('plan_watering_path.py', ['--screening', co['screening'], '--resume-candidates', co['screening'].parent,
                 '--config', timing, '--output-dir', d/'result', '--iterations', budget['iterations'],
                 '--seconds', budget['seconds_per_goal'], '--optimize-goals', budget['optimize_goals']], d)
            if read_json(d/'result/path_report.json')['status'] != 'PATH_PLANNING_COMPLETE':
                raise ValueError('No feasible weighted path found within search budget')
            return dict(report=d/'result/path_report.json', trajectory=d/'result/trajectory_q.npz')
        pl = stages.execute('05_path', plan)
        def refine(d):
            call('attitude_escape_search.py', ['--path', pl['report'].parent, '--config', timing,
                 '--output-dir', d/'result', '--seconds', budget['refinement_seconds']], d)
            if read_json(d/'result/path_report.json')['status'] != 'PATH_PLANNING_COMPLETE':
                raise ValueError('Attitude refinement failed')
            return dict(report=d/'result/path_report.json', trajectory=d/'result/trajectory_q.npz', line=d/'result/nozzle_path_line.csv')
        re = stages.execute('06_attitude', refine)
        if coordination is not None:
            refined_path = re['report'].parent
            def coordinate(d):
                call('coordinated_motion.py', ['--path', refined_path, '--config', timing,
                     '--coordination-config', coordination, '--output-dir', d/'result'], d)
                if read_json(d/'result/path_report.json')['status'] != 'PATH_PLANNING_COMPLETE':
                    raise ValueError('Whole-arm coordination search failed')
                return dict(report=d/'result/path_report.json', trajectory=d/'result/trajectory_q.npz',
                            line=d/'result/nozzle_path_line.csv', coordination=d/'result/coordination_report.json',
                            joint_plot=d/'result/joint_coordination.png')
            re = stages.execute('06b_coordination', coordinate)
        if natural is not None:
            natural_source=re['report'].parent
            def natural_refine(d):
                call('natural_motion.py',['--path',natural_source,'--config',timing,
                    '--coordination-config',natural,'--output-dir',d/'result'],d)
                selection=read_json(d/'result/natural_selection.json')
                chosen=Path(selection['path'])
                artifacts=dict(report=chosen/'path_report.json',trajectory=chosen/'trajectory_q.npz',
                    line=chosen/'nozzle_path_line.csv',selection=d/'result/natural_selection.json',
                    natural_search=d/'result/natural_search_report.json')
                if (chosen/'joint_coordination.png').is_file():artifacts['joint_plot']=chosen/'joint_coordination.png'
                if (chosen/'joint_velocity_comparison.png').is_file():artifacts['velocity_plot']=chosen/'joint_velocity_comparison.png'
                if (chosen/'timing_plan.json').is_file():artifacts['timing_plan']=chosen/'timing_plan.json'
                return artifacts
            re=stages.execute('06c_natural',natural_refine)
        if demonstration is not None:
            demonstration_source=re['report'].parent
            prior_artifacts=dict(re)
            def demonstrate(d):
                call('demonstration_motion.py',['--path',demonstration_source,'--config',timing,
                    '--settings',demonstration,'--output-dir',d/'result','--no-promote'],d)
                selection=read_json(d/'result/demonstration_selection.json');chosen=Path(selection['path'])
                artifacts=dict(prior_artifacts,report=chosen/'path_report.json',trajectory=chosen/'trajectory_q.npz',
                    line=chosen/'nozzle_path_line.csv',demonstration_selection=d/'result/demonstration_selection.json',
                    demonstration_search=d/'result/demonstration_search_report.json')
                for key,name in (('joint_plot','joint_coordination.png'),('velocity_plot','joint_velocity_comparison.png'),
                                 ('timing_plan','timing_plan.json')):
                    if (chosen/name).is_file():artifacts[key]=chosen/name
                if (chosen/'demonstration_verification.json').is_file():
                    artifacts['demonstration_verification']=chosen/'demonstration_verification.json'
                return artifacts
            re=stages.execute('06d_demonstration',demonstrate)
        def validate(d):
            call('validate_watering_trajectory.py', ['--path', re['report'].parent, '--scene', built['scene'],
                 '--config', timing, '--output-dir', d/'result'], d)
            report = d/'result/validation_report.json'
            if read_json(report)['status'] != 'SAMPLED_REPLAY_CHECKS_PASSED':
                raise ValueError('Dynamic collision/tracking checks failed; path is not approved')
            return dict(report=report, trajectory=d/'result/timed_trajectory.npz')
        va = stages.execute('07_dynamics', validate)
        # Renderer writes into the validation directory; it does not alter trajectory/report.
        def render(d):
            call('render_trajectory_replay.py', ['--replay', va['report'].parent, '--fps', 15], d)
            return {k: va['report'].parent/v for k,v in dict(animation='trajectory_replay.gif',
                    preview='trajectory_preview.png', line='nozzle_path_line.png', closeup='nozzle_soil_closeup.png').items()}
        rendered = stages.execute('08_render', render) if settings.get('render', True) else {}
        if 'joint_plot'in re:
            rendered['joint_coordination'] = re['joint_plot']
        if 'velocity_plot'in re:rendered['joint_velocity']=re['velocity_plot']
        report = dict(status='SIMULATION_JOB_SUCCESS', run=args.run, finished_utc=now(), real_robot_ready=False,
                      global_optimum_certified=False, scene=str(built['scene']), scene_report=str(built['report']), path_report=str(re['report']),
                      line_csv=str(re['line']), timed_trajectory=str(va['trajectory']), validation_report=str(va['report']),
                      media={k:str(v) for k,v in rendered.items()}, validation=read_json(va['report']),
                      mesh_fingerprint=read_json(built['report'])['mesh_fingerprint'],
                      limitation='Sampled collision/dynamics checks; no continuous swept-volume or physical damage guarantee.')
        if natural is not None:
            report.update(natural_selection=str(re['selection']),natural_search_report=str(re['natural_search']))
        if demonstration is not None:
            report.update(demonstration_selection=str(re['demonstration_selection']),
                          demonstration_search_report=str(re['demonstration_search']))
        scene_report = read_json(built['report'])
        if coordination is not None:
            from motion_review import review_motion
            report['motion_style_review'] = review_motion(read_json(re['report']))
            if not report['motion_style_review']['accepted']:
                print('MOTION_STYLE_NEEDS_REVIEW: numerical replay passed; accepted elbow/wrist style was not reproduced', flush=True)
        report['watering_surface_source'] = scene_report.get('watering_surface_source', 'semantic_mesh_substrate')
        if 'virtual_substrate' in scene_report:
            report.update(virtual_substrate=scene_report['virtual_substrate'], measured_soil_available=False,
                          limitation=report['limitation']+' Watering target is an explicitly assumed pot-interior surface, not photographed soil.')
        write_json(out/'job_report.json', report)
        stages.state.update(status='SIMULATION_JOB_SUCCESS', finished_utc=now())
        write_json(stages.path, stages.state)
        (out/'SIMULATION_JOB_SUCCESS').write_text(now()+'\n', encoding='ascii')
        (out/'JOB_FAILED.json').unlink(missing_ok=True)
        print('SIMULATION_JOB_SUCCESS', out, flush=True)
    except BaseException as exc:
        write_json(out/'JOB_FAILED.json', dict(error=str(exc), time=now(), stage=stages.state.get('current_stage')))
        raise


if __name__ == '__main__':
    main()
