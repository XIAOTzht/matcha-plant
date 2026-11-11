"""Optional visibly coordinated route, within the verified baseline risk ceiling.

This is an explicit presentation preference, never a claim of lower optimal
cost. Broad early excursions end before the final soil approach. Travel and
reversal caps prevent rewarding loops; dynamics remain independently checked.
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from plant_scene import read_json,write_json
from trajectory_tools import make_timing_plan,time_parameterize
from plan_watering_path import CollisionChecker,timed_path_score
from natural_motion import (validate_natural_settings,natural_metrics,motion_passes,
                           trajectory_from_replay,actuator_preview)
from attitude_escape_search import export_path


def validate_presentation_settings(s):
    for name in ('route_departure_mm','added_lift_mm','elbow_span_deg','roll_span_deg'):
        low,high=s['min_'+name],s['desired_'+name]
        if not np.isfinite([low,high]).all()or not 0<low<=high:raise ValueError('Invalid presentation target: '+name)
    for name in ('max_tip_length_ratio','max_joint_travel_ratio'):
        if not np.isfinite(s[name])or s[name]<1:raise ValueError('Invalid presentation travel cap')


def tip_trace(checker,path):
    sid=checker.model.site('nozzle_tip').id
    result=[]
    for q in path:
        checker.set_q(q);result.append(checker.data.site_xpos[sid].copy())
    return np.asarray(result)


def route_metrics(tips,joints,baseline_tips,baseline_joints):
    tips=np.asarray(tips);baseline_tips=np.asarray(baseline_tips)
    arc=lambda p:float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    travel=lambda p:float(np.abs(np.diff(p,axis=0)).sum())
    return dict(route_departure_mm=float(cKDTree(baseline_tips).query(tips)[0].max()*1000),
        added_lift_mm=float((tips[:,2].max()-baseline_tips[:,2].max())*1000),
        tip_length_mm=arc(tips)*1000,baseline_tip_length_mm=arc(baseline_tips)*1000,
        tip_length_ratio=arc(tips)/max(arc(baseline_tips),1e-12),
        joint_travel_ratio=travel(joints)/max(travel(baseline_joints),1e-12))


def presentation_passes(route,motion,s):
    return bool(route['route_departure_mm']>=s['min_route_departure_mm']
        and route['added_lift_mm']>=s['min_added_lift_mm']
        and motion['joint_span_deg'][2]>=s['min_elbow_span_deg']
        and motion['joint_span_deg'][4]>=s['min_roll_span_deg']
        and route['tip_length_ratio']<=s['max_tip_length_ratio']
        and route['joint_travel_ratio']<=s['max_joint_travel_ratio'])


def presentation_score(route,motion,s):
    # Saturate each observable effect: longer/wilder routes earn no extra credit.
    values=[route['route_departure_mm']/s['desired_route_departure_mm'],
        route['added_lift_mm']/s['desired_added_lift_mm'],
        motion['joint_span_deg'][2]/s['desired_elbow_span_deg'],
        motion['joint_span_deg'][4]/s['desired_roll_span_deg']]
    return float(np.minimum(values,1.).mean())


def proposals(base,seed):
    u=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(base,axis=0),axis=1))];u/=u[-1]
    # Joint offsets in the arm's own coordinates; no plant-specific mesh indices.
    seeds=np.array([[8,-10,-10,8,18],[15,-12,-10,10,25],[-8,-10,-15,12,25],
        [15,-20,-20,18,35],[10,-15,-12,14,30]],float)
    draws=qmc.Sobol(5,scramble=True,seed=seed).random_base2(5)
    offsets=np.vstack([seeds,draws*np.array([28,-18,-18,18,32])+np.array([-10,-5,-5,2,8])])
    for index,offset in enumerate(offsets):
        # Full smooth lobe, and a compact lobe preserving the final three poses.
        for compact in (True,False):
            w=u/.72 if compact else u
            bump=np.sin(np.pi*np.clip(w,0,1))**2
            if not compact:bump=np.exp(-.5*((u-.35)/.24)**2)*np.sin(np.pi*u)
            p=base+bump[:,None]*np.deg2rad(offset)
            p[[0,-1]]=base[[0,-1]]
            yield p,f'early_elbow_wrist_side_arc_{index}_'+('compact'if compact else 'broad')


def main():
    root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path',type=Path,required=True);ap.add_argument('--baseline-replay',type=Path)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=root/'configs/weighted_trajectory_validation_v5.json')
    ap.add_argument('--settings',type=Path,default=root/'configs/demonstration_motion.json')
    ap.add_argument('--no-promote',action='store_true');args=ap.parse_args()
    source=args.path.resolve();out=args.output_dir.resolve();cfg=read_json(args.config);s=read_json(args.settings)
    validate_natural_settings(s);ps=s['presentation'];validate_presentation_settings(ps)
    if out.exists()and any(out.iterdir()):raise ValueError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True);report=read_json(source/'path_report.json')
    replay=args.baseline_replay or out/'baseline_replay';validator=root/'scripts/validate_watering_trajectory.py'
    if args.baseline_replay is None:
        subprocess.run([sys.executable,'-u',str(validator),'--path',str(source),'--config',str(args.config),
            '--output-dir',str(replay),'--no-promote'],check=True)
    validation=read_json(replay/'validation_report.json');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    scene=root/report['scene_relative']
    if (validation['status']!='SAMPLED_REPLAY_CHECKS_PASSED'or validation['config']!=cfg
        or sha(source/'trajectory_q.npz')!=validation['source_trajectory_sha256']
        or sha(source/'path_report.json')!=validation['source_path_report_sha256']
        or sha(scene)!=validation['scene_sha256']):raise ValueError('Baseline verification mismatch')
    baseline=dict(persistence_cost=validation['contact_risk']['simulated']['persistence_cost'],
        motion=natural_metrics(trajectory_from_replay(replay),s))
    checker=CollisionChecker(root,scene,cfg['hard_margin_mm']/1000)
    if read_json(scene.parent/'scene_report.json')['mesh_fingerprint']!=validation['mesh_fingerprint']:
        raise ValueError('Geometry mismatch')
    with np.load(source/'trajectory_q.npz')as z:
        base=z['q_arm_rad'].copy()
        if not np.allclose(z['q_gripper_rad'],checker.gripper_rad):raise ValueError('Grasp mismatch')
    saved=read_json(source/report['timing_plan_relative'])if report.get('timing_plan_relative')else make_timing_plan(base,cfg)
    if report.get('timing_plan_relative')and (sha(source/report['timing_plan_relative'])!=report['timing_plan_sha256']
        or report['timing_plan_sha256']!=validation.get('timing_plan_sha256')):raise ValueError('Baseline timing checksum mismatch')
    btr=time_parameterize(base,cfg,.02,saved);btips=tip_trace(checker,btr['q_ref'])
    begin=time.perf_counter();rows=[];attempts=0;rejected={}
    def reject(name):rejected[name]=rejected.get(name,0)+1
    for path,label in proposals(base,s['seed']):
        if time.perf_counter()-begin>=s['search_seconds']:break
        if np.any(path<checker.q_lo)or np.any(path>checker.q_hi):reject('joint_limits');continue
        length=np.linalg.norm(np.diff(path,axis=0),axis=1)
        blength=np.linalg.norm(np.diff(base,axis=0),axis=1)
        nominal=np.array(saved['segment_duration_s'])*np.maximum(length/np.maximum(blength,1e-8),.7)**.55
        for factor in (1.1,1.,.9):
            if time.perf_counter()-begin>=s['search_seconds']:break
            attempts+=1;h=nominal*factor;h[-1]=min(h[-1],saved['segment_duration_s'][-1]*.95)
            plan=make_timing_plan(path,cfg,h);plan['actuator_feedforward_mode']=s['actuator_feedforward_mode']
            tr=time_parameterize(path,cfg,.04,plan);m=natural_metrics(tr,s)
            if not motion_passes(m,s,baseline['motion']):reject('motion');continue
            route=route_metrics(tip_trace(checker,tr['q_ref']),tr['q_ref'],btips,btr['q_ref'])
            if not presentation_passes(route,m,ps):reject('presentation');continue
            if not all(checker.check(q,evaluate_risk=False)[0]for q in tr['q_ref'][::3]):reject('hard_preview');continue
            score=timed_path_score(checker,path,cfg,.08,plan)
            if not np.isfinite(score['objective']):reject('hard_interpolation');continue
            if score['persistence_cost']>baseline['persistence_cost']:reject('exposure');continue
            preview,actual=actuator_preview(checker,path,cfg,plan);am=natural_metrics(actual,s)
            if not preview['passed']or not motion_passes(am,s,baseline['motion']):reject('actuators');continue
            value=presentation_score(route,m,ps)
            rows.append(dict(path=path,timing_plan=plan,source=label,motion=am,presentation=route,
                presentation_score=value,**score))
            print(f'DEMONSTRATION candidate={attempts} risk={score["persistence_cost"]:.5f} duration={m["duration_s"]:.2f} score={value:.3f} departure_mm={route["route_departure_mm"]:.1f} lift_mm={route["added_lift_mm"]:.1f}',flush=True)
            write_json(out/'search_progress.json',dict(attempts=attempts,elapsed_s=time.perf_counter()-begin,
                rejected=rejected,candidates=[{k:v for k,v in r.items()if k not in ('path','timing_plan')}for r in rows]))
    diagnostics=dict(attempts=attempts,elapsed_s=time.perf_counter()-begin,settings=s,
        baseline_score=baseline,coarse_feasible=len(rows),rejected=rejected)
    write_json(out/'demonstration_search_report.json',diagnostics)
    rows.sort(key=lambda r:(-r['presentation_score'],r['motion']['normalized_jerk'],r['persistence_cost']))
    checked=[];selected=None
    for index,item in enumerate(rows[:s['finalists']]):
        dense=timed_path_score(checker,item['path'],cfg,.02,item['timing_plan'])
        if not np.isfinite(dense['objective'])or dense['persistence_cost']>baseline['persistence_cost']:continue
        item.update(dense);item['goal_id']=report['goal_id'];directory=out/'candidates'/f'candidate_{index:03d}'
        directory.mkdir(parents=True)
        export_path(root,directory,checker,source,report,item,[item],diagnostics,cfg,args.config.resolve())
        updated=read_json(directory/'path_report.json');updated.pop('coordination',None);updated.pop('attitude_escape',None)
        updated.update(planner='weighted RRT* + bounded demonstration posture search + fixed C3 timing',
            natural_acceptance=dict(settings=s,baseline=baseline),natural_search=diagnostics,selected_motion=item['motion'],
            presentation_preference=True,presentation=item['presentation'],presentation_score=item['presentation_score'],
            objectives=dict(selection='saturated side arc, lift and joint participation within actual baseline exposure ceiling',
                            global_optimum_certified=False),risk_optimized=False,risk_constrained=True)
        write_json(directory/'path_report.json',updated)
        result=subprocess.run([sys.executable,'-u',str(validator),'--path',str(directory),'--config',str(args.config),
            '--output-dir',str(directory/'replay'),'--no-promote'])
        checked.append(dict(path=str(directory),exit_code=result.returncode))
        if result.returncode==0:
            with np.load(directory/'replay/timed_trajectory.npz')as z:
                actual_route=route_metrics(z['simulated_tip_m'],z['q_actual_rad'][:,:5],btips,btr['q_ref'])
            verified=read_json(directory/'replay/validation_report.json')
            presentation_ok=presentation_passes(actual_route,verified['natural_motion'],ps)
            write_json(directory/'demonstration_verification.json',dict(passed=presentation_ok,
                actual_route=actual_route,settings=ps,baseline_path=str(source)))
            checked[-1]['actual_presentation_passed']=presentation_ok
            if not presentation_ok:continue
            selected=directory;break
    status='DEMONSTRATION_MOTION_VERIFIED'if selected else 'DEMONSTRATION_NO_IMPROVEMENT'
    write_json(out/'demonstration_selection.json',dict(status=status,path=str(selected or source),
        replay=str(selected/'replay'if selected else replay),trials=checked,baseline=baseline,settings=s))
    if selected:
        from render_joint_coordination import plot
        plot(selected)
        write_json(selected/'demonstration_search_report.json',dict(**diagnostics,verification_trials=checked))
        if not args.no_promote:
            for name,folder,filename in (('path',selected,'path_report.json'),('replay',selected/'replay','validation_report.json')):
                write_json(root/f'configs/latest_{name}.json',dict(output=folder.relative_to(root).as_posix(),
                    report=(folder/filename).relative_to(root).as_posix(),status=read_json(folder/filename)['status']))
    print(status,selected or source,flush=True)


if __name__=='__main__':main()
