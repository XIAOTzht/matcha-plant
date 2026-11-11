"""Natural five-joint refinement, retaining a verified baseline on regression."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.stats import qmc
from trajectory_tools import time_parameterize,make_timing_plan,actuator_command
from plant_scene import read_json,write_json
from plan_watering_path import CollisionChecker,timed_path_score
from attitude_escape_search import export_path


def validate_natural_settings(s):
    if s.get('selection_mode')!='natural':raise ValueError('Natural mode settings required')
    for name in ('search_seconds','max_duration_s','reversal_threshold_deg','min_elbow_span_deg','min_wrist_roll_span_deg','max_interior_low_speed_s'):
        if not isinstance(s[name],(int,float))or not np.isfinite(s[name])or s[name]<=0:raise ValueError(name)
    for name in ('finalists','rounds','max_reversals_per_joint'):
        if type(s[name])is not int or s[name]<1:raise ValueError(name)
    if not 0<s['low_speed_fraction']<1:raise ValueError('Invalid low speed fraction')
    if type(s['seed'])is not int or s['seed']<0:raise ValueError('Invalid random seed')
    if s.get('actuator_feedforward_mode','legacy')not in ('legacy','inverse_dynamics'):
        raise ValueError('Invalid actuator feedforward mode')


def significant_reversals(values,threshold_deg=3.):
    """Directional-change hysteresis: tiny reverse excursions never count."""
    x=np.asarray(values,float);threshold=np.deg2rad(threshold_deg)
    pivot=float(x[0]);extreme=pivot;direction=0;count=0
    for value in x[1:]:
        if direction==0:
            if abs(value-pivot)>=threshold:direction=int(np.sign(value-pivot));extreme=float(value)
        elif direction*(value-extreme)>=0:extreme=float(value)
        elif direction*(extreme-value)>=threshold:
            count+=1;direction=-direction;extreme=float(value)
    return count


def natural_metrics(trajectory,settings):
    t=trajectory['time_s'];q=trajectory['q_ref'][:,:5];v=trajectory['qd_ref'][:,:5];a=trajectory['qdd_ref'][:,:5]
    moving=trajectory['phase']==1;ids=np.flatnonzero(moving)
    duration=float(t[ids[-1]]-t[ids[0]])if len(ids)>1 else 0.
    speed=np.linalg.norm(v,axis=1);peak=float(speed.max())
    # Trim acceleration/deceleration shoulders; measure only interior hesitation.
    interior=moving&(t>=t[ids[0]]+.5)&(t<=t[ids[-1]]-.5)
    low=interior&(speed<settings['low_speed_fraction']*peak)
    runs=[];start=None
    for k,flag in enumerate(low):
        if flag and start is None:start=k
        if start is not None and (not flag or k==len(low)-1):
            end=k if flag else k-1
            runs.append(float(t[end]-t[start]));start=None
    jerk=trajectory.get('qddd_ref')
    if jerk is None:jerk=np.gradient(a,t,axis=0)
    else:jerk=jerk[:,:5]
    span=np.ptp(q,axis=0)
    # Dimensionless squared jerk: invariant to uniform time stretching.
    integral=np.trapezoid(jerk[ids]**2,t[ids],axis=0)
    normalized=integral*duration**5/np.maximum(span,np.deg2rad(3))**2
    return dict(duration_s=float(t[-1]),longest_interior_low_speed_s=max(runs,default=0.),
        interior_low_speed_total_s=float(np.trapezoid(low.astype(float),t)),
        normalized_jerk=float(np.mean(normalized)),normalized_jerk_per_joint=normalized.tolist(),
        joint_span_deg=np.rad2deg(span).tolist(),significant_reversals=[significant_reversals(q[:,j],settings['reversal_threshold_deg'])for j in range(5)],
        peak_speed_time_s=[float(t[np.argmax(abs(v[:,j]))])for j in range(5)])


def trajectory_from_replay(directory):
    with np.load(Path(directory)/'timed_trajectory.npz')as z:
        return dict(time_s=z['time_s'],q_ref=z['q_actual_rad'][:,:5],qd_ref=z['qd_actual_rad_s'][:,:5],
            qdd_ref=z['qdd_actual_rad_s2'][:,:5],phase=z['phase'])


def motion_passes(metrics,settings,baseline=None):
    if metrics['duration_s']>settings['max_duration_s']+1e-8:return False
    if max(metrics['significant_reversals'])>settings['max_reversals_per_joint']:return False
    if any(metrics['joint_span_deg'][i]+1e-6<value for i,value in ((2,settings['min_elbow_span_deg']),(4,settings['min_wrist_roll_span_deg']))):return False
    if baseline is not None:
        if metrics['longest_interior_low_speed_s']>settings['max_interior_low_speed_s']+1e-8:return False
        if metrics['normalized_jerk']>baseline['normalized_jerk']*(1+1e-6):return False
    return True


def actuator_preview(checker,path,config,plan):
    """Cheap actual actuator check before full plant triangle/contact queries.

    Uses the same model, integrator, position lead and gravity compensation as
    the full validator. This cannot certify plant clearance or contact risk.
    """
    import mujoco
    model=checker.model;data=mujoco.MjData(model)
    inverse=mujoco.MjData(model)if plan.get('actuator_feedforward_mode')=='inverse_dynamics'else None
    names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
    qa=np.array([model.joint(n).qposadr[0]for n in names]);va=np.array([model.joint(n).dofadr[0]for n in names])
    ai=np.array([model.actuator(n).id for n in names]);kp=model.actuator_gainprm[ai,0]
    tr=time_parameterize(path,config,float(model.opt.timestep),plan)
    data.qpos[qa]=np.r_[path[0],config['gripper_rad']];data.ctrl[ai]=data.qpos[qa]
    mujoco.mj_forward(model,data);acc=[];vel=[];pos=[];ranges_ok=True
    for k,t in enumerate(tr['time_s']):
        mujoco.mj_forward(model,data)
        command=actuator_command(model,data,config,np.r_[tr['q_ref'][k],config['gripper_rad']],
            np.r_[tr['qd_ref'][k],0.],np.r_[tr['qdd_ref'][k],0.],qa,va,ai,inverse)
        ranges=model.actuator_ctrlrange[ai];ranges_ok=ranges_ok and bool(np.all(command>=ranges[:,0])and np.all(command<=ranges[:,1]))
        data.ctrl[ai]=command;mujoco.mj_forward(model,data)
        acc.append(data.qacc[va[:5]].copy());vel.append(data.qvel[va[:5]].copy());pos.append(data.qpos[qa[:5]].copy())
        if k<len(tr['time_s'])-1:mujoco.mj_step(model,data)
    a=np.array(acc);v=np.array(vel);q=np.array(pos)
    ratio=max(float(np.max(abs(a)/config['acceleration_limit_rad_s2'])),float(np.max(abs(v)/config['velocity_limit_rad_s'])))
    result=dict(peak_acceleration_rad_s2=np.max(abs(a),axis=0).tolist(),limit_ratio=ratio,
        passed=bool(ratio<=1 and ranges_ok and np.isfinite(q).all()and not np.any(data.warning.number)
            and np.max(abs(q-tr['q_ref']))<=np.deg2rad(config['joint_tracking_tolerance_deg'])))
    actual=dict(time_s=tr['time_s'],q_ref=q,qd_ref=v,qdd_ref=a,phase=tr['phase'])
    return result,actual


def repaired_timing_proposals(path,config,settings,seed):
    h=np.array(seed['segment_duration_s'])
    original=dict(seed);original['actuator_feedforward_mode']=settings.get('actuator_feedforward_mode','legacy')
    yield original
    draws=qmc.Sobol(len(h),scramble=True,seed=settings['seed']+17).random_base2(7)
    for draw in draws:
        factors=1+draw*.5
        # Preserve the last approach duration to avoid increasing contact time;
        # vary the preceding joints' reversals through segment timing instead.
        factors[-1]=.92+draw[-1]*.09
        plan=make_timing_plan(path,config,h*factors)
        plan['actuator_feedforward_mode']=settings.get('actuator_feedforward_mode','legacy')
        yield plan


def timing_proposals(path,config):
    seed=make_timing_plan(path,config,smooth=False)
    h=np.asarray(seed['segment_duration_s'])
    for factor in (1.,.9,.8):
        yield make_timing_plan(path,config,h*factor,smooth=True)
    # Redistribute segment times at fixed total: prioritize longer geometric
    # transitions without reserving a full stop-like interval for tiny knots.
    length=np.linalg.norm(np.diff(path,axis=0),axis=1)
    allocated=np.maximum(length,.01)**.75
    allocated=allocated/allocated.sum()*h.sum()
    for mix in (.35,.65):
        yield make_timing_plan(path,config,(h*(1-mix)+allocated*mix)*.8,smooth=True)


def geometry_proposals(path,settings,round_id):
    yield path.copy(),'same_geometry_smooth_derivatives'
    # Delete every single redundant knot, plus pairs. All proposals are rechecked.
    for index in range(1,len(path)-1):
        yield np.delete(path,index,axis=0),f'remove_knot_{index}'
    if len(path)>5:
        for indices in ((2,6),(2,5),(1,6)):
            if max(indices)<len(path)-1:yield np.delete(path,indices,axis=0),f'remove_knots_{indices}'
    u=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))];u/=u[-1]
    draws=qmc.Sobol(d=10,scramble=True,seed=settings['seed']+round_id).random_base2(3)*2-1
    for n,draw in enumerate(draws):
        # Smooth monotone phase changes overlap, with optional early wrist motion.
        powers=np.exp(draw[:5]*.22)
        p=np.column_stack([np.interp(u**powers[j],u,path[:,j])for j in range(5)])
        p+=np.sin(np.pi*u[:,None])*draw[5:]*np.deg2rad([2,4,4,5,6])*(.7**round_id)
        p[[0,-1]]=path[[0,-1]]
        yield p,f'overlapping_joint_phases_{n}'


def key(item):
    m=item['motion']
    return (round(m['longest_interior_low_speed_s'],2),m['normalized_jerk'],item['persistence_cost'],m['duration_s'])


def summaries(rows):
    return [{k:v for k,v in r.items()if k not in ('path','timing_plan')}for r in rows]


def search(checker,base,config,settings,out,baseline):
    start=time.perf_counter();candidates=[];seen=set();rejected={};attempts=0;anchors=[base]
    for round_id in range(settings['rounds']):
        for anchor in anchors:
            for path,label in geometry_proposals(anchor,settings,round_id):
                if time.perf_counter()-start>=settings['search_seconds']:break
                if np.any(path<checker.q_lo)or np.any(path>checker.q_hi):continue
                pathkey=path.tobytes()
                if pathkey in seen:continue
                seen.add(pathkey)
                if not all(checker.edge_cost(a,b,max_step_rad=.025)[0]for a,b in zip(path[:-1],path[1:])):
                    rejected['hard_edges']=rejected.get('hard_edges',0)+1;continue
                for plan in timing_proposals(path,config):
                    if time.perf_counter()-start>=settings['search_seconds']:break
                    attempts+=1
                    plan['actuator_feedforward_mode']=settings.get('actuator_feedforward_mode','legacy')
                    tr=time_parameterize(path,config,.02,plan);m=natural_metrics(tr,settings)
                    if not motion_passes(m,settings):
                        rejected['motion']=rejected.get('motion',0)+1;continue
                    # Keep nearly admissible candidates as branching seeds; final
                    # natural improvement thresholds are enforced after dynamics.
                    score=timed_path_score(checker,path,config,dt=.08,timing_plan=plan)
                    if not np.isfinite(score['objective']):
                        rejected['hard_interpolation']=rejected.get('hard_interpolation',0)+1;continue
                    if score['persistence_cost']>baseline['persistence_cost']*1.01:
                        rejected['exposure']=rejected.get('exposure',0)+1;continue
                    item=dict(path=path,timing_plan=plan,source=label,motion=m,**score)
                    candidates.append(item)
                    print(f"NATURAL candidate={attempts} knots={len(path)} risk={score['persistence_cost']:.3f} slow={m['longest_interior_low_speed_s']:.3f} jerk={m['normalized_jerk']:.1f} seconds={m['duration_s']:.2f}",flush=True)
                    write_json(out/'search_progress.json',dict(attempts=attempts,elapsed_s=time.perf_counter()-start,candidates=summaries(candidates),rejected=rejected))
            if time.perf_counter()-start>=settings['search_seconds']:break
        # Distinct representatives for flow, risk, and joint participation.
        beam=[]
        for ranked in (sorted(candidates,key=key),sorted(candidates,key=lambda x:x['persistence_cost']),
                       sorted(candidates,key=lambda x:-min(x['motion']['joint_span_deg'][2]/30,x['motion']['joint_span_deg'][4]/40))):
            for row in ranked:
                if not any(np.array_equal(row['path'],p)for p in beam):beam.append(row['path']);break
        anchors=beam
        if not anchors or time.perf_counter()-start>=settings['search_seconds']:break
    return sorted(candidates,key=key),dict(attempts=attempts,coarse_feasible=len(candidates),rejected=rejected,elapsed_s=time.perf_counter()-start)


def main(args=None):
    root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=root/'configs/weighted_trajectory_validation_v5.json')
    ap.add_argument('--coordination-config',type=Path,default=root/'configs/natural_motion.json')
    ap.add_argument('--baseline-replay',type=Path);ap.add_argument('--seconds',type=float)
    ap.add_argument('--seed-path',type=Path,help='Optional same-scene refinement seed; baseline stays authoritative')
    args=ap.parse_args(args);source=args.path.resolve();out=args.output_dir.resolve()
    if out.exists()and any(out.iterdir()):raise ValueError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True);settings=read_json(args.coordination_config);config=read_json(args.config)
    if args.seconds is not None:settings['search_seconds']=args.seconds
    validate_natural_settings(settings)
    report=read_json(source/'path_report.json')
    if report['status']!='PATH_PLANNING_COMPLETE':raise ValueError('Completed baseline required')
    baseline_replay=args.baseline_replay
    validate=root/'scripts/validate_watering_trajectory.py'
    if baseline_replay is None:
        baseline_replay=out/'baseline_replay'
        subprocess.run([sys.executable,'-u',str(validate),'--path',str(source),'--config',str(args.config),'--output-dir',str(baseline_replay),'--no-promote'],check=True)
    validation=read_json(baseline_replay/'validation_report.json')
    if (validation['status']!='SAMPLED_REPLAY_CHECKS_PASSED'or validation['source_trajectory_sha256']!=hashlib.sha256((source/'trajectory_q.npz').read_bytes()).hexdigest()
        or validation['source_path_report_sha256']!=hashlib.sha256((source/'path_report.json').read_bytes()).hexdigest()):
        raise ValueError('Baseline replay does not verify the supplied path')
    if validation['config']!=config:raise ValueError('Baseline was validated with different timing/limits')
    if validation['scene_sha256']!=hashlib.sha256((root/report['scene_relative']).read_bytes()).hexdigest():
        raise ValueError('Baseline scene has changed')
    metrics=natural_metrics(trajectory_from_replay(baseline_replay),settings)
    baseline=dict(persistence_cost=validation['contact_risk']['simulated']['persistence_cost'],motion=metrics)
    print('NATURAL_BASELINE',baseline,flush=True)
    checker=CollisionChecker(root,root/report['scene_relative'],config['hard_margin_mm']/1000)
    if read_json(checker.scene_dir/'scene_report.json')['mesh_fingerprint']!=validation['mesh_fingerprint']:
        raise ValueError('Baseline validation belongs to another geometry')
    with np.load(source/'trajectory_q.npz')as z:
        base=z['q_arm_rad'].copy()
        if not np.allclose(z['q_gripper_rad'],checker.gripper_rad):raise ValueError('Grasp mismatch')
    if args.seed_path:
        seed_report=read_json(args.seed_path/'path_report.json')
        if seed_report['mesh_fingerprint']!=report['mesh_fingerprint']:raise ValueError('Seed scene mismatch')
        with np.load(args.seed_path/'trajectory_q.npz')as z:seed_path=z['q_arm_rad'].copy()
        if not np.array_equal(seed_path[[0,-1]],base[[0,-1]]):raise ValueError('Seed endpoint mismatch')
        candidates=[];begin=time.perf_counter()
        seed=read_json(args.seed_path/'timing_plan.json')
        for plan in repaired_timing_proposals(seed_path,config,settings,seed):
            if time.perf_counter()-begin>=settings['search_seconds']:break
            preview,actual=actuator_preview(checker,seed_path,config,plan)
            if not preview['passed']:continue
            m=natural_metrics(actual,settings)
            if not motion_passes(m,settings,metrics):continue
            score=timed_path_score(checker,seed_path,config,.08,plan)
            if not np.isfinite(score['objective'])or score['persistence_cost']>baseline['persistence_cost']:continue
            candidates.append(dict(path=seed_path,timing_plan=plan,source='actuator_aware_segment_timing',motion=m,**score))
            print('NATURAL_REPAIRED',len(candidates),score,m,flush=True)
            if len(candidates)>=4:break
        candidates.sort(key=key)
        diagnostics=dict(source_seed=str(args.seed_path),coarse_feasible=len(candidates),elapsed_s=time.perf_counter()-begin)
    else:
        candidates,diagnostics=search(checker,base,config,settings,out,baseline)
    # Bound actuator-aware timing repair as part of the same search budget.
    repair_deadline=time.perf_counter()+max(0.,settings['search_seconds']-diagnostics['elapsed_s'])
    diagnostics.update(baseline_score=baseline,settings=settings)
    write_json(out/'natural_search_report.json',diagnostics)
    checked=[];selected=None;dense_count=0
    for index,item in enumerate(candidates):
        if dense_count>=settings['finalists']:break
        if not motion_passes(item['motion'],settings,metrics):continue
        if item['persistence_cost']>baseline['persistence_cost']:continue
        preview,actual=actuator_preview(checker,item['path'],config,item['timing_plan'])
        if not preview['passed']:
            repaired=None
            for plan in repaired_timing_proposals(item['path'],config,settings,item['timing_plan']):
                if time.perf_counter()>repair_deadline:break
                preview,actual=actuator_preview(checker,item['path'],config,plan)
                if not preview['passed']:continue
                m=natural_metrics(actual,settings)
                if not motion_passes(m,settings,metrics):continue
                score=timed_path_score(checker,item['path'],config,.08,plan)
                if not np.isfinite(score['objective'])or score['persistence_cost']>baseline['persistence_cost']:continue
                repaired=dict(**item);repaired.update(timing_plan=plan,motion=m,**score);break
            if repaired is None:continue
            item=repaired
        dense_count+=1
        dense=timed_path_score(checker,item['path'],config,dt=.02,timing_plan=item['timing_plan'])
        if not np.isfinite(dense['objective'])or dense['persistence_cost']>baseline['persistence_cost']:continue
        item.update(dense);directory=out/'candidates'/f'candidate_{index:03d}';directory.mkdir(parents=True)
        item['goal_id']=report['goal_id']
        export_path(root,directory,checker,source,report,item,[item],diagnostics,config,args.config.resolve())
        updated=read_json(directory/'path_report.json');updated.pop('coordination',None);updated.pop('attitude_escape',None)
        updated.update(planner='weighted RRT* + natural joint phase, knot reduction and fixed C3 timing search',
            selected_motion=item['motion'],natural_acceptance=dict(settings=settings,baseline=baseline),risk_optimized=False,
            risk_constrained=True,natural_search=diagnostics,presentation_preference=False,
            objectives=dict(selection='interior hesitation, then dimensionless jerk, under actual baseline exposure ceiling',
                            global_optimum_certified=False))
        write_json(directory/'path_report.json',updated)
        print('NATURAL_DYNAMICS',directory,flush=True)
        result=subprocess.run([sys.executable,'-u',str(validate),'--path',str(directory),'--config',str(args.config),
            '--output-dir',str(directory/'replay'),'--no-promote'])
        checked.append(dict(path=str(directory),exit_code=result.returncode))
        if result.returncode==0:
            selected=directory;break
    status='NATURAL_MOTION_VERIFIED'if selected else 'NATURAL_SEARCH_NO_IMPROVEMENT'
    chosen=selected or source;replay=selected/'replay'if selected else baseline_replay
    write_json(out/'natural_selection.json',dict(status=status,path=str(chosen),replay=str(replay),
        trials=checked,dense_candidates=dense_count,baseline=baseline))
    if selected:
        from render_joint_coordination import plot
        plot(selected)
        write_json(selected/'natural_search_report.json',dict(**diagnostics,verification_trials=checked))
        for name,file,status_value in (('path','path_report.json','PATH_PLANNING_COMPLETE'),('replay','validation_report.json','SAMPLED_REPLAY_CHECKS_PASSED')):
            folder=chosen if name=='path'else replay
            write_json(root/f'configs/latest_{name}.json',dict(output=folder.relative_to(root).as_posix(),report=(folder/file).relative_to(root).as_posix(),status=status_value))
    print(status,chosen,flush=True)


if __name__=='__main__':main()
