"""Whole-trajectory joint coordination with explicit contact/time tradeoffs.

Broad, low-frequency joint offsets and independent joint phase schedules can
turn the wrist before entering foliage, change elbow posture and use side
approaches. A task-position nullspace proposal also varies posture without
arbitrarily moving the nozzle. Every selected interpolated path is rechecked.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.stats import qmc
from plan_watering_path import ARM_JOINTS, CollisionChecker, timed_path_score
from attitude_escape_search import export_path
from trajectory_tools import time_parameterize
from plant_scene import read_json, write_json


def arc_parameter(path):
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if distance[-1]<1e-10:raise ValueError('A nonzero start-to-goal route is required')
    # Consecutive duplicate knots carry no extra motion information.
    keep=np.r_[True,np.diff(distance)>1e-10]
    return distance[keep]/distance[-1],np.asarray(path)[keep]


def controls(path):
    u,p=arc_parameter(path)
    phase=np.unique(np.r_[u,np.linspace(0,1,7)])
    return phase,np.column_stack([np.interp(phase,u,p[:,j])for j in range(5)])


def motion_metrics(path,settings):
    q=np.asarray(path);delta=np.diff(q,axis=0)
    span=np.ptp(q,axis=0);desired=np.deg2rad(settings['desired_joint_span_deg'])
    participation=np.minimum(span/desired,1.)
    travel=np.abs(delta).sum(axis=0);excess=np.maximum(travel-span,0.)
    # Saturated span, not accumulated travel: no reward for looping in place.
    coverage=.55*float(participation.mean())+.45*float(participation.min())
    reversal=float(np.sum(excess)/max(np.sum(travel),1e-10))
    return dict(joint_span_deg=np.rad2deg(span).tolist(),joint_travel_deg=np.rad2deg(travel).tolist(),
        joint_participation=participation.tolist(),active_joints=int((span>=np.deg2rad(10)).sum()),
        joint_travel_rad=float(travel.sum()),coordination_score=float(coverage-.12*reversal),
        reversal_fraction=reversal)


def validate_settings(s):
    if s['selection_mode'] not in ('coordinated','efficient'):raise ValueError('Unknown selection mode')
    for key in ('desired_joint_span_deg','excursion_scale_deg'):
        v=np.asarray(s[key],float)
        if v.shape!=(5,) or not np.isfinite(v).all() or np.any(v<=0):raise ValueError(key)
    for key in ('relative_exposure_allowance','absolute_exposure_allowance'):
        if not np.isfinite(s[key]) or s[key]<0:raise ValueError(key)
    for key in ('max_duration_ratio','max_joint_travel_ratio'):
        if not np.isfinite(s[key]) or s[key]<1:raise ValueError(key)
    for key in ('rounds','proposals_per_round','finalists'):
        if not isinstance(s[key],int) or s[key]<1:raise ValueError(key)
    if not 0<s['final_dt_s']<=s['coarse_dt_s']<=.1 or not np.isfinite(s['search_seconds']) or s['search_seconds']<=0:
        raise ValueError('Invalid sampling or time budget')


def path_proposals(base,lo,hi,s,round_id=0):
    phase,anchor=controls(base);u,p=arc_parameter(base)
    sampler=qmc.Sobol(d=15,scramble=True,seed=int(s['seed'])+round_id)
    draws=sampler.random_base2(int(np.ceil(np.log2(s['proposals_per_round']))))[:s['proposals_per_round']]*2-1
    scales=np.deg2rad(s['excursion_scale_deg'])*(0.8**round_id)
    for index,draw in enumerate(draws):
        if index%3==0:
            # Positive monotone warps sequence joints differently, with exact endpoints.
            power=np.exp(draw[:5]*.65)
            warped=phase[:,None]**power[None,:]
            path=np.column_stack([np.interp(warped[:,j],u,p[:,j])for j in range(5)])
            kind='independent_joint_phases'
        else:
            # Separate early/late lobes allow elbow and wrist to turn at different times.
            centers=.32+.25*draw[5:10]
            widths=.22+.05*draw[10:15]
            bump=np.exp(-.5*((phase[:,None]-centers)/widths)**2)*np.sin(np.pi*phase[:,None])
            offsets=bump*draw[:5]*scales
            if index%3==2:
                offsets+=.35*np.sin(2*np.pi*phase[:,None])*draw[5:10]*scales
            path=anchor+offsets;kind='whole_arm_smooth_excursion'
        path[0]=base[0];path[-1]=base[-1]
        if np.any(path<lo) or np.any(path>hi):continue
        yield dict(path=path,source=kind,round=round_id,proposal=index)


def nullspace_proposal(checker,base,s,sign=1):
    phase,path=controls(base);result=path.copy();sid=checker.model.site('nozzle_tip').id
    cols=[checker.model.jnt_dofadr[j]for j in checker.joint_ids]
    for i in range(1,len(path)-1):
        checker.set_q(path[i]);target=checker.data.site_xpos[sid].copy()
        jac=np.zeros((3,checker.model.nv));mujoco.mj_jacSite(checker.model,checker.data,jac,None,sid)
        j=jac[:,cols];projector=np.eye(5)-np.linalg.pinv(j,rcond=1e-5)@j
        offset=projector@np.deg2rad([8,-25,30,-30,45])*sign*np.sin(np.pi*phase[i])
        anchor=np.clip(path[i]+offset,checker.q_lo+1e-6,checker.q_hi-1e-6)
        def residual(q):
            checker.set_q(q)
            return np.r_[(checker.data.site_xpos[sid]-target)/.0001,.002*(q-anchor)]
        sol=least_squares(residual,anchor,bounds=(checker.q_lo,checker.q_hi),max_nfev=60)
        checker.set_q(sol.x)
        if np.linalg.norm(checker.data.site_xpos[sid]-target)>.00025:return None
        result[i]=sol.x
    return dict(path=result,source='nozzle_position_nullspace',round=0,proposal=sign)


def select_candidate(scored,s):
    feasible=[c for c in scored if np.isfinite(c['objective'])]
    if not feasible:raise ValueError('No valid candidate')
    efficient=min(feasible,key=lambda c:c['objective'])
    lowest=min(c['persistence_cost']for c in feasible)
    limit=lowest*(1+s['relative_exposure_allowance'])+s['absolute_exposure_allowance']
    baselines=[c['persistence_cost']for c in feasible if c['source']=='unchanged_baseline']
    baseline_ceiling=min(baselines)if baselines else None
    if baseline_ceiling is not None:limit=min(limit,baseline_ceiling)
    # Constraints reference an actual minimum-exposure route, never a changing beam.
    reference=min(feasible,key=lambda c:(c['persistence_cost'],c['duration_s']))
    eligible=[c for c in feasible if c['persistence_cost']<=limit+1e-9
        and c['duration_s']<=reference['duration_s']*s['max_duration_ratio']+1e-9
        and c['motion']['joint_travel_rad']<=reference['motion']['joint_travel_rad']*s['max_joint_travel_ratio']+1e-9]
    selected=efficient if s['selection_mode']=='efficient' else max(eligible,key=lambda c:(c['motion']['coordination_score'],-c['objective']))
    return selected,dict(mode=s['selection_mode'],minimum_evaluated_persistence=lowest,
        persistence_allowance_limit=limit,efficient_objective=efficient['objective'],
        baseline_persistence_ceiling=baseline_ceiling,
        eligible_candidates=len(eligible),selected_objective_delta=selected['objective']-efficient['objective'],
        selected_persistence_delta=selected['persistence_cost']-lowest,
        duration_reference_s=reference['duration_s'],global_optimum_certified=False)


def summarize(item):
    return {k:v for k,v in item.items() if k!='path'}


def search(checker,base,timing,s,out):
    validate_settings(s);started=time.perf_counter()
    baseline=dict(path=base,source='unchanged_baseline',**timed_path_score(checker,base,timing,dt=s['final_dt_s']),
                  motion=motion_metrics(base,s))
    if not np.isfinite(baseline['objective']):raise ValueError('Baseline path fails on this scene')
    print('COORDINATION_BASELINE',summarize(baseline),flush=True)
    all_candidates=[baseline];rejected=[];anchor=base;attempts=0
    for round_id in range(s['rounds']):
        proposals=list(path_proposals(anchor,checker.q_lo,checker.q_hi,s,round_id))
        if round_id==0:
            for sign in (-1,1):
                item=nullspace_proposal(checker,base,s,sign)
                if item is not None:proposals.insert(0,item)
        for item in proposals:
            if attempts and time.perf_counter()-started>s['search_seconds']:break
            attempts+=1;path=item['path'];motion=motion_metrics(path,s)
            if motion['joint_travel_rad']>baseline['motion']['joint_travel_rad']*s['max_joint_travel_ratio']:
                rejected.append(dict(attempt=attempts,reason='excess joint travel'));continue
            trajectory=time_parameterize(path,timing,s['coarse_dt_s'])
            if trajectory['time_s'][-1]>baseline['duration_s']*s['max_duration_ratio']:
                rejected.append(dict(attempt=attempts,reason='excess duration'));continue
            # Check knots and intermediate poses before the full coarse timed score.
            quick=np.concatenate([np.linspace(a,b,5)for a,b in zip(path[:-1],path[1:])])
            if not all(checker.check(q,evaluate_risk=False)[0]for q in quick):
                rejected.append(dict(attempt=attempts,reason='hard constraint'));continue
            score=timed_path_score(checker,path,timing,dt=s['coarse_dt_s'])
            if not np.isfinite(score['objective']):
                rejected.append(dict(attempt=attempts,reason='interpolated hard constraint'));continue
            item.update(score,motion=motion);all_candidates.append(item)
            print(f"COORDINATION candidate={attempts} round={round_id} risk={score['persistence_cost']:.4f} seconds={score['duration_s']:.2f} participation={motion['coordination_score']:.3f}",flush=True)
            write_json(out/'search_progress.json',dict(attempts=attempts,elapsed_s=time.perf_counter()-started,
                candidates=[summarize(c)for c in all_candidates],rejected=rejected))
        # Improve risk first; next round can explore coordinated alternatives around it.
        anchor=min(all_candidates,key=lambda c:c['objective'])['path']
        if time.perf_counter()-started>s['search_seconds']:break
    # Keep both low-exposure and expressive contenders. Coarse scores never certify a path.
    provisional,selection=select_candidate(all_candidates,s)
    shortlist=[baseline,provisional]
    ranked=sorted(all_candidates,key=lambda c:c['persistence_cost'])
    expressive=sorted(all_candidates,key=lambda c:-c['motion']['coordination_score'])
    for low,high in zip(ranked,expressive):
        shortlist.extend((low,high))
    final=[];seen=set()
    for item in shortlist:
        key=item['path'].tobytes()
        if key in seen:continue
        seen.add(key)
        if not all(checker.edge_cost(a,b,max_step_rad=.025)[0]for a,b in zip(item['path'][:-1],item['path'][1:])):
            rejected.append(dict(source=item['source'],reason='dense edge verification'));continue
        score=timed_path_score(checker,item['path'],timing,dt=s['final_dt_s'])
        if not np.isfinite(score['objective']):continue
        item.update(score);final.append(item)
        print('COORDINATION_FINALIST',len(final),summarize(item),flush=True)
        if len(final)>=s['finalists']:break
    selected,selection=select_candidate(final,s)
    diagnostics=dict(attempts=attempts,coarse_feasible=len(all_candidates),verified_finalists=len(final),
        rejected=rejected,baseline_score=summarize(baseline),selection=selection,
        elapsed_s=time.perf_counter()-started,settings=s)
    return selected,final,diagnostics


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=root/'configs/weighted_trajectory_validation_v5.json')
    ap.add_argument('--coordination-config',type=Path,default=root/'configs/coordinated_motion.json')
    ap.add_argument('--seconds',type=float)
    ap.add_argument('--baseline-replay',type=Path,help='Natural mode: reuse a verified baseline replay')
    args=ap.parse_args();source=args.path.resolve();out=args.output_dir.resolve();config_path=args.config.resolve()
    if read_json(args.coordination_config).get('selection_mode')=='natural':
        from natural_motion import main as natural_main
        return natural_main()
    if out.exists()and any(out.iterdir()):raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);report=read_json(source/'path_report.json')
    if report['status']!='PATH_PLANNING_COMPLETE':raise ValueError('Completed baseline required')
    config=read_json(config_path);settings=read_json(args.coordination_config)
    if args.seconds is not None:settings['search_seconds']=args.seconds
    validate_settings(settings)
    checker=CollisionChecker(root,(root/report['scene_relative']).resolve(),config['hard_margin_mm']/1000)
    scene_report=read_json(checker.scene_dir/'scene_report.json')
    if report['mesh_fingerprint']!=scene_report['mesh_fingerprint']:raise ValueError('Baseline belongs to another mesh')
    with np.load(source/'trajectory_q.npz')as z:
        base=z['q_arm_rad'];grip=z['q_gripper_rad']
    if not np.allclose(grip,checker.gripper_rad)or not np.isclose(config['gripper_rad'],checker.gripper_rad):raise ValueError('Grasp mismatch')
    if not np.array_equal(base[0],report['start_q_rad'])or not np.array_equal(base[-1],report['goal_q_rad']):raise ValueError('Endpoint mismatch')
    selected,scored,diagnostics=search(checker,base,config,settings,out)
    for item in scored:item['goal_id']=report['goal_id']
    export_path(root,out,checker,source,report,selected,scored,diagnostics,config,config_path)
    updated=read_json(out/'path_report.json');updated.pop('attitude_escape',None)
    updated.update(planner='weighted RRT* and whole-trajectory coordinated posture/phase search',
        coordination=diagnostics,selected_motion=selected['motion'],
        risk_optimized=bool(settings['selection_mode']=='efficient'),
        risk_constrained=True,
        objectives=dict(contact='original weighted distance and continuous-exposure penalty',
            efficient='persistence + 0.05 * duration',selection=diagnostics['selection']),
        presentation_preference=settings['selection_mode']=='coordinated',
        better_than_baseline_exposure=selected['persistence_cost']<diagnostics['baseline_score']['persistence_cost'])
    from motion_review import review_motion
    updated['motion_style_review'] = review_motion(updated)
    write_json(out/'path_report.json',updated);write_json(out/'coordination_report.json',diagnostics)
    write_json(out/'coordination_config.json',settings)
    print('MOTION_STYLE_REVIEW', updated['motion_style_review'], flush=True)
    from render_joint_coordination import plot
    plot(out)
    write_json(root/'configs/latest_path.json',dict(output=out.relative_to(root).as_posix(),
        report=(out/'path_report.json').relative_to(root).as_posix(),status=updated['status']))
    print('COORDINATED_PATH_COMPLETE',out,flush=True)


if __name__=='__main__':main()
