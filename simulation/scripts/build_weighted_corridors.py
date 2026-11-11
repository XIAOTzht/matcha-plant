"""Generate alternative approach corridors and wrist postures as planner seeds.

The routes are evaluated with unchanged plant weights. Cartesian waypoints
are only search proposals; every accepted edge uses full robot hard checks.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import least_squares
from plant_scene import read_json,write_json
from plan_watering_path import CollisionChecker,ARM_JOINTS,path_cost,timed_path_score
from explore_watering_poses import position_cone_residual,spray_hits_soil


def screening_report(checker,selected,out):
    grouped={}
    for pose in selected:
        cid=pose['target_id']
        branch={**pose,'accepted_screening':True,'ik_success':True,'ik_residual_scaled':0.}
        if cid not in grouped:
            grouped[cid]=dict(**branch,checked_branches=[])
            grouped[cid].update(joint_names=list(ARM_JOINTS),q_gripper_rad=checker.gripper_rad)
            grouped[cid]['id']=cid
        grouped[cid]['checked_branches'].append(branch)
    report=dict(status='GOAL_SCREENING_COMPLETE',scene=str(checker.scene),
        scene_relative=checker.scene.relative_to(checker.root).as_posix(),entries=list(grouped.values()),
        orientation='downward 20 degree cone; free wrist and arm posture',not_a_trajectory=True)
    write_json(out/'goal_screening_report.json',report)
    return report


def corridor_seeds(checker,pose,count=3):
    """Side approaches with free mid-route tilt, lifted above hard objects."""
    start=np.array(checker.config['planning']['start_q_arm_rad'],dtype=float)
    goal=np.asarray(pose['q_arm_rad']);target=np.asarray(pose['target_m'])
    sid=checker.model.site('nozzle_tip').id
    checker.set_q(start);start_tip=checker.data.site_xpos[sid].copy()
    # Move away from the soft plant centroid in the horizontal plane.
    soft=checker.xyz[np.unique(checker.faces[~checker.face_hard])]
    center=np.median(soft,axis=0)
    outward=target[:2]-center[:2]
    if np.linalg.norm(outward)<.001:outward=np.array([-1.,0.])
    outward/=np.linalg.norm(outward)
    for variant in range(count):
        side=np.array([-outward[1],outward[0]])
        offset=(variant%3-1)*.04
        away=outward*(.07+.03*(variant//3))+side*offset
        approach=target+np.r_[away,.035+.02*(variant//3)]
        middle=target+np.r_[away*1.3,.09]
        lifted=start_tip+np.array([-.02,0,.12])
        points=[lifted,middle,approach,target]
        path=[start];anchor=start.copy();valid=True
        for k,point in enumerate(points[:-1]):
            # Partial final-posture bias encourages wrist rotation early,
            # while position and risk decide if the posture is useful.
            anchors=[anchor,.5*anchor+.5*goal,goal]
            proposals=[]
            for initial in anchors:
                result=least_squares(position_cone_residual,initial,bounds=(checker.q_lo,checker.q_hi),
                    args=(checker,point,initial,-.3),max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                checker.set_q(result.x)
                error=np.linalg.norm(checker.data.site_xpos[sid]-point)
                if error>.002:continue
                feasible,risk=checker.check(result.x)
                if not feasible:continue
                proposals.append((risk+.05*np.linalg.norm(result.x-anchor),result.x))
            if not proposals:valid=False;break
            anchor=min(proposals,key=lambda x:x[0])[1];path.append(anchor)
        if valid:
            path=np.vstack([*path,goal])
            yield path,f'task_space_side_corridor_{variant}'


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--poses',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--postures',type=int,default=8);ap.add_argument('--corridors',type=int,default=6)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--max-endpoint-risk',type=float,default=float('inf'),
                    help='Optional search budget filter; soft contact is not a hard obstacle by default')
    args=ap.parse_args();source=read_json(args.poses);out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    checker=CollisionChecker(root,root/source['scene_relative'])
    selected=[]
    rejected=[]
    for pose in source['entries']:
        q=np.asarray(pose['q_arm_rad'])
        if any(np.linalg.norm(q-np.asarray(p['q_arm_rad']))<.13 for p in selected):continue
        if pose['risk_proxy']>args.max_endpoint_risk:continue
        hits,ray=spray_hits_soil(checker,q)
        if not hits:
            rejected.append(dict(target_id=pose['target_id'],ray=ray));continue
        pose['spray_ray']=ray
        selected.append(pose)
        if len(selected)>=args.postures:break
    screening_report(checker,selected,out)
    write_json(root/'configs/latest_screening.json',dict(output=out.relative_to(root).as_posix(),
        report=(out/'goal_screening_report.json').relative_to(root).as_posix(),status='GOAL_SCREENING_COMPLETE'))
    write_json(out/'rejected_spray_rays.json',rejected)
    config=read_json(args.config or root/'configs/weighted_trajectory_validation.json')
    candidates=[];saved={}
    start=np.array(checker.config['planning']['start_q_arm_rad'])
    for pose in selected:
        proposals=[(np.vstack([start,pose['q_arm_rad']]),'direct_seed')]
        proposals+=list(corridor_seeds(checker,pose,args.corridors))
        for path,label in proposals:
            spatial=path_cost(checker,path)
            if not np.isfinite(spatial):continue
            score=timed_path_score(checker,path,config,dt=.06)
            if not np.isfinite(score['objective']):continue
            index=len(candidates);saved[f'q_{index}']=path
            candidates.append(dict(goal_id=pose['target_id'],source=label,spatial_cost=spatial,**score))
            print(f"CORRIDOR goal={pose['target_id']} {label} score={score['objective']:.3f}",flush=True)
        write_json(out/'seed_progress.json',candidates)
    np.savez_compressed(out/'candidate_paths.npz',**saved)
    write_json(out/'candidate_scores.json',candidates)
    write_json(out/'path_report.json',dict(status='SEARCH_SEEDS_ONLY',scene_relative=source['scene_relative'],
        mesh_fingerprint=read_json(checker.scene_dir/'scene_report.json')['mesh_fingerprint']))
    print('CORRIDOR_SEEDS_COMPLETE',len(candidates),flush=True)


if __name__=='__main__':main()
