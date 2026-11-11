"""Optimize soil point and tool attitude jointly using the actual plant weights.

The wrist is not held at its initial angle. Position IK with two free joint
coordinates samples the redundant posture family; only a downward spray cone
is required. Finite sampling does not certify an optimal endpoint.
"""
from pathlib import Path
from datetime import datetime
import argparse
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from plant_scene import read_json,write_json
from plan_watering_path import CollisionChecker,ARM_JOINTS
from spray_target import SprayRayQuery


def spray_hits_soil(checker,q):
    checker.set_q(q);sid=checker.model.site('nozzle_tip').id
    if not hasattr(checker,'spray_query'):
        checker.spray_query=SprayRayQuery(checker.xyz,checker.faces,checker.face_labels)
    ray=checker.spray_query.first_hit(checker.data.site_xpos[sid],-checker.data.site_xmat[sid].reshape(3,3)[:,2])
    return bool(ray['first_hit_class']==checker.policy['watering_target']['class_id']
                and .5<ray['range_mm']<40.),ray


def position_cone_residual(q,checker,target,anchor,minimum_cosine):
    checker.set_q(q)
    site=checker.model.site('nozzle_tip').id
    tip=checker.data.site_xpos[site]
    down=checker.data.site_xmat[site].reshape(3,3)[2,2]
    # Position is much tighter than nullspace preference. Outside the allowed
    # cone the hinge residual steers the spray down without fixing its yaw.
    return np.r_[(tip-target)/.0001, max(0.,minimum_cosine-down)/.002,
                 .002*(q-anchor)/(checker.q_hi-checker.q_lo)]


def endpoint_search(checker,targets,output,seeds_per_target=48,max_tilt_deg=20.,seed=20261024):
    rng=np.random.default_rng(seed);model=checker.model
    minimum_cosine=np.cos(np.deg2rad(max_tilt_deg))
    # A slight interior bias gives the dynamics check numerical room at 20 deg.
    solve_cosine=np.cos(np.deg2rad(max_tilt_deg-1.))
    candidates=[];summaries=[];started=time.perf_counter()
    start=np.asarray(checker.config.get('planning',{}).get('start_q_arm_rad',[0]*5),dtype=float)
    known=[]
    for ti,entry in enumerate(targets):
        target=np.asarray(entry['nozzle_tip_candidate_world_m'])
        pan=-np.arctan2(target[1],max(target[0]-.06,1e-6))
        anchors=[start,np.array([pan,-.4,.1,.35,1.62]),np.array([pan,.5,-1.,.5,1.62])]
        anchors+=known[:6]
        anchors+=list(rng.uniform(checker.q_lo,checker.q_hi,(max(0,seeds_per_target-len(anchors)),5)))
        solutions=[]
        for anchor in anchors:
            result=least_squares(position_cone_residual,np.clip(anchor,checker.q_lo,checker.q_hi),
                bounds=(checker.q_lo,checker.q_hi),args=(checker,target,anchor,solve_cosine),
                max_nfev=150,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            q=result.x
            if any(np.linalg.norm(q-old)<.03 for old in solutions):continue
            checker.set_q(q);site=model.site('nozzle_tip').id
            error=float(np.linalg.norm(checker.data.site_xpos[site]-target))
            down=float(checker.data.site_xmat[site].reshape(3,3)[2,2])
            if error>.00025 or down<minimum_cosine:continue
            solutions.append(q.copy())
            valid,detail=checker.check(q,details=True)
            if not valid:continue
            candidates.append(dict(id=ti,target_id=int(entry['id']),q_arm_rad=q.tolist(),
                tip_m=checker.data.site_xpos[site].tolist(),target_m=target.tolist(),
                soil_surface_world_m=entry['soil_surface_world_m'],source_face_index=int(entry['source_face_index']),
                position_error_m=error,spray_down_cosine=down,
                height_above_soil_mm=1000*(checker.data.site_xpos[site][2]-entry['soil_surface_world_m'][2]),
                **detail))
        group=[c for c in candidates if c['target_id']==entry['id']]
        group.sort(key=lambda c:c['risk_proxy'])
        if group:known=[np.asarray(c['q_arm_rad']) for c in group[:3]]+known[:3]
        summaries.append(dict(target_id=int(entry['id']),postures=len(group),
                              minimum_endpoint_exposure=min((c['risk_proxy'] for c in group),default=None)))
        print(f"POSE_TARGET {ti+1}/{len(targets)} valid={len(group)} best={summaries[-1]['minimum_endpoint_exposure']}",flush=True)
        write_json(output/'pose_search_progress.json',dict(targets=summaries,elapsed_s=time.perf_counter()-started))
    candidates.sort(key=lambda c:c['risk_proxy'])
    write_json(output/'pose_candidates.json',dict(scene_relative=checker.scene.relative_to(checker.root).as_posix(),
        max_tilt_deg=max_tilt_deg,elapsed_s=time.perf_counter()-started,summaries=summaries,entries=candidates))
    return candidates


def refine_low_risk_poses(checker,targets,candidates,output,rounds=160,seed=123):
    """Search neighboring real soil faces and redundant arm poses together."""
    from scipy.spatial import cKDTree
    rng=np.random.default_rng(seed)
    soil_ids=np.flatnonzero(checker.face_labels==checker.policy['watering_target']['class_id'])
    centers=checker.xyz[checker.faces[soil_ids]].mean(axis=1)
    tree=cKDTree(centers[:,:2])
    lo,hi=checker.q_lo,checker.q_hi
    chosen=[]
    for c in candidates:
        if c['target_id'] not in [x['target_id'] for x in chosen]:chosen.append(c)
        if len(chosen)>=4:break
    for base in chosen:
        best=base
        for it in range(rounds):
            # Randomize all joints including wrist flex/roll; project back to
            # a valid real-soil target, not a free point outside the flowerpot.
            anchor=np.clip(np.asarray(best['q_arm_rad'])+rng.normal(0,[.06,.16,.22,.20,.22]),lo,hi)
            xy=np.asarray(best['soil_surface_world_m'])[:2]+rng.normal(0,.002,2)
            nearest=int(tree.query(xy)[1]);soil=centers[nearest];target=soil+[0,0,.01]
            result=least_squares(position_cone_residual,anchor,bounds=(lo,hi),
                args=(checker,target,anchor,np.cos(np.deg2rad(19.))),max_nfev=90,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            checker.set_q(result.x);sid=checker.model.site('nozzle_tip').id
            error=float(np.linalg.norm(checker.data.site_xpos[sid]-target))
            down=float(checker.data.site_xmat[sid].reshape(3,3)[2,2])
            if error>.00025 or down<np.cos(np.deg2rad(20)):continue
            valid,detail=checker.check(result.x,details=True)
            if not valid or detail['risk_proxy']>=best['risk_proxy']:continue
            target_id=len(targets)
            entry=dict(id=target_id,source_face_index=int(soil_ids[nearest]),
                soil_surface_world_m=soil.tolist(),nozzle_tip_candidate_world_m=target.tolist(),height_above_soil_mm=10.)
            targets.append(entry)
            best=dict(id=target_id,target_id=target_id,q_arm_rad=result.x.tolist(),
                tip_m=checker.data.site_xpos[sid].tolist(),target_m=target.tolist(),
                soil_surface_world_m=soil.tolist(),source_face_index=int(soil_ids[nearest]),
                position_error_m=error,spray_down_cosine=down,height_above_soil_mm=1000*(checker.data.site_xpos[sid][2]-soil[2]),**detail)
            candidates.append(best)
            print(f"POSE_REFINE target={target_id} risk={best['risk_proxy']:.5f}",flush=True)
            if best['risk_proxy']<1e-6:break
    candidates.sort(key=lambda c:c['risk_proxy'])
    write_json(checker.scene_dir/'watering_candidates.json',dict(status='GEOMETRIC_CANDIDATES_WITH_REFINED_SOIL_FACES',units='m',entries=targets))
    write_json(output/'refined_pose_candidates.json',dict(scene_relative=checker.scene.relative_to(checker.root).as_posix(),
               max_tilt_deg=20.,entries=candidates))
    return candidates


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=40);ap.add_argument('--tilt-deg',type=float,default=20.)
    ap.add_argument('--refine-from',type=Path);ap.add_argument('--refine-rounds',type=int,default=160)
    args=ap.parse_args();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    checker=CollisionChecker(root,args.scene.resolve())
    targets=read_json(checker.scene_dir/'watering_candidates.json')['entries']
    if args.refine_from:
        candidates=read_json(args.refine_from)['entries']
        candidates=refine_low_risk_poses(checker,targets,candidates,out,args.refine_rounds)
    else:
        candidates=endpoint_search(checker,targets,out,args.seeds,args.tilt_deg)
    print('POSE_SEARCH_COMPLETE',len(candidates),flush=True)


if __name__=='__main__':main()
