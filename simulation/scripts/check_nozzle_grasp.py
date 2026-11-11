"""Check attached nozzle against native proxies and original finger triangles.

Visual mesh checks sample the entire cylindrical solid, not its convex hull.
They are diagnostic sampled distances, not an intersection certificate.
"""
from pathlib import Path
import argparse
import igl
import mujoco
import numpy as np
from plant_scene import read_json,write_json
from plan_watering_path import ARM_JOINTS,allowed_grasp_contact


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True)
    args=ap.parse_args();scene=args.scene.resolve()
    config=read_json(scene.parent/'config_snapshot.json')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    for name,q in zip(ARM_JOINTS,config.get('planning',{}).get('start_q_arm_rad',[0]*5)):
        data.qpos[model.joint(name).qposadr]=q
    grip=config['nozzle']['gripper_rad'];data.qpos[model.joint('gripper').qposadr]=grip
    mujoco.mj_forward(model,data)
    tool=model.body('placeholder_nozzle').id
    ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==tool and model.geom_contype[i]]
    others=[i for i in range(model.ngeom) if model.geom_bodyid[i]!=tool
            and (model.geom_contype[i] or model.geom_conaffinity[i]) and model.geom(i).name!='floor']
    native=[]
    for a in ids:
        for b in others:
            distance=float(mujoco.mj_geomDistance(model,data,a,b,1,None))
            native.append(dict(tool=model.geom(a).name,other=model.geom(b).name or f'geom_{b}',
                               gap_mm=distance*1000,allowed_grasp=allowed_grasp_contact(model,a,b,distance,grip,config['nozzle'])))
    illegal=[r for r in native if r['gap_mm'] < -.0001 and not r['allowed_grasp']]
    fingers={model.body(n).id for n in ('gripper','moving_jaw_so101_v1')}
    meshes=[]
    for i in range(model.ngeom):
        if model.geom_bodyid[i] not in fingers or model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mid=int(model.geom_dataid[i]);va=model.mesh_vertadr[mid];fa=model.mesh_faceadr[mid]
        vertices=np.asarray(model.mesh_vert[va:va+model.mesh_vertnum[mid]],dtype=float)
        vertices=vertices@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
        faces=np.asarray(model.mesh_face[fa:fa+model.mesh_facenum[mid]],dtype=np.int64)
        tree=igl.AABB();tree.init(vertices,faces)
        meshes.append((i,vertices,faces,tree))
    sampled=[]
    for a in ids:
        radius,half=model.geom_size[a,:2]
        angles=np.linspace(0,2*np.pi,max(40,int(np.ceil(2*np.pi*radius/.0005))),endpoint=False)
        axial=np.linspace(-half,half,max(2,int(np.ceil(2*half/.0005))+1))
        rings=np.linspace(0,radius,max(2,int(np.ceil(radius/.0005))+1))
        z,r,theta=np.meshgrid(axial,rings,angles,indexing='ij')
        local=np.c_[r.ravel()*np.cos(theta.ravel()),r.ravel()*np.sin(theta.ravel()),z.ravel()]
        world=local@data.geom_xmat[a].reshape(3,3).T+data.geom_xpos[a]
        for b,vertices,faces,tree in meshes:
            # Exact distances at sample points, against original mesh triangles.
            sq,_,_=tree.squared_distance(vertices,faces,world)
            sampled.append(dict(tool=model.geom(a).name,other=f'geom_{b}',
                                mesh=model.mesh(int(model.geom_dataid[b])).name,
                                sampled_solid_points=len(world),sampled_min_surface_gap_mm=float(np.sqrt(max(0,sq.min()))*1000)))
    report=dict(status='NATIVE_GRASP_CHECK_PASSED' if not illegal else 'NATIVE_GRASP_CHECK_FAILED',
                gripper_rad=grip,native_min_unpermitted_gap_mm=min(r['gap_mm'] for r in native if not r['allowed_grasp']),
                native_contacts=[r for r in native if r['gap_mm']<1],illegal_contacts=illegal,
                sampled_original_finger_mesh_distances=sampled,
                limitation='Original visual triangle distances are sampled on a <=0.5 mm cylinder-solid grid; this is not a watertight or continuous intersection proof. Native collision proxies govern planning. Rigid attachment does not model frictional grip stability.')
    write_json(scene.parent/'nozzle_grasp_geometry.json',report)
    print(dict(status=report['status'],min_unpermitted_gap_mm=report['native_min_unpermitted_gap_mm'],
               min_sampled_finger_surface_gap_mm=min(r['sampled_min_surface_gap_mm'] for r in sampled)),flush=True)
    if illegal:raise ValueError('Unexpected nozzle/robot intersection')


if __name__=='__main__':main()
