"""Render baseline and selected nozzle lines from saved actual simulation states."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from plant_scene import read_json
from replay_watering_viewer import load_replay,apply_frame,add_path_line
from render_trajectory_replay import font


def main():
    root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selected',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    _,r,scene,arrays=load_replay(root,args.selected)
    _,br,bscene,barrays=load_replay(root,args.baseline)
    if r['mesh_fingerprint']!=br['mesh_fingerprint']or r['scene_sha256']!=br['scene_sha256']:
        raise ValueError('Comparison requires the same scene and model')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    apply_frame(model,data,arrays,len(arrays['time_s'])-1)
    im=Image.new('RGB',(1920,810),'#111c2c');d=ImageDraw.Draw(im)
    d.text((24,13),'SO-101 | actual nozzle route comparison',font=font(28),fill='white')
    d.text((24,54),'Gray: accepted natural route   |   Blue: selected side arc, elbow lift and early wrist turn',font=font(22),fill='#c7e4ef')
    option=mujoco.MjvOption();mujoco.mjv_defaultOption(option);option.geomgroup[3:5]=0;option.sitegroup[:]=0
    with mujoco.Renderer(model,height=610,width=960)as renderer:
        for index,(azimuth,elevation)in enumerate(((135,-25),(55,-30))):
            cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
            cam.lookat[:]=[.23,.06,.25];cam.distance=.95;cam.azimuth=azimuth;cam.elevation=elevation
            renderer.update_scene(data,camera=cam,scene_option=option)
            add_path_line(renderer.scene,barrays['simulated_tip_m'],rgba=(.55,.56,.60,1.),radius=.0017)
            add_path_line(renderer.scene,arrays['simulated_tip_m'],rgba=(.05,.7,1.,1.),radius=.0016)
            im.paste(Image.fromarray(renderer.render()),(index*960,100))
    arc=lambda a:np.linalg.norm(np.diff(a,axis=0),axis=1).sum()*1000
    d.text((24,729),f"Nozzle arc: {arc(barrays['simulated_tip_m']):.1f} -> {arc(arrays['simulated_tip_m']):.1f} mm     Duration: {br['duration_s']:.2f} -> {r['duration_s']:.2f} s",font=font(24),fill='white')
    d.text((24,770),f"Actual geometric contact cost: {br['contact_risk']['simulated']['persistence_cost']:.4f} -> {r['contact_risk']['simulated']['persistence_cost']:.4f}  |  Same endpoints and weights; presentation preference, no global-optimum claim",font=font(19),fill='#c7e4ef')
    args.output.parent.mkdir(parents=True,exist_ok=True);im.save(args.output)
    print('DEMONSTRATION_COMPARISON_RENDERED',args.output)


if __name__=='__main__':main()
