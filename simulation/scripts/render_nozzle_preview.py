"""Render the real MJCF nozzle and rear grasp from multiple camera angles."""
from pathlib import Path
import argparse
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plant_scene import read_json
from plan_watering_path import ARM_JOINTS


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    config = read_json(args.scene.parent/'config_snapshot.json')
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    for name,q in zip(ARM_JOINTS,config.get('planning',{}).get('start_q_arm_rad',[0]*5)):
        data.qpos[model.joint(name).qposadr] = q
    data.qpos[model.joint('gripper').qposadr] = config['nozzle']['gripper_rad']
    mujoco.mj_forward(model,data)
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('plant_class_'):
            model.geom_rgba[i,3] = 0
    option = mujoco.MjvOption(); mujoco.mjv_defaultOption(option)
    option.geomgroup[3:5] = 0; option.sitegroup[:] = 0
    handle = data.geom_xpos[model.geom('nozzle_grasp_handle').id].copy()
    tip = data.site_xpos[model.site('nozzle_tip').id].copy()
    length = config['nozzle'].get('total_length_mm',config['nozzle'].get('length_mm'))
    fontfile = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(fontfile),22) if fontfile.exists() else ImageFont.load_default()
    sheet = Image.new('RGB',(1800,860),'#142133')
    with mujoco.Renderer(model,height=720,width=900) as renderer:
        for k,(az,el) in enumerate(((90,-18),(205,-20))):
            camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = .55*handle+.45*tip
            camera.distance=.30; camera.azimuth=az; camera.elevation=el
            renderer.update_scene(data,camera=camera,scene_option=option)
            sheet.paste(Image.fromarray(renderer.render()),(k*900,140))
    draw=ImageDraw.Draw(sheet)
    for k in range(2):
        draw.text((k*900+20,16),f'SO-101 | rear-end grasp | {length:g} mm nozzle',fill='white',font=font)
        draw.text((k*900+20,49),'18 mm grip | 8 mm extension | stepped metal spray head',fill='#d9e9f4',font=font)
        draw.text((k*900+20,82),'Rigid attached placeholder; wrist start roll = 90 degrees',fill='#ffcd78',font=font)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.output)
    print(f'NOZZLE_PREVIEW: {args.output}',flush=True)


if __name__=='__main__':main()
