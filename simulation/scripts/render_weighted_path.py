"""Render the saved continuous nozzle route and the held-tool close-up."""
from pathlib import Path
import argparse
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plant_scene import read_json
from plan_watering_path import ARM_JOINTS
from replay_watering_viewer import add_path_line


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path', type=Path)
    args = ap.parse_args()
    directory = args.path or root/read_json(root/'configs/latest_path.json')['output']
    report = read_json(directory/'path_report.json')
    model = mujoco.MjModel.from_xml_path(str(root/report['scene_relative']))
    config = read_json((root/report['scene_relative']).parent/'config_snapshot.json')
    data = mujoco.MjData(model)
    arrays = np.load(directory/'nozzle_path_line.npz')
    points, qs = arrays['tip_m'], arrays['q_arm_rad']
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt)
    opt.geomgroup[3:5] = 0; opt.sitegroup[:] = 0
    face = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(face), 23) if face.exists() else ImageFont.load_default()
    sheet = Image.new('RGB', (1800, 1380), '#142133')
    with mujoco.Renderer(model, height=620, width=900) as renderer:
        for panel, (azimuth, fraction) in enumerate(((135, 0.), (135, .55), (135, 1.), (20, 1.))):
            k = int(fraction*(len(qs)-1))
            for name, q in zip(ARM_JOINTS, qs[k]): data.qpos[model.joint(name).qposadr] = q
            data.qpos[model.joint('gripper').qposadr] = report['gripper_rad']
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [.22, .10, .17]; camera.distance = .78
            camera.azimuth = azimuth; camera.elevation = -24
            renderer.update_scene(data, camera=camera, scene_option=opt)
            add_path_line(renderer.scene, points, radius=.0015)
            frame = Image.fromarray(renderer.render())
            x, y = (panel%2)*900, (panel//2)*690
            sheet.paste(frame, (x, y+70))
            draw = ImageDraw.Draw(sheet)
            draw.text((x+18, y+14), f"SO-101 | {['Start', 'Approach', 'Watering pose', 'Reverse view'][panel]}", fill='white', font=font)
            draw.text((x+18, y+42), 'Blue line: complete nozzle-tip route', fill='#5dceff', font=font)
    sheet.save(directory/'nozzle_path_line.png')
    # Tool detail at the configured start, with the plant hidden for visibility.
    data.qpos[:] = 0
    for name, q in zip(ARM_JOINTS, config.get('planning', {}).get('start_q_arm_rad', [0]*5)):
        data.qpos[model.joint(name).qposadr] = q
    data.qpos[model.joint('gripper').qposadr] = report['gripper_rad']
    mujoco.mj_forward(model, data)
    for gid in range(model.ngeom):
        if model.geom(gid).name.startswith('plant_class_'): model.geom_rgba[gid, 3] = 0
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    handle = data.geom_xpos[model.geom('nozzle_grasp_handle').id]
    tip = data.site_xpos[model.site('nozzle_tip').id]
    cam.lookat[:] = .55*handle + .45*tip
    length_m = config['nozzle'].get('total_length_mm', config['nozzle'].get('length_mm',60))/1000
    cam.distance = max(.18, length_m*2.5); cam.azimuth = 90; cam.elevation = -15
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        Image.fromarray(renderer.render()).save(directory/'nozzle_grasp_closeup.png')
    print(f"WEIGHTED_PATH_RENDERED: {directory/'nozzle_path_line.png'}")


if __name__ == '__main__': main()
