"""Export labeled preview PNG and animated GIF from saved simulated states."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from replay_watering_viewer import apply_frame, load_replay, add_path_line
from plant_scene import write_json, read_json


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def render_frame(renderer, model, data, arrays, index, width, height):
    apply_frame(model, data, arrays, index)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.23, .10, .17]; camera.distance = .83; camera.azimuth = 135; camera.elevation = -25
    option = mujoco.MjvOption(); mujoco.mjv_defaultOption(option)
    option.geomgroup[3:5] = 0; option.sitegroup[5] = 0
    renderer.update_scene(data, camera=camera, scene_option=option)
    # Complete continuous route is visible from the first frame.
    add_path_line(renderer.scene, arrays["simulated_tip_m"], rgba=(.05,.65,1.,.95))
    if index>1:
        add_path_line(renderer.scene, arrays["simulated_tip_m"][:index+1],rgba=(.05,.95,.95,1.),radius=.0013)
    image = Image.fromarray(renderer.render())
    canvas = Image.new("RGB", (width, height+114), "#111c2c"); canvas.paste(image, (0, 114))
    d = ImageDraw.Draw(canvas)
    t = arrays["time_s"][index]; total = arrays["time_s"][-1]
    hard_gap = 1000*arrays["simulated_hard_clearance_min_m"][index]
    exposure = arrays["simulated_instantaneous_exposure"][index]
    d.text((18, 12), f"SO-101 | timed simulation replay | {t:.2f} / {total:.2f} s", font=font(22), fill="white")
    d.text((18, 43), f"Conservative hard gap: {hard_gap:.2f} mm    Soft exposure proxy: {exposure:.1f}",
           font=font(18), fill="#d9e9f4")
    d.text((18, 72), "Blue line: full nozzle path | leaf/flower/stem weighted soft contact | hard objects forbidden",
           font=font(15), fill="#ffcd78")
    d.rectangle((18, 101, width-18, 105), fill="#33435c")
    d.rectangle((18, 101, 18+int((width-36)*t/total), 105), fill="#2fb8f4")
    return canvas


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    if not 1 <= args.fps <= 60:
        raise ValueError("FPS must be between 1 and 60")
    directory, report, scene, arrays = load_replay(root, args.replay)
    nozzle = read_json(scene.parent/'config_snapshot.json')['nozzle']
    length_mm = nozzle.get('total_length_mm', nozzle.get('length_mm', 60))
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    width, height = 960, 610
    time = arrays["time_s"]
    frames = []
    sample_times = np.linspace(0, time[-1], max(2, int(np.ceil(time[-1]*args.fps))+1))
    indices = np.minimum(len(time)-1, np.searchsorted(time, sample_times))
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for k, index in enumerate(indices):
            frames.append(render_frame(renderer, model, data, arrays, int(index), width, height))
            if k % 25 == 0:
                print(f"RENDER {k+1}/{len(indices)}", flush=True)
    durations = [int(round(1000/args.fps))]*len(frames)
    frames[0].save(directory / "trajectory_replay.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False)
    sheet = Image.new("RGB", (width*2, (height+114)*2), "#111c2c")
    picked = [0, int(.45*(len(frames)-1)), int(.7*(len(frames)-1)), len(frames)-1]
    for k, i in enumerate(picked):
        sheet.paste(frames[i], ((k%2)*width, (k//2)*(height+114)))
    sheet.save(directory / "trajectory_preview.png")
    frames[-1].save(directory / "nozzle_path_line.png")
    # An end-pose close-up exposes the millimetre soil gap that is hard to see
    # from the whole-arm camera. Render unmodified saved dynamics states.
    tip = arrays["simulated_tip_m"][-1]
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = tip + [0, 0, .012]
    camera.distance = .20; camera.azimuth = 315; camera.elevation = -18
    option = mujoco.MjvOption(); mujoco.mjv_defaultOption(option)
    option.geomgroup[3:5] = 0; option.sitegroup[:] = 0
    apply_frame(model, data, arrays, len(time)-1)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        closeup = Image.fromarray(renderer.render())
    canvas = Image.new("RGB", (960, 800), "#111c2c"); canvas.paste(closeup, (0, 80))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"Nozzle end pose | soil height: {report['final_height_above_soil_mm']:.2f} mm", font=font(22), fill="white")
    draw.text((18, 44), f"Saved simulated pose | soil stays a hard obstacle | adjustable {length_mm:g} mm nozzle", font=font(17), fill="#d9e9f4")
    canvas.save(directory / "nozzle_soil_closeup.png")
    pointer = root / "configs/latest_replay.json"
    import json
    latest = json.loads(pointer.read_text(encoding="utf-8-sig"))
    if latest["output"] == directory.relative_to(root).as_posix():
        for key, name in (("preview", "trajectory_preview.png"), ("animation", "trajectory_replay.gif"),
                          ("closeup", "nozzle_soil_closeup.png"), ("trajectory", "timed_trajectory.npz")):
            latest[key] = (directory / name).relative_to(root).as_posix()
        latest["guide"] = "docs/TIMED_REPLAY.md"
        write_json(pointer, latest)
    print(f"REPLAY_RENDERED: {directory / 'trajectory_replay.gif'}", flush=True)


if __name__ == "__main__":
    main()
