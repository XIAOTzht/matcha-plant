"""Compare saved dynamics states at the same main-arm passage and final pose."""
from pathlib import Path
import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from plant_scene import write_json
from render_trajectory_replay import font
from replay_watering_viewer import load_replay, apply_frame, add_path_line


def render_panel(model, arrays, index, lookat, camera_distance, azimuth, line=False):
    data = mujoco.MjData(model)
    apply_frame(model, data, arrays, index)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = camera_distance
    camera.azimuth = azimuth
    camera.elevation = -18
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3:5] = 0
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, width=900, height=620) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        if line:
            add_path_line(renderer.scene, arrays["simulated_tip_m"])
        return Image.fromarray(renderer.render())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_dir, old_report, old_scene, old = load_replay(root, args.before)
    new_dir, new_report, new_scene, new = load_replay(root, args.after)
    if old_report["mesh_fingerprint"] != new_report["mesh_fingerprint"]:
        raise ValueError("Comparison needs the same plant geometry")
    models = [mujoco.MjModel.from_xml_path(str(scene)) for scene in (old_scene, new_scene)]
    before_i = int(np.argmax(old["simulated_instantaneous_exposure"]))
    # Shoulder pan/lift/elbow identify the same passage. The wrist is deliberately
    # free to differ, so it is excluded from correspondence matching.
    first3 = old["q_actual_rad"][before_i,:3]
    after_i = int(np.argmin(np.linalg.norm(new["q_actual_rad"][:,:3]-first3,axis=1)))
    lookat = old["simulated_tip_m"][before_i] + [0,0,.035]
    canvas = Image.new("RGB", (1800, 1560), "#111c2c")
    draw = ImageDraw.Draw(canvas)
    for column, (label, model, report, arrays, index) in enumerate(zip(
            ("Before: wrist turns too late", "After: wrist turns before the flower"),
            models, (old_report,new_report), (old,new), (before_i,after_i))):
        x = column*900
        for row, (frame_i, center, distance, azimuth) in enumerate(
                ((index, lookat, .36, 315),
                 (len(arrays["time_s"])-1, [.25,.17,.19], .72, 135))):
            y = row*780
            panel = render_panel(model,arrays,frame_i,center,distance,azimuth,line=row==1)
            canvas.paste(panel,(x,y+160))
            draw.text((x+18,y+12), label if row==0 else "Complete route and final soil approach",
                      font=font(23),fill="white")
            draw.text((x+18,y+46),f"Saved dynamics | t={arrays['time_s'][frame_i]:.3f} s",
                      font=font(20),fill="#d9e9f4")
            q_deg = np.rad2deg(arrays["q_actual_rad"][frame_i,:5])
            draw.text((x+18,y+78),
                      f"Wrist pitch: {q_deg[3]:.1f} deg | wrist roll: {q_deg[4]:.1f} deg",
                      font=font(20),fill="#ffcd78")
            text=(f"Instantaneous soft risk: {arrays['simulated_instantaneous_exposure'][frame_i]:.2f}"
                  if row==0 else
                  f"Total persistence cost: {report['contact_risk']['simulated']['persistence_cost']:.3f}")
            draw.text((x+18,y+111),text,font=font(20),fill="#d9e9f4")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output)
    write_json(args.output.with_suffix(".json"),dict(
        before=old_dir.relative_to(root).as_posix(),
        after=new_dir.relative_to(root).as_posix(),
        before_sample=before_i,after_sample=after_i,
        before_time_s=float(old["time_s"][before_i]),
        after_time_s=float(new["time_s"][after_i]),
        main_arm_match_error_rad=float(np.linalg.norm(new["q_actual_rad"][after_i,:3]-first3)),
        before_q_deg=np.rad2deg(old["q_actual_rad"][before_i]).tolist(),
        after_q_deg=np.rad2deg(new["q_actual_rad"][after_i]).tolist(),
        before_cost=old_report["contact_risk"]["simulated"]["persistence_cost"],
        after_cost=new_report["contact_risk"]["simulated"]["persistence_cost"],
        before_overlap_duration_s=old_report["contact_risk"]["simulated"]["overlap_duration_s"],
        after_overlap_duration_s=new_report["contact_risk"]["simulated"]["overlap_duration_s"],
        interpretation="Identical plant and weights; time-sampled geometry proxies, not measured physical damage."))
    print(f"ATTITUDE_COMPARISON_RENDERED: {args.output}",flush=True)


if __name__ == "__main__":
    main()
