from pathlib import Path
import sys
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
scene = root / "assets" / "robot" / "robotstudio_so101" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

print(f"scene={scene}")
print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} geoms={model.ngeom} meshes={model.nmesh}")
print("joints:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
    lo, hi = np.rad2deg(model.jnt_range[i])
    print(f"  {name}: {lo:.3f} .. {hi:.3f} deg")

# Neutral configuration; run the actuator model without a user command.
if model.nu:
    data.ctrl[:] = 0.0
for _ in range(1000):
    mujoco.mj_step(model, data)

if not np.isfinite(data.qpos).all():
    raise SystemExit("SO101_CHECK_FAILED: non-finite qpos")
print(f"sim_time={data.time:.3f}s contacts={data.ncon}")
print("SO101_CHECK_OK")
