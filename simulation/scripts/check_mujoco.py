import sys
from pathlib import Path
import mujoco
xml=Path(sys.argv[1])
model=mujoco.MjModel.from_xml_path(str(xml))
data=mujoco.MjData(model)
z0=float(data.qpos[2])
for _ in range(1000): mujoco.mj_step(model,data)
z1=float(data.qpos[2])
print(f'MODEL_OK bodies={model.nbody} geoms={model.ngeom} joints={model.njnt}')
print(f'SIM_OK steps=1000 initial_z={z0:.6f} final_z={z1:.6f} time={data.time:.3f}')
if not (z1<z0 and z1>0.05): raise SystemExit('SMOKE_TEST_FAILED')
