from pathlib import Path
import struct
import zlib
import mujoco
import numpy as np

root = Path(__file__).resolve().parents[1]
scene = root / "assets" / "robot" / "robotstudio_so101" / "scene.xml"
out = root / "outputs" / "so101_preview.png"
out.parent.mkdir(parents=True, exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
if model.nq >= 6:
    data.qpos[1:6] = np.deg2rad([25.0, -35.0, 65.0, 25.0, 5.0])
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera=-1)
rgb = np.asarray(renderer.render(), dtype=np.uint8)

def chunk(kind, payload):
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)

h, w, c = rgb.shape
raw = b''.join(b'\x00' + row.tobytes() for row in rgb)
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(raw, 9))
png += chunk(b'IEND', b'')
out.write_bytes(png)
print(f"PREVIEW_OK: {out} ({w}x{h})")
