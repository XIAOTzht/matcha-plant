"""Linux runtime and model check for server V3 inference."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "MobileSAM"))

required = ["numpy", "scipy", "PIL", "open3d", "torch", "torchvision",
            "transformers", "cv2", "sklearn", "plyfile", "mobile_sam"]
failed = []
for name in required:
    try:
        mod = __import__(name)
        print(f"OK {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"FAIL {name}: {exc}")
for path in [ROOT / "models/grounding-dino-tiny/model.safetensors",
             ROOT / "models/clipseg-rd64-refined/model.safetensors",
             ROOT / "models/mobile_sam_parallel.pt"]:
    if path.is_file() and path.stat().st_size > 0:
        print(f"OK asset {path} {path.stat().st_size} bytes")
    else:
        failed.append(("asset", str(path)))
        print(f"FAIL asset {path}")
try:
    import torch
    print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception:
    pass
if failed:
    raise SystemExit(1)
print("PLANTSEM_SERVER_RUNTIME_OK")
