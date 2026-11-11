"""Check the local plant segmentation runtime and model assets."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if sys.platform == 'win32' or __import__('os').environ.get('PLANTSEG_USE_VENDOR') == '1':
    sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'third_party' / 'MobileSAM'))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    required = ['numpy', 'scipy', 'PIL', 'open3d', 'torch', 'torchvision',
                'transformers', 'cv2', 'sklearn', 'plyfile', 'mobile_sam']
    status = {}
    for name in required:
        try:
            module = importlib.import_module(name)
            status[name] = {'ok': True, 'version': getattr(module, '__version__', None)}
        except Exception as exc:
            status[name] = {'ok': False, 'error': str(exc)}
    assets = {
        'grounding_dino': ROOT / 'models' / 'grounding-dino-tiny' / 'model.safetensors',
        'clipseg': ROOT / 'models' / 'clipseg-rd64-refined' / 'model.safetensors',
        'mobile_sam': ROOT / 'models' / 'mobile_sam_parallel.pt',
    }
    asset_status = {name: {'exists': path.exists(), 'path': str(path),
                           'bytes': path.stat().st_size if path.exists() else 0}
                    for name, path in assets.items()}
    result = {'python': sys.executable, 'packages': status, 'assets': asset_status}
    if args.input:
        from plantseg import mesh_data
        try:
            xyz, tri, rgb = mesh_data(args.input)
            result['input'] = {'path': str(args.input.resolve()), 'vertices': len(xyz),
                               'faces': len(tri), 'bounds': [xyz.min(0).tolist(), xyz.max(0).tolist()],
                               'has_rgb': True}
        except Exception as exc:
            result['input'] = {'path': str(args.input.resolve()), 'ok': False, 'error': str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    package_ok = all(item['ok'] for item in status.values()) and all(item['exists'] for item in asset_status.values())
    input_ok = not args.input or result.get('input', {}).get('has_rgb', False)
    raise SystemExit(0 if package_ok and input_ok else 1)


if __name__ == '__main__':
    main()
