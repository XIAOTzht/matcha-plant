"""Automatic detail cameras from spatial tiles and SAM3's pot mask."""
import argparse
import json
from pathlib import Path
import plantseg as p
import numpy as np
from PIL import Image
from refine_detail import camera_crop, render_camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=896)
    args=parser.parse_args()
    source=json.loads((args.base/'manifest.json').read_text(encoding='utf8'))
    inference=json.loads((args.predictions/'inference_config.json').read_text(encoding='utf8'))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'views').mkdir(exist_ok=True)
    recipe=dict(revision=1,size=args.size,base_manifest_sha256=p.sha256(args.base/'manifest.json'),
                prediction_cache_key=inference['cache_key'])
    if (args.output/'manifest.json').exists():
        old=json.loads((args.output/'manifest.json').read_text(encoding='utf8'))
        if old.get('crop_recipe')!=recipe:raise ValueError('Detail camera recipe changed')
        for v in old['views']:
            for suffix in ['.png','_rays.npz']:
                if not (args.output/'views'/f"{v['name']}{suffix}").exists():
                    raise ValueError('Incomplete camera cache; use a new output directory')
        return
    if p.sha256(source['input'])!=source['input_sha256']:raise ValueError('Input changed')
    xyz,tri,rgb=p.mesh_data(source['input']);scene=p.scene_from_arrays(xyz,tri)
    cameras=[]
    for parent in source['views'][::2]:
        name=parent['name']
        with np.load(args.base/'views'/f'{name}_rays.npz') as ray:foreground=ray['face_id']>=0
        y,x=np.where(foreground)
        if not len(x):continue
        left,right=x.min(),x.max();top,bottom=y.min(),y.max()
        side=float(max(right-left,bottom-top)*.62)
        boxes=[]
        for ry in [.28,.72]:
            for rx in [.28,.72]:
                cx=left+(right-left)*rx;cy=top+(bottom-top)*ry
                boxes.append(('uniform_detail',[cx-side/2,cy-side/2,cx+side/2,cy+side/2]))
        with np.load(args.predictions/'views'/f'{name}_scores.npz') as scores:
            if str(scores['cache_key'])!=inference['cache_key']:raise ValueError('Stale scores')
            pot=(scores['scores'][...,4]>.4)&foreground
        y,x=np.where(pot)
        if len(x)>100:
            side=float(max(np.ptp(x),np.ptp(y))*1.35)
            cx=(x.min()+x.max())/2;cy=y.min()+.25*side
            boxes.append(('pot_opening',[cx-side/2,cy-side/2,cx+side/2,cy+side/2]))
        for kind,box in boxes:
            cam=camera_crop(parent,box,source['size'])
            cam.update(name=f'detail_{len(cameras):03d}',parent=name,kind=kind)
            image,fid,bary=render_camera(scene,tri,rgb,cam,args.size)
            Image.fromarray(image).save(args.output/'views'/f"{cam['name']}.png")
            np.savez_compressed(args.output/'views'/f"{cam['name']}_rays.npz",face_id=fid,barycentric=bary)
            cameras.append(cam)
            print('Rendered',cam['name'],kind,flush=True)
    p.save_json(args.output/'manifest.json',dict(source,views=cameras,size=args.size,
        source_directory=str(args.base.resolve()),sam3_crop_selection=True,crop_recipe=recipe))


if __name__=='__main__':main()
