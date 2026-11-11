"""Depth-tested before/after weight surfaces; render actual mesh RGB only.

Optional preview tool requires Open3D. No classifiers or physical models load.
Camera framing selects the plant by semantic IDs, never by a plant-specific box.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS/'plant_semantics'))
sys.path.insert(0,str(TOOLS/'plant_semantics_with_car'))
import plantseg as p
from with_car_semantics import mesh


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path,required=True)
    ap.add_argument('--coordinate-system',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    old,xyz,tri=mesh(args.before);new,px,pt=mesh(args.after)
    if not np.array_equal(xyz,px)or not np.array_equal(tri,pt):
        raise ValueError('Comparison requires identical original geometry')
    coord=json.loads(args.coordinate_system.read_text(encoding='utf-8-sig'))
    upright=((xyz-np.asarray(coord['board_center_mm']))@np.asarray(coord['board_frame_axes'])).astype('f4')
    soft=np.isin(new['vertex'].data['label'],[1,2,3])
    faces=tri[soft[tri].all(1)]
    ids=np.unique(faces)
    inverse=np.full(len(xyz),-1,np.int32);inverse[ids]=np.arange(len(ids))
    faces=inverse[faces];points=upright[ids]
    scene=p.scene_from_arrays(points,faces)
    rgb=[]
    for model in (old,new):
        rgb.append(np.column_stack([model['vertex'].data[k]for k in ('red','green','blue')])[ids].astype('f4')/255)
    canvas=Image.new('RGB',(2000,2100),'white')
    from refine_detail import render_camera
    for row,az in enumerate((0,135)):
        image,fid,bary,camera=p.render_view(scene,points,faces,rgb[0],az,22,1000)
        for col,colors in enumerate(rgb):
            if col:image,_,_=render_camera(scene,faces,colors,camera,1000)
            canvas.paste(Image.fromarray(image),(col*1000,row*1050+40))
            ImageDraw.Draw(canvas).text((col*1000+15,row*1050+12),
                'Before: previous weight surface' if col==0 else
                'After: revised organ labels and weights',fill='black')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output,quality=95)
    print('WEIGHT_SURFACE_COMPARISON',args.output,flush=True)


if __name__=='__main__':main()
