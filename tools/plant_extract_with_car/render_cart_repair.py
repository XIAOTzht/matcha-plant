"""Optional Open3D raycast comparison with identical cameras for changed meshes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
tools=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(tools/'plant_semantics'),str(tools/'plant_semantics_with_car')]
import plantseg as p
from refine_detail import render_camera
from with_car_semantics import mesh


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','coordinate-system','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--overview',action='store_true')
    args=ap.parse_args();c=json.loads(args.coordinate_system.read_text(encoding='utf-8-sig'))
    models=[]
    for path in (args.before,args.after):
        ply,xyz,tri=mesh(path);q=(xyz-np.array(c['board_center_mm']))@np.array(c['board_frame_axes'])
        colors=np.column_stack([ply['vertex'].data[k]for k in ('red','green','blue')]).astype(float)/255
        select=np.ones(len(q),bool) if args.overview else ply['vertex'].data['label']==6
        faces=tri[select[tri].all(1)];ids=np.unique(faces);inverse=np.full(len(q),-1,np.int32);inverse[ids]=np.arange(len(ids))
        q=q[ids];faces=inverse[faces];colors=colors[ids]
        normal=np.cross(q[faces[:,1]]-q[faces[:,0]],q[faces[:,2]]-q[faces[:,0]])
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        models.append((p.scene_from_arrays(q,faces),q,faces,colors,normal))
    angles=[(40,-28),(0,-90),(0,0)] if not args.overview else [(35,20),(140,20)]
    size=850;canvas=Image.new('RGB',(size*2,len(angles)*(size+44)),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',24)
    except OSError:font=ImageFont.load_default(size=24)
    for row,(az,el)in enumerate(angles):
        camera=None
        for col,(scene,q,tri,color,normal)in enumerate(models):
            if camera is None:im,fid,_,camera=p.render_view(scene,q,tri,color,az,el,size)
            else:im,fid,_=render_camera(scene,tri,color,camera,size)
            valid=fid>=0;light=-np.array(camera['direction'])+np.array([.2,.4,.2]);light/=np.linalg.norm(light)
            shade=.52+.48*np.abs(normal[fid[valid]]@light)
            im[valid]=np.rint(im[valid]*shade[:,None]).astype('u1')
            canvas.paste(Image.fromarray(im),(col*size,row*(size+44)+44))
            draw.text((col*size+15,row*(size+44)+10),'Before' if col==0 else'After: inferred flat underside',fill='black',font=font)
    canvas.save(args.output,quality=95);print('CART_REPAIR_COMPARISON',args.output,flush=True)


if __name__=='__main__':main()
