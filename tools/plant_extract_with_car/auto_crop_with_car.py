#!/usr/bin/env python3
"""Automatic mesh crop for one plant on a BLUE wheeled cart above a flat support.

An explicit color/geometry prior, not semantic segmentation. No fixed world
coordinates or physical scale; the original plant-only crop remains unchanged.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from auto_extract_plant import load_mesh
from auto_crop_mesh import voxel_clusters, filter_components, save_result, draw_points


def masks(rgb):
    r,g,b=np.asarray(rgb,dtype=float).T
    blue=(b>r*1.4)&(b>g*1.15)&(b>60)
    green=(g>r*1.10)&(g>b*1.07)&(g>25)
    pink=(r>g*1.35)&(b>g*1.05)&(r>65)
    # Blue body panels and yellow wheels must not become plant/flower seeds.
    return blue,green|pink


def fit_blue_plane(points,plant,rng,trials=400):
    span=np.linalg.norm(np.diff(np.percentile(points,[1,99],axis=0),axis=0)[0])
    tolerance=span*.006
    fitpoints=points if len(points)<=12000 else points[rng.choice(len(points),12000,replace=False)]
    best=None;best_count=0
    for _ in range(trials):
        tri=fitpoints[rng.choice(len(fitpoints),3,replace=False)]
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
        if length<1e-12:continue
        n/=length;d=-tri[0]@n
        near=np.abs(fitpoints@n+d)<tolerance
        if near.sum()>best_count:best_count=int(near.sum());best=near
    if best is None or best_count<len(fitpoints)*.20:
        raise ValueError('Blue cluster does not contain a convincing planar chassis')
    p=fitpoints[best];origin=p.mean(0)
    _,sv,vh=np.linalg.svd(p-origin,full_matrices=False);up=vh[-1]
    if sv[1]<sv[0]*.18:raise ValueError('Blue plate is too narrow/degenerate')
    if np.median((plant-origin)@up)<0:up=-up
    return origin,up,tolerance


def detect_legacy(points,rgb,rng,voxel_ratio=.006,candidate_rank=1):
    blue,botanical=masks(rgb)
    if blue.sum()<100 or botanical.sum()<50:
        raise ValueError('Need a visible blue cart and green/pink plant colors')
    extent=np.linalg.norm(np.diff(np.percentile(points,[5,95],axis=0),axis=0)[0])
    if extent<=0:raise ValueError('Degenerate scene')
    bp=points[blue];plants=points[botanical]
    labels,counts,*_=voxel_clusters(bp,extent*voxel_ratio)
    options=[]
    for k in np.argsort(counts)[-10:][::-1]:
        if counts[k]<100:continue
        p=bp[labels==k];center=np.median(p,axis=0)
        span=np.linalg.norm(np.diff(np.percentile(p,[1,99],axis=0),axis=0)[0])
        botanical_near=plants[np.linalg.norm(plants-center,axis=1)<span*1.35]
        if len(botanical_near)<50:continue
        try:origin,up,tol=fit_blue_plane(p,botanical_near,rng)
        except ValueError:continue
        heights=(botanical_near-origin)@up
        canopy=botanical_near[(heights>span*.16)&(heights<span*2)]
        if len(canopy)<50:continue
        u=np.eye(3)[np.argmin(abs(up))];u=np.cross(up,u);u/=np.linalg.norm(u)
        w=np.cross(u,up);basis=np.column_stack((u,w))
        _,ev=np.linalg.eigh(np.cov(((p-origin)@basis).T));u=basis@ev[:,-1]
        w=np.cross(u,up);frame=np.column_stack((u,up,w))
        q=(p-origin)@frame
        lo,hi=np.percentile(q,[.2,99.8],axis=0);length=max((hi-lo)[[0,2]])
        cq=(canopy-origin)@frame
        centered=((cq[:,[0,2]]>=lo[[0,2]]-length*.12)&
                  (cq[:,[0,2]]<=hi[[0,2]]+length*.12)).all(axis=1)
        canopy=canopy[centered]
        if len(canopy)<50:continue
        options.append(dict(blue=p,canopy=canopy,origin=origin,frame=frame,
                            lo=lo,hi=hi,length=length,score=float(np.sqrt(len(p)*len(canopy)))))
    options.sort(key=lambda v:v['score'],reverse=True)
    if len(options)<candidate_rank:raise ValueError('No blue cart with a plant above it found')
    chosen=options[candidate_rank-1]
    chosen['candidates']=[dict(score=o['score'],blue_samples=len(o['blue']),
                              plant_samples=len(o['canopy']),center=o['origin'].tolist()) for o in options]
    return chosen


def detect(points,rgb,rng,voxel_ratio=.006,candidate_rank=1,plate_mm=(135.,110.),
           max_height=4.,max_radius=2.):
    from cart_geometry import detect_geometry
    return detect_geometry(points,rgb,rng,voxel_ratio,candidate_rank,plate_mm,max_height,max_radius)


def support_and_bounds(points,rgb,det,padding=.06,clearance=.004):
    origin,frame,L=det['origin'],det['frame'],det['length']
    q=(points-origin)@frame
    lo,hi=det['lo'].copy(),det['hi'].copy()
    # Look BELOW the blue chassis, never at the upper plant support board.
    local=((q[:,[0,2]]>=lo[[0,2]]-L*.12)&(q[:,[0,2]]<=hi[[0,2]]+L*.12)).all(axis=1)
    below=local&(q[:,1]<-L*.04)&(q[:,1]>-L*.60)
    if below.sum()<200:raise ValueError('No visible support under cart; refusing an unsafe lower crop')
    bins=np.arange(-L*.60, -L*.04+L*.004, L*.004)
    hist,edges=np.histogram(q[below,1],bins=bins)
    peak=int(np.argmax(gaussian_filter1d(hist.astype(float),1)))
    h=(edges[peak]+edges[peak+1])/2
    band=below&(abs(q[:,1]-h)<L*.010)
    p=q[band]
    if len(p)<100:raise ValueError('Weak lower support plane')
    _,sv,vh=np.linalg.svd(p-p.mean(0),full_matrices=False)
    n=vh[-1]
    if n[1]<0:n=-n
    if n[1]<.98 or sv[1]/np.sqrt(len(p))<L*.12:
        raise ValueError('Support below chassis is not broad/parallel enough')
    d=-p.mean(0)@n
    residual=p@n+d
    cut=float(max(clearance*L,np.percentile(residual,99)+L*.002))
    canopy=(det['canopy']-origin)@frame
    clo,chi=np.percentile(canopy,[.2,99.8],axis=0)
    lo[[0,2]]=np.minimum(lo[[0,2]],clo[[0,2]])-padding*L
    hi[[0,2]]=np.maximum(hi[[0,2]],chi[[0,2]])+padding*L
    lo[1]=-L*.65;hi[1]=max(hi[1],chi[1])+L*.05
    return lo,hi,n,float(d),cut,dict(height_peak=float(h),samples=len(p),
        normal_in_cart_frame=n.tolist(),offset=float(d),clearance=cut,
        residual_p99=float(np.percentile(residual,99)))


def run(args):
    start=time.monotonic();out=args.output_dir.resolve();src=args.input.resolve()
    if src.is_relative_to(out):raise ValueError('Output must be a separate subdirectory')
    if args.samples<10000 or not 0<args.voxel_ratio<.1 or not 0<=args.padding<.3 or not 0<=args.clearance<.05:
        raise ValueError('Invalid detector parameters')
    if args.candidate_rank<1 or args.min_component_faces<1:raise ValueError('Invalid rank/component parameter')
    if not args.plate_length_mm >= args.plate_width_mm > 0 or args.max_height_cart_lengths <= .3 or args.max_radius_cart_lengths <= .3:
        raise ValueError('Invalid upper plate dimensions or geometric search limits')
    out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ['plant_car.ply','report.json']):raise FileExistsError('Choose a fresh output directory')
    print('Loading original mesh',flush=True);ply,xyz,faces=load_mesh(src);v=ply['vertex'].data
    if not all(k in v.dtype.names for k in ('red','green','blue')):raise ValueError('RGB vertex colors required')
    rng=np.random.default_rng(42);ids=rng.choice(len(xyz),min(args.samples,len(xyz)),replace=False)
    sample=xyz[ids];rgb=np.column_stack([v[k][ids] for k in ('red','green','blue')])
    if not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255 or (rgb.dtype.kind=='f' and rgb.max()<=1):
        raise ValueError('Expected RGB values in 0..255')
    print('Detecting fixed blue cart and UPPER plate; plant colors unrestricted',flush=True)
    det=detect(sample,rgb,rng,args.voxel_ratio,args.candidate_rank,
               (args.plate_length_mm,args.plate_width_mm),args.max_height_cart_lengths,args.max_radius_cart_lengths)
    print(f"Upper plate verified; geometry proposal: {len(det['canopy']):,} samples, no plant color filter",flush=True)
    lo,hi,n,d,clearance,support=support_and_bounds(sample,rgb,det,args.padding,args.clearance)
    origin,frame=det['origin'],det['frame']
    print('Cropping above the support UNDER the wheels',flush=True)
    keep=np.zeros(len(xyz),bool)
    for st in range(0,len(xyz),500000):
        q=(xyz[st:st+500000]-origin)@frame
        keep[st:st+len(q)]=((q>=lo)&(q<=hi)).all(1)&(q@n+d>clearance)
    chunks=[]
    for st in range(0,len(faces),500000):
        tri=faces[st:st+500000];chunks.append(tri[keep[tri].all(1)])
    cropped=np.concatenate(chunks)
    if len(cropped)<100:raise ValueError('Empty crop')
    before=len(cropped);cropped,components=filter_components(cropped,args.min_component_faces)
    if len(cropped)<100:raise ValueError('Empty crop after component cleanup')
    print(f'Writing {len(cropped):,} triangles',flush=True)
    used=save_result(ply,cropped,out/'plant_car.ply')
    # Preserve exact source indexing for later labels, scaling and reproducibility.
    np.save(out/'source_vertex_indices.npy',used)
    preview=rng.choice(used,min(250000,len(used)),replace=False)
    colors=np.column_stack([v[k][preview] for k in ('red','green','blue')])
    draw_points(xyz[preview],colors,out/'plant_car_preview.jpg','Plant + pot + cart',frame,origin)
    sq=(sample-origin)@frame;delta=hi-lo
    near=((sq>=lo-delta*.2)&(sq<=hi+delta*.2)).all(1)
    col=np.where(keep[ids,None],rgb,np.array([190,65,60],dtype=np.uint8))
    draw_points(sample[near],col[near],out/'detection_preview.jpg','Removed geometry in red',frame,origin)
    kept_blue=masks(rgb)[0]&keep[ids]
    warnings=['Cart/upper-board anchored geometry; flower and leaf colors are unrestricted. Inspect previews.',
              'Nearby objects connected to the plant, severe reconstruction gaps or ambiguous carts can still fail or require review.',
              'At wheel/support contact, the cut may remove a thin tire strip. No hole filling is performed.',
              'Original coordinates preserved. This output is NOT metric calibrated or semantically labeled.']
    report=dict(version='car-crop-0.2.0',input=str(src),output=str(out/'plant_car.ply'),
        input_vertices=len(xyz),input_faces=len(faces),output_vertices=len(used),output_faces=len(cropped),
        faces_before_component_filter=before,components=components,candidates=det['candidates'],
        frame_origin=origin.tolist(),frame_axes=frame.tolist(),crop_lower=lo.tolist(),crop_upper=hi.tolist(),
        lower_support=support,blue_samples_retained=int(kept_blue.sum()),
        detected_blue_samples=len(det['blue']),parameters=vars(args).copy(),warnings=warnings,
        upper_plate_proposal=det['upper_plate'],geometry_growth=det['growth'],
        rejected_cart_candidates=det['rejected_candidates'],
        elapsed_seconds=round(time.monotonic()-start,2),
        method='Fixed blue cart + verified 135:110 upper plate + color-independent board-anchored geometry + lower support plane + padded crop; original coordinates/properties preserved. Bounding crop, not semantic segmentation.')
    report['parameters']={k:str(v) if isinstance(v,Path) else v for k,v in report['parameters'].items()}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"AUTO_CROP_WITH_CAR_SUCCESS: {out/'plant_car.ply'}",flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--samples',type=int,default=300000)
    p.add_argument('--voxel-ratio',type=float,default=.006)
    p.add_argument('--candidate-rank',type=int,default=1)
    p.add_argument('--padding',type=float,default=.06,help='Horizontal margin / chassis span')
    p.add_argument('--clearance',type=float,default=.004,help='Minimum clearance / chassis span')
    p.add_argument('--min-component-faces',type=int,default=20)
    p.add_argument('--plate-length-mm',type=float,default=135.)
    p.add_argument('--plate-width-mm',type=float,default=110.)
    p.add_argument('--max-height-cart-lengths',type=float,default=4.,help='Fail for objects touching this upper search limit')
    p.add_argument('--max-radius-cart-lengths',type=float,default=2.,help='Horizontal search radius; connected background may cause refusal')
    return p.parse_args()


if __name__=='__main__':run(cli())
