#!/usr/bin/env python3
"""Estimate a rigid turntable axis from foreground matches with known 30deg steps.

This is angle-constrained self-calibration, NOT a measured camera calibration.
Object coordinates equal camera-0 coordinates. Radius is an arbitrary unit.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def axis_geometry(params):
    a, b, bearing = params[:3]
    axis = np.array([a, -1.0, b]); axis /= np.linalg.norm(axis)
    front = np.array([0., 0., 1.])
    front -= axis * (front @ axis); front /= np.linalg.norm(front)
    right = np.cross(front, axis); right /= np.linalg.norm(right)
    center = np.cos(bearing)*front + np.sin(bearing)*right
    return axis, center


def make_poses(params, sign, count=12):
    axis, center = axis_geometry(params)
    poses = []
    for i in range(count):
        R = Rotation.from_rotvec(axis * np.deg2rad(sign*30*i)).as_matrix()
        E = np.eye(4); E[:3,:3] = R; E[:3,3] = center - R@center
        poses.append(E)
    return np.array(poses)


def skew(x):
    return np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])


def residuals(params, data, sign, focal, pp):
    axis, center = axis_geometry(params)
    K = np.array([[focal,0,pp[0]],[0,focal,pp[1]],[0,0,1.]])
    ki = np.linalg.inv(K)
    residual = []
    for d in data:
        R = Rotation.from_rotvec(axis * np.deg2rad(sign*30*(d['j']-d['i']))).as_matrix()
        E = skew(center - R@center) @ R
        F = ki.T @ E @ ki
        p, q = d['p'], d['q']
        fp = p @ F.T; fq = q @ F
        err = np.sum(q * fp, axis=1)
        den = np.sqrt(fp[:,0]**2+fp[:,1]**2+fq[:,0]**2+fq[:,1]**2+1e-30)
        residual.append(err/den)
    return np.concatenate(residual)


def collect_matches(images, masks, max_size=1200):
    cv2.setRNGSeed(314159)
    features=[]
    sift=cv2.SIFT_create(nfeatures=6000, contrastThreshold=.025)
    scale = None
    for i,(src,mask_path) in enumerate(zip(images,masks)):
        # imdecode handles non-ASCII Windows paths.
        im=cv2.imdecode(np.fromfile(str(src),np.uint8),cv2.IMREAD_COLOR)
        mask=cv2.imdecode(np.fromfile(str(mask_path),np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None or mask.shape != im.shape[:2]:
            raise ValueError(f'Image/mask missing or different sizes: {src}')
        h,w=im.shape[:2]
        s=min(1.,max_size/max(h,w))
        target=(int(round(w*s)),int(round(h*s)))
        if scale is not None and target != scale[0]:
            raise ValueError('All 12 images must have the same dimensions')
        scale=(target,s,(w,h))
        im=cv2.resize(im,target,interpolation=cv2.INTER_AREA)
        mask=cv2.resize(mask,target,interpolation=cv2.INTER_NEAREST)
        mask=(mask>180).astype(np.uint8)*255
        # Stay inside foreground; avoid silhouette edge matching.
        mask=cv2.erode(mask,np.ones((7,7),np.uint8))
        # Exclude the repetitive pot texture from axis estimation. Flowers/leaves
        # provide less ambiguous correspondences; the pot remains in reconstruction.
        foreground_y=np.flatnonzero(mask.any(axis=1))
        if len(foreground_y):
            cut=int(foreground_y[0]+.68*(foreground_y[-1]-foreground_y[0]))
            mask[cut:]=0
        kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask)
        if desc is None or len(kp)<30:
            raise RuntimeError(f'Too few foreground features in {src.name}')
        features.append((np.array([k.pt for k in kp]),desc))
        print(f'Foreground features {i:03d}: {len(kp)}',flush=True)
    bf=cv2.BFMatcher(cv2.NORM_L2)
    data=[]
    for i in range(12):
        for step in (1,2):
            j=(i+step)%12
            pi,di=features[i]; pj,dj=features[j]
            forward=bf.knnMatch(di,dj,k=2)
            backward=bf.knnMatch(dj,di,k=2)
            reverse={m.queryIdx:m.trainIdx for pair in backward if len(pair)==2
                     for m,n in [pair] if m.distance<.72*n.distance}
            matches=[m for pair in forward if len(pair)==2 for m,n in [pair]
                     if m.distance<.72*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
            if len(matches)<12:
                continue
            p=np.array([pi[m.queryIdx] for m in matches]); q=np.array([pj[m.trainIdx] for m in matches])
            F,inliers=cv2.findFundamentalMat(p,q,cv2.FM_RANSAC,1.5,.999)
            if F is None or inliers is None:
                continue
            ok=inliers.ravel().astype(bool)
            if ok.sum()<12:
                continue
            p=p[ok]; q=q[ok]
            # Limit the influence of repetitive pot patterns in any single pair.
            if len(p)>180:
                ids=np.random.default_rng(i*12+j).choice(len(p),180,replace=False)
                p=p[ids]; q=q[ids]
            data.append({'i':i,'j':j,'p':np.c_[p,np.ones(len(p))],'q':np.c_[q,np.ones(len(q))]})
            print(f'Matches {i:03d}->{j:03d}: {len(p)}',flush=True)
    covered={d[k] for d in data for k in ('i','j')}
    if len(covered)!=12 or len(data)<12:
        raise RuntimeError('Foreground matching did not reliably cover all 12 views; no poses exported')
    return data,scale


def validate_candidate(params, data, sign, focal, pp):
    poses=make_poses(params,sign)
    K=np.array([[focal,0,pp[0]],[0,focal,pp[1]],[0,0,1.]])
    stats=[]; points=[]; positive=0; total=0
    errors=np.abs(residuals(params,data,sign,focal,pp)); offset=0
    for d in data:
        n=len(d['p']); e=errors[offset:offset+n]; offset+=n
        p,q=d['p'][:,:2],d['q'][:,:2]
        P=K@poses[d['i']][:3]; Q=K@poses[d['j']][:3]
        x=cv2.triangulatePoints(P,Q,p.T.copy(),q.T.copy()).T
        valid=np.abs(x[:,3])>1e-10
        xyz=x[:,:3]/np.where(valid,x[:,3],1)[:,None]
        zi=(xyz@poses[d['i']][:3,:3].T+poses[d['i']][:3,3])[:,2]
        zj=(xyz@poses[d['j']][:3,:3].T+poses[d['j']][:3,3])[:,2]
        good=valid & np.isfinite(xyz).all(1) & (zi>0) & (zj>0) & (e<2.5)
        positive+=int(good.sum()); total+=n
        if good.any(): points.append(xyz[good])
        stats.append({'i':d['i'],'j':d['j'],'matches':n,'median_error_px':float(np.median(e)),
                      'positive_depth_inliers':int(good.sum())})
    return {'median_sampson_px':float(np.median(errors)),
            'p90_sampson_px':float(np.percentile(errors,90)),
            'positive_depth_inlier_fraction':positive/max(1,total),
            'positive_depth_inliers':positive,'pair_stats':stats},points


def estimate_axis(data, focal, pp):
    # Repeated pot motifs generate mutually consistent but incorrect pair matches.
    # Fit a *single* rigid turntable model with global robust sampling, rather than
    # treating all pairwise fundamental-matrix inliers as correct correspondences.
    candidates=[]
    rng=np.random.default_rng(20261016)
    for sign in (1,-1):
        for tilt in (-.35,0.,.35):
            fit=least_squares(residuals,[0.,tilt,0.],args=(data,sign,focal,pp),
                              bounds=([-.5,-.9,-.5],[.5,.9,.5]),loss='soft_l1',f_scale=1.5,
                              max_nfev=180)
            stats,points=validate_candidate(fit.x,data,sign,focal,pp)
            score=stats['median_sampson_px']+10*(1-stats['positive_depth_inlier_fraction'])
            candidates.append((score,fit.x,sign,stats,points))
        sampled=[]
        for trial in range(120):
            ids=rng.choice(len(data),size=4,replace=False)
            subset=[]
            for idx in ids:
                d=data[idx]; k=rng.integers(len(d['p']))
                subset.append({'i':d['i'],'j':d['j'],'p':d['p'][k:k+1],'q':d['q'][k:k+1]})
            fit=least_squares(residuals,[0.,-.2,0.],args=(subset,sign,focal,pp),
                              bounds=([-.5,-.9,-.5],[.5,.9,.5]),loss='soft_l1',f_scale=1.,max_nfev=50)
            e=np.abs(residuals(fit.x,data,sign,focal,pp))
            sampled.append((float(np.minimum(e**2,2.5**2).mean()),fit.x))
        for _,x0 in sorted(sampled,key=lambda k:k[0])[:4]:
            x=x0.copy()
            for _ in range(3):
                subset=[]
                for d in data:
                    e=np.abs(residuals(x,[d],sign,focal,pp)); ok=e<3.
                    if ok.any(): subset.append({**d,'p':d['p'][ok],'q':d['q'][ok]})
                fit=least_squares(residuals,x,args=(subset,sign,focal,pp),
                                  bounds=([-.5,-.9,-.5],[.5,.9,.5]),loss='soft_l1',f_scale=1.,max_nfev=100)
                x=fit.x
            stats,points=validate_candidate(x,data,sign,focal,pp)
            score=stats['median_sampson_px']+10*(1-stats['positive_depth_inlier_fraction'])
            candidates.append((score,x,sign,stats,points))
    candidates.sort(key=lambda c:c[0])
    _,params,sign,stats,points=candidates[0]
    return params,sign,stats,points


def load_reference_focal(sfm, matcha, original_size):
    module_path=matcha/'mast3r'/'colmap'/'read_write_model.py'
    spec=importlib.util.spec_from_file_location('colmap_reader',module_path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sparse=sfm/'sparse'/'0'
    if (sparse/'cameras.bin').is_file():
        cams=module.read_cameras_binary(str(sparse/'cameras.bin'))
    else:
        cams=module.read_cameras_text(str(sparse/'cameras.txt'))
    estimates=[]
    for cam in cams.values():
        f=cam.params[0] if cam.model=='SIMPLE_PINHOLE' else float(np.mean(cam.params[:2]))
        estimates.append(f*max(original_size)/max(cam.width,cam.height))
    f=float(np.median(estimates))
    if not .25*max(original_size)<f<5*max(original_size):
        raise ValueError(f'Implausible reference focal: {f}')
    return f


def export_dataset(out, images, masks, width, height, focal, params, sign, report):
    out.mkdir(parents=True,exist_ok=False)
    sparse=out/'sparse'/'0'; sparse.mkdir(parents=True)
    (out/'images').mkdir(); (out/'masks').mkdir()
    poses=make_poses(params,sign)
    with (sparse/'cameras.bin').open('wb') as f:
        f.write(struct.pack('<QiiQQdddd',1,1,1,width,height,focal,focal,width/2,height/2))
    lines=['# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]',f'1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}']
    (sparse/'cameras.txt').write_text('\n'.join(lines)+'\n')
    image_lines=['# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME; observations on next line']
    with (sparse/'images.bin').open('wb') as f:
        f.write(struct.pack('<Q',12))
        for i,(src,mask,E) in enumerate(zip(images,masks,poses)):
            q=Rotation.from_matrix(E[:3,:3]).as_quat()[[3,0,1,2]]
            t=E[:3,3]
            f.write(struct.pack('<idddddddi',i+1,*q,*t,1))
            f.write(src.name.encode()+b'\0'+struct.pack('<Q',0))
            image_lines.append(' '.join(map(str,[i+1,*q,*t,1,src.name]))+'\n')
            shutil.copy2(src,out/'images'/src.name)
            shutil.copy2(mask,out/'masks'/mask.name)
    (sparse/'images.txt').write_text('\n'.join(image_lines)+'\n')
    (sparse/'points3D.bin').write_bytes(struct.pack('<Q',0))
    report.update({'width':width,'height':height,'estimated_focal_px':focal,
                   'axis_in_camera0':axis_geometry(params)[0].tolist(),
                   'axis_closest_point_camera0_radius1':axis_geometry(params)[1].tolist(),
                   'object_rotation_sign':sign,'angles_degrees':{p.name:30*i for i,p in enumerate(images)},
                   'world_to_camera':poses.tolist(),
                   'calibration_status':'Angle-constrained estimate, not measured camera calibration. '
                   'Centered principal point, square pixels, no lens distortion; unit turntable radius. '
                   'Axis and rotation sign estimated from foreground SIFT matches.'})
    (out/'turntable_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Human-readable camera distribution for checking before launching training.
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(750,750),'white'); draw=ImageDraw.Draw(im)
    centers=np.linalg.inv(poses)[:,:3,3]; axis,c=axis_geometry(params)
    e1=centers[0]-c; e1/=np.linalg.norm(e1); e2=np.cross(axis,e1)
    draw.ellipse((150,150,600,600),outline='gray',width=2)
    draw.ellipse((368,368,382,382),fill='green')
    for i,point in enumerate(centers):
        uv=375+225*np.array([(point-c)@e1,(point-c)@e2])
        x,y=map(float,uv); draw.line((x,y,375,375),fill='lightgray')
        draw.ellipse((x-6,y-6,x+6,y+6),fill='blue')
        draw.text((x+8,y),f'{i:03d}: {i*30} deg',fill='black')
    draw.text((20,20),'Estimated camera orbit: top view (radius = 1, arbitrary scale)',fill='black')
    # MAtCha treats ANY image in the dataset root as an input-photo directory,
    # skipping images/ and calibration loading. Diagnostics must stay nested.
    (out/'diagnostics').mkdir(exist_ok=True)
    im.save(out/'diagnostics'/'camera_orbit.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-dir',type=Path,required=True)
    p.add_argument('--masks-dir',type=Path,required=True)
    p.add_argument('--reference-sfm',type=Path)
    p.add_argument('--matcha-dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--focal-px',type=float,help='Optional measured/estimated original-image focal')
    p.add_argument('--assumed-elevation',type=float,
                   help='Explicit approximate experiment: use this camera elevation instead of fitting the axis')
    p.add_argument('--rotation-sign',type=int,choices=(-1,1),default=1)
    args=p.parse_args()
    images=[args.images_dir/f'img_{i:03d}.jpg' for i in range(12)]
    masks=[args.masks_dir/f'img_{i:03d}_mask.png' for i in range(12)]
    if args.output_dir.exists(): raise FileExistsError('Use a new dataset output directory')
    for src in images+masks:
        if not src.is_file(): raise FileNotFoundError(src)
    from PIL import Image
    with Image.open(images[0]) as im: size=im.size
    for src in images+masks:
        with Image.open(src) as im:
            if im.size!=size: raise ValueError(f'Image/mask dimensions mismatch: {src}')
    focal=args.focal_px
    if focal is None:
        if args.reference_sfm is None or args.matcha_dir is None:
            raise ValueError('Supply --focal-px or --reference-sfm and --matcha-dir')
        focal=load_reference_focal(args.reference_sfm,args.matcha_dir,size)
    if not np.isfinite(focal) or focal<=0: raise ValueError('Invalid focal')
    if args.assumed_elevation is not None:
        if not -40<=args.assumed_elevation<=60:
            raise ValueError('Assumed elevation outside plausible bounds')
        params=np.array([0.,-np.tan(np.deg2rad(args.assumed_elevation)),0.])
        report={'axis_source':'ASSUMED, NOT CALIBRATED','assumed_camera_elevation_degrees':args.assumed_elevation,
                'focal_source':'user-provided' if args.focal_px else str(args.reference_sfm)}
        export_dataset(args.output_dir,images,masks,*size,focal,params,args.rotation_sign,report)
        print(f'TURNTABLE30_APPROXIMATE_DATASET_READY: {args.output_dir}',flush=True)
        return
    data,(target,scale,size)=collect_matches(images,masks)
    pp=np.array(target)/2
    print(f'Fitting turntable axis; original-image focal = {focal:.2f}px (estimated)',flush=True)
    params,sign,stats,points=estimate_axis(data,focal*scale,pp)
    print(json.dumps(stats,indent=2),flush=True)
    if (stats['median_sampson_px']>2.5 or stats['positive_depth_inlier_fraction']<.55
            or stats['positive_depth_inliers']<150):
        raise RuntimeError('Known-angle fit is unreliable: check real angles, rigidity, masks and intrinsics; no dataset exported')
    # Every image must have evidence consistent with the angular model.
    reliable={s[k] for s in stats['pair_stats'] if s['positive_depth_inliers']>=10
              and s['median_error_px']<3 for k in ('i','j')}
    if len(reliable)!=12:
        raise RuntimeError(f'Only {len(reliable)}/12 views fit the angular model; no dataset exported')
    stats['focal_source']='user-provided' if args.focal_px else str(args.reference_sfm)
    export_dataset(args.output_dir,images,masks,*size,focal,params,sign,stats)
    print(f'TURNTABLE30_DATASET_READY: {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
