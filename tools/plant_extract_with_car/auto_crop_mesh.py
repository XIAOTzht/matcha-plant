#!/usr/bin/env python3
"""Crop a colored plant-and-pot mesh using only a PLY, on CPU.

Heuristic for one potted plant on a locally planar table/floor. It detects a
compact cluster of botanical colors, fits a nearby support plane, and crops an
oriented box above it. It is not a general 3-D semantic segmentation model.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from auto_extract_plant import load_mesh


def color_seeds(rgb, mode):
    """HSV-like, deterministic rules; pot itself need not have botanical colors."""
    c = np.asarray(rgb, dtype=float)/255.0
    high, low = c.max(axis=1), c.min(axis=1)
    delta = high-low
    saturation = delta/np.maximum(high, 1e-8)
    # Green includes fairly dark leaves; reject near-gray surfaces.
    r, g, b = c.T
    green = (g > r*1.10) & (g > b*1.07) & (saturation > .18) & (g > .08)
    # Red/magenta/pink/yellow/blue are potential flowers. Greenish/cyan objects
    # are deliberately not flower seeds. These rules remain an explicit prior.
    pink = (r > g*1.3) & (b > g*1.05) & (r > .25)
    red = (r > g*1.7) & (r > b*1.4) & (r > .3)
    yellow = (r > b*1.6) & (g > b*1.5) & (r > .4) & (g > .32)
    blue = (b > r*1.35) & (b > g*1.2) & (b > .3)
    flowers = (pink | red | yellow | blue) & (saturation > .25)
    return green if mode == "green" else flowers if mode == "flowers" else green | flowers


def voxel_clusters(points, voxel):
    """26-neighbor components on occupied voxels; avoid dense point graphs."""
    cells = np.floor((points-points.min(axis=0))/voxel).astype(np.int64)
    occupied, inverse = np.unique(cells, axis=0, return_inverse=True)
    pairs = cKDTree(occupied).query_pairs(1.75, output_type="ndarray")
    if len(pairs):
        graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8),
                            (pairs[:,0], pairs[:,1])), shape=(len(occupied),len(occupied))).tocsr()
        n, voxel_labels = connected_components(graph, directed=False)
    else:
        n, voxel_labels = len(occupied), np.arange(len(occupied))
    labels = voxel_labels[inverse]
    return labels, np.bincount(labels, minlength=n), occupied, inverse


def detect_seed_cluster(points, rgb, mode, voxel_ratio, candidate_rank):
    flags = color_seeds(rgb, mode)
    seeds = points[flags]
    if len(seeds) < 50:
        raise ValueError("Too few botanical color samples. Try --seed-mode flowers or green; "
                         "uncolored/gray plants are not supported by this detector.")
    # Quantiles prevent a few far-away reconstruction artifacts from setting scale.
    extent = np.linalg.norm(np.diff(np.percentile(points,[5,95],axis=0),axis=0)[0])
    if extent <= 0:
        raise ValueError("Degenerate scene extent")
    voxel = extent*voxel_ratio
    labels, counts, _, _ = voxel_clusters(seeds,voxel)
    order = np.argsort(counts)[::-1]
    order = [i for i in order if counts[i] >= max(40, len(points)*0.0002)]
    if not order or candidate_rank > len(order):
        raise ValueError("No sufficiently large colored cluster found; adjust --voxel-ratio")
    summaries = []
    for i in order[:10]:
        p = seeds[labels==i]
        summaries.append({"samples":int(len(p)),"center":np.median(p,axis=0).tolist()})
    chosen = order[candidate_rank-1]
    core = seeds[labels==chosen]
    lo, hi = np.percentile(core,[2,98],axis=0)
    span = np.linalg.norm(hi-lo)
    if span <= voxel*2:
        raise ValueError("Color seed cluster is too small to estimate a plant")
    # Include nearby detached flowers/leaves as well as the dominant seed cluster.
    tree = cKDTree(core[::max(1,len(core)//10000)])
    distance = tree.query(seeds, workers=1)[0]
    group = seeds[distance < span*.30]
    warnings = []
    if candidate_rank == 1 and len(order)>1 and counts[order[1]] > counts[chosen]*.6:
        warnings.append("Multiple large color clusters: target is ambiguous; inspect detection_preview.jpg "
                        "or choose --candidate-rank 2")
    return group, span, summaries, warnings, voxel


def fit_support_plane(points, seeds, span, rng, trials):
    center = np.median(seeds,axis=0)
    near = points[np.linalg.norm(points-center,axis=1)<span*1.7]
    if len(near)<500:
        raise ValueError("Insufficient geometry near the plant to identify a support plane")
    if len(near)>16000:
        near=near[rng.choice(len(near),16000,replace=False)]
    threshold=span*.008
    remaining=near.copy()
    candidates=[]
    for _ in range(4):
        if len(remaining)<200:
            break
        best_count=0; best=None
        for _ in range(trials):
            sample=remaining[rng.choice(len(remaining),3,replace=False)]
            normal=np.cross(sample[1]-sample[0],sample[2]-sample[0])
            length=np.linalg.norm(normal)
            if length<1e-12:
                continue
            normal/=length; offset=-sample[0]@normal
            inliers=np.abs(remaining@normal+offset)<threshold
            count=int(inliers.sum())
            if count>best_count:
                best_count=count; best=(normal,offset,inliers)
        if best is None or best_count<150:
            break
        normal,offset,inliers=best
        fit=remaining[inliers]
        _,_,vh=np.linalg.svd(fit-fit.mean(axis=0),full_matrices=False)
        normal=vh[-1];offset=-fit.mean(axis=0)@normal
        if np.median(seeds@normal+offset)<0:
            normal=-normal;offset=-offset
        heights=seeds@normal+offset
        hlo,hmid,hhi=np.percentile(heights,[2,50,98])
        # A support plane must lie underneath most of the colored canopy, have
        # nearby canopy, and contain meaningful area in BOTH tangent directions.
        _,sv,_=np.linalg.svd(fit-fit.mean(axis=0),full_matrices=False)
        wide_enough=sv[1]/np.sqrt(len(fit)) > span*.13
        valid=(hlo>span*.04 and hmid<span*1.7 and hhi<span*2.5 and wide_enough)
        candidate={"normal":normal,"offset":float(offset),"inliers":int(best_count),
                   "canopy_heights": [float(hlo),float(hmid),float(hhi)], "valid":bool(valid)}
        candidates.append(candidate)
        remaining=remaining[~inliers]
    valid=[c for c in candidates if c["valid"]]
    if not valid:
        raise ValueError("No convincing table/floor under the plant. This method needs a visible "
                         "support plane; it cannot safely crop this scene automatically.")
    plane=max(valid,key=lambda c:c["inliers"])
    return plane,candidates


def estimate_frame(seeds, plane, padding, bottom_clearance):
    up=plane["normal"]
    center=np.median(seeds,axis=0)
    origin=center-(center@up+plane["offset"])*up
    helper=np.eye(3)[np.argmin(np.abs(up))]
    axis_x=np.cross(up,helper);axis_x/=np.linalg.norm(axis_x)
    axis_z=np.cross(axis_x,up)
    basis=np.column_stack((axis_x,axis_z))
    planar=(seeds-origin)@basis
    cov=np.cov(planar.T)
    _,ev=np.linalg.eigh(cov)
    axis_x=basis@ev[:,-1]
    axis_x/=np.linalg.norm(axis_x)
    axis_z=np.cross(axis_x,up)
    frame=np.column_stack((axis_x,up,axis_z))
    q=(seeds-origin)@frame
    lo,hi=np.percentile(q,[.5,99.5],axis=0)
    height=hi[1]
    if height<=0:
        raise ValueError("Invalid estimated plant height")
    margin=height*padding
    # No fixed coordinates: scale, rotation and bounds come from this mesh.
    lower=np.array([lo[0]-margin,height*bottom_clearance,lo[2]-margin])
    upper=np.array([hi[0]+margin,hi[1]+height*.08,hi[2]+margin])
    return origin,frame,lower,upper


def crop_faces(xyz,faces,origin,frame,lower,upper,chunk_size):
    keep=np.zeros(len(xyz),dtype=bool)
    for start in range(0,len(xyz),chunk_size):
        q=(xyz[start:start+chunk_size]-origin)@frame
        keep[start:start+len(q)]=((q>=lower)&(q<=upper)).all(axis=1)
    batches=[]
    for start in range(0,len(faces),chunk_size):
        tri=faces[start:start+chunk_size]
        batches.append(tri[keep[tri].all(axis=1)])
    return np.concatenate(batches),keep


def filter_components(faces,min_faces):
    used,inv=np.unique(faces,return_inverse=True)
    tri=inv.reshape(-1,3)
    edges=np.concatenate((tri[:,:2],tri[:,1:],tri[:,[2,0]]))
    graph=coo_matrix((np.ones(len(edges),dtype=np.uint8),(edges[:,0],edges[:,1])),
                     shape=(len(used),len(used))).tocsr()
    count,labels=connected_components(graph,directed=False)
    fl=labels[tri[:,0]];sizes=np.bincount(fl,minlength=count)
    keep=sizes[fl]>=min_faces
    return faces[keep], {"before":int(count),"after":int((sizes>=min_faces).sum()),
                         "min_faces":min_faces,"largest_face_counts":sorted(sizes.tolist(),reverse=True)[:20]}


def save_result(ply,faces,output):
    used=np.unique(faces)
    remap=np.full(len(ply["vertex"].data),-1,dtype=np.int32)
    remap[used]=np.arange(len(used),dtype=np.int32)
    ff=np.empty(len(faces),dtype=[("vertex_indices","i4",(3,))])
    ff["vertex_indices"]=remap[faces]
    PlyData([PlyElement.describe(np.array(ply["vertex"].data[used]),"vertex"),
             PlyElement.describe(ff,"face")],text=False,byte_order="<").write(str(output))
    return used


def draw_points(xyz,rgb,path,title,frame,origin,limits=None):
    q=(xyz-origin)@frame
    cell=(600,650)
    out=Image.new("RGB",(cell[0]*3,cell[1]),"#252a32")
    directions=[(0,1),(2,1),(0,2)]
    for i,(a,b) in enumerate(directions):
        tile=Image.new("RGB",cell,"#252a32");draw=ImageDraw.Draw(tile)
        lo,hi=np.percentile(q[:,[a,b]],[.2,99.8],axis=0) if limits is None else (limits[0][[a,b]],limits[1][[a,b]])
        scale=min((cell[0]-50)/max(hi[0]-lo[0],1e-9),(cell[1]-90)/max(hi[1]-lo[1],1e-9))
        xy=(q[:,[a,b]]-(lo+hi)/2)*scale+np.array(cell)/2
        xy[:,1]=cell[1]-xy[:,1]
        # Draw sampled mesh vertices, not synthesized surfaces.
        for j in np.argsort(q[:,3-a-b]):
            x,y=xy[j]
            if 0<=x<cell[0] and 35<=y<cell[1]-20:
                draw.point((int(x),int(y)),fill=tuple(map(int,rgb[j])))
        draw.text((12,12),title+f" / axes {a},{b}",fill="white")
        out.paste(tile,(i*cell[0],0))
    out.save(path)


def run(args):
    start=time.monotonic()
    if args.samples<5000 or args.ransac_trials<20 or args.chunk_size<1:
        raise ValueError("Insufficient samples/trials or invalid chunk size")
    if not 0<args.voxel_ratio<.1 or not 0<args.padding<1 or not 0<=args.bottom_clearance<.2:
        raise ValueError("Invalid voxel, padding or clearance ratio")
    if args.candidate_rank<1 or args.min_component_faces<1:
        raise ValueError("Invalid candidate/component setting")
    args.input=args.input.resolve()
    if args.output_dir is None:
        args.output_dir=args.input.parent/(args.input.stem+"_plant_auto_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out=args.output_dir.resolve()
    if args.input.is_relative_to(out):
        raise ValueError("Use a separate output directory to protect the source mesh")
    out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ("plant.ply","report.json","plant_preview.jpg","detection_preview.jpg")):
        raise FileExistsError("Output already contains results; choose a fresh --output-dir")
    print(f"Loading: {args.input}",flush=True)
    ply,xyz,faces=load_mesh(args.input)
    vertices=ply["vertex"].data
    if not all(k in vertices.dtype.names for k in ("red","green","blue")):
        raise ValueError("This detector requires red/green/blue vertex colors")
    rng=np.random.default_rng(args.random_seed)
    ids=rng.choice(len(xyz),min(args.samples,len(xyz)),replace=False)
    sample=xyz[ids]
    rgb=np.column_stack([vertices[k][ids] for k in ("red","green","blue")])
    if rgb.min()<0 or rgb.max()>255 or (rgb.dtype.kind=="f" and rgb.max()<=1):
        raise ValueError("Expected vertex RGB colors in 0..255")
    print(f"Finding colored plant candidates in {len(sample):,} samples...",flush=True)
    seeds,span,candidates,warnings,voxel=detect_seed_cluster(sample,rgb,args.seed_mode,args.voxel_ratio,args.candidate_rank)
    print(f"Seed cluster: {len(seeds):,} samples. Fitting nearby support planes...",flush=True)
    plane,planes=fit_support_plane(sample,seeds,span,rng,args.ransac_trials)
    origin,frame,lower,upper=estimate_frame(seeds,plane,args.padding,args.bottom_clearance)
    print("Cropping original mesh above support plane...",flush=True)
    cropped,_=crop_faces(xyz,faces,origin,frame,lower,upper,args.chunk_size)
    if len(cropped)<100:
        raise ValueError("Fewer than 100 triangles survived automatic crop; inspect input/parameters")
    cropped,components=filter_components(cropped,args.min_component_faces)
    if len(cropped)<100:
        raise ValueError("Too few triangles after component filtering")
    used=save_result(ply,cropped,out/"plant.ply")
    preview_ids=rng.choice(used,min(180000,len(used)),replace=False)
    preview_rgb=np.column_stack([vertices[k][preview_ids] for k in ("red","green","blue")])
    draw_points(xyz[preview_ids],preview_rgb,out/"plant_preview.jpg","Plant and pot",frame,origin)
    # Wider view shows what was removed and helps spot wrong object selection.
    q=(sample-origin)@frame
    delta=upper-lower
    local=((q>=lower-delta*.45)&(q<=upper+delta*.45)).all(axis=1)
    inside=((q>=lower)&(q<=upper)).all(axis=1)
    detection_colors=np.where(inside[:,None],rgb,np.array([190,70,65],dtype=np.uint8))
    draw_points(sample[local],detection_colors[local],out/"detection_preview.jpg",
                "Crop region: original colors / outside: red",frame,origin,
                (lower-delta*.45,upper+delta*.45))
    if len(cropped)/len(faces)>.85:
        warnings.append("Most input faces survived; input may already be isolated or crop may be too broad")
    report={"input":str(args.input),"output":str(out/"plant.ply"),
            "input_vertices":len(xyz),"input_faces":len(faces),
            "output_vertices":len(used),"output_faces":len(cropped),
            "seed_samples":len(seeds),"candidates":candidates,"voxel_size":voxel,
            "frame_origin":origin.tolist(),"frame_axes":frame.tolist(),
            "crop_lower":lower.tolist(),"crop_upper":upper.tolist(),
            "support_planes":[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()} for p in planes],
            "components":components,"warnings":warnings,
            "parameters":{k:getattr(args,k) for k in ("seed_mode","voxel_ratio","candidate_rank","padding",
                                                       "bottom_clearance","min_component_faces","random_seed")},
            "elapsed_seconds":round(time.monotonic()-start,2),
            "method":"Colored seed cluster + local RANSAC support plane + oriented box crop. "
                     "Heuristic, no semantic model. Original vertex properties and world coordinates preserved. "
                     "Face attributes/UV textures not exported. No retraining, filling or smoothing."}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    for warning in warnings:
        print("WARNING:",warning,flush=True)
    print(f"AUTO_CROP_FINISHED: {out/'plant.ply'}\n"
          f"{len(used):,} vertices, {len(cropped):,} faces; {report['elapsed_seconds']}s",flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path,help="Complete colored MAtCha triangle mesh PLY")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--seed-mode",choices=["auto","green","flowers"],default="auto")
    p.add_argument("--candidate-rank",type=int,default=1,help="Select a colored cluster, largest first")
    p.add_argument("--samples",type=int,default=200000)
    p.add_argument("--voxel-ratio",type=float,default=.006)
    p.add_argument("--padding",type=float,default=.14,help="Horizontal padding / estimated plant height")
    p.add_argument("--bottom-clearance",type=float,default=.012,help="Cut above support plane / estimated height")
    p.add_argument("--min-component-faces",type=int,default=20,help="Use 1 to keep all detached fragments")
    p.add_argument("--ransac-trials",type=int,default=400)
    p.add_argument("--chunk-size",type=int,default=500000)
    p.add_argument("--random-seed",type=int,default=42)
    return p.parse_args()


if __name__=="__main__":
    try:
        run(cli())
    except Exception as error:
        print(f"AUTO_CROP_FAILED: {error}",file=sys.stderr,flush=True)
        sys.exit(1)
