"""Automatic leaf boundary repair from mesh cores and multiview SAM masks.

There are no hand-picked boxes, clicks or color thresholds. All masks are
prompted by connected, confident leaf cores from the five-class result. Only
nearby vertices connected by mesh edges can change; other organs need several
independent camera confirmations. Existing leaf labels are never removed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import plantseg as p
from PIL import Image, ImageDraw, ImageFont

REVISION = 3


def surface_graph(xyz, tri):
    from scipy import sparse
    edge=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    a,b=edge.T
    length=np.linalg.norm(xyz[a]-xyz[b],axis=1).clip(1e-9)
    return sparse.coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),
                             shape=(len(xyz),len(xyz))).tocsr()


def leaf_cores(xyz,tri,labels,scores,max_organs=32):
    """Connected leaf patches chosen by surface area, not triangle density."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    area=np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]]),axis=1)*.5
    va=np.zeros(len(xyz),np.float64)
    for j in range(3):np.add.at(va,tri[:,j],area/3)
    a=tri[:,[0,1,2]].ravel();b=tri[:,[1,2,0]].ravel()
    keep=(labels[a]==2)&(labels[b]==2)
    graph=sparse.coo_matrix((np.ones(2*keep.sum(),np.uint8),
                            (np.r_[a[keep],b[keep]],np.r_[b[keep],a[keep]])),
                           shape=(len(xyz),len(xyz))).tocsr()
    _,component=connected_components(graph,directed=False)
    core=(labels==2)&(scores[:,1]>=.65)
    core_area=np.bincount(component,weights=va*core)
    minimum=max(area.sum()*.0005,va[labels==2].sum()*.004)
    selected=np.flatnonzero(core_area>=minimum)
    selected=selected[np.argsort(core_area[selected])[::-1]][:max_organs]
    organs=[]
    for comp in selected:
        members=np.flatnonzero((component==comp)&(labels==2))
        if len(members)<12:continue
        organs.append(dict(vertices=members,core=members[core[members]],
                           area=float(va[members].sum())))
    return organs,va


def prompt_points(core,negative=None):
    from scipy.ndimage import distance_transform_edt,label
    blobs,num=label(core)
    if not num:return None,None,None
    sizes=np.bincount(blobs.ravel());sizes[0]=0
    core=blobs==sizes.argmax()
    if core.sum()<80:return None,None,None
    distance=distance_transform_edt(core)
    work=distance.copy();h,w=core.shape;yy,xx=np.ogrid[:h,:w]
    points=[];labels=[]
    for _ in range(3):
        y,x=np.unravel_index(work.argmax(),work.shape)
        if work[y,x]<2:break
        points.append([x,y]);labels.append(1)
        work*=np.clip(np.sqrt((xx-x)**2+(yy-y)**2)/(min(h,w)*.13),0,1)
    if not points:return None,None,None
    if negative is not None:
        dist=distance_transform_edt(negative)
        for _ in range(3):
            y,x=np.unravel_index(dist.argmax(),dist.shape)
            if dist[y,x]<3:break
            points.append([x,y]);labels.append(0)
            dist*=np.clip(np.sqrt((xx-x)**2+(yy-y)**2)/(min(h,w)*.10),0,1)
    return np.asarray(points,np.float32),np.asarray(labels,np.int32),core


def select_mask(masks,quality,core,valid):
    recall=(masks&core).sum((1,2))/max(core.sum(),1)
    ratio=(masks&valid).sum((1,2))/max(core.sum(),1)
    score=np.clip(quality,0,1)+recall*.3-np.maximum(ratio-2,0)*.12
    acceptable=(quality>=.85)&(recall>=.85)&(ratio<=15)
    if not acceptable.any():return None,None
    score[~acceptable]=-np.inf
    index=int(score.argmax())
    return masks[index]&valid,dict(quality=float(quality[index]),core_recall=float(recall[index]),
                                   expansion_ratio=float(ratio[index]))


def multiview_accept(labels,bud,positive,observed,distance,scale):
    ratio=positive/np.maximum(observed,1)
    return (np.isin(labels,[1,3])&~bud&(positive>=3)&(ratio>=.70)&
            (distance<=scale*.035))


def infer(args,source,manifest,xyz,tri,rgb,data):
    from scipy.sparse.csgraph import dijkstra
    from refine_detail import camera_crop,render_camera
    import torch
    from mobile_sam import sam_model_registry,SamPredictor
    torch.set_num_threads(args.threads)
    labels=data['vertex_labels'];scores=data['class_scores']
    organs,va=leaf_cores(xyz,tri,labels,scores,max_organs=args.max_organs)
    scale=float(np.linalg.norm(np.ptp(xyz,axis=0)))
    config=dict(revision=REVISION,source_labels_sha256=p.sha256(source/'labels.npz'),
                source_mesh_sha256=p.sha256(manifest['input']),sam_sha256=p.sha256(args.sam),
                size=args.size,organ_limit=args.max_organs,view_count=len(manifest['views']),
                max_distance_fraction=.035,min_views=3,min_agreement=.70,
                mask_boundary_tolerance_pixels=2)
    p.save_json(args.output/'boundary_config.json',config)
    if not organs:
        return np.zeros(len(xyz),bool),np.zeros(len(xyz),np.uint16),np.zeros(len(xyz),np.uint16),[],va
    graph=surface_graph(xyz,tri)
    scene=p.scene_from_arrays(xyz,tri)
    cameras=list(manifest['views'])
    # The original above-ground views often see a leaf back only once. Add
    # independent oblique underside directions instead of lowering vote count.
    up_axis=manifest['views'][0]['up_axis']
    for elevation in [-25,-55,-80]:
        for azimuth in [0,90,180,270]:
            _,_,_,camera=p.render_view(scene,xyz,tri,rgb,azimuth,elevation,8,up_axis)
            camera['name']=f'underside_{azimuth}_{abs(elevation)}'
            cameras.append(camera)
    config['cameras']=cameras
    config['view_count']=len(cameras)
    p.save_json(args.output/'boundary_config.json',config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictor=SamPredictor(sam_model_registry['vit_t'](checkpoint=str(args.sam)).to(device).eval())
    repaired=np.zeros(len(xyz),bool)
    best_positive=np.zeros(len(xyz),np.uint16);best_observed=np.zeros(len(xyz),np.uint16)
    records=[]
    for organ_id,organ in enumerate(organs[:args.max_organs]):
        seeds=organ['vertices'];is_core=np.zeros(len(xyz),bool);is_core[organ['core']]=True
        # Geodesic distance follows mesh edges, never a 3D nearest neighbour.
        distance=dijkstra(graph,directed=False,indices=seeds,min_only=True,limit=scale*.035)
        nearby=np.flatnonzero(np.isfinite(distance))
        positive=np.zeros(len(nearby),np.uint16);observed=np.zeros(len(nearby),np.uint16)
        points=xyz[seeds]
        for camera in cameras:
            delta=points-np.asarray(camera['origin'])
            px=delta@np.asarray(camera['right'])/camera['extent']*manifest['size']+manifest['size']/2
            py=-delta@np.asarray(camera['screen_up'])/camera['extent']*manifest['size']+manifest['size']/2
            side=max(np.ptp(px),np.ptp(py))*1.35
            if side<4:continue
            cx=(px.min()+px.max())/2;cy=(py.min()+py.max())/2
            crop=camera_crop(camera,[cx-side/2,cy-side/2,cx+side/2,cy+side/2],manifest['size'])
            image,fid,bary=render_camera(scene,tri,rgb,crop,args.size)
            valid=fid>=0;corners=tri[fid[valid]]
            vertex=corners[np.arange(len(corners)),bary[valid].argmax(1)]
            core=np.zeros(fid.shape,bool);core[valid]=is_core[vertex]
            # Separate confident organs may reject an overbroad SAM candidate.
            # Do not use old stem labels as negative clicks: the false leaf
            # rim itself can already have a confident stem posterior.
            negative=np.zeros(fid.shape,bool)
            vlabels=labels[vertex]
            protected=data['bud_protected'][vertex]|np.isin(vlabels,[4,5])
            protected|=(vlabels==3)&(scores[vertex,2]>.96)&(distance[vertex]>scale*.015)
            negative[valid]=protected
            prompts,prompt_labels,core=prompt_points(core,negative)
            if prompts is None:continue
            predictor.set_image(image)
            masks,quality,_=predictor.predict(point_coords=prompts,point_labels=prompt_labels,multimask_output=True)
            mask,diagnostic=select_mask(masks,quality,core,valid)
            name=f"leaf_{organ_id:02d}_{camera['name']}"
            if mask is None:
                records.append(dict(name=name,accepted=False));continue
            # Small rasterization uncertainty at a silhouette must not leave
            # a systematic one-pixel stem ring on the 3D leaf. Dilation never
            # labels background, and still needs three independent cameras.
            from scipy.ndimage import binary_dilation
            mask=binary_dilation(mask,iterations=2)&valid
            # Depth visibility is tested for each original vertex. Every base
            # camera contributes at most one vote to this leaf.
            pixel=np.zeros((*fid.shape,6),np.float32);pixel[...,2]=mask
            sampled,seen=p.sample_visible_vertices(scene,xyz[nearby],crop,pixel)
            positive+=(sampled[:,2]>=.75)&seen
            observed+=seen
            before=image.copy();before[valid]=p.COLORS[labels[vertex]]
            overlay=image.copy();overlay[mask]=(overlay[mask]*.35+p.COLORS[2]*.65).astype(np.uint8)
            thumb=Image.new('RGB',(960,345),'white');draw=ImageDraw.Draw(thumb)
            for col,(title,picture) in enumerate([('Original RGB',image),('Before',before),('Automatic mask',overlay)]):
                thumb.paste(Image.fromarray(picture).resize((320,320)),(col*320,25))
                draw.text((col*320+4,5),title,fill='black')
            thumb.save(args.output/'boundary_views'/f'{name}.jpg',quality=91)
            records.append(dict(name=name,accepted=True,positive_points=int((prompt_labels==1).sum()),
                                negative_points=int((prompt_labels==0).sum()),**diagnostic))
            print('Leaf boundary',name,'positive',int(((sampled[:,2]>=.75)&seen).sum()),flush=True)
        # Count an original connected leaf patch as a seed; expansion remains
        # bounded by the explicit geodesic limit and the independent masks.
        accepted=multiview_accept(labels[nearby],data['bud_protected'][nearby],positive,observed,distance[nearby],scale)
        repaired[nearby[accepted]]=True
        better=positive>best_positive[nearby]
        best_positive[nearby[better]]=positive[better];best_observed[nearby[better]]=observed[better]
        print('Leaf organ',organ_id,'vertices changed',int(accepted.sum()),flush=True)
    np.savez_compressed(args.output/'boundary_votes.npz',changed=repaired,
                        positive=best_positive,observed=best_observed)
    return repaired,best_positive,best_observed,records,va


def preview(args,manifest,tri,old,new):
    from refine_detail import camera_crop,render_camera
    source=args.source
    base=Path(manifest['base_directory'])
    fontpath=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(fontpath),20) if fontpath.exists() else ImageFont.load_default(size=20)
    width=480;selected=[manifest['views'][i] for i in [0,2,4,6,8,10,12,13]]
    sheet=Image.new('RGB',(width*3,45+width*len(selected)),'white');draw=ImageDraw.Draw(sheet)
    for i,title in enumerate(['原始颜色','优化前','叶缘优化后']):draw.text((width*i+10,10),title,font=font,fill='black')
    for row,camera in enumerate(selected):
        name=camera['name'];original=Image.open(base/'views'/f'{name}.png').convert('RGB')
        rays=np.load(base/'views'/f'{name}_rays.npz');fid=rays['face_id'];bary=rays['barycentric']
        valid=fid>=0;corners=tri[fid[valid]];vertex=corners[np.arange(len(corners)),bary[valid].argmax(1)]
        pics=[original]
        for label in [old,new]:
            im=np.asarray(original).copy();im[valid]=p.COLORS[label[vertex]];pics.append(Image.fromarray(im))
        for col,im in enumerate(pics):sheet.paste(im.resize((width,width)),(col*width,45+row*width))
        draw.text((6,50+row*width),name,font=font,fill='black')
    sheet.save(args.output/'comparison_preview.jpg',quality=94)
    compact=Image.new('RGB',(1440,45+480*2),'white')
    compact.paste(sheet.crop((0,0,1440,45+480)),(0,0))
    compact.paste(sheet.crop((0,45+480*7,1440,45+480*8)),(0,45+480))
    compact.save(args.output/'comparison_compact.jpg',quality=94)
    # Also show close views selected by visibility of the detected leaf cores.
    xyz,_,rgb=p.mesh_data(manifest['input'])
    scores=np.load(source/'labels.npz')['class_scores']
    organs,_=leaf_cores(xyz,tri,old,scores,max_organs=4)
    if not organs:return
    scene=p.scene_from_arrays(xyz,tri)
    close=Image.new('RGB',(1800,45+600*len(organs)*2),'white')
    draw=ImageDraw.Draw(close)
    for col,title in enumerate(['原始颜色','优化前','优化后']):draw.text((600*col+10,10),title,font=font,fill='black')
    row=0
    for organ in organs:
        selected_cameras=[]
        for subset in [manifest['views'][:12],manifest['views'][12:]]:
            candidates=[]
            for camera in subset:
                rays=np.load(base/'views'/f"{camera['name']}_rays.npz")
                fid=rays['face_id'];valid=fid>=0
                organ_faces=np.isin(tri[fid[valid]],organ['vertices']).any(1)
                candidates.append((int(organ_faces.sum()),camera))
            if candidates:selected_cameras.append(max(candidates,key=lambda item:item[0])[1])
        for camera in selected_cameras:
            points=xyz[organ['vertices']];delta=points-np.asarray(camera['origin'])
            px=delta@np.asarray(camera['right'])/camera['extent']*manifest['size']+manifest['size']/2
            py=-delta@np.asarray(camera['screen_up'])/camera['extent']*manifest['size']+manifest['size']/2
            side=max(np.ptp(px),np.ptp(py))*1.35
            cx=(px.min()+px.max())/2;cy=(py.min()+py.max())/2
            crop=camera_crop(camera,[cx-side/2,cy-side/2,cx+side/2,cy+side/2],manifest['size'])
            original,fid,bary=render_camera(scene,tri,rgb,crop,900)
            valid=fid>=0;corners=tri[fid[valid]];vertex=corners[np.arange(len(corners)),bary[valid].argmax(1)]
            pics=[original]
            for label in [old,new]:
                im=original.copy();im[valid]=p.COLORS[label[vertex]];pics.append(im)
            for col,im in enumerate(pics):close.paste(Image.fromarray(im).resize((600,600)),(col*600,45+row*600))
            row+=1
    close.crop((0,0,1800,45+row*600)).save(args.output/'leaf_boundary_comparison.jpg',quality=94)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sam',type=Path,default=p.ROOT/'models/mobile_sam_parallel.pt')
    parser.add_argument('--size',type=int,default=768)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--max-organs',type=int,default=32)
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if args.source==args.output:raise ValueError('Use a separate output directory for boundary refinement')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'boundary_views').mkdir(exist_ok=True)
    manifest=json.loads((args.source/'manifest.json').read_text(encoding='utf8'))
    if p.sha256(manifest['input'])!=manifest['input_sha256']:raise ValueError('Source mesh identity changed')
    if (args.output/'manifest.json').exists():
        previous=json.loads((args.output/'manifest.json').read_text(encoding='utf8'))
        if previous['input_sha256']!=manifest['input_sha256']:
            raise ValueError('Boundary output belongs to another input; use a new output directory')
    xyz,tri,rgb=p.mesh_data(manifest['input'])
    data=dict(np.load(args.source/'labels.npz'))
    changed,positive,observed,records,va=infer(args,args.source,manifest,xyz,tri,rgb,data)
    from complete_five import closed_face_labels
    old=data['vertex_labels'].copy();label=old.copy();label[changed]=2
    # These are heuristic evidence values, not semantic probabilities.
    agreement=positive/np.maximum(observed,1)
    scores=data['class_scores'].copy()
    weight=np.clip(agreement[changed]*.6,.42,.60)
    scores[changed]*=(1-weight[:,None]);scores[changed,1]+=weight
    faces,boundary=closed_face_labels(tri,label,scores)
    data.update(vertex_labels=label,face_labels=faces,class_scores=scores,face_boundary=boundary,
                leaf_boundary_refined=changed,leaf_mask_positive_views=positive,leaf_mask_observed_views=observed)
    rank=np.sort(scores,axis=1);data['margin']=rank[:,-1]-rank[:,-2];data['evidence']=scores.max(1)
    # Uncertainty is retained, including changed vertices. Masks are not ground truth.
    np.savez_compressed(args.output/'labels.npz',**data)
    for name,semantic in [('semantic_mesh.ply',True),('labeled_original_rgb.ply',False)]:
        p.write_labeled_ply(args.output/name,xyz,tri,rgb,label,data['evidence'],semantic,
                            uncertain=data['uncertain'],inferred=data['inferred'],face_labels=faces)
    report=json.loads((args.source/'report.json').read_text(encoding='utf8'))
    report.update(method=report['method']+' + automatic multiview leaf boundary masks',
                  boundary_source=str(args.source),leaf_boundary_changed_vertices=int(changed.sum()),
                  leaf_boundary_changed_area_fraction=float(va[changed].sum()/va.sum()),
                  boundary_transition_count={str(cls):int((changed&(old==cls)).sum()) for cls in range(1,6)},
                  boundary_views=len(records),boundary_accepted_views=sum(r['accepted'] for r in records),
                  class_vertex_count={c['name']:int((label==c['id']).sum()) for c in p.CLASSES[1:]},
                  class_surface_area_fraction={c['name']:float(va[label==c['id']].sum()/va.sum()) for c in p.CLASSES[1:]},
                  unknown_vertices=int((label==0).sum()),unknown_faces=int((faces==0).sum()))
    p.save_json(args.output/'report.json',report)
    p.save_json(args.output/'manifest.json',dict(manifest,boundary_source=str(args.source),
                boundary_source_labels_sha256=p.sha256(args.source/'labels.npz')))
    p.save_json(args.output/'boundary_records.json',records)
    p.save_json(args.output/'classes.json',p.CLASSES[1:])
    p.save_json(args.output/'inference_config.json',dict(recipe_revision=REVISION,
                boundary_config_sha256=p.sha256(args.output/'boundary_config.json'),
                code_sha256=p.sha256(Path(__file__)),complete_closed_set=True,manual_annotations=False,
                uncertainty_separate=True,source_labels_sha256=p.sha256(args.source/'labels.npz')))
    preview(args,manifest,tri,old,label)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
