"""Fuse SAM3 masks with depth visibility and mesh-edge completion.

The existing segmentation is used only for the comparison panels. Its labels
never enter this SAM3-only result. Scores are heuristic, not probabilities.
"""
import argparse
import json
from pathlib import Path

import plantseg as p
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def project(args, xyz, tri, manifest):
    cache = args.output/'projected_evidence.npz'
    configs = [p.sha256(args.predictions/k/'inference_config.json') for k in ['global','detail']]
    for kind,folder in [('global',args.base),('detail',args.detail)]:
        config=json.loads((args.predictions/kind/'inference_config.json').read_text(encoding='utf8'))
        if config['source_manifest_sha256']!=p.sha256(folder/'manifest.json'):
            raise ValueError('SAM3 scores and camera manifest do not match')
        if config['input_sha256']!=manifest['input_sha256']:
            raise ValueError('SAM3 scores belong to another mesh')
    from complete_five import key
    signature = key(dict(configs=configs,revision=1))
    if cache.exists():
        data = dict(np.load(cache))
        if str(data.pop('cache_key')) != signature:
            raise ValueError('SAM3 projection cache is stale')
        return data
    detail = json.loads((args.detail/'manifest.json').read_text(encoding='utf8'))
    scene = p.scene_from_arrays(xyz,tri)
    n = len(xyz)
    total = np.zeros((n,6),np.float32)
    count = np.zeros(n,np.uint16)
    votes = np.zeros((n,6),np.uint16)
    support = np.zeros((n,6),np.uint16)
    bud_total = np.zeros(n,np.float32)
    bud_count = np.zeros(n,np.uint16)
    for camera in manifest['views']:
        relevant = [('global',camera)] + [('detail',v) for v in detail['views'] if v['parent']==camera['name']]
        values = np.zeros_like(total)
        seen = np.zeros(n,bool)
        bud = np.zeros(n,np.float32)
        for kind,cam in relevant:
            path = args.predictions/kind/'views'/f"{cam['name']}_scores.npz"
            if not path.exists():
                raise FileNotFoundError(f'Incomplete SAM3 inference: {path}')
            with np.load(path) as cached:
                config = json.loads((args.predictions/kind/'inference_config.json').read_text(encoding='utf8'))
                if str(cached['cache_key']) != config['cache_key']:
                    raise ValueError('Stale SAM3 2D prediction')
                sampled,observed = p.sample_visible_vertices(scene,xyz,cam,cached['scores'])
                values = np.maximum(values,sampled)
                extra = np.zeros_like(cached['scores'])
                extra[...,3] = cached['bud']
                sampled_bud,_ = p.sample_visible_vertices(scene,xyz,cam,extra)
                bud = np.maximum(bud,sampled_bud[:,3])
                seen |= observed
        # Overlapping crops of a single camera count as one observation.
        total += values
        count += seen
        label = values.argmax(1)
        label[values.max(1)<.15] = 0
        indices = np.flatnonzero(seen)
        votes[indices,label[indices]] += 1
        support += values >= .15
        bud_total += bud
        bud_count += bud >= .15
        print('SAM3 projected',camera['name'],'visible',int(seen.sum()),flush=True)
    data = dict(mean=total/np.maximum(count[:,None],1),view_count=count,votes=votes,
                support_count=support,bud_total=bud_total,bud_count=bud_count)
    np.savez_compressed(cache,**data,cache_key=signature)
    return data


def render_label(image, fid, bary, tri, labels):
    canvas = np.asarray(image).copy()
    valid = fid >= 0
    corners = tri[fid[valid]]
    v = corners[np.arange(len(corners)),bary[valid].argmax(1)]
    canvas[valid] = p.COLORS[labels[v]]
    return Image.fromarray(canvas)


def comparison(args, tri, labels, result_title='SAM3 + 三维补全'):
    old = np.load(args.compare/'labels.npz')['vertex_labels'] if args.compare else None
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    for kind,folder in [('global',args.base),('detail',args.detail)]:
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        rows=[]
        for camera in manifest['views']:
            name = camera['name']
            original = Image.open(folder/'views'/f'{name}.png').convert('RGB')
            ray=np.load(folder/'views'/f'{name}_rays.npz')
            before=render_label(original,ray['face_id'],ray['barycentric'],tri,old) if old is not None else original
            after=render_label(original,ray['face_id'],ray['barycentric'],tri,labels)
            row=Image.new('RGB',(1440,520),'white');draw=ImageDraw.Draw(row)
            for col,(title,img) in enumerate([('原始颜色',original),('现有 v2' if old is not None else '原始颜色',before),(result_title,after)]):
                draw.text((col*480+12,6),f'{name}  {title}',font=font,fill='black')
                row.paste(img.resize((480,480)),(col*480,35))
            row.save(args.output/f'comparison_{name}.jpg',quality=93)
            rows.append(row.resize((1080,390)))
        sheet=Image.new('RGB',(1080,len(rows)*390),'white')
        for i,row in enumerate(rows):sheet.paste(row,(0,i*390))
        sheet.save(args.output/f'comparison_{kind}.jpg',quality=91)
    if old is not None:
        transition=np.bincount(old.astype(int)*6+labels.astype(int),minlength=36).reshape(6,6)
        p.save_json(args.output/'v2_comparison.json',dict(
            changed_vertices=int((old!=labels).sum()),transition_rows_v2_columns_sam3=transition.tolist(),
            classes=p.CLASSES,reference_is_ground_truth=False,accuracy_measured=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--detail',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.base/'manifest.json').read_text(encoding='utf8'))
    if p.sha256(base['input'])!=base['input_sha256']:raise ValueError('Input mesh changed')
    xyz,tri,rgb=p.mesh_data(base['input'])
    projected=project(args,xyz,tri,base)
    from complete_five import appearance_likelihood,mesh_graph,fill_unobserved_components,diffuse,closed_face_labels
    from refine_detail import pot_envelope
    from scipy.special import softmax
    import cv2
    means=projected['mean'][:,1:]
    count=projected['view_count']
    raw=means.argmax(1).astype(np.uint8)+1
    ranking=np.sort(means,axis=1)
    agreement=projected['votes'][np.arange(len(xyz)),raw]/np.maximum(count,1)
    direct=(ranking[:,-1]>.20)&(ranking[:,-1]-ranking[:,-2]>.05)&(count>=2)&(agreement>=.5)
    seeds=np.where(direct,raw,0).astype(np.uint8)
    bud=(projected['bud_count']>=2)&(projected['bud_total']/np.maximum(count,1)>.18)
    seeds[bud]=0
    lab=cv2.cvtColor(rgb[:,None,:].astype(np.float32),cv2.COLOR_RGB2LAB)[:,0,:]
    appearance,diagnostic=appearance_likelihood(lab,seeds,np.random.default_rng(20261020))
    # Semantic masks dominate. Color is learned from this scene's own seeds.
    logits=np.log(means+.025)*2.0+appearance*1.1
    envelope=pot_envelope(xyz,seeds,base['views'][0]['up_axis'])
    if envelope:
        logits[~envelope['substrate'],4]=-20
        logits[~envelope['pot'],3]=-20
    logits[bud,2]+=4
    unary=softmax(logits,axis=1).astype(np.float32)
    anchors=direct&(unary.max(1)>.94)
    graph=mesh_graph(xyz,tri,lab)
    unary,propagated=fill_unobserved_components(graph,unary,means.max(1)>.12)
    strength=(.15+unary.max(1)*.6).astype(np.float32)
    strength[means.max(1)<.12]=.025
    posterior=diffuse(graph,unary,strength,anchors,80)
    labels=posterior.argmax(1).astype(np.uint8)+1
    faces,boundary=closed_face_labels(tri,labels,posterior)
    rank=np.sort(posterior,axis=1);margin=rank[:,-1]-rank[:,-2]
    uncertain=(~direct)|(margin<.18)
    data=dict(vertex_labels=labels,face_labels=faces,class_scores=posterior,
              evidence=posterior.max(1),margin=margin,uncertain=uncertain,inferred=~direct,
              view_count=count,bud_protected=bud,face_boundary=boundary,
              raw_vertex_labels=raw,raw_direct=direct,mean_evidence=projected['mean'],
              propagated_unobserved=propagated)
    np.savez_compressed(args.output/'labels.npz',**data)
    for name,semantic in [('semantic_mesh.ply',True),('labeled_original_rgb.ply',False)]:
        p.write_labeled_ply(args.output/name,xyz,tri,rgb,labels,data['evidence'],semantic,
                            uncertain=uncertain,inferred=~direct,face_labels=faces)
    p.save_json(args.output/'manifest.json',dict(base,base_directory=str(args.base.resolve()),
                detail_directory=str(args.detail.resolve()),prediction_directory=str(args.predictions.resolve()),
                status='closed_set_all_vertices_assigned'))
    p.save_json(args.output/'classes.json',p.CLASSES[1:])
    area=np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]]),axis=1)*.5
    va=np.zeros(len(xyz),np.float64)
    for j in range(3):np.add.at(va,tri[:,j],area/3)
    report=dict(method='SAM3 text masks + depth-aware multiview fusion + scene appearance + mesh-edge diffusion',
                vertex_count=len(xyz),face_count=len(tri),unknown_vertices=int((labels==0).sum()),
                unknown_faces=int((faces==0).sum()),
                class_vertex_count={c['name']:int((labels==c['id']).sum()) for c in p.CLASSES[1:]},
                class_surface_area_fraction={c['name']:float(va[labels==c['id']].sum()/va.sum()) for c in p.CLASSES[1:]},
                direct_evidence_vertex_fraction=float(direct.mean()),
                uncertain_vertex_fraction=float(uncertain.mean()),appearance_seeds=diagnostic,
                semantic_accuracy_measured=False,cross_plant_validation=False,
                manual_annotations=False,foundation_model_finetuning=False,
                previous_semantic_labels_used_for_fusion=False,
                existing_automatically_selected_crop_cameras_reused=not bool(json.loads((args.detail/'manifest.json').read_text(encoding='utf8')).get('sam3_crop_selection')),
                scores_are_calibrated_probabilities=False)
    p.save_json(args.output/'report.json',report)
    p.save_json(args.output/'inference_config.json',dict(code_sha256=p.sha256(__file__),
                complete_closed_set=True,manual_annotations=False,uncertainty_separate=True))
    comparison(args,tri,labels)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
