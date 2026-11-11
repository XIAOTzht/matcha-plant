"""Experimental, training-free plant mesh semantics from rendered views.

Original mesh indices are retained. GroundingDINO boxes prompt MobileSAM.
All scores are heuristic evidence, NOT calibrated semantic probabilities.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
# The distributed vendor tree contains Windows-only compiled extensions. Keep
# it for Windows, but use normal Linux wheels on the server.
if os.name == 'nt' or os.environ.get('PLANTSEG_USE_VENDOR') == '1':
    sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'third_party' / 'MobileSAM'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
from PIL import Image, ImageDraw

CLASSES = [
    {'id': 0, 'name': 'unknown', 'zh': '未知/证据不足', 'color': [140, 140, 140]},
    {'id': 1, 'name': 'stem', 'zh': '枝干、茎、叶柄、花梗', 'color': [245, 155, 35]},
    {'id': 2, 'name': 'leaf', 'zh': '叶片', 'color': [40, 190, 85]},
    {'id': 3, 'name': 'flower', 'zh': '花、花苞', 'color': [235, 75, 165]},
    {'id': 4, 'name': 'pot', 'zh': '花盆、托盘', 'color': [55, 130, 240]},
    {'id': 5, 'name': 'substrate', 'zh': '土壤、基质', 'color': [140, 95, 55]},
]
COLORS = np.array([c['color'] for c in CLASSES], dtype=np.uint8)
PHRASES = [('stem', 1), ('branch', 1), ('leaf', 2),
           ('flower', 3), ('flower head', 3), ('flower center', 3),
           ('petal', 3), ('bud', 3), ('pot', 4),
           ('saucer', 4), ('soil', 5), ('moss', 5)]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def mesh_data(path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    tri = np.asarray(mesh.triangles, dtype=np.int32)
    if not len(tri) or not mesh.has_vertex_colors():
        raise ValueError('This backend requires a triangle mesh with vertex RGB colors.')
    rgb = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite mesh coordinates')
    return xyz, tri, rgb


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def scene_from_arrays(xyz, tri):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(tri.astype(np.uint32), dtype=o3d.core.Dtype.UInt32))
    return scene


def render_view(scene, xyz, tri, rgb, azimuth, elevation, size, up_axis='y'):
    import open3d as o3d
    up = np.eye(3, dtype=np.float32)['xyz'.index(up_axis)]
    east = np.eye(3, dtype=np.float32)[('xyz'.index(up_axis) + 1) % 3]
    north = np.cross(up, east)
    az, el = np.deg2rad([azimuth, elevation])
    outward = math.cos(el) * (math.cos(az)*east + math.sin(az)*north) + math.sin(el)*up
    forward = -outward
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, forward)
    center = (xyz.min(0) + xyz.max(0)) * .5
    centered = xyz - center
    ux, vy = centered @ right, centered @ screen_up
    plane_center = center + (ux.min()+ux.max())*.5*right + (vy.min()+vy.max())*.5*screen_up
    extent = max(np.ptp(ux), np.ptp(vy)) * 1.08
    depth_offset = np.linalg.norm(np.ptp(xyz, axis=0)) * 2
    xx, yy = np.meshgrid((np.arange(size)+.5-size/2)*extent/size,
                         (np.arange(size)+.5-size/2)*extent/size)
    origin = plane_center + depth_offset*outward
    starts = origin + xx[..., None]*right - yy[..., None]*screen_up
    directions = np.broadcast_to(forward, starts.shape)
    rays = np.concatenate([starts, directions], axis=-1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=8)
    fid = hit['primitive_ids'].numpy().astype(np.int64)
    valid = fid != 4294967295
    fid[~valid] = -1
    uv = hit['primitive_uvs'].numpy()
    weights = np.concatenate([1-uv.sum(-1, keepdims=True), uv], axis=-1)
    image = np.full((size, size, 3), 242, dtype=np.uint8)
    values = (rgb[tri[fid[valid]]] * weights[valid, :, None]).sum(1)
    image[valid] = np.rint(np.clip(values, 0, 1)*255).astype(np.uint8)
    camera = {'azimuth': azimuth, 'elevation': elevation, 'up_axis': up_axis,
              'origin': origin.tolist(), 'direction': forward.tolist(),
              'right': right.tolist(), 'screen_up': screen_up.tolist(),
              'extent': float(extent), 'pixel_size_world_units': float(extent/size)}
    return image, fid.astype(np.int32), weights.astype(np.float32), camera


def prepare(args):
    out = Path(args.output)
    if (out / 'manifest.json').exists():
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if manifest['input_sha256'] != sha256(args.input):
            raise ValueError('Output directory belongs to a different input mesh')
        print('Prepared views already exist; use a new output directory to change rendering.', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out/'views').mkdir(exist_ok=True)
    xyz, tri, rgb = mesh_data(args.input)
    scene = scene_from_arrays(xyz, tri)
    angles = [(i*360/args.azimuths, e) for e in [12, 48] for i in range(args.azimuths)]
    angles += [(0, 85), (0, -70)]
    cameras = []
    tiles = []
    for i, (az, el) in enumerate(angles):
        image, fid, weights, camera = render_view(scene, xyz, tri, rgb, az, el, args.size, args.up)
        name = f'view_{i:03d}'
        Image.fromarray(image).save(out/'views'/f'{name}.png')
        np.savez_compressed(out/'views'/f'{name}_rays.npz', face_id=fid, barycentric=weights)
        camera['name'] = name
        camera['visible_faces'] = int(len(np.unique(fid[fid>=0])))
        cameras.append(camera)
        tile = Image.fromarray(image).resize((280, 280))
        ImageDraw.Draw(tile).text((8, 8), f'{name} az={az:.0f} el={el:.0f}', fill='black')
        tiles.append(tile)
        print('Rendered', name, camera['visible_faces'], 'faces', flush=True)
    sheet = Image.new('RGB', (280*4, 280*math.ceil(len(tiles)/4)), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i%4)*280, (i//4)*280))
    sheet.save(out/'render_contact_sheet.jpg')
    save_json(out/'classes.json', CLASSES)
    save_json(out/'manifest.json', {'input': str(Path(args.input).resolve()),
              'input_sha256': sha256(args.input), 'vertex_count': len(xyz), 'face_count': len(tri),
              'size': args.size, 'views': cameras, 'class_count': 5,
              'status': 'rendered_only', 'semantics_validated': False})


def load_backends(args):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from mobile_sam import sam_model_registry, SamPredictor
    torch.set_num_threads(args.threads)
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(str(args.detector), local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(args.detector), local_files_only=True, disable_custom_kernels=True).to(device).eval()
    sam = sam_model_registry['vit_t'](checkpoint=str(args.sam)).to(device).eval()
    return processor, detector, SamPredictor(sam), device


def load_dense(args,device):
    from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation
    processor=CLIPSegProcessor.from_pretrained(str(args.dense),local_files_only=True,use_fast=False)
    model=CLIPSegForImageSegmentation.from_pretrained(str(args.dense),local_files_only=True).to(device).eval()
    return processor,model


DENSE_PHRASES=[('plant stem',1),('plant stalk',1),('plant leaf',2),('leaf',2),
               ('flower',3),('flower head',3),('flower center',3),
               ('flower petals',3),('flower buds',3),('pot',4),('potting soil',5)]


def dense_evidence(image,processor,model,device):
    import torch
    import torch.nn.functional as F
    # Uniform tiles improve sampling of thin parts without selecting regions manually.
    w,h=image.size
    tiles=[(0,0,w,h),(0,0,w,round(h*.67)),(0,round(h*.40),w,h)]
    total=np.zeros((h,w,6),np.float32)
    weight=np.zeros((h,w),np.float32)
    for ti,(x1,y1,x2,y2) in enumerate(tiles):
        tile=image.crop((x1,y1,x2,y2))
        predictions=[]
        for start in range(0,len(DENSE_PHRASES),4):
            phrases=[s for s,_ in DENSE_PHRASES[start:start+4]]
            inputs=processor(text=phrases,images=[tile]*len(phrases),padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():pred=model(**inputs).logits
            pred=F.interpolate(pred[:,None],(y2-y1,x2-x1),mode='bilinear',align_corners=False)[:,0].sigmoid()
            predictions.append(pred.cpu().numpy())
        pred=np.concatenate(predictions)
        scores=np.zeros((y2-y1,x2-x1,6),np.float32)
        for i,(_,cls) in enumerate(DENSE_PHRASES):scores[...,cls]=np.maximum(scores[...,cls],pred[i])
        scale=1.0 if ti==0 else .5
        total[y1:y2,x1:x2]+=scores*scale
        weight[y1:y2,x1:x2]+=scale
    return total/weight[...,None]


def detect(image, processor, detector, device, threshold, phrases=None):
    import torch
    from torchvision.ops import nms
    phrases = PHRASES if phrases is None else phrases
    text = '. '.join(p for p, _ in phrases) + '.'
    inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    # Token spans make each prompt's class explicit; decoded phrases can merge adjacent classes.
    offset = processor.tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    spans, start = [], 0
    for phrase, cls in phrases:
        end = start + len(phrase)
        tokens = [i for i, (a, b) in enumerate(offset) if b > start and a < end and b > a]
        spans.append((phrase, cls, tokens))
        start = end + 2
    with torch.inference_mode():
        pred = detector(**inputs)
    logits = pred.logits[0].sigmoid()
    boxes = pred.pred_boxes[0]
    cx, cy, w, h = boxes.unbind(-1)
    boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)
    boxes *= torch.tensor([image.width, image.height, image.width, image.height], device=device)
    records = []
    for phrase, cls, tokens in spans:
        scores = logits[:, tokens].max(-1).values
        idx = torch.where(scores >= threshold)[0]
        if not len(idx):
            continue
        keep = nms(boxes[idx], scores[idx], .5)
        for j in idx[keep[:12]].tolist():
            b = boxes[j].detach().cpu().numpy()
            b[[0, 2]] = b[[0, 2]].clip(0, image.width-1)
            b[[1, 3]] = b[[1, 3]].clip(0, image.height-1)
            if b[2]-b[0] < 2 or b[3]-b[1] < 2:
                continue
            records.append({'phrase': phrase, 'class_id': cls, 'box': b.tolist(),
                            'detector_score': float(scores[j])})
    return records


def evidence_labels(scores, minimum=.20, margin=.035):
    ranking = np.sort(scores, axis=-1)
    top = ranking[..., -1]
    gap = top-ranking[..., -2]
    labels = scores.argmax(-1).astype(np.uint8)
    labels[(top < minimum) | (gap < margin)] = 0
    return labels, top, gap


def predict(args):
    out = Path(args.output)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    processor, detector, sam, device = load_backends(args)
    dense_processor,dense_model=load_dense(args,device)
    config = {'backend': 'GroundingDINO-tiny + CLIPSeg + MobileSAM',
              'detector_sha256': sha256(Path(args.detector)/'model.safetensors'),
              'dense_sha256':sha256(Path(args.dense)/'model.safetensors'),
              'sam_sha256': sha256(args.sam), 'device': device, 'phrases': PHRASES,
              'dense_phrases':DENSE_PHRASES,'recipe_revision':5,
              'box_threshold': args.box_threshold, 'scores_are_probabilities': False,
              'trained_or_finetuned_on_user_data': False}
    config_key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['cache_key']=config_key
    old=out/'inference_config.json'
    if old.exists() and not args.repredict:
        previous=json.loads(old.read_text(encoding='utf-8'))
        if previous.get('cache_key') != config_key:
            raise ValueError('Inference configuration changed. Use --repredict explicitly or a new output folder.')
    save_json(old,config)
    for view in manifest['views'][:args.limit or None]:
        name = view['name']
        target = out/'views'/f'{name}_scores.npz'
        if target.exists() and not args.repredict:
            print('Cached', name, flush=True)
            continue
        start = time.time()
        image = Image.open(out/'views'/f'{name}.png').convert('RGB')
        rgb = np.asarray(image)
        fid = np.load(out/'views'/f'{name}_rays.npz')['face_id']
        dense=dense_evidence(image,dense_processor,dense_model,device)
        detections = detect(image, processor, detector, device, args.box_threshold)
        sam.set_image(rgb)
        scores = dense*.6
        scores[fid<0]=0
        records = []
        for det in detections:
            masks, quality, _ = sam.predict(box=np.array(det['box']), multimask_output=True)
            cls = det['class_id']
            options=[]
            for candidate in masks:
                candidate=candidate&(fid>=0)
                semantic=dense[candidate].mean(0) if candidate.any() else np.zeros(6)
                competitors=np.delete(semantic,[0,cls])
                options.append(float(semantic[cls]-competitors.max()))
            k = int(np.argmax(np.asarray(options)+np.asarray(quality)*.15))
            mask = masks[k] & (fid >= 0)
            # Restrict support to the detector box; no image-specific or manual prompts.
            x1,y1,x2,y2 = det['box']
            yy,xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            mask &= (xx>=x1-2)&(xx<=x2+2)&(yy>=y1-2)&(yy<=y2+2)
            quality_score = float(np.clip(quality[k], 0, 1))
            score = det['detector_score'] * quality_score
            semantic=dense[mask].mean(0) if mask.any() else np.zeros(6)
            best_other=float(np.delete(semantic,[0,cls]).max())
            # A local detection may rescue the dense model's second-ranked class.
            # Whole-plant proposals must agree with dense semantics instead.
            local=mask.sum()<.12*(fid>=0).sum()
            rank=int((semantic[1:]>semantic[cls]).sum()+1)
            validated=bool(semantic[cls]>=.15 and
                           (semantic[cls]>=best_other-.06 or (local and rank<=2)))
            det.update(sam_quality=quality_score, pixels=int(mask.sum()), evidence=score)
            det.update(dense_class_mean=float(semantic[cls]),dense_other_max=best_other,semantic_validated=validated)
            records.append(det)
            if validated:
                supported=.6*dense[...,cls]+.7*score
                scores[...,cls][mask]=np.maximum(scores[...,cls][mask],supported[mask])
        labels, _, _ = evidence_labels(scores)
        np.savez_compressed(target, scores=scores.astype(np.float16),dense=dense.astype(np.float16),cache_key=config_key)
        Image.fromarray(labels).save(out/'views'/f'{name}_labels.png')
        overlay = rgb.copy()
        fg = fid >= 0
        overlay[fg] = (rgb[fg]*.4 + COLORS[labels[fg]]*.6).astype(np.uint8)
        Image.fromarray(overlay).save(out/'views'/f'{name}_overlay.jpg')
        save_json(out/'views'/f'{name}_detections.json', records)
        print(name, 'detections', len(records), 'class pixels', np.bincount(labels[fg], minlength=6).tolist(),
              'seconds', round(time.time()-start, 1), flush=True)


def accumulate_vertex_view(tri, fid, barycentric, pixel_scores, vertex_count):
    """Each view contributes at most one averaged observation per mesh vertex."""
    valid = fid >= 0
    indices = tri[fid[valid]]
    bary = np.maximum(barycentric[valid], 0)
    values = pixel_scores[valid].astype(np.float32)
    total = np.zeros((vertex_count, 6), np.float32)
    weight = np.zeros(vertex_count, np.float32)
    for corner in range(3):
        np.add.at(weight, indices[:, corner], bary[:, corner])
        for cls in range(1, 6):
            np.add.at(total[:, cls], indices[:, corner], bary[:, corner]*values[:, cls])
    observed = weight > 1e-6
    total[observed] /= weight[observed, None]
    return total, observed


def sample_visible_vertices(scene,xyz,camera,pixel_scores):
    """Project every original vertex, then ray-test its own depth for occlusion.

    This avoids losing tiny original triangles which fall between rendered pixels.
    """
    import open3d as o3d
    from scipy.ndimage import map_coordinates
    origin=np.array(camera['origin'],np.float32)
    direction=np.array(camera['direction'],np.float32)
    right=np.array(camera['right'],np.float32)
    up=np.array(camera['screen_up'],np.float32)
    h,w=pixel_scores.shape[:2]
    delta=xyz-origin
    depth=delta@direction
    x=(delta@right)/camera['extent']*w+w/2-.5
    y=-(delta@up)/camera['extent']*h+h/2-.5
    observed=(x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)&(depth>0)
    starts=xyz-depth[:,None]*direction
    rays=np.concatenate([starts,np.broadcast_to(direction,starts.shape)],axis=1).astype(np.float32)
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)
    first=hit['t_hit'].numpy()
    tolerance=float(np.linalg.norm(np.ptp(xyz,axis=0))*1e-5)
    observed &= np.isfinite(first)&(np.abs(first-depth)<=tolerance)
    values=np.zeros((len(xyz),6),np.float32)
    indices=np.where(observed)[0]
    for cls in range(1,6):
        values[indices,cls]=map_coordinates(pixel_scores[...,cls].astype(np.float32),
                           [y[indices],x[indices]],order=1,mode='nearest')
    return values,observed


def write_labeled_ply(path, xyz, tri, rgb, labels, evidence, semantic_colors=False,
                      uncertain=None, inferred=None, face_labels=None):
    color = COLORS[labels] if semantic_colors else np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red','u1'), ('green','u1'),
             ('blue','u1'), ('label','u1'), ('evidence','<f4')]
    metadata = {key:value for key,value in [('uncertain',uncertain),('inferred',inferred)] if value is not None}
    for key in metadata:
        dtype.append((key,'u1'))
    vertices = np.empty(len(xyz), dtype=dtype)
    for i, key in enumerate(['x', 'y', 'z']): vertices[key] = xyz[:, i]
    for i, key in enumerate(['red', 'green', 'blue']): vertices[key] = color[:, i]
    vertices['label'], vertices['evidence'] = labels, evidence
    for key,value in metadata.items():
        vertices[key] = value
    face_dtype = [('count','u1'), ('indices','<i4',(3,))]
    if face_labels is not None:
        face_dtype.append(('label','u1'))
    faces = np.empty(len(tri), dtype=face_dtype)
    faces['count'], faces['indices'] = 3, tri
    if face_labels is not None:
        faces['label'] = face_labels
    extra_vertex_header = ''.join(f'property uchar {key}\n' for key in metadata)
    extra_face_header = 'property uchar label\n' if face_labels is not None else ''
    header = ('ply\nformat binary_little_endian 1.0\ncomment evidence is not calibrated probability\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar label\nproperty float evidence\n'
              f'{extra_vertex_header}element face {len(tri)}\nproperty list uchar int vertex_indices\n{extra_face_header}end_header\n')
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii')); vertices.tofile(f); faces.tofile(f)


def fuse(args):
    from scipy import sparse
    out = Path(args.output)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if sha256(manifest['input']) != manifest['input_sha256']:
        raise ValueError('Source mesh changed after rendering')
    xyz, tri, rgb = mesh_data(manifest['input'])
    total = np.zeros((len(xyz), 6), np.float32)
    count = np.zeros(len(xyz), np.uint16)
    votes = np.zeros((len(xyz), 6), np.uint16)
    scene=scene_from_arrays(xyz,tri) if args.vertex_rays else None
    done = []
    for view in manifest['views']:
        name = view['name']
        score_path = out/'views'/f'{name}_scores.npz'
        if not score_path.exists(): continue
        ray = np.load(out/'views'/f'{name}_rays.npz')
        cached=np.load(score_path)
        inference=json.loads((out/'inference_config.json').read_text(encoding='utf-8'))
        if 'cache_key' not in cached or str(cached['cache_key'])!=inference['cache_key']:
            raise ValueError(f'{name}: stale predictions; rerun predict --repredict')
        scores = cached['scores']
        if args.vertex_rays:
            scores,observed=sample_visible_vertices(scene,xyz,view,scores)
        else:
            scores, observed = accumulate_vertex_view(tri, ray['face_id'], ray['barycentric'], scores, len(xyz))
        labels, _, _ = evidence_labels(scores)
        total += scores
        count += observed
        indices = np.where(observed)[0]
        votes[indices, labels[indices]] += 1
        done.append(name)
        print('Fused',name,'visible vertices',int(observed.sum()),flush=True)
    if not done: raise ValueError('No predicted views found')
    mean = total / np.maximum(count[:, None], 1)
    label, evidence, margin = evidence_labels(mean, args.min_evidence, args.margin)
    agreement = votes[np.arange(len(xyz)), label] / np.maximum(count, 1)
    accepted = (count >= args.min_views) & (agreement >= args.min_agreement) & (label != 0)
    label[~accepted] = 0
    direct = label != 0
    # Fill only holes without a direct raster sample, along actual mesh edges, with unanimous
    # already-accepted neighbours. Never use unrestricted Euclidean nearest neighbours.
    filled = np.zeros(len(xyz), bool)
    if args.fill_steps:
        a = tri[:, [0,1,2]].ravel(); b = tri[:, [1,2,0]].ravel()
        adjacency = sparse.coo_matrix((np.ones(2*len(a), np.float32),
                     (np.concatenate([a,b]),np.concatenate([b,a]))), shape=(len(xyz),len(xyz))).tocsr()
        adjacency.data[:] = 1
        for _ in range(args.fill_steps):
            support = np.column_stack([adjacency @ (label==cls).astype(np.float32) for cls in range(1,6)])
            candidates = (label==0)&(count==0)&((support>0).sum(1)==1)&(support.sum(1)>=2)
            label[candidates] = support[candidates].argmax(1)+1
            filled |= candidates
    mixed_faces = np.any(label[tri] != label[tri[:, :1]], axis=1)
    face_labels = label[tri[:, 0]].copy()
    face_labels[mixed_faces] = 0
    area = np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]], xyz[tri[:,2]]-xyz[tri[:,0]]), axis=1)*.5
    vertex_area = np.zeros(len(xyz),np.float64)
    for j in range(3):np.add.at(vertex_area,tri[:,j],area/3)
    np.savez_compressed(out/'labels.npz', vertex_labels=label, face_labels=face_labels,
                        mean_evidence=mean, evidence=evidence, margin=margin,
                        view_count=count, agreement=agreement.astype(np.float32),
                        direct=direct, propagated=filled, face_boundary=mixed_faces)
    write_labeled_ply(out/'semantic_mesh.ply',xyz,tri,rgb,label,evidence,True)
    write_labeled_ply(out/'labeled_original_rgb.ply',xyz,tri,rgb,label,evidence,False)
    counts = np.bincount(label,minlength=6)
    report = {'status':'experimental_not_accuracy_validated', 'processed_views':done,
              'total_prepared_views':len(manifest['views']), 'source_sha256':manifest['input_sha256'],
              'vertex_count':len(xyz), 'face_count':len(tri),
              'class_vertex_count':{c['name']:int(counts[c['id']]) for c in CLASSES},
              'class_surface_area_fraction':{c['name']:float(vertex_area[label==c['id']].sum()/vertex_area.sum()) for c in CLASSES},
              'unknown_vertex_fraction':float((label==0).mean()),
              'unobserved_vertex_fraction':float((count==0).mean()),
              'propagated_vertex_fraction':float(filled.mean()),
              'min_views':args.min_views,'min_agreement':args.min_agreement,
              'min_evidence':args.min_evidence,'min_margin':args.margin,
              'visibility_sampling':'per_vertex_rays' if args.vertex_rays else 'pixel_triangle_barycentric',
              'accuracy_measured':False,'scores_are_probabilities':False,
              'notes':['No manual image-specific prompts, finetuning, or color thresholds.',
                       'Multi-view agreement may repeat the same model error.',
                       'Unknown must not be counted as a correct semantic prediction.',
                       'PLY faces/coordinates/order preserved; alpha is omitted in output copies.',
                       'Mixed-label triangle faces remain unknown; vertex labels preserve boundary evidence.']}
    save_json(out/'report.json',report)
    review_sheet(out, manifest, tri, label)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def review_sheet(out, manifest, tri, labels, destination=None):
    chosen = [v for v in manifest['views'] if (out/'views'/f"{v['name']}_scores.npz").exists()]
    sheet = Image.new('RGB',(1000,330*len(chosen)), 'white')
    for row, view in enumerate(chosen):
        name=view['name']
        image=Image.open(out/'views'/f'{name}.png').convert('RGB')
        ray=np.load(out/'views'/f'{name}_rays.npz'); fid=ray['face_id']; bary=ray['barycentric']; valid=fid>=0
        corners=tri[fid[valid]]; picked=corners[np.arange(len(corners)),bary[valid].argmax(1)]
        output=np.asarray(image).copy();output[valid]=COLORS[labels[picked]]
        cols=[image,Image.open(out/'views'/f'{name}_overlay.jpg'),Image.fromarray(output)]
        for col,img in enumerate(cols):sheet.paste(img.resize((330,330)),(col*335,row*330))
        ImageDraw.Draw(sheet).text((4,row*330+4),name,fill='black')
    sheet.save((Path(destination) if destination else out)/'review_contact_sheet.jpg')


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','predict','fuse','run'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=896)
    parser.add_argument('--azimuths',type=int,default=6)
    parser.add_argument('--up',choices=['x','y','z'],default='y')
    parser.add_argument('--detector',type=Path,default=ROOT/'models'/'grounding-dino-tiny')
    parser.add_argument('--sam',type=Path,default=ROOT/'models'/'mobile_sam_parallel.pt')
    parser.add_argument('--dense',type=Path,default=ROOT/'models'/'clipseg-rd64-refined')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--box-threshold',type=float,default=.22)
    parser.add_argument('--limit',type=int,default=0,help='Only process first N views; 0 = all')
    parser.add_argument('--repredict',action='store_true',help='Explicitly replace cached predictions')
    parser.add_argument('--min-evidence',type=float,default=.20)
    parser.add_argument('--margin',type=float,default=.035)
    parser.add_argument('--min-views',type=int,default=2)
    parser.add_argument('--min-agreement',type=float,default=.6)
    parser.add_argument('--fill-steps',type=int,default=0)
    parser.add_argument('--vertex-rays',action=argparse.BooleanOptionalAction,default=True,
                        help='Ray-test every original vertex to avoid raster sampling holes')
    args=parser.parse_args()
    if args.command in ['prepare','run']:
        if args.input is None:parser.error('--input is required for prepare/run')
        prepare(args)
    if args.command in ['predict','run']:predict(args)
    if args.command in ['fuse','run']:fuse(args)


if __name__=='__main__':main()
