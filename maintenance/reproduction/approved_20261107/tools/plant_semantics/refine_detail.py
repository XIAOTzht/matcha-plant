"""Automatic detail rendering and shape-aware zero-shot experiment.

Uses only model-generated labels and scale-normalized geometry for crop selection.
Overlapping crops share the original camera vote to avoid inflating view counts.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import plantseg as p
from PIL import Image, ImageDraw

RENDER_REVISION = 1


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def check_source(source_path, manifest):
    if str(source_path.resolve()) != manifest['source_directory']:
        raise ValueError('Detail output belongs to a different source directory')
    if p.sha256(manifest['input']) != manifest['input_sha256']:
        raise ValueError('Source mesh changed')
    expected = manifest['source_labels_sha256']
    if p.sha256(source_path/'labels.npz') != expected:
        raise ValueError('Baseline labels changed after selecting detail views; use a new output directory')


def shape_features(xyz, spacing=.004, radii=(.012, .025)):
    """Multi-scale PCA on a spatially uniform sample, independent of mesh density."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(float)))
    cloud = cloud.voxel_down_sample(scale * spacing)
    points = np.asarray(cloud.points)
    tree = cKDTree(points)
    features = []
    for radius in radii:
        distance, index = tree.query(points, k=96, distance_upper_bound=scale*radius, workers=8)
        ok = np.isfinite(distance)
        neighbours = points[np.minimum(index, len(points)-1)]
        count = ok.sum(1)
        mean = (neighbours*ok[..., None]).sum(1)/np.maximum(count[:, None], 1)
        delta = (neighbours-mean[:, None, :])*ok[..., None]
        covariance = np.einsum('nki,nkj->nij', delta, delta)/np.maximum(count[:, None, None], 1)
        eigen = np.linalg.eigvalsh(covariance).clip(0)
        linearity = (eigen[:, 2]-eigen[:, 1])/np.maximum(eigen[:, 2], 1e-12)
        planarity = (eigen[:, 1]-eigen[:, 0])/np.maximum(eigen[:, 2], 1e-12)
        scattering = eigen[:, 0]/np.maximum(eigen[:, 2], 1e-12)
        features.append(np.column_stack([linearity, planarity, scattering]))
    _, nearest = tree.query(xyz, workers=8)
    return np.concatenate(features, axis=1)[nearest].astype(np.float32)


def camera_crop(camera, box, source_size):
    """Square image-space crop expressed as another orthographic camera."""
    x1, y1, x2, y2 = box
    width = x2-x1
    if abs(width-(y2-y1)) > 1e-5:
        raise ValueError('Detail cameras require square crops')
    pixel = camera['extent']/source_size
    origin = (np.array(camera['origin']) +
              ((x1+x2)/2-source_size/2)*pixel*np.array(camera['right']) -
              ((y1+y2)/2-source_size/2)*pixel*np.array(camera['screen_up']))
    return dict(camera, origin=origin.tolist(), extent=width*pixel,
                source_box=[float(v) for v in box])


def render_camera(scene, tri, rgb, camera, size):
    import open3d as o3d
    xx, yy = np.meshgrid((np.arange(size)+.5-size/2)*camera['extent']/size,
                         (np.arange(size)+.5-size/2)*camera['extent']/size)
    starts = (np.array(camera['origin']) + xx[..., None]*np.array(camera['right']) -
              yy[..., None]*np.array(camera['screen_up']))
    directions = np.broadcast_to(np.array(camera['direction']), starts.shape)
    rays = np.concatenate([starts, directions], axis=-1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=8)
    fid = hit['primitive_ids'].numpy().astype(np.int64)
    valid = fid != 4294967295
    fid[~valid] = -1
    uv = hit['primitive_uvs'].numpy()
    bary = np.concatenate([1-uv.sum(-1, keepdims=True), uv], axis=-1)
    image = np.full((size, size, 3), 242, np.uint8)
    values = (rgb[tri[fid[valid]]] * bary[valid, :, None]).sum(1)
    image[valid] = np.rint(values.clip(0, 1)*255).astype(np.uint8)
    return image, fid.astype(np.int32), bary.astype(np.float32)


def detail_boxes(fid, tri, features, labels, size):
    """Pot context + high-linearity plant regions, all selected automatically."""
    valid = fid >= 0
    corners = tri[fid[valid]]
    line = np.zeros(fid.shape, np.float32)
    line[valid] = features[corners, 0].mean(1)
    pot = np.zeros(fid.shape, bool)
    pot[valid] = (labels[corners] == 4).mean(1) >= 2/3
    boxes = []
    if pot.sum() >= 100:
        yy, xx = np.where(pot)
        # Include the entire pot and plant attachment as semantic context.
        side = float(max(np.ptp(xx), np.ptp(yy))*1.35)
        cx = float((xx.min()+xx.max())/2)
        cy = float(yy.min() + .25*side)
        boxes.append(('pot_opening', [cx-side/2, cy-side/2, cx+side/2, cy+side/2]))
    plant = valid & ~pot
    priority = (line-.55).clip(0)*plant
    # Uniform non-maximum selection prevents hand-picked image regions.
    from scipy.ndimage import uniform_filter
    side = round(size*.36)
    support = uniform_filter(priority, size=side, mode='constant')
    for _ in range(3):
        y, x = np.unravel_index(np.argmax(support), support.shape)
        if support[y, x] < .0003:
            break
        boxes.append(('linear_parts', [x-side/2, y-side/2, x+side/2, y+side/2]))
        yy, xx = np.ogrid[:size, :size]
        support[(xx-x)**2+(yy-y)**2 < (side*.65)**2] = 0
    return boxes


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'views').mkdir(exist_ok=True)
    source = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    if p.sha256(source['input']) != source['input_sha256']:
        raise ValueError('Source mesh changed')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        check_source(args.source, old)
        if old.get('render_revision') != RENDER_REVISION or old['size'] != args.size:
            raise ValueError('Detail render recipe changed; use a new output directory')
        print('Prepared detail views already exist', flush=True)
        return
    xyz, tri, rgb = p.mesh_data(source['input'])
    features = shape_features(xyz)
    np.savez_compressed(out/'geometry.npz', features=features,
                        columns=['linearity_small','planarity_small','scattering_small',
                                 'linearity_large','planarity_large','scattering_large'])
    labels = np.load(args.source/'labels.npz')['vertex_labels']
    scene = p.scene_from_arrays(xyz, tri)
    cameras, tiles = [], []
    selected = [v for i, v in enumerate(source['views']) if i%2 == 0]
    for parent in selected:
        ray = np.load(args.source/'views'/f"{parent['name']}_rays.npz")
        boxes = detail_boxes(ray['face_id'], tri, features, labels, source['size'])
        for kind, box in boxes:
            camera = camera_crop(parent, box, source['size'])
            camera.update(name=f"detail_{len(cameras):03d}", parent=parent['name'], kind=kind)
            image, fid, bary = render_camera(scene, tri, rgb, camera, args.size)
            Image.fromarray(image).save(out/'views'/f"{camera['name']}.png")
            np.savez_compressed(out/'views'/f"{camera['name']}_rays.npz", face_id=fid, barycentric=bary)
            cameras.append(camera)
            tile = Image.fromarray(image).resize((280, 280))
            ImageDraw.Draw(tile).text((5, 5), f"{camera['name']} {kind} {parent['name']}", fill='black')
            tiles.append(tile)
            print('Rendered', camera['name'], kind, 'parent', parent['name'], flush=True)
    sheet = Image.new('RGB', (4*280, math.ceil(len(tiles)/4)*280), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i%4)*280, (i//4)*280))
    sheet.save(out/'render_contact_sheet.jpg')
    p.save_json(out/'manifest.json', dict(source, views=cameras, size=args.size,
                source_directory=str(args.source.resolve()),
                source_labels_sha256=p.sha256(args.source/'labels.npz'),
                render_revision=RENDER_REVISION,
                status='detail_rendered_only', semantics_validated=False))
    p.save_json(out/'classes.json', p.CLASSES)


def dense_single(image, processor, model, device):
    import torch
    import torch.nn.functional as F
    scores = np.zeros((image.height, image.width, 6), np.float32)
    for start in range(0, len(p.DENSE_PHRASES), 4):
        phrases = p.DENSE_PHRASES[start:start+4]
        inputs = processor(text=[s for s, _ in phrases], images=[image]*len(phrases),
                           padding=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        pred = F.interpolate(logits[:, None], (image.height, image.width),
                             mode='bilinear', align_corners=False)[:, 0].sigmoid().cpu().numpy()
        for i, (_, cls) in enumerate(phrases):
            scores[..., cls] = np.maximum(scores[..., cls], pred[i])
    return scores


def predict(args):
    import torch
    out = args.output
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    check_source(args.source, manifest)
    processor, detector, sam, device = p.load_backends(args)
    dense_processor, model = p.load_dense(args, device)
    config = dict(method='automatic closeups, frozen CLIPSeg+GroundingDINO+MobileSAM',
                  detector_sha256=p.sha256(args.detector/'model.safetensors'),
                  dense_sha256=p.sha256(args.dense/'model.safetensors'),
                  sam_sha256=p.sha256(args.sam), phrases=p.PHRASES,
                  dense_phrases=p.DENSE_PHRASES, box_threshold=args.box_threshold,
                  manifest_sha256=p.sha256(out/'manifest.json'), device=device,
                  recipe_revision=1, camera_votes_shared_with_parent=True,
                  manual_annotations=False, accuracy_measured=False)
    config['cache_key'] = signature(config)
    if (out/'inference_config.json').exists():
        old = json.loads((out/'inference_config.json').read_text(encoding='utf-8'))
        if old.get('cache_key') != config['cache_key']:
            raise ValueError('Detail inference configuration changed; use a new output directory')
    p.save_json(out/'inference_config.json', config)
    for camera in manifest['views'][:args.limit or None]:
        name = camera['name']
        target = out/'views'/f'{name}_scores.npz'
        if target.exists():
            if str(np.load(target)['cache_key']) != config['cache_key']:
                raise ValueError(f'Stale cached prediction {name}')
            print('Cached', name, flush=True)
            continue
        start = time.time()
        image = Image.open(out/'views'/f'{name}.png').convert('RGB')
        rgb = np.asarray(image)
        fid = np.load(out/'views'/f'{name}_rays.npz')['face_id']
        dense = dense_single(image, dense_processor, model, device)
        detections = p.detect(image, processor, detector, device, args.box_threshold)
        sam.set_image(rgb)
        scores = dense*.6
        scores[fid < 0] = 0
        for det in detections:
            masks, quality, _ = sam.predict(box=np.array(det['box']), multimask_output=True)
            cls = det['class_id']
            options = []
            for mask in masks:
                mask = mask & (fid >= 0)
                means = dense[mask].mean(0) if mask.any() else np.zeros(6)
                options.append(means[cls]-np.delete(means, [0, cls]).max())
            k = int(np.argmax(np.array(options)+quality*.15))
            mask = masks[k] & (fid >= 0)
            yy, xx = np.ogrid[:fid.shape[0], :fid.shape[1]]
            x1, y1, x2, y2 = det['box']
            mask &= (xx >= x1-2)&(xx <= x2+2)&(yy >= y1-2)&(yy <= y2+2)
            means = dense[mask].mean(0) if mask.any() else np.zeros(6)
            rank = int((means[1:] > means[cls]).sum()+1)
            local = mask.sum() < .16*(fid >= 0).sum()
            accepted = bool(means[cls] >= .15 and
                            (means[cls] >= np.delete(means, [0, cls]).max()-.06 or (local and rank <= 2)))
            score = float(det['detector_score']*np.clip(quality[k], 0, 1))
            det.update(pixels=int(mask.sum()), semantic_validated=accepted, evidence=score,
                       sam_quality=float(quality[k]), dense_class_mean=float(means[cls]))
            if accepted:
                scores[..., cls][mask] = np.maximum(scores[..., cls][mask], .6*dense[..., cls][mask]+.7*score)
        labels, _, _ = p.evidence_labels(scores)
        np.savez_compressed(target, scores=scores.astype(np.float16), dense=dense.astype(np.float16),
                            cache_key=config['cache_key'])
        Image.fromarray(labels).save(out/'views'/f'{name}_labels.png')
        overlay = rgb.copy()
        fg = fid >= 0
        overlay[fg] = (rgb[fg]*.4+p.COLORS[labels[fg]]*.6).astype(np.uint8)
        Image.fromarray(overlay).save(out/'views'/f'{name}_overlay.jpg')
        p.save_json(out/'views'/f'{name}_detections.json', detections)
        print(name, np.bincount(labels[fg], minlength=6).tolist(), 'seconds', round(time.time()-start, 1), flush=True)


def pot_envelope(xyz, labels, up_axis):
    """Conservative single-container envelope from the zero-shot pot seeds.

    A geometric eligibility constraint, never a positive substrate label.
    Reject the constraint when the inferred container is not compact/round.
    """
    up = 'xyz'.index(up_axis)
    horizontal = [i for i in range(3) if i != up]
    seed = xyz[labels == 4]
    if len(seed) < 500:
        return None
    low, high = np.quantile(seed[:, up], [.01, .995])
    selected = seed[(seed[:, up] < np.quantile(seed[:, up], .85))]
    lo, hi = np.quantile(selected[:, horizontal], [.01, .99], axis=0)
    center = (lo+hi)/2
    axes = (hi-lo)/2
    if axes.min() <= 0 or axes.max()/axes.min() > 1.8:
        return None
    radial = np.sqrt((((xyz[:, horizontal]-center)/axes)**2).sum(1))
    radius = float(np.mean(axes))
    # Restrict soil evidence to the inner opening and a band near the rim.
    substrate = (radial < .90)&(xyz[:, up] > high-.25*radius)&(xyz[:, up] < high+.10*radius)
    pot = (radial < 1.25)&(xyz[:, up] < high+.12*radius)&(xyz[:, up] > low-.12*radius)
    return dict(substrate=substrate, pot=pot, center=center.tolist(), axes=axes.tolist(),
                rim_height=float(high), bottom_height=float(low), up_axis=up_axis,
                applicability='heuristic compact single pot; not validated on arbitrary containers')


def comparison_sheet(out, detail, tri, before, after):
    from PIL import ImageFont
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    chinese = font_path.exists()
    font = ImageFont.truetype(str(font_path),22) if chinese else ImageFont.load_default(size=22)
    small = ImageFont.truetype(str(font_path),18) if chinese else ImageFont.load_default(size=18)
    # One pot opening and linear region from each of the first two camera heights.
    selected = []
    elevations = sorted(set(v['elevation'] for v in detail['views']))[:2]
    for elevation in elevations:
        group = [v for v in detail['views'] if v['elevation'] == elevation]
        for kind in ['pot_opening','linear_parts']:
            matching = [v for v in group if v['kind'] == kind and
                        (out/'views'/f"{v['name']}_overlay.jpg").exists()]
            if matching:
                selected.append(matching[0])
    if not selected:
        return
    width = 360
    sheet = Image.new('RGB',(width*4,80+width*len(selected)+56),'white')
    draw = ImageDraw.Draw(sheet)
    titles = ['原始颜色','局部二维预测','原方案三维结果','局部放大＋几何约束'] if chinese else ['Original RGB','Local 2D predictions','Baseline 3D labels','Closeup + geometry']
    for i,title in enumerate(titles):
        draw.text((i*width+10,12),title,font=font,fill='black')
    for row,view in enumerate(selected):
        name = view['name']
        original = Image.open(out/'views'/f'{name}.png').convert('RGB')
        ray = np.load(out/'views'/f'{name}_rays.npz')
        fid,bary = ray['face_id'],ray['barycentric']
        ok = fid >= 0
        corners = tri[fid[ok]]
        vertices = corners[np.arange(len(corners)),bary[ok].argmax(1)]
        images = [original,Image.open(out/'views'/f'{name}_overlay.jpg')]
        for label in [before,after]:
            array = np.asarray(original).copy()
            array[ok] = p.COLORS[label[vertices]]
            images.append(Image.fromarray(array))
        for col,img in enumerate(images):
            sheet.paste(img.resize((width,width)),(col*width,60+row*width))
        draw.text((8,60+row*width+5),name,font=small,fill='black')
    legend = ['未知','茎/枝/花梗','叶片','花/花苞','花盆/托盘','土壤/基质'] if chinese else [c['name'] for c in p.CLASSES]
    y = 65+len(selected)*width
    for i,cls in enumerate(p.CLASSES):
        x = i*235
        draw.rectangle((x+6,y,x+28,y+22),fill=tuple(cls['color']))
        draw.text((x+36,y-2),legend[i],font=small,fill='black')
    sheet.save(out/'comparison_preview.jpg',quality=92)


def fuse(args):
    out = args.output
    source = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    detail = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    check_source(args.source, detail)
    source_inference = json.loads((args.source/'inference_config.json').read_text(encoding='utf-8'))
    inference = json.loads((out/'inference_config.json').read_text(encoding='utf-8'))
    if inference['manifest_sha256'] != p.sha256(out/'manifest.json'):
        raise ValueError('Detail cameras changed after inference')
    xyz, tri, rgb = p.mesh_data(source['input'])
    scene = p.scene_from_arrays(xyz, tri)
    baseline = np.load(args.source/'labels.npz')
    features = np.load(out/'geometry.npz')['features']
    envelope = pot_envelope(xyz, baseline['vertex_labels'], source['views'][0]['up_axis'])
    total = np.zeros((len(xyz), 6), np.float32)
    count = np.zeros(len(xyz), np.uint16)
    votes = np.zeros((len(xyz), 6), np.uint16)
    detail_count = np.zeros(len(xyz), np.uint16)
    # A line-like region can receive a small stem bonus only if the model itself
    # gives stem evidence. This also occurs on petal edges, so no hard assignment.
    tubular = ((features[:, 3]-.45)/.40).clip(0, 1)
    for parent in source['views']:
        name = parent['name']
        cached = np.load(args.source/'views'/f'{name}_scores.npz')
        if str(cached['cache_key']) != source_inference['cache_key']:
            raise ValueError(f'Stale parent prediction {name}')
        values, seen = p.sample_visible_vertices(scene, xyz, parent, cached['scores'])
        local_sum = np.zeros_like(values)
        local_weight = np.zeros(len(xyz), np.float32)
        for camera in [v for v in detail['views'] if v['parent'] == name]:
            path = out/'views'/f"{camera['name']}_scores.npz"
            if not path.exists():
                continue
            cached_detail = np.load(path)
            if str(cached_detail['cache_key']) != inference['cache_key']:
                raise ValueError(f'Stale cached prediction {path.name}')
            scores = cached_detail['scores']
            local, observed = p.sample_visible_vertices(scene, xyz, camera, scores)
            # Taper crop edges where recognition lacks surrounding context.
            origin = np.array(camera['origin'])
            dx = ((xyz-origin)@np.array(camera['right']))/camera['extent']
            dy = ((xyz-origin)@np.array(camera['screen_up']))/camera['extent']
            border = (.5-np.maximum(np.abs(dx), np.abs(dy)))
            weight = (border/.12).clip(0, 1).astype(np.float32)*observed
            local_sum += local*weight[:, None]
            local_weight += weight
        selected = local_weight > 0
        detail_count += selected
        local_sum[selected] /= local_weight[selected, None]
        # Crops of the same view share one vote. Their 2D errors are correlated.
        values[selected] = .25*values[selected]+.75*local_sum[selected]
        seen |= selected
        if envelope is not None:
            values[~envelope['substrate'], 5] = 0
            values[~envelope['pot'], 4] = 0
        if args.shape_bonus:
            support = (values[:, 1] > .18)&(values[:, 1] >= values[:, 2]*.8)
            values[:, 1] += .08*tubular*support
        label, _, _ = p.evidence_labels(values)
        total += values
        count += seen
        index = np.where(seen)[0]
        votes[index, label[index]] += 1
        print('Fused', name, 'detail vertices', int(selected.sum()), flush=True)
    mean = total/np.maximum(count[:, None], 1)
    label, evidence, margin = p.evidence_labels(mean)
    agreement = votes[np.arange(len(xyz)), label]/np.maximum(count, 1)
    label[(count < 2)|(agreement < .6)] = 0
    boundary = np.any(label[tri] != label[tri[:, :1]], axis=1)
    face_labels = label[tri[:, 0]].copy()
    face_labels[boundary] = 0
    area = np.linalg.norm(np.cross(xyz[tri[:, 1]]-xyz[tri[:, 0]], xyz[tri[:, 2]]-xyz[tri[:, 0]]), axis=1)*.5
    vertex_area = np.zeros(len(xyz), np.float64)
    for j in range(3):
        np.add.at(vertex_area, tri[:, j], area/3)
    np.savez_compressed(out/'labels.npz', vertex_labels=label, face_labels=face_labels,
                       mean_evidence=mean, evidence=evidence, margin=margin, view_count=count,
                       detail_view_count=detail_count, agreement=agreement.astype(np.float32),
                       direct=label != 0, propagated=np.zeros(len(xyz), bool), face_boundary=boundary)
    p.write_labeled_ply(out/'semantic_mesh.ply', xyz, tri, rgb, label, evidence, True)
    p.write_labeled_ply(out/'labeled_original_rgb.ply', xyz, tri, rgb, label, evidence, False)
    counts = np.bincount(label, minlength=6)
    report = dict(status='experimental_not_accuracy_validated', source_sha256=source['input_sha256'],
                  vertex_count=len(xyz), face_count=len(tri), detail_views=len(detail['views']),
                  detail_predictions=sum((out/'views'/f"{v['name']}_scores.npz").exists() for v in detail['views']),
                  class_vertex_count={c['name']:int(counts[c['id']]) for c in p.CLASSES},
                  class_surface_area_fraction={c['name']:float(vertex_area[label == c['id']].sum()/vertex_area.sum()) for c in p.CLASSES},
                  unknown_vertex_fraction=float((label == 0).mean()), unobserved_vertex_fraction=float((count == 0).mean()),
                  accuracy_measured=False, cross_plant_validation=False, scores_are_probabilities=False,
                  manual_annotations=False, finetuned_foundation_models=False,
                  shape_bonus=args.shape_bonus, crop_weight=.75,
                  pot_envelope={k:v for k,v in envelope.items() if k not in ['substrate','pot']} if envelope else None,
                  notes=['Crop views share votes with their parent camera.',
                         'Pot geometry only removes impossible evidence; it never assigns substrate by itself.',
                         'All thresholds selected during this single-plant experiment; not an independent test.',
                         'Positive shape bonus also responds to leaf/petal edges; inspect boundaries.',
                         'Unknown coverage and class counts are not semantic accuracy.'])
    p.save_json(out/'report.json', report)
    p.review_sheet(out, detail, tri, label)
    # Retain both full and local review sheets under separate names.
    (out/'review_contact_sheet.jpg').replace(out/'detail_review_contact_sheet.jpg')
    p.review_sheet(args.source, source, tri, label, destination=out)
    comparison_sheet(out, detail, tri, baseline['vertex_labels'], label)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','predict','fuse'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=896)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--box-threshold', type=float, default=.22)
    parser.add_argument('--detector', type=Path, default=p.ROOT/'models'/'grounding-dino-tiny')
    parser.add_argument('--dense', type=Path, default=p.ROOT/'models'/'clipseg-rd64-refined')
    parser.add_argument('--sam', type=Path, default=p.ROOT/'models'/'mobile_sam_parallel.pt')
    parser.add_argument('--shape-bonus', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
