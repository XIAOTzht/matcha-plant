#!/usr/bin/env python3
"""Reclassify complete organs from RGB multiview masks plus 3D shape checks.

Consumes a six-class metric source pair. Never infers identity from palette RGB.
Preserves vertices, faces, hard classes, source results and uncertainty history.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import plantseg as p
from organ_shape_rules import (mask_shape, coherent_mask, mask_evidence, decide_relabels, extend_surface_consensus)
from organ_surface_consistency import regularize_surface, refine_stem_rims

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'tools/plant_semantics_with_car'))
sys.path.insert(0, str(PROJECT/'tools/plant_organ_instances'))
from with_car_semantics import mesh, sha, read_json, save_json, fresh, face_assignment, COLORS
from split_plant_organs import write_fields

REVISION = 'organ-rgb-shape-review-2.0'
# Explicit organ names reduce the detector's tendency to call an entire
# arrangement "flower" and pale broad leaves "petal". No species/color rule.
REVIEW_PHRASE_GROUPS = [
    [('plant stem',1),('plant branch',1)],
    [('leaf',2),('plant leaf',2),('leaf blade',2)],
    [('dried leaf',2),('curled leaf',2),('broad leaf',2)],
    [('flower head',3),('flower bloom',3),('flower seed head',3),('flower bud',3)],
]
REVIEW_PHRASES = [entry for group in REVIEW_PHRASE_GROUPS for entry in group]


def geometric_features(xyz, threads=8):
    """Uniform voxel sampling and multiscale PCA, independent of triangle count."""
    from scipy.spatial import cKDTree
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Empty/degenerate plant geometry')
    cells = np.floor((xyz-xyz.min(0))/(scale*.003)).astype(np.int32)
    _, keep = np.unique(cells, axis=0, return_index=True)
    points = xyz[np.sort(keep)]
    tree = cKDTree(points)
    features = []
    for radius in (.014, .045):
        if len(points) < 3:
            raise ValueError('Too few spatial samples for shape review')
        distance, index = tree.query(points, k=min(128, len(points)),
                                     distance_upper_bound=scale*radius, workers=threads)
        ok = np.isfinite(distance)
        neighbours = points[np.minimum(index, len(points)-1)]
        count = np.maximum(ok.sum(1), 1)
        mean = (neighbours*ok[..., None]).sum(1)/count[:, None]
        delta = (neighbours-mean[:, None])*ok[..., None]
        covariance = np.einsum('nki,nkj->nij', delta, delta)/count[:, None, None]
        eigen = np.linalg.eigvalsh(covariance).clip(0)
        largest = np.maximum(eigen[:, 2], 1e-12)
        features.append(np.column_stack([(eigen[:, 2]-eigen[:, 1])/largest,
            (eigen[:, 1]-eigen[:, 0])/largest, eigen[:, 0]/largest]))
    _, nearest = tree.query(xyz, workers=threads)
    return np.concatenate(features, axis=1)[nearest].astype(np.float32), scale


def score_masks(image, fid, bary, xyz, tri, features, scale, labels, detector, sam, dense_backend, args):
    from scipy.ndimage import binary_erosion
    processor, model, _, device = detector
    dense_processor, dense_model = dense_backend
    dense = p.dense_evidence(Image.fromarray(image), dense_processor, dense_model, device)
    # DINO proposals depend on the joint text context. A long mixed-class
    # sentence suppresses some pale/curled leaves even when their individual
    # prompts detect them. Groups are independent proposals, never extra votes.
    detections = []
    for phrases in REVIEW_PHRASE_GROUPS:
        detections.extend(p.detect(Image.fromarray(image), processor, model, device,
                                   args.box_threshold, phrases=phrases))
    valid = fid >= 0
    corners = tri[fid[valid]]
    vertices = corners[np.arange(len(corners)), bary[valid].argmax(1)]
    pixel_vertex = np.full(fid.shape, -1, np.int32); pixel_vertex[valid] = vertices
    valid &= np.isin(labels[pixel_vertex.clip(0)], [1,2,3])
    sam.set_image(image)
    scores = np.zeros((*fid.shape, 6), np.float32)
    records = []
    # Synonyms often return identical boxes; do not run SAM repeatedly for them.
    selected = []
    for det in sorted(detections, key=lambda d: -d['detector_score']):
        a = np.asarray(det['box'])
        duplicate = False
        for other in selected:
            if other['class_id'] != det['class_id']:
                continue
            b = np.asarray(other['box'])
            inter = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
            union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
            if inter/max(union, 1e-9) > .85:
                duplicate = True
                break
        if not duplicate:
            selected.append(det)
    for det in selected:
        cls = det['class_id']
        if cls not in (1, 2, 3):
            continue
        masks, quality, _ = sam.predict(box=np.asarray(det['box']), multimask_output=True)
        best = None
        for mask, q in zip(masks, quality):
            mask = coherent_mask(mask, valid)
            x1, y1, x2, y2 = det['box']
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            mask &= (xx >= x1-2) & (xx <= x2+2) & (yy >= y1-2) & (yy <= y2+2)
            if mask.sum() < 80 or mask.sum() > valid.sum()*.65:
                continue
            ids = pixel_vertex[mask]
            shape = mask_shape(mask, xyz[ids], features[ids], scale)
            evidence = mask_evidence(cls, shape, det['detector_score'], float(q), dense[mask].mean(0))
            if evidence and (best is None or evidence > best[0]):
                best = (evidence, mask, shape, float(q))
        if best is None:
            continue
        value, mask, shape, quality = best
        # Whole mask is the voting unit. Boundary confidence falls slightly;
        # there is no pixelwise PCA-driven class switch within an organ.
        interior = binary_erosion(mask, iterations=1)
        scores[..., cls][mask] = np.maximum(scores[..., cls][mask], value*.90)
        scores[..., cls][interior] = np.maximum(scores[..., cls][interior], value)
        records.append(dict(det, sam_quality=quality, shape=shape, accepted_score=value,
                            pixels=int(mask.sum())))
    return scores, records


def infer(xyz, tri, rgb, features, scale, labels, output, args, initial=None):
    import torch
    from refine_detail import camera_crop, render_camera
    torch.set_num_threads(args.threads)
    backend = p.load_backends(args)
    dense_backend = p.load_dense(args, backend[-1])
    scene = p.scene_from_arrays(xyz, tri)
    total = np.zeros((len(xyz), 6), np.float32)
    votes = np.zeros((len(xyz), 6), np.uint16)
    observed = np.zeros(len(xyz), np.uint16)
    records = []
    cameras = []
    completed=0
    if initial is not None:
        total,votes,observed,completed=initial
        if (total.shape!=(len(xyz),6) or votes.shape!=total.shape or observed.shape!=(len(xyz),)
                or not 0<=completed<=24):
            raise ValueError('Checkpoint geometry/camera count mismatch')
        print('ORGAN_REVIEW_RECOVERED_CAMERAS',completed,flush=True)
    for elevation in (-30, 25, 65):
        for azimuth in range(0, 360, 45):
            image, fid, bary, camera = p.render_view(scene, xyz, tri, rgb, azimuth, elevation,
                                                    args.size, up_axis='y')
            name = f'view_{len(cameras):03d}'
            camera['name'] = name; cameras.append(camera)
            if len(cameras)<=completed:
                files=sorted((output/'views').glob(name+'*_masks.json'))
                if not files:
                    raise ValueError('Missing completed view masks: '+name)
                records.append(dict(camera=name,accepted_masks=sum(len(read_json(f))for f in files),
                                    visible_vertices=None,renders=len(files),recovered=True))
                continue
            # Four overlapping crops expose entire broad leaves/flower heads at
            # higher resolution. All crops share ONE vote with their parent view.
            renders = [(name, image, fid, bary, camera)]
            if elevation == 25 and azimuth % 90 == 0 and not args.no_detail:
                side = args.size*.65
                for k, (cx, cy) in enumerate(((.325,.325),(.675,.325),(.325,.675),(.675,.675))):
                    box = [cx*args.size-side/2, cy*args.size-side/2,
                           cx*args.size+side/2, cy*args.size+side/2]
                    detail = camera_crop(camera, box, args.size)
                    detail.update(name=f'{name}_detail{k}', pixel_size_world_units=detail['extent']/args.size)
                    im, f, b = render_camera(scene, tri, rgb, detail, args.size)
                    renders.append((detail['name'], im, f, b, detail))
            values = np.zeros_like(total); seen = np.zeros(len(xyz), bool)
            mask_count = 0
            for render_name, image, fid, bary, active_camera in renders:
                started=time.monotonic()
                print('ORGAN_REVIEW_RENDER_START',render_name,flush=True)
                mixed = getattr(args,'precision','float32') == 'mixed' and backend[-1] == 'cuda'
                context = torch.autocast('cuda',dtype=torch.float16) if mixed else nullcontext()
                with context:
                    scores, masks = score_masks(image, fid, bary, xyz, tri, features, scale, labels,
                        backend, backend[2], dense_backend, args)
                if not np.isfinite(scores).all():
                    raise ValueError('Non-finite organ evidence; retry with --precision float32')
                sample, visible = p.sample_visible_vertices(scene, xyz, active_camera, scores)
                values = np.maximum(values, sample); seen |= visible
                Image.fromarray(image).save(output/'views'/f'{render_name}_rgb.jpg', quality=93)
                im = image.copy(); accepted = scores.max(2) >= .12
                im[accepted] = (.25*im[accepted]+.75*p.COLORS[scores.argmax(2)[accepted]]).astype(np.uint8)
                Image.fromarray(im).save(output/'views'/f'{render_name}_mask.jpg', quality=93)
                save_json(output/'views'/f'{render_name}_masks.json', masks)
                mask_count += len(masks)
                print('ORGAN_REVIEW_RENDER', render_name, 'masks', len(masks),
                      'seconds',round(time.monotonic()-started,2),flush=True)
            total += values
            winner, _, _ = p.evidence_labels(values, minimum=.12, margin=.025)
            rows = np.flatnonzero(seen); votes[rows, winner[rows]] += 1; observed += seen
            records.append(dict(camera=name, accepted_masks=mask_count, visible_vertices=int(seen.sum()), renders=len(renders)))
            np.savez_compressed(output/'review_evidence_checkpoint.npz', total=total, votes=votes, observed=observed,
                                completed_cameras=np.int32(len(cameras)))
            print('ORGAN_REVIEW_VIEW', name, 'masks', mask_count, 'visible', int(seen.sum()), flush=True)
    save_json(output/'cameras.json', cameras)
    return total, votes, observed, records


def preview(xyz, old, new, output, coordinate_system):
    frame = np.asarray(coordinate_system['board_frame_axes'], float)
    aligned = (xyz-np.asarray(coordinate_system['board_center_mm'])) @ frame
    plant = np.isin(old, [1, 2, 3])
    ids = np.flatnonzero(plant)
    if len(ids) > 300000:
        ids = np.random.default_rng(8).choice(ids, 300000, False)
    q = aligned[ids]; width, height = 650, 750
    canvas = Image.new('RGB', (width*3, height*2), (242, 244, 247))
    for row, angle in enumerate((25, 145)):
        a = np.deg2rad(angle)
        rot = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        points = q @ rot
        lo, hi = points.min(0), points.max(0)
        factor = min((width-70)/max(hi[0]-lo[0], 1e-9), (height-90)/max(hi[1]-lo[1], 1e-9))
        xx = np.clip(((points[:, 0]-(lo[0]+hi[0])/2)*factor+width/2).astype(int), 0, width-1)
        yy = np.clip((height-40-(points[:, 1]-lo[1])*factor).astype(int), 35, height-1)
        order = np.argsort(points[:, 2])
        for col, (title, colors) in enumerate([
            ('Before: orange stem / green leaf / pink flower', COLORS[old[ids]]),
            ('After: independent RGB masks + organ geometry', COLORS[new[ids]]),
            ('Changes: yellow = relabelled; gray = retained',
             np.where((old[ids] != new[ids])[:, None], [245, 175, 30], [130, 135, 143]))]):
            tile = np.full((height, width, 3), 242, np.uint8)
            tile[yy[order], xx[order]] = colors[order]
            picture = Image.fromarray(tile); ImageDraw.Draw(picture).text((12, 12), title, fill=(25, 25, 30))
            canvas.paste(picture, (col*width, row*height))
    canvas.save(output/'semantic_comparison.jpg', quality=94)


def surface_preview(xyz,tri,rgb,old,new,output):
    """Depth-tested surface comparison, including a closeup of the lower plant."""
    from refine_detail import camera_crop,render_camera
    scene=p.scene_from_arrays(xyz,tri)
    canvas=Image.new('RGB',(900*3,950*3),'white')
    for row,(az,el) in enumerate(((0,20),(135,20),(0,15))):
        im,fid,bary,camera=p.render_view(scene,xyz,tri,rgb,az,el,900)
        if row==2:
            camera=camera_crop(camera,[180,310,720,850],900)
            im,fid,bary=render_camera(scene,tri,rgb,camera,900)
        valid=fid>=0
        corners=tri[fid[valid]]
        ids=corners[np.arange(len(corners)),bary[valid].argmax(1)]
        for col,(title,colors) in enumerate((('Original RGB',None),('Before: orange stem / green leaf / pink flower',COLORS[old]),
                                            ('After: orange stem / green leaf / pink flower',COLORS[new]))):
            tile=im.copy()
            if colors is not None:
                tile[valid]=colors[ids]
            image=Image.fromarray(tile)
            canvas.paste(image,(col*900,row*950+40))
            ImageDraw.Draw(canvas).text((col*900+12,row*950+12),title,fill='black')
    canvas.save(output/'semantic_surface_comparison.jpg',quality=95)


def export(source, output, data, old, new, means, agreement, records, xyz, tri, ply, coordinates):
    changed = old != new
    original_scores = data['class_scores'].copy()
    data['shape_review_original_scores'] = original_scores
    data['shape_review_original_labels'] = old
    data['vertex_labels'] = new
    data['uncertain'] = np.asarray(data['uncertain'], bool) | changed
    data['inferred'] = np.asarray(data['inferred'], bool) | changed
    data['label_source'][changed] = 4
    if 'contact_uncertain' in data:
        data['contact_uncertain'] = data['contact_uncertain'].astype(bool) | changed
    # Retain the original posterior for unchanged vertices. Reviewed scores are
    # separate heuristic evidence; normalise only changes for downstream fields.
    ids = np.flatnonzero(changed)
    if len(ids):
        update = means[ids, 1:4]
        data['class_scores'][ids] = 0
        data['class_scores'][ids, :3] = update/np.maximum(update.sum(1, keepdims=True), 1e-8)
        data['class_score_valid'][ids] = True
    data['organ_review_mean_evidence'] = means
    data['organ_review_agreement'] = agreement
    data['organ_review_changed'] = changed
    faces, boundary = face_assignment(tri, new)
    # Preserve every previous face decision that does not touch a changed vertex.
    old_faces=data['face_labels'].copy()
    touched = changed[tri].any(1) & ~np.isin(old_faces,[4,5,6])
    data['face_labels'][touched] = faces[touched]
    data['face_boundary'][touched] = boundary[touched]
    data['face_uncertain'][touched] = True
    np.savez_compressed(output/'labels.npz', **data)
    vertex_fields = dict(label=new, uncertain=data['uncertain'].astype('u1'),
        inferred=data['inferred'].astype('u1'), label_source=data['label_source'])
    face_fields = dict(label=data['face_labels'], uncertain=data['face_uncertain'].astype('u1'))
    checks = {}
    for name, colors in [('semantic_mesh_mm.ply', COLORS[new]), ('labeled_original_rgb_mm.ply', None)]:
        write_fields(output/name, ply, vertex_fields, colors=colors, face_fields=face_fields)
        reloaded, positions, faces_out = mesh(output/name)
        v, f = reloaded['vertex'].data, reloaded['face'].data
        expected_rgb = colors if colors is not None else np.column_stack([ply['vertex'].data[k] for k in ('red','green','blue')])
        checks[name] = dict(xyz_exact=bool(np.array_equal(positions, xyz)), faces_exact=bool(np.array_equal(faces_out, tri)),
            labels_exact=bool(np.array_equal(v['label'], new)), face_labels_exact=bool(np.array_equal(f['label'], data['face_labels'])),
            rgb_exact=bool(np.array_equal(np.column_stack([v[k] for k in ('red','green','blue')]), expected_rgb)),
            uncertainty_exact=bool(np.array_equal(v['uncertain'], data['uncertain'])))
        if not all(checks[name].values()):
            raise ValueError('Exported PLY validation failed: '+name)
    for name in ('coordinate_system.json', 'classes.json'):
        if (source/name).is_file():
            (output/name).write_bytes((source/name).read_bytes())
    report = read_json(source/'report.json') if (source/'report.json').is_file() else {}
    report.update(version=REVISION, status='REVIEW_REQUIRED', source=str(source),
        source_labels_sha256=sha(source/'labels.npz'), source_mesh_sha256=sha(source/'labeled_original_rgb_mm.ply'),
        vertices=len(xyz), faces=len(tri), changed_vertices=int(changed.sum()),
        class_counts={str(k): int((new == k).sum()) for k in range(7)},
        transitions={f'{a}->{b}': int(((old == a) & (new == b)).sum())
                     for a in (1, 2, 3) for b in (1, 2, 3) if a != b},
        hard_vertices_exact=bool(np.array_equal(old[np.isin(old, [4, 5, 6])], new[np.isin(old, [4, 5, 6])])),
        hard_faces_exact=bool(np.array_equal(old_faces[np.isin(old_faces,[4,5,6])],data['face_labels'][np.isin(old_faces,[4,5,6])])),
        geometry_unchanged=True, historical_uncertainty_preserved=True,
        semantic_accuracy_measured=False, cameras=records,
        limitations=['Geometry cannot uniquely distinguish every leaf and petal.',
                     'Surface regularization can infer labels without an independent vote at every vertex; inspect fused surfaces.',
                     'Changed labels are uncertain until the comparison views are reviewed.',
                     'Previous weights and paths belong to previous labels and must be regenerated.'])
    report.setdefault('label_source_ids', {})['4'] = 'RGB multiview organ evidence and surface-consistency inference'
    names = ['unknown','stem','leaf','flower','pot','substrate','cart']
    report['class_vertex_counts'] = {name:int((new==k).sum()) for k,name in enumerate(names)}
    report['uncertain_vertices'] = int(data['uncertain'].sum())
    report['uncertain_faces'] = int(data['face_uncertain'].sum())
    if not report['hard_vertices_exact'] or not report['hard_faces_exact']:
        raise ValueError('Hard classes changed unexpectedly')
    save_json(output/'validation.json', dict(status='PASS', checks=checks, semantic_accuracy_measured=False))
    save_json(output/'report.json', report)
    preview(xyz, old, new, output, coordinates)
    (output/'ORGAN_SEMANTIC_REVIEW_EXPORTED').write_text(REVISION+'\n', encoding='ascii')
    print('ORGAN_SEMANTIC_REVIEW_EXPORTED', output, 'changed', int(changed.sum()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    ap.add_argument('--precision',choices=['mixed','float32'],default='mixed',
                    help='Mixed CUDA inference reduces peak memory; CPU always uses float32')
    ap.add_argument('--box-threshold', type=float, default=.20)
    ap.add_argument('--no-detail', action='store_true', help='Skip localized render crops (full views still run)')
    ap.add_argument('--evidence-dir',type=Path,
        help='Explicitly reuse a completed matching RGB review; validate source/model/inference recipe before new postprocessing')
    ap.add_argument('--detector', type=Path, default=p.ROOT/'models/grounding-dino-tiny')
    ap.add_argument('--sam', type=Path, default=p.ROOT/'models/mobile_sam_parallel.pt')
    ap.add_argument('--dense', type=Path, default=p.ROOT/'models/clipseg-rd64-refined')
    args = ap.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output in source.parents or args.size < 64:
        raise ValueError('Use a separate, fresh output and resolution >=64')
    # The semantic-palette mesh cannot be substituted here. Inspecting class
    # colors as if they were original RGB would reinforce exactly the old errors.
    rgb_mesh = source/'labeled_original_rgb_mm.ply'
    if not rgb_mesh.is_file():
        raise ValueError(f'Original RGB is required: {rgb_mesh}; fetch it from the server')
    ply, xyz, tri = mesh(rgb_mesh)
    with np.load(source/'labels.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    old = data['vertex_labels'].copy()
    if not np.array_equal(ply['vertex'].data['label'], old):
        raise ValueError('Source mesh and labels disagree')
    if not np.array_equal(ply['face'].data['label'], data['face_labels']):
        raise ValueError('Source face labels and NPZ disagree')
    if not np.array_equal(data['class_score_ids'], np.arange(1,7)):
        raise ValueError('Expected class scores ordered stem,leaf,flower,pot,soil,cart')
    coordinates = read_json(source/'coordinate_system.json')
    if coordinates['units'] != 'mm':
        raise ValueError('Expected metric coordinates')
    fresh(output); (output/'views').mkdir()
    save_json(output/'review_config.json', dict(version=REVISION,
        code_sha256=sha(Path(__file__)), rules_sha256=sha(Path(__file__).with_name('organ_shape_rules.py')),
        consistency_sha256=sha(Path(__file__).with_name('organ_surface_consistency.py')),
        phrases=REVIEW_PHRASES, phrase_groups=REVIEW_PHRASE_GROUPS,
        source_labels_sha256=sha(source/'labels.npz'),
        detector_sha256=sha(args.detector/'model.safetensors'),
        dense_sha256=sha(args.dense/'model.safetensors'), sam_sha256=sha(args.sam),
        size=args.size, local_crops=not args.no_detail, box_threshold=args.box_threshold,
        precision=args.precision,
        scores_are_probabilities=False))
    frame = np.asarray(coordinates['board_frame_axes'], float)
    upright = (xyz-np.asarray(coordinates['board_center_mm'])) @ frame
    keep_faces = ~np.any(old[tri] == 6, axis=1)
    plant_faces = tri[keep_faces]
    plant_ids = np.unique(plant_faces)
    remap = np.full(len(xyz), -1, np.int32); remap[plant_ids] = np.arange(len(plant_ids))
    positions = upright[plant_ids].astype(np.float32); faces = remap[plant_faces]
    vertex = ply['vertex'].data
    rgb = np.column_stack([vertex[k] for k in ('red', 'green', 'blue')])[plant_ids].astype(np.float32)/255
    print('ORGAN_SHAPE_FEATURES', len(plant_ids), flush=True)
    features, scale = geometric_features(positions, args.threads)
    if args.evidence_dir:
        previous=args.evidence_dir.resolve()
        previous_config=read_json(previous/'review_config.json')
        previous_report=read_json(previous/'report.json')
        current=read_json(output/'review_config.json')
        required=('phrase_groups','source_labels_sha256','detector_sha256','dense_sha256',
                  'sam_sha256','size','local_crops','box_threshold','precision')
        if any(previous_config.get(k)!=current.get(k)for k in required):
            raise ValueError('Cached RGB inference recipe/source/models differ; recompute evidence')
        if previous_report['source_mesh_sha256']!=sha(rgb_mesh):
            raise ValueError('Cached evidence belongs to different original geometry/RGB')
        if previous_config['rules_sha256']!=current['rules_sha256']:
            raise ValueError('Mask evidence rules differ; recompute inference')
        with np.load(previous/'review_evidence_checkpoint.npz',allow_pickle=False)as z:
            if int(z['completed_cameras'])!=24:
                raise ValueError('Evidence is incomplete')
            total,votes,observed=z['total'].copy(),z['votes'].copy(),z['observed'].copy()
        if total.shape!=(len(positions),6)or votes.shape!=total.shape or observed.shape!=(len(positions),):
            raise ValueError('Evidence/geometry dimensions differ')
        records=previous_report['cameras']
        for name in ('review_evidence_checkpoint.npz','cameras.json'):
            shutil.copy2(previous/name,output/name)
        save_json(output/'evidence_provenance.json',dict(source=str(previous),
            checkpoint_sha256=sha(previous/'review_evidence_checkpoint.npz'),
            config_sha256=sha(previous/'review_config.json'),source_and_inference_validated=True))
        print('ORGAN_REVIEW_EXISTING_EVIDENCE_VERIFIED',previous,flush=True)
    else:
        total, votes, observed, records = infer(positions, faces, rgb, features, scale, old[plant_ids], output, args)
    reviewed, changed, means, agreement = decide_relabels(old[plant_ids],
        data['class_scores'][plant_ids], total, votes, observed)
    reviewed,extension=extend_surface_consensus(positions,faces,rgb,old[plant_ids],
        total,votes,observed,features,scale,reviewed)
    data['surface_consensus_extended']=np.zeros(len(xyz),bool)
    data['surface_consensus_extended'][plant_ids]=extension
    print('ORGAN_SURFACE_EXTENSION',int(extension.sum()),flush=True)
    print('ORGAN_SURFACE_REGULARIZATION_START',flush=True)
    consistent,consistency_report=regularize_surface(positions,faces,rgb,reviewed,
        total,votes,features,scale)
    data['organ_consistency_changed']=np.zeros(len(xyz),bool)
    data['organ_consistency_changed'][plant_ids]=consistent!=reviewed
    reviewed=consistent
    save_json(output/'surface_consistency_report.json',consistency_report)
    print('ORGAN_SURFACE_REGULARIZATION_DONE',consistency_report['changed_vertices'],flush=True)
    print('ORGAN_STEM_RIM_REVIEW_START',flush=True)
    rim_reviewed,rim_report=refine_stem_rims(positions,faces,rgb,reviewed,
        total,votes,features,scale)
    data['stem_rim_changed']=np.zeros(len(xyz),bool)
    data['stem_rim_changed'][plant_ids]=rim_reviewed!=reviewed
    reviewed=rim_reviewed
    save_json(output/'stem_rim_report.json',rim_report)
    print('ORGAN_STEM_RIM_REVIEW_DONE',rim_report['changed_vertices'],flush=True)
    new = old.copy(); new[plant_ids] = reviewed
    full_means = np.zeros((len(xyz), 6), np.float32); full_means[plant_ids] = means
    full_agreement = np.zeros(len(xyz), np.float32); full_agreement[plant_ids] = agreement
    export(source, output, data, old, new, full_means, full_agreement, records, xyz, tri, ply, coordinates)
    surface_preview(positions,faces,rgb,old[plant_ids],new[plant_ids],output)


if __name__ == '__main__':
    main()
