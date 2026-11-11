"""Complete narrow stem-colored rims supported by an existing leaf surface.

This is a bounded, single-pass geometric heuristic after multiview segmentation.
It never reads RGB. Original leaf seeds stay frozen; newly repaired points cannot
seed more growth. A separate source is required so reruns do not widen the rim.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import plantseg as p
from refine_leaf_boundary import leaf_cores, surface_graph

REVISION = 1
MAX_DISTANCE = .004
SUPPORT_RADIUS = .009
VOXEL_SIZE = .001
MIN_SUPPORT = .40
MIN_NEIGHBORS = 12


def complete_rim(xyz, tri, labels, bud, leaf_region, *, max_distance=MAX_DISTANCE,
                 support_radius=SUPPORT_RADIUS, voxel_size=VOXEL_SIZE,
                 min_support=MIN_SUPPORT, threads=8):
    """Return repair mask and diagnostics; all distances use bbox-diagonal units.

    Only stem/leaf edges may carry paths. The Euclidean support estimate is
    secondary to this geodesic gate, so a nearby disconnected branch cannot be
    relabeled. Voxel sampling reduces dependence on mesh tessellation density.
    A true attachment inside the small band can remain ambiguous.
    """
    import open3d as o3d
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    changed = np.zeros(len(xyz), bool)
    empty = dict(indices=np.empty(0, np.int64), distance=np.empty(0, np.float32),
                 support=np.empty(0, np.float32), neighbors=np.empty(0, np.int32))
    seeds = np.flatnonzero(leaf_region & (labels == 2) & ~bud)
    if not len(seeds):
        return changed, empty
    points = np.asarray(xyz, np.float64)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        return changed, empty
    # Normalization also makes voxel bins stable under translation and scaling.
    points = (points - points.min(axis=0)) / scale
    graph = surface_graph(points, tri)
    allowed = np.isin(labels, [1, 2]) & ~bud
    rows = np.repeat(np.arange(len(points)), np.diff(graph.indptr))
    graph.data[~(allowed[rows] & allowed[graph.indices])] = 0
    graph.eliminate_zeros()
    distance = dijkstra(graph, directed=False, indices=seeds, min_only=True,
                        limit=max_distance)
    ids = np.flatnonzero((labels == 1) & ~bud & (distance <= max_distance))
    if not len(ids):
        return changed, empty

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    samples = np.asarray(cloud.voxel_down_sample(voxel_size).points)
    _, nearest = cKDTree(points).query(samples, workers=threads)
    sample_leaf = leaf_region[nearest] & (labels[nearest] == 2) & ~bud[nearest]
    tree = cKDTree(samples)
    support = np.zeros(len(ids), np.float32)
    counts = np.zeros(len(ids), np.int32)
    # Keep memory bounded for large plant meshes.
    for start in range(0, len(ids), 4096):
        stop = min(start + 4096, len(ids))
        dist, index = tree.query(points[ids[start:stop]], k=256,
                                 distance_upper_bound=support_radius, workers=threads)
        valid = np.isfinite(dist)
        index = np.minimum(index, len(samples) - 1)
        weight = np.exp(-.5 * (dist / (support_radius * .6)) ** 2) * valid
        total = weight.sum(axis=1).clip(1e-12)
        support[start:stop] = (weight * sample_leaf[index]).sum(axis=1) / total
        counts[start:stop] = valid.sum(axis=1)
    accept = (support >= min_support) & (counts >= MIN_NEIGHBORS)
    changed[ids[accept]] = True
    return changed, dict(indices=ids, distance=distance[ids].astype(np.float32),
                         support=support, neighbors=counts)


def rim_preview(output, manifest, xyz, tri, rgb, old, new, organs):
    """Choose review views by visible changes, including surrounding branches."""
    from PIL import Image, ImageDraw, ImageFont
    from refine_detail import camera_crop, render_camera

    scene = p.scene_from_arrays(xyz, tri)
    changed = old != new
    rows = []
    for organ_id, organ in enumerate(organs):
        candidates = []
        for camera in manifest['views']:
            delta = xyz[organ['vertices']] - np.asarray(camera['origin'])
            px = delta @ np.asarray(camera['right']) / camera['extent'] * manifest['size'] + manifest['size']/2
            py = -delta @ np.asarray(camera['screen_up']) / camera['extent'] * manifest['size'] + manifest['size']/2
            side = max(np.ptp(px), np.ptp(py)) * 1.35
            if side <= 0:
                continue
            cx, cy = (px.min()+px.max())/2, (py.min()+py.max())/2
            crop = camera_crop(camera, [cx-side/2, cy-side/2, cx+side/2, cy+side/2], manifest['size'])
            original, fid, bary = render_camera(scene, tri, rgb, crop, 700)
            valid = fid >= 0
            corners = tri[fid[valid]]
            vertex = corners[np.arange(len(corners)), bary[valid].argmax(1)]
            count = int(changed[vertex].sum())
            pictures = [original]
            for labels in (old, new):
                im = original.copy()
                im[valid] = p.COLORS[labels[vertex]]
                pictures.append(im)
            candidates.append((count, camera['name'], pictures))
        candidates.sort(key=lambda row: row[0], reverse=True)
        rows.extend((organ_id, *row) for row in candidates[:2])
    if not rows:
        return
    fontpath = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(fontpath), 23) if fontpath.exists() else ImageFont.load_default(size=23)
    width = 600
    sheet = Image.new('RGB', (width*3, 50 + 635*len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(['原始颜色', '修正前', '窄叶缘补全后']):
        draw.text((col*width+10, 10), title, fill='black', font=font)
    for row, (organ_id, count, name, pictures) in enumerate(rows):
        y = 50 + row*635
        draw.text((10, y+3), f'Leaf {organ_id+1} / {name}', fill='black', font=font)
        for col, picture in enumerate(pictures):
            sheet.paste(Image.fromarray(picture).resize((width, width)), (col*width, y+35))
    sheet.save(output/'leaf_rim_comparison.jpg', quality=95)
    sheet.crop((0, 0, width*3, 50+635)).save(output/'leaf_rim_compact.jpg', quality=95)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.source == args.output:
        raise ValueError('Use a separate output directory for rim completion')
    manifest = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    if p.sha256(manifest['input']) != manifest['input_sha256']:
        raise ValueError('Source mesh identity changed')
    data = dict(np.load(args.source/'labels.npz'))
    if 'leaf_rim_refined' in data:
        raise ValueError('Rim completion is single-pass; use the original boundary-stage source')
    if (args.output/'manifest.json').exists():
        previous = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
        if previous['input_sha256'] != manifest['input_sha256']:
            raise ValueError('Output belongs to another input; use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    xyz, tri, rgb = p.mesh_data(manifest['input'])
    old = data['vertex_labels'].copy()
    organs, va = leaf_cores(xyz, tri, old, data['class_scores'])
    region = np.zeros(len(xyz), bool)
    for organ in organs:
        region[organ['vertices']] = True
    changed, diagnostic = complete_rim(xyz, tri, old, data['bud_protected'], region,
                                       threads=args.threads)
    print(f'Leaf rim: {changed.sum()} changes / {len(diagnostic["indices"])} candidates', flush=True)
    label = old.copy()
    label[changed] = 2
    scores = data['class_scores'].copy()
    scores[changed] *= .45
    scores[changed, 1] += .55
    from complete_five import closed_face_labels
    faces, boundary = closed_face_labels(tri, label, scores)
    rank = np.sort(scores, axis=1)
    data.update(vertex_labels=label, face_labels=faces, class_scores=scores,
                face_boundary=boundary, leaf_rim_refined=changed,
                margin=rank[:, -1]-rank[:, -2], evidence=scores.max(axis=1),
                uncertain=data['uncertain'] | changed)
    np.savez_compressed(args.output/'labels.npz', **data)
    np.savez_compressed(args.output/'rim_diagnostics.npz', **diagnostic)
    for name, semantic in [('semantic_mesh.ply', True), ('labeled_original_rgb.ply', False)]:
        p.write_labeled_ply(args.output/name, xyz, tri, rgb, label, data['evidence'], semantic,
                            uncertain=data['uncertain'], inferred=data['inferred'], face_labels=faces)
    config = dict(revision=REVISION, max_distance_fraction=MAX_DISTANCE,
                  support_radius_fraction=SUPPORT_RADIUS, voxel_size_fraction=VOXEL_SIZE,
                  min_leaf_support=MIN_SUPPORT, min_neighbors=MIN_NEIGHBORS,
                  max_neighbors=256, single_pass=True, input_rgb_used=False,
                  seed_leaf_regions=len(organs), source_directory=str(args.source),
                  source_labels_sha256=p.sha256(args.source/'labels.npz'),
                  code_sha256=p.sha256(Path(__file__)),
                  boundary_code_sha256=p.sha256(Path(__file__).with_name('refine_leaf_boundary.py')))
    p.save_json(args.output/'rim_config.json', config)
    report = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    report.update(method=report['method']+' + bounded leaf surface rim completion',
                  rim_source=str(args.source), leaf_rim_changed_vertices=int(changed.sum()),
                  leaf_rim_candidate_vertices=len(diagnostic['indices']),
                  leaf_rim_changed_area_fraction=float(va[changed].sum()/max(va.sum(), 1e-12)),
                  rim_transition_count={str(cls): int((changed & (old == cls)).sum()) for cls in range(1, 6)},
                  class_vertex_count={c['name']: int((label == c['id']).sum()) for c in p.CLASSES[1:]},
                  class_surface_area_fraction={c['name']: float(va[label == c['id']].sum()/max(va.sum(), 1e-12)) for c in p.CLASSES[1:]},
                  uncertain_vertex_fraction=float(data['uncertain'].mean()),
                  unknown_vertices=int((label == 0).sum()), unknown_faces=int((faces == 0).sum()))
    report['limitations'].append('Narrow rim completion is heuristic; true petiole junctions and fused surfaces can remain ambiguous.')
    p.save_json(args.output/'report.json', report)
    p.save_json(args.output/'manifest.json', dict(manifest, rim_source=str(args.source),
                rim_source_labels_sha256=config['source_labels_sha256']))
    p.save_json(args.output/'classes.json', p.CLASSES[1:])
    p.save_json(args.output/'inference_config.json', dict(recipe='boundary_then_rim', revision=REVISION,
                rim_config_sha256=p.sha256(args.output/'rim_config.json'),
                source_inference_config_sha256=p.sha256(args.source/'inference_config.json'),
                complete_closed_set=True, manual_annotations=False, uncertainty_separate=True))
    for name in ['boundary_config.json', 'boundary_records.json', 'boundary_votes.npz']:
        if (args.source/name).exists():
            shutil.copy2(args.source/name, args.output/name)
    from refine_leaf_boundary import preview
    preview(args, manifest, tri, old, label)
    rim_preview(args.output, manifest, xyz, tri, rgb, old, label, organs[:4])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
