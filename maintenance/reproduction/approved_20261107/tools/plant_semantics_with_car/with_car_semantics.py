#!/usr/bin/env python3
"""V3 adapter: plate-relative cart rule, upright plant inference, exact mm merge.

This is not a trained cart detector. It assumes the pot and plant are above the
upper plate, with cart, black cover and support hardware below that plate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image, ImageDraw

VERSION = 'with-car-semantics-0.1.1'
CLASSES = [
    dict(id=0, name='unknown', color=[140, 140, 140]),
    dict(id=1, name='stem', color=[245, 155, 35]),
    dict(id=2, name='leaf', color=[40, 190, 85]),
    dict(id=3, name='flower', color=[235, 75, 165]),
    dict(id=4, name='pot', color=[55, 130, 240]),
    dict(id=5, name='substrate', color=[140, 95, 55]),
    dict(id=6, name='cart', color=[40, 210, 220]),
]
COLORS = np.array([c['color'] for c in CLASSES], dtype=np.uint8)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def mesh(path):
    ply = PlyData.read(str(path), known_list_len={'face': {'vertex_indices': 3}})
    v = ply['vertex'].data
    xyz = np.column_stack([v[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    tri = np.asarray(ply['face'].data['vertex_indices'], dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri):
        raise ValueError('Triangle mesh required')
    if not np.isfinite(xyz).all() or tri.min() < 0 or tri.max() >= len(xyz):
        raise ValueError('Invalid mesh geometry')
    return ply, xyz, tri


def fresh(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Use a fresh output directory: {directory}')
    directory.mkdir(parents=True, exist_ok=True)


def preview(xyz, rgb, path, title):
    ids = np.random.default_rng(3).choice(len(xyz), min(180000, len(xyz)), replace=False)
    q, rgb = xyz[ids], rgb[ids]
    image = Image.new('RGB', (1500, 650), (35, 39, 46))
    draw = ImageDraw.Draw(image)
    for panel, axes in enumerate(((0, 1), (2, 1), (0, 2))):
        xy = q[:, axes]
        lo, hi = xy.min(0), xy.max(0)
        factor = min(460/max(hi[0]-lo[0], 1e-9), 570/max(hi[1]-lo[1], 1e-9))
        pix = (xy-(lo+hi)/2)*factor
        px = np.clip(np.rint(pix[:, 0]+panel*500+250), panel*500, panel*500+499).astype(int)
        py = np.clip(np.rint(345-pix[:, 1]), 45, 649).astype(int)
        # Stable depth ordering for an orthogonal point preview.
        depth = ({0, 1, 2}-set(axes)).pop()
        for i in np.argsort(q[:, depth]):
            draw.point((int(px[i]), int(py[i])), fill=tuple(map(int, rgb[i])))
        draw.text((panel*500+10, 10), title, fill='white')
    image.save(path)


def frame_from_scale(scale):
    frame = np.array(scale['board_frame_axes'], dtype=float)
    center = np.array(scale['board_center_original'], dtype=float)
    factor = float(scale['scale_mm_per_model_unit'])
    if (frame.shape != (3, 3) or center.shape != (3,) or not np.isfinite(center).all()
            or not np.allclose(frame.T@frame, np.eye(3), atol=1e-5)
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError('Invalid scale report frame or factor')
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1
    return frame, center, factor


def split_vertices(xyz, scale, split_mm, band_mm):
    frame, center, factor = frame_from_scale(scale)
    aligned = (xyz-center)@frame
    height_mm = aligned[:, 1]*factor
    cart = height_mm <= split_mm
    boundary = np.abs(height_mm-split_mm) <= band_mm
    return aligned, cart, boundary, frame, center, factor


def prepare(args):
    if not 0 <= args.split_mm <= 10 or not 0 < args.band_mm <= 20:
        raise ValueError('Invalid split/review band in millimetres')
    src, metric, report = args.input.resolve(), args.metric.resolve(), args.scale_report.resolve()
    out = args.output.resolve()
    config = dict(version=VERSION, input=str(src), metric=str(metric), scale_report=str(report),
                  input_sha256=sha(src), metric_sha256=sha(metric), scale_sha256=sha(report),
                  split_mm=args.split_mm, boundary_band_mm=args.band_mm)
    manifest_path = out/'prepare.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if all(old.get(k) == v for k, v in config.items()):
            for name, digest in old['artifacts_sha256'].items():
                if sha(out/name) != digest:
                    raise ValueError(f'Changed prepared artifact: {name}')
            print('WITH_CAR_PREPARE_REUSED', flush=True)
            return
        raise ValueError('Existing preparation belongs to different input/configuration')
    fresh(out)
    print('Checking original and millimetre meshes', flush=True)
    ply, xyz, tri = mesh(src)
    metric_ply, mm, mm_tri = mesh(metric)
    scale = read_json(report)
    if len(xyz) != scale['vertices'] or len(tri) != scale['faces']:
        raise ValueError('Scale report does not match crop counts')
    aligned, cart, boundary, frame, center, factor = split_vertices(xyz, scale, args.split_mm, args.band_mm)
    if not np.array_equal(tri, mm_tri) or not np.allclose(mm, xyz*factor, rtol=2e-7, atol=.0001):
        raise ValueError('Millimetre mesh does not match original geometry/order and scale')
    for key in ('red', 'green', 'blue'):
        if not np.array_equal(ply['vertex'].data[key], metric_ply['vertex'].data[key]):
            raise ValueError('Original and metric mesh colors differ')
    plant_faces = np.flatnonzero((~cart[tri]).all(axis=1))
    if len(plant_faces) < 100 or cart.sum() < 100:
        raise ValueError('Implausible plant/cart split')
    used = np.unique(tri[plant_faces])
    mapping = np.full(len(xyz), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    sub_tri = mapping[tri[plant_faces]]
    vertices = ply['vertex'].data[used].copy()
    for i, key in enumerate(('x', 'y', 'z')):
        vertices[key] = aligned[used, i]
    # Normals, if present, must follow the rotation for inference geometry.
    if all(k in vertices.dtype.names for k in ('nx', 'ny', 'nz')):
        normals = np.column_stack([vertices[k] for k in ('nx', 'ny', 'nz')])@frame
        for i, key in enumerate(('nx', 'ny', 'nz')):
            vertices[key] = normals[:, i]
    face = np.empty(len(sub_tri), dtype=[('vertex_indices', '<i4', (3,))])
    face['vertex_indices'] = sub_tri
    PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(face, 'face')], text=False,
            comments=['V3 inference only: upright y, original model units; see prepare.json']).write(str(out/'plant_upright.ply'))
    orphan = (~cart)&(mapping < 0)
    np.savez_compressed(out/'mapping.npz', plant_vertex_indices=used, plant_face_indices=plant_faces,
                        cart_mask=cart, contact_uncertain=boundary, unprocessed_plant=orphan)
    rgb = np.column_stack([ply['vertex'].data[k] for k in ('red', 'green', 'blue')])
    show = rgb.copy()
    show[cart] = COLORS[6]
    show[boundary|orphan] = [255, 190, 30]
    preview(aligned, show, out/'cart_partition_preview.jpg', 'Cyan=cart; yellow=review band')
    preview(aligned[used], rgb[used], out/'plant_input_preview.jpg', 'Plant + pot for V3 (upright)')
    transform = np.eye(4)
    transform[:3, :3] = frame.T
    transform[:3, 3] = -frame.T@center
    config.update(vertex_count=len(xyz), face_count=len(tri), plant_vertex_count=len(used),
                  plant_face_count=len(sub_tri), cart_vertex_count=int(cart.sum()),
                  boundary_review_vertices=int(boundary.sum()), unprocessed_plant_vertices=int(orphan.sum()),
                  original_to_inference=transform.tolist(), inference_to_original=np.linalg.inv(transform).tolist(),
                  original_to_mm=np.diag([factor, factor, factor, 1.]).tolist(),
                  rule='cart = height <= split_mm relative to detected upper support plate',
                  limitations=['No learned cart detector or rigid template registration.',
                               'Assumes all plant and pot geometry is above the upper plate.',
                               'Dangling leaves below plate or tall cart accessories require a different separation.',
                               'Contact strip may contain pot bottom or plate screws; marked uncertain.',
                               'All full-mesh triangles are retained at final merge.'],
                  artifacts_sha256={name:sha(out/name) for name in ('plant_upright.ply', 'mapping.npz')})
    save_json(manifest_path, config)
    print(f'WITH_CAR_PREPARE_SUCCESS: {len(used):,} plant vertices; {cart.sum():,} cart vertices', flush=True)


def face_assignment(tri, labels):
    corner = labels[tri]
    boundary = np.any(corner != corner[:, :1], axis=1)
    result = np.zeros(len(tri), dtype=np.uint8)
    for cls in range(1, 7):
        result[(corner == cls).sum(axis=1) >= 2] = cls
    return result, boundary


def hardware_mask(xyz_mm, scale):
    """Small cylinders around the four measured plate corners, not flower color.

    Radius includes a fastener inset from each plate corner. Heights are in mm
    relative to the reference plane, so the pot/flowers above are excluded.
    """
    frame, center, factor = frame_from_scale(scale)
    corners = np.asarray(scale['estimated_corners_original'], dtype=float)
    if corners.shape != (4, 3) or not np.isfinite(corners).all():
        raise ValueError('Four finite calibrated plate corners required')
    local = (xyz_mm-center*factor)@frame
    cq = (corners-center)@frame*factor
    width = float(min(scale['known_plate_mm']))
    radius = min(14., .12*width)
    distance = np.full(len(local), np.inf)
    for corner in cq:
        distance = np.minimum(distance, np.linalg.norm(local[:, [0, 2]]-corner[[0, 2]], axis=1))
    height = local[:, 1]
    mask = (distance <= radius)&(height >= -3.)&(height <= 8.)
    edge = mask&((distance > radius-1.)|(height > 7.))
    return mask, edge, dict(radius_mm=radius, min_height_mm=-3., max_height_mm=8.,
                            rule='four calibrated upper-plate corner cylinders; geometry, no color threshold',
                            limitation='Assumes no plant part rests within 8 mm above plate corners; inspect preview')


def override_hardware(xyz_mm, tri, data, scale):
    mask, edge, config = hardware_mask(xyz_mm, scale)
    before = data['vertex_labels'].copy()
    changed = mask&(before != 6)
    data['vertex_labels'][mask] = 6
    # Source 3 identifies the added hardware rule, distinct from V3 evidence.
    data['label_source'][mask] = 3
    data['inferred'][mask] = True
    data['uncertain'] |= edge
    data['class_scores'][mask] = np.nan
    data['class_score_valid'][mask] = False
    affected = changed[tri].any(axis=1)
    replacement, _ = face_assignment(tri[affected], data['vertex_labels'])
    data['face_labels'][affected] = replacement
    _, boundary = face_assignment(tri, data['vertex_labels'])
    data['face_boundary'] = boundary
    data['face_uncertain'] |= data['uncertain'][tri].any(axis=1)|boundary|(data['face_labels'] == 0)
    data['cart_hardware_mask'] = mask
    data['cart_hardware_changed'] = changed
    stats = dict(config, selected_vertices=int(mask.sum()), changed_vertices=int(changed.sum()),
                 changed_faces=int(affected.sum()),
                 previous_labels={str(i):int((changed&(before == i)).sum()) for i in range(7)})
    return stats


def merged_fields(n, tri, mapping, data):
    ids, cart = mapping['plant_vertex_indices'], mapping['cart_mask'].astype(bool)
    labels = np.zeros(n, dtype=np.uint8)
    uncertain = np.ones(n, dtype=bool)
    inferred = np.ones(n, dtype=bool)
    source = np.zeros(n, dtype=np.uint8)  # 0=unprocessed, 1=V3, 2=geometric cart rule
    scores = np.full((n, 6), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if (np.asarray(data['vertex_labels']).shape != (len(ids),) or
            not np.isin(data['vertex_labels'], range(1, 6)).all() or
            np.asarray(data['class_scores']).shape != (len(ids), 5) or
            not np.isfinite(data['class_scores']).all()):
        raise ValueError('Expected complete V3 five-class labels/scores for prepared plant mesh')
    for name in ('uncertain', 'inferred'):
        if np.asarray(data[name]).shape != (len(ids),):
            raise ValueError(f'Invalid V3 {name} shape')
    labels[cart], uncertain[cart], source[cart] = 6, False, 2
    labels[ids], uncertain[ids], inferred[ids], source[ids] = data['vertex_labels'], data['uncertain'], data['inferred'], 1
    scores[ids, :5], scores[ids, 5], valid[ids] = data['class_scores'], 0., True
    uncertain |= mapping['contact_uncertain'].astype(bool)
    face_labels, face_boundary = face_assignment(tri, labels)
    face_uncertain = uncertain[tri].any(axis=1)|face_boundary|(face_labels == 0)
    return dict(vertex_labels=labels, face_labels=face_labels, uncertain=uncertain,
                inferred=inferred, label_source=source, class_scores=scores,
                class_score_valid=valid, class_score_ids=np.arange(1, 7, dtype=np.uint8),
                face_boundary=face_boundary, face_uncertain=face_uncertain,
                contact_uncertain=mapping['contact_uncertain'].astype(bool),
                plant_vertex_indices=ids)


def labeled_ply(source, path, data, semantic):
    base = source['vertex'].data
    extra = [('label', 'u1'), ('uncertain', 'u1'), ('inferred', 'u1'), ('label_source', 'u1')]
    if any(k in base.dtype.names for k, _ in extra):
        raise ValueError('Use the unlabeled metric source mesh')
    vertices = np.empty(len(base), dtype=[(k, base.dtype.fields[k][0]) for k in base.dtype.names]+extra)
    for key in base.dtype.names:
        vertices[key] = base[key]
    for name, key in [('label', 'vertex_labels'), ('uncertain', 'uncertain'),
                      ('inferred', 'inferred'), ('label_source', 'label_source')]:
        vertices[name] = data[key]
    if semantic:
        colors = COLORS[data['vertex_labels']]
        for i, key in enumerate(('red', 'green', 'blue')):
            vertices[key] = colors[:, i]
    base_face = source['face'].data
    faces = np.empty(len(base_face), dtype=[(k, base_face.dtype.fields[k][0]) for k in base_face.dtype.names]+[('label', 'u1'), ('uncertain', 'u1')])
    for key in base_face.dtype.names:
        faces[key] = base_face[key]
    faces['label'], faces['uncertain'] = data['face_labels'], data['face_uncertain']
    elements = [PlyElement.describe(vertices, 'vertex'), PlyElement.describe(faces, 'face')]
    elements += [e for e in source.elements if e.name not in ('vertex', 'face')]
    PlyData(elements, text=False, comments=list(source.comments)+[
        'coordinate_unit: millimeter', 'label 6: cart (geometric rule, not model confidence)',
        'label 0: unprocessed or ambiguous face; inspect uncertainty before planning']).write(str(path))


def merge(args):
    prep, final, out = args.prepared.resolve(), args.plant_result.resolve(), args.output.resolve()
    cfg = read_json(prep/'prepare.json')
    for key, path in [('input_sha256', cfg['input']), ('metric_sha256', cfg['metric']),
                      ('scale_sha256', cfg['scale_report'])]:
        if sha(path) != cfg[key]:
            raise ValueError(f'Changed prepared input: {path}')
    for name, digest in cfg['artifacts_sha256'].items():
        if sha(prep/name) != digest:
            raise ValueError(f'Changed preparation: {name}')
    validation = read_json(final/'validation.json')
    if validation['input_sha256'] != cfg['artifacts_sha256']['plant_upright.ply']:
        raise ValueError('V3 validation belongs to a different plant mesh')
    input_ply, input_xyz, input_tri = mesh(prep/'plant_upright.ply')
    source, xyz, tri = mesh(cfg['metric'])
    with np.load(prep/'mapping.npz') as m:
        mapping = {k:m[k] for k in m.files}
    with np.load(final/'labels.npz') as d:
        data = {k:d[k] for k in ('vertex_labels', 'class_scores', 'uncertain', 'inferred', 'face_labels')}
    # Do not trust a copied validation marker: verify actual V3 output geometry/labels.
    exported, ex, et = mesh(final/'labeled_original_rgb.ply')
    if not np.array_equal(input_xyz, ex) or not np.array_equal(input_tri, et):
        raise ValueError('V3 output geometry or vertex order changed')
    if not np.array_equal(exported['vertex'].data['label'], data['vertex_labels']):
        raise ValueError('V3 labels NPZ does not match exported mesh')
    if not np.array_equal(exported['face'].data['label'], data['face_labels']):
        raise ValueError('V3 face labels mismatch')
    combined = merged_fields(len(xyz), tri, mapping, data)
    # Preserve V3 face decisions for the faces processed by the five-class model.
    combined['face_labels'][mapping['plant_face_indices']] = data['face_labels']
    scale = read_json(cfg['scale_report'])
    hardware = override_hardware(xyz, tri, combined, scale)
    fresh(out)
    np.savez_compressed(out/'labels.npz', **combined)
    print('Writing six-class full meshes in millimetres', flush=True)
    for name, semantic in [('semantic_mesh_mm.ply', True), ('labeled_original_rgb_mm.ply', False)]:
        labeled_ply(source, out/name, combined, semantic)
    checks = {}
    for name, semantic in [('semantic_mesh_mm.ply', True), ('labeled_original_rgb_mm.ply', False)]:
        exported, ex, et = mesh(out/name)
        v, face = exported['vertex'].data, exported['face'].data
        checks[name] = dict(xyz_exact=bool(np.array_equal(xyz, ex)), triangles_exact=bool(np.array_equal(tri, et)),
                           vertex_labels_exact=bool(np.array_equal(v['label'], combined['vertex_labels'])),
                           face_labels_exact=bool(np.array_equal(face['label'], combined['face_labels'])),
                           uncertainty_exact=bool(np.array_equal(v['uncertain'], combined['uncertain'])),
                           face_uncertainty_exact=bool(np.array_equal(face['uncertain'], combined['face_uncertain'])))
        colors = np.column_stack([v[k] for k in ('red', 'green', 'blue')])
        expected = COLORS[combined['vertex_labels']] if semantic else np.column_stack([source['vertex'].data[k] for k in ('red', 'green', 'blue')])
        checks[name]['colors_exact'] = bool(np.array_equal(colors, expected))
        if not all(checks[name].values()):
            raise ValueError(f'Failed merged mesh validation: {checks[name]}')
    save_json(out/'validation.json', dict(status='PASS', metric_input_sha256=cfg['metric_sha256'], checks=checks))
    save_json(out/'classes.json', CLASSES)
    scale = read_json(cfg['scale_report'])
    save_json(out/'coordinate_system.json', dict(units='mm', reference='original reconstruction origin and orientation',
        original_to_mm=cfg['original_to_mm'], inference_to_original=cfg['inference_to_original'],
        board_center_mm=(np.array(scale['board_center_original'])*scale['scale_mm_per_model_unit']).tolist(),
        board_frame_axes=scale['board_frame_axes'], robot_registration_available=False,
        note='Metric scale is known. Robot/world registration, collision clearance and uncertainty handling remain separate planning steps.'))
    report = dict(version=VERSION, prepared=str(prep), plant_result=str(final), units='mm',
                  vertex_count=len(xyz), face_count=len(tri),
                  class_vertex_counts={c['name']:int((combined['vertex_labels'] == c['id']).sum()) for c in CLASSES},
                  uncertain_vertices=int(combined['uncertain'].sum()), uncertain_faces=int(combined['face_uncertain'].sum()),
                  label_source_ids={'0':'unprocessed', '1':'V3 plant inference', '2':'plate-relative cart geometry rule', '3':'upper plate corner hardware rule'},
                  scores_note='V3 first five channels only; cart scores are NaN with class_score_valid=False. No calibrated probabilities.',
                  semantic_accuracy_measured=False, cart_template_registration=False,
                  contact_split_mm=cfg['split_mm'], contact_review_band_mm=cfg['boundary_band_mm'],
                  cart_hardware=hardware, limitations=cfg['limitations'])
    save_json(out/'report.json', report)
    frame, center, factor = frame_from_scale(scale)
    aligned = (xyz-center*factor)@frame
    preview(aligned, COLORS[combined['vertex_labels']], out/'semantic_preview.jpg', 'Six classes; cyan=cart; gray=unknown')
    color = COLORS[combined['vertex_labels']].copy()
    color[combined['uncertain']] = [255, 190, 30]
    preview(aligned, color, out/'uncertainty_preview.jpg', 'Yellow=uncertain; review before planning')
    save_json(out/'result_paths.json', {k:str(out/v) for k,v in dict(semantic_mesh='semantic_mesh_mm.ply',
        original_rgb='labeled_original_rgb_mm.ply', labels='labels.npz', coordinates='coordinate_system.json',
        validation='validation.json').items()})
    (out/'WITH_CAR_SEMANTICS_SUCCESS').write_text(VERSION+'\n', encoding='ascii')
    print(f'WITH_CAR_SEMANTICS_SUCCESS: {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for key in ('input', 'metric', 'scale-report', 'output'):
        prep.add_argument('--'+key, type=Path, required=True)
    prep.add_argument('--split-mm', type=float, default=2.)
    prep.add_argument('--band-mm', type=float, default=3.)
    prep.set_defaults(func=prepare)
    combine = commands.add_parser('merge')
    for key in ('prepared', 'plant-result', 'output'):
        combine.add_argument('--'+key, type=Path, required=True)
    combine.set_defaults(func=merge)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
