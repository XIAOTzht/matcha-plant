#!/usr/bin/env python3
"""Repair corner fasteners in an existing full six-class result; CPU, no inference."""
import argparse
from pathlib import Path
import numpy as np
from with_car_semantics import (mesh, read_json, save_json, fresh, sha, override_hardware,
    labeled_ply, frame_from_scale, preview, COLORS, VERSION)


def run(args):
    source, target = args.source.resolve(), args.output.resolve()
    if target == source or source.is_relative_to(target):
        raise ValueError('Output must be separate from the existing result')
    scale = read_json(args.scale_report)
    ply, xyz, tri = mesh(source/'labeled_original_rgb_mm.ply')
    with np.load(source/'labels.npz') as cached:
        data = {k:cached[k].copy() for k in cached.files}
    if not np.array_equal(ply['vertex'].data['label'], data['vertex_labels']):
        raise ValueError('Source NPZ and mesh labels do not match')
    if not np.array_equal(ply['face'].data['label'], data['face_labels']):
        raise ValueError('Source NPZ and mesh face labels do not match')
    if len(xyz) != scale['vertices'] or len(tri) != scale['faces']:
        raise ValueError('Calibration and result counts do not match')
    # Check the actual metric positions against the calibration source, not just counts.
    metric, mx, mt = mesh(args.metric)
    if not np.array_equal(xyz, mx) or not np.array_equal(tri, mt):
        raise ValueError('Metric geometry/order differs from semantic result')
    for key in ('red', 'green', 'blue'):
        if not np.array_equal(ply['vertex'].data[key], metric['vertex'].data[key]):
            raise ValueError('Original RGB mismatch')
    coordinates = read_json(source/'coordinate_system.json')
    if not np.allclose(coordinates['original_to_mm'], np.diag([
        scale['scale_mm_per_model_unit']]*3+[1.]), rtol=1e-6):
        raise ValueError('Calibration scale differs from source coordinates')
    before = data['vertex_labels'].copy()
    before_face = data['face_labels'].copy()
    stats = override_hardware(xyz, tri, data, scale)
    fresh(target)
    np.savez_compressed(target/'labels.npz', **data)
    checks = {}
    for name, semantic in [('semantic_mesh_mm.ply', True), ('labeled_original_rgb_mm.ply', False)]:
        labeled_ply(metric, target/name, data, semantic)
        exported, ex, et = mesh(target/name)
        v, f = exported['vertex'].data, exported['face'].data
        expected = COLORS[data['vertex_labels']] if semantic else np.column_stack([metric['vertex'].data[k] for k in ('red', 'green', 'blue')])
        checks[name] = dict(xyz_exact=bool(np.array_equal(ex, xyz)), faces_exact=bool(np.array_equal(et, tri)),
            labels_exact=bool(np.array_equal(v['label'], data['vertex_labels'])),
            face_labels_exact=bool(np.array_equal(f['label'], data['face_labels'])),
            colors_exact=bool(np.array_equal(np.column_stack([v[k] for k in ('red', 'green', 'blue')]), expected)),
            uncertainty_exact=bool(np.array_equal(v['uncertain'], data['uncertain'])))
        if not all(checks[name].values()):
            raise ValueError('Export consistency check failed')
    mask = data['cart_hardware_mask']
    unchanged_outside = bool(np.array_equal(before[~mask], data['vertex_labels'][~mask]))
    untouched_faces = ~data['cart_hardware_changed'][tri].any(axis=1)
    faces_outside = bool(np.array_equal(before_face[untouched_faces], data['face_labels'][untouched_faces]))
    if not unchanged_outside or not faces_outside:
        raise ValueError('Labels outside hardware region changed')
    save_json(target/'validation.json', dict(status='PASS', checks=checks,
        labels_outside_hardware_exact=unchanged_outside, unaffected_face_labels_exact=faces_outside))
    report = read_json(source/'report.json')
    report.update(version=VERSION, repaired_from=str(source), source_labels_sha256=sha(source/'labels.npz'),
        scale_report_sha256=sha(args.scale_report), cart_hardware=stats,
        uncertain_vertices=int(data['uncertain'].sum()), uncertain_faces=int(data['face_uncertain'].sum()))
    classes = read_json(source/'classes.json')
    report['class_vertex_counts'] = {c['name']:int((data['vertex_labels'] == c['id']).sum()) for c in classes}
    report['label_source_ids']['3'] = 'upper plate corner hardware rule'
    save_json(target/'report.json', report)
    save_json(target/'classes.json', classes)
    save_json(target/'coordinate_system.json', coordinates)
    frame, center, factor = frame_from_scale(scale)
    aligned = (xyz-center*factor)@frame
    preview(aligned, COLORS[data['vertex_labels']], target/'semantic_preview.jpg', 'Repaired six classes; cyan=cart')
    color = COLORS[data['vertex_labels']].copy(); color[data['uncertain']] = [255, 190, 30]
    preview(aligned, color, target/'uncertainty_preview.jpg', 'Yellow=uncertain; hardware flags preserved')
    # Closer view of all four fasteners, with the pot retained for context.
    roi = (aligned[:, 1] > -5)&(aligned[:, 1] < 20)
    preview(aligned[roi], COLORS[data['vertex_labels']][roi], target/'hardware_preview.jpg', 'Plate corners: fasteners belong to cart')
    save_json(target/'result_paths.json', {k:str(target/v) for k,v in dict(
        semantic_mesh='semantic_mesh_mm.ply', original_rgb='labeled_original_rgb_mm.ply', labels='labels.npz',
        coordinates='coordinate_system.json', validation='validation.json').items()})
    (target/'WITH_CAR_SEMANTICS_SUCCESS').write_text(VERSION+'\n', encoding='ascii')
    print(stats, flush=True)
    print(f'CART_HARDWARE_FIX_SUCCESS: {target}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'scale-report', 'metric'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
