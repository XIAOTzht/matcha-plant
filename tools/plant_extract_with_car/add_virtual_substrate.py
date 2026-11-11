#!/usr/bin/env python3
"""Append an explicitly assumed pot-interior surface; never relabel a scan.

Opt-in only. The pot is a compact, approximately convex single container in a
calibrated metric frame. The inferred fill height is not a measured soil level.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plant_semantics_with_car'))
from with_car_semantics import mesh, read_json, save_json, sha, fresh, COLORS

VERSION = 'virtual-substrate-1.0'


def inset_polygon(points, inset):
    hull = ConvexHull(points)
    polygon = points[hull.vertices].copy()
    for equation in hull.equations:
        normal, offset = equation[:2], equation[2] + inset
        result = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            fa, fb = a @ normal + offset, b @ normal + offset
            if fa <= 1e-9:
                result.append(a)
            if (fa <= 0) != (fb <= 0):
                result.append(a + (b-a) * fa/(fa-fb))
        polygon = np.asarray(result).reshape(-1, 2)
        if len(polygon) < 3:
            raise ValueError('Wall inset leaves no pot-interior surface')
    return polygon


def estimate_surface(q, labels, rim_offset_mm=10., wall_inset_mm=6., height_mm=None):
    if (not np.isfinite([rim_offset_mm, wall_inset_mm]).all()
            or not 3 <= rim_offset_mm <= 30 or not 3 <= wall_inset_mm <= 20):
        raise ValueError('Rim offset must be 3..30 mm and wall inset 3..20 mm')
    pot = q[labels == 4]
    if len(pot) < 100:
        raise ValueError('Not enough pot geometry to infer a single opening')
    # Uniform voxel support avoids letting triangle tessellation determine height.
    _, ids = np.unique(np.floor(pot/.75).astype('i8'), axis=0, return_index=True)
    pot = pot[ids]
    bottom, rim = np.percentile(pot[:, 1], [1, 99.5])
    height = float(rim-rim_offset_mm if height_mm is None else height_mm)
    if not np.isfinite(height) or not bottom+5 < height < rim-2:
        raise ValueError('Assumed fill level must lie inside the pot, below its rim')
    band = pot[np.abs(pot[:, 1]-height) <= 2.5][:, [0, 2]]
    if len(band) < 40:
        raise ValueError('Pot walls do not support a reliable cross section at this height')
    low, high = np.percentile(band, [1, 99], axis=0)
    extent = high-low
    if extent.min() < 20 or extent.max()/extent.min() > 2.5:
        raise ValueError('Expected one compact pot; provide a different capture/template')
    middle = (low+high)/2
    angle = np.sort(np.arctan2(band[:, 1]-middle[1], band[:, 0]-middle[0]))
    if np.max(np.diff(np.r_[angle, angle[0]+2*np.pi])) > np.deg2rad(75):
        raise ValueError('Pot cross section is missing a broad angular sector')
    quadrants = (band[:, 0] > middle[0]).astype(int)*2 + (band[:, 1] > middle[1])
    if min(np.bincount(quadrants, minlength=4)) < 5:
        raise ValueError('Incomplete pot wall coverage around the opening')
    # Trim scan outliers, then explicitly assume a convex inner opening.
    band = band[((band >= low) & (band <= high)).all(1)]
    polygon = inset_polygon(band, wall_inset_mm)
    if ConvexHull(polygon).volume < 100:
        raise ValueError('Assumed opening is too small')
    return dict(version=VERSION, units='mm', frame='calibrated upper plate, Y up',
        surface_source='assumed_virtual_substrate', measured=False,
        assumed_height_mm=height, estimated_rim_height_mm=float(rim),
        estimated_bottom_height_mm=float(bottom), rim_offset_mm=float(rim-height),
        wall_inset_mm=float(wall_inset_mm), footprint_xz_mm=polygon.tolist(),
        basis='Pot wall cross section with an inward offset; height is an explicit fill-level assumption',
        limitations=['Not observed or reconstructed soil; simulation target only.',
            'Single compact convex container assumed; inset is not measured wall thickness.',
            'Hidden pot contents and actual fill height remain unknown.',
            'Original plant surfaces remain obstacles; the spray ray must hit substrate first.'])


def surface_mesh(template, spacing_mm=2.):
    if not np.isfinite(spacing_mm) or not .5 <= spacing_mm <= 5:
        raise ValueError('Surface spacing must be .5..5 mm')
    polygon = np.asarray(template['footprint_xz_mm'], float)
    hull = ConvexHull(polygon)
    points = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        count = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing_mm)))
        points.extend(a+(b-a)*np.arange(count)[:, None]/count)
    lo, hi = polygon.min(0), polygon.max(0)
    xx, zz = np.meshgrid(np.arange(lo[0], hi[0], spacing_mm), np.arange(lo[1], hi[1], spacing_mm))
    grid = np.column_stack([xx.ravel(), zz.ravel()])
    inside = np.all(grid @ hull.equations[:, :2].T+hull.equations[:, 2] <= -1e-7, axis=1)
    points = np.unique(np.round(np.concatenate([points, grid[inside]]), 8), axis=0)
    triangles = Delaunay(points).simplices[:, [0, 2, 1]]  # +Y-facing in X,Y,Z.
    q = np.column_stack([points[:, 0], np.full(len(points), template['assumed_height_mm']), points[:, 1]])
    return q, triangles


def append_arrays(data, nv, nf, added_vertices, added_faces):
    """Keep the original prefix, including every prior synthetic provenance mask."""
    result = {}
    for key, a in data.items():
        if key == 'class_score_ids':
            result[key] = a.copy()
            continue
        if key in ('plant_vertex_indices', 'plant_face_indices'):
            start, count = (nv, added_vertices) if key == 'plant_vertex_indices' else (nf, added_faces)
            result[key] = np.r_[a, np.arange(start, start+count, dtype=a.dtype)]
            continue
        if not a.ndim:
            raise ValueError('Unsupported semantic scalar: '+key)
        if key.startswith('face_') or key == 'cart_face_geometry_inferred':
            expected, count = nf, added_faces
        else:
            expected, count = nv, added_vertices
        if len(a) != expected:
            raise ValueError('Unrecognized semantic array shape: '+key)
        extra = np.zeros((count,)+a.shape[1:], dtype=a.dtype)
        if a.dtype.kind == 'f':
            extra.fill(np.nan)  # No photographed or classifier evidence is invented.
        result[key] = np.concatenate([a, extra])
    for key in ('vertex_labels', 'uncertain', 'inferred', 'label_source', 'class_scores',
                'class_score_valid', 'face_labels', 'face_uncertain', 'face_boundary'):
        if key not in result:
            raise ValueError('Missing semantic field: '+key)
    result['vertex_labels'][nv:] = 5
    result['face_labels'][nf:] = 5
    for key in ('uncertain', 'inferred', 'contact_uncertain'):
        if key in result:
            result[key][nv:] = True
    result['label_source'][nv:] = 6
    result['face_uncertain'][nf:] = True
    result['virtual_substrate'] = np.r_[np.zeros(nv, bool), np.ones(added_vertices, bool)]
    result['face_virtual_substrate'] = np.r_[np.zeros(nf, bool), np.ones(added_faces, bool)]
    return result


def extended_element(element, count, extra_fields):
    # Rebuild a packed dtype: mmap triangle lists contain an unnamed count byte.
    fields = [(k, element.dtype.fields[k][0]) for k in element.dtype.names]
    fields.extend((k, dtype) for k, dtype in extra_fields if k not in element.dtype.names)
    result = np.zeros(len(element)+count, dtype=fields)
    for key in element.dtype.names:
        result[key][:len(element)] = element[key]
    return result


def run(args):
    src, out = args.source.resolve(), args.output.resolve()
    if not args.assume_pot_interior_substrate:
        raise ValueError('Explicit --assume-pot-interior-substrate is required')
    if out == src or out in src.parents or src in out.parents:
        raise ValueError('Use a separate sibling/result directory')
    ply, xyz, triangles = mesh(src/'labeled_original_rgb_mm.ply')
    semantic_ply, semantic_xyz, semantic_tri = mesh(src/'semantic_mesh_mm.ply')
    if not np.array_equal(xyz, semantic_xyz) or not np.array_equal(triangles, semantic_tri):
        raise ValueError('Source semantic and original RGB geometry disagree')
    with np.load(src/'labels.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    nv, nf = len(xyz), len(triangles)
    if any(np.any(data[k] == 5) for k in ('vertex_labels', 'face_labels')):
        raise ValueError('Existing substrate labels found; do not replace real soil with an assumption')
    for element, key in [('vertex', 'vertex_labels'), ('face', 'face_labels')]:
        if not np.array_equal(ply[element].data['label'], data[key]):
            raise ValueError('Source RGB mesh and semantic labels disagree')
        if not np.array_equal(semantic_ply[element].data['label'], data[key]):
            raise ValueError('Source semantic PLY and NPZ labels disagree')
    coords = read_json(src/'coordinate_system.json')
    frame = np.asarray(coords['board_frame_axes'], float)
    center = np.asarray(coords['board_center_mm'], float)
    if (coords.get('units') != 'mm' or frame.shape != (3, 3) or center.shape != (3,)
            or not np.isfinite(center).all() or not np.allclose(frame.T@frame, np.eye(3), atol=1e-6)):
        raise ValueError('A valid calibrated millimeter frame is required')
    q = (xyz-center)@frame
    template = estimate_surface(q, data['vertex_labels'], args.rim_offset_mm, args.wall_inset_mm, args.height_mm)
    added, faces = surface_mesh(template, args.spacing_mm)
    nd = append_arrays(data, nv, nf, len(added), len(faces))
    positions = np.concatenate([xyz, added@frame.T+center])
    full_faces = np.concatenate([triangles, faces+nv])
    vertices = extended_element(ply['vertex'].data, len(added), [('geometry_inferred', 'u1'), ('virtual_substrate', 'u1')])
    vertices['geometry_inferred'][nv:] = 1
    vertices['virtual_substrate'][nv:] = 1
    for i, key in enumerate(('x', 'y', 'z')):
        vertices[key][nv:] = positions[nv:, i]
    for key in ('label', 'uncertain', 'inferred', 'label_source'):
        vertices[key] = nd['vertex_labels' if key == 'label' else key]
    for key, value in zip(('red', 'green', 'blue'), (125, 89, 51)):
        vertices[key][nv:] = value
    if 'alpha' in vertices.dtype.names:
        vertices['alpha'][nv:] = 255
    for i, key in enumerate(('nx', 'ny', 'nz')):
        if key in vertices.dtype.names:
            vertices[key][nv:] = frame[i, 1]
    face_data = extended_element(ply['face'].data, len(faces),
        [('uncertain', 'u1'), ('geometry_inferred', 'u1'), ('virtual_substrate', 'u1')])
    face_data['vertex_indices'] = full_faces
    face_data['label'] = nd['face_labels']
    face_data['uncertain'][nf:] = 1
    face_data['geometry_inferred'][nf:] = 1
    face_data['virtual_substrate'][nf:] = 1
    fresh(out)
    checks = dict(original_positions_exact=True, original_triangles_exact=True,
        original_labels_exact=True, original_rgb_exact=True, prior_provenance_preserved=True)
    for name, semantic in [('labeled_original_rgb_mm.ply', False), ('semantic_mesh_mm.ply', True)]:
        v = vertices.copy()
        if semantic:
            for i, key in enumerate(('red', 'green', 'blue')):
                v[key] = COLORS[nd['vertex_labels'], i]
        PlyData([PlyElement.describe(v, 'vertex'), PlyElement.describe(face_data, 'face')], text=False,
            comments=['ASSUMED VIRTUAL SUBSTRATE; virtual_substrate=1 is not reconstructed soil']).write(str(out/name))
        check, cx, ct = mesh(out/name)
        checks[name+'_roundtrip'] = bool(np.array_equal(ct, full_faces)
            and np.array_equal(check['vertex'].data['label'], nd['vertex_labels'])
            and np.array_equal(check['face'].data['label'], nd['face_labels']))
        checks['original_positions_exact'] &= bool(np.array_equal(cx[:nv], xyz))
        checks['original_triangles_exact'] &= bool(np.array_equal(ct[:nf], triangles))
        if not semantic:
            for key in ply['vertex'].data.dtype.names:
                checks['prior_provenance_preserved'] &= bool(np.array_equal(check['vertex'].data[key][:nv], ply['vertex'].data[key]))
            for key in ply['face'].data.dtype.names:
                checks['prior_provenance_preserved'] &= bool(np.array_equal(check['face'].data[key][:nf], ply['face'].data[key]))
            checks['original_rgb_exact'] = all(np.array_equal(check['vertex'].data[key][:nv], ply['vertex'].data[key])
                                               for key in ('red', 'green', 'blue'))
    for key, a in data.items():
        if key not in ('plant_vertex_indices', 'plant_face_indices'):
            checks['original_labels_exact'] &= bool(np.array_equal(nd[key][:len(a)], a, equal_nan=True))
    if not all(checks.values()):
        raise ValueError('Append-only export validation failed: '+str(checks))
    np.savez_compressed(out/'labels.npz', **nd)
    np.savez_compressed(out/'source_mapping.npz', vertex_source_index=np.r_[np.arange(nv), np.full(len(added), -1)],
        face_source_index=np.r_[np.arange(nf), np.full(len(faces), -1)])
    for name in ('coordinate_system.json', 'classes.json', 'cart_template.json'):
        if (src/name).is_file():
            shutil.copyfile(src/name, out/name)
    # Retain older repair lineage separately; its mapping refers to the old mesh.
    for name in ('source_mapping.npz', 'report.json'):
        if (src/name).is_file():
            shutil.copyfile(src/name, out/('upstream_'+name))
    template.update(vertex_start=nv, vertex_count=len(added), face_start=nf, face_count=len(faces),
        spacing_mm=args.spacing_mm, original_mesh_sha256=sha(src/'labeled_original_rgb_mm.ply'),
        original_labels_sha256=sha(src/'labels.npz'), original_source=str(src),
        explicit_assumption_authorized=True)
    save_json(out/'virtual_substrate.json', template)
    old_report = read_json(src/'report.json') if (src/'report.json').is_file() else {}
    source_ids = dict(old_report.get('label_source_ids', {}))
    source_ids['6'] = 'Explicitly assumed pot-interior substrate, not classifier evidence'
    report = dict(version=VERSION, status='ASSUMED_SURFACE_EXPORTED', units='mm',
        surface_source=template['surface_source'], virtual_substrate='virtual_substrate.json',
        original_geometry_unchanged=True, original_labels_unchanged=True,
        original_source=str(src), vertex_count=len(vertices), face_count=len(full_faces),
        class_vertex_counts={str(k): int((nd['vertex_labels'] == k).sum()) for k in range(7)},
        label_source_ids=source_ids,
        real_robot_ready=False, measured_soil_available=False, limitations=template['limitations'])
    save_json(out/'report.json', report)
    save_json(out/'validation.json', dict(status='PASS', checks=checks))
    (out/'VIRTUAL_SUBSTRATE_EXPORTED').write_text(VERSION+'\n', encoding='ascii')
    print('VIRTUAL_SUBSTRATE_EXPORTED', out, 'height_mm='+str(template['assumed_height_mm']), flush=True)
    return template


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--assume-pot-interior-substrate', action='store_true')
    ap.add_argument('--rim-offset-mm', type=float, default=10.)
    ap.add_argument('--wall-inset-mm', type=float, default=6.)
    ap.add_argument('--spacing-mm', type=float, default=2.)
    ap.add_argument('--height-mm', type=float, help='Explicit assumed height in the calibrated board frame')
    run(ap.parse_args())
