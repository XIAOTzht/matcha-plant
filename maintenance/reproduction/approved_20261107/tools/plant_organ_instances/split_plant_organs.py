#!/usr/bin/env python3
"""Propose leaf/flower instances for any semantically labelled triangle PLY."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import colorsys
import numpy as np
from plyfile import PlyData, PlyElement, PlyListProperty
from scipy.spatial import cKDTree
from organ_geometry import split_organs


def fingerprint(xyz, faces, labels):
    h = hashlib.sha256()
    for a, dtype in ((xyz, '<f8'), (faces, '<i4'), (labels, '<i4')):
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
    return h.hexdigest()


def read_mesh(mesh, label_file=None, label_field='label', mm_per_unit=1.):
    ply = PlyData.read(str(mesh))
    v = ply['vertex'].data
    if 'face' not in ply or not len(ply['face'].data):
        raise ValueError('A triangle mesh is required; a Gaussian/point-cloud PLY is not an organ surface.')
    lists = ply['face'].data['vertex_indices']
    if any(len(f) != 3 for f in lists):
        raise ValueError('Triangulate the mesh first.')
    faces = np.stack(lists).astype(np.int32)
    xyz = np.column_stack([v[k] for k in ('x', 'y', 'z')]).astype(float) * mm_per_unit
    if label_file:
        with np.load(label_file, allow_pickle=False) as z:
            labels = np.asarray(z['vertex_labels'], dtype=np.int32)
        if label_field in v.dtype.names and not np.array_equal(v[label_field], labels):
            raise ValueError('PLY and NPZ semantic labels disagree. Use the matching source pair.')
    elif label_field in v.dtype.names:
        labels = np.asarray(v[label_field], dtype=np.int32)
    else:
        raise ValueError('Supply labels.npz or a PLY vertex label field.')
    if len(labels) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid coordinates or label length.')
    if np.any(faces < 0) or np.any(faces >= len(xyz)):
        raise ValueError('Invalid triangle indices.')
    return ply, xyz, faces, labels


def write_fields(path, ply, fields, colors=None, face_fields=None):
    """Preserve all original elements/properties; append only new fields."""
    elements = []
    for element in ply.elements:
        extras = fields if element.name == 'vertex' else (face_fields or {}) if element.name == 'face' else {}
        if not extras and (element.name != 'vertex' or colors is None):
            elements.append(element)
            continue
        old = element.data
        additions = {k: np.asarray(a) for k, a in extras.items()}
        if element.name == 'vertex' and colors is not None:
            additions.update(zip(('red', 'green', 'blue'), np.asarray(colors, np.uint8).T))
        dtype = [(name, old.dtype.fields[name][0]) for name in old.dtype.names if name not in additions]
        dtype += [(name, a.dtype) for name, a in additions.items()]
        data = np.empty(len(old), dtype=dtype)
        for name in old.dtype.names:
            if name not in additions:
                data[name] = old[name]
        for name, a in additions.items():
            if len(a) != len(old):
                raise ValueError(f'Wrong length: {name}')
            data[name] = a
        length_types = {p.name: p.len_dtype for p in element.properties if isinstance(p, PlyListProperty)}
        value_types = {p.name: p.val_dtype for p in element.properties if isinstance(p, PlyListProperty)}
        elements.append(PlyElement.describe(data, element.name, len_types=length_types, val_types=value_types,
                                            comments=list(element.comments)))
    PlyData(elements, text=False, byte_order='<', comments=list(ply.comments) + ['organ fields are provisional geometric proposals']).write(str(path))


def instance_colors(ids):
    rgb = np.full((len(ids), 3), 195, np.uint8)
    rgb[ids < 0] = [65, 65, 65]
    for k in np.unique(ids[ids > 0]):
        rgb[ids == k] = np.rint(np.array(colorsys.hsv_to_rgb((int(k) * .61803398875) % 1, .68, .95)) * 255)
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--labels', type=Path)
    ap.add_argument('--label-field', default='label')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--leaf-id', type=int, default=2)
    ap.add_argument('--flower-id', type=int, default=3)
    ap.add_argument('--mm-per-unit', type=float, default=1.)
    ap.add_argument('--min-area-mm2', type=float, default=10.)
    ap.add_argument('--marker-separation', type=float, default=1., help='Larger gives fewer geometric parts; no fixed organ count.')
    ap.add_argument('--max-pixels', type=int, default=600)
    ap.add_argument('--seed-file', type=Path, help='JSON list: {class_id, xyz_mm:[x,y,z]} or {vertex_index}. Coordinates in original metric frame.')
    ap.add_argument('--export-parts', action='store_true', help='Also export individual proposed parts as PLY files with source vertex indices.')
    args = ap.parse_args()
    if args.mm_per_unit <= 0 or args.min_area_mm2 <= 0 or args.marker_separation <= 0 or args.max_pixels < 64:
        ap.error('Scale, minimum area and marker separation must be positive; max-pixels >=64.')
    if args.leaf_id == args.flower_id:
        ap.error('Leaf and flower IDs must differ.')
    out = args.output_dir.resolve()
    if out == args.mesh.resolve().parent:
        ap.error('Use a separate output directory to protect source files.')
    if (out / 'organ_instances.npz').exists():
        ap.error('Output already exists. Choose a new directory to preserve previous proposals.')
    print('Reading labelled mesh; coordinates interpreted in millimeters', flush=True)
    ply, xyz, faces, labels = read_mesh(args.mesh, args.labels, args.label_field, args.mm_per_unit)
    seeds = json.loads(args.seed_file.read_text(encoding='utf-8-sig')) if args.seed_file else []
    for s in seeds:
        if 'vertex_index' not in s:
            members = np.flatnonzero(labels == int(s['class_id']))
            if not len(members):
                raise ValueError('Seed refers to a missing class.')
            _, j = cKDTree(xyz[members]).query(s['xyz_mm'])
            s['vertex_index'] = int(members[j])
        i = int(s['vertex_index'])
        if i < 0 or i >= len(xyz) or labels[i] not in (args.leaf_id, args.flower_id):
            raise ValueError('Every seed must lie on a requested leaf/flower vertex.')
        if 'class_id' in s and labels[i] != int(s['class_id']):
            raise ValueError('Seed class and mesh label disagree.')
    arrays, records = split_organs(xyz, faces, labels, (args.leaf_id, args.flower_id),
        args.min_area_mm2, args.marker_separation, args.max_pixels, seeds)
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(xyz, faces, labels)
    np.savez_compressed(out / 'organ_instances.npz', **arrays, vertex_labels=labels,
                        mesh_fingerprint=np.array(fp), mm_per_unit=np.array(args.mm_per_unit))
    fields = {'organ_id': arrays['vertex_instance_id'], 'center_score': arrays['vertex_center_score'],
              'organ_area_mm2': arrays['vertex_organ_area_mm2'], 'organ_review': arrays['vertex_instance_review'].astype('u1')}
    corners = arrays['vertex_instance_id'][faces]
    homogeneous = np.all(corners == corners[:, :1], axis=1)
    face_ids = np.where(homogeneous, corners[:, 0], -2).astype('i4')
    write_fields(out / 'organ_instances.ply', ply, fields, instance_colors(arrays['vertex_instance_id']),
                 {'organ_id': face_ids, 'organ_boundary': (~homogeneous).astype('u1')})
    if args.export_parts:
        parts = out / 'parts'
        parts.mkdir(exist_ok=True)
        for r in records:
            indices = np.flatnonzero(arrays['vertex_instance_id'] == r['instance_id'])
            remap = np.full(len(xyz), -1, np.int32)
            remap[indices] = np.arange(len(indices))
            ft = remap[faces]
            ft = ft[np.all(ft >= 0, axis=1)]
            data = np.empty(len(indices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'i4'), ('source_vertex_index', 'i4')])
            for k in ('x', 'y', 'z', 'red', 'green', 'blue'):
                if k in ply['vertex'].data.dtype.names:
                    data[k] = ply['vertex'].data[k][indices]
                else:
                    data[k] = 180
            data['label'] = labels[indices]
            data['source_vertex_index'] = indices
            fd = np.empty(len(ft), dtype=[('vertex_indices', 'i4', (3,))])
            fd['vertex_indices'] = ft
            path = parts / f'class_{r["class_id"]}_part_{r["instance_id"]:04d}.ply'
            PlyData([PlyElement.describe(data, 'vertex'), PlyElement.describe(fd, 'face',
                len_types={'vertex_indices': 'u1'}, val_types={'vertex_indices': 'i4'})], text=False, byte_order='<').write(str(path))
            r['part_mesh'] = str(path.resolve())
            r['part_coordinate_units'] = 'source model units (unchanged); area fields use mm2'
    report = dict(version='organ-proposals-0.1.0', status='REVIEW_REQUIRED', mesh=str(args.mesh.resolve()),
        mesh_fingerprint=fp, units='mm', mm_per_unit=args.mm_per_unit, leaf_id=args.leaf_id, flower_id=args.flower_id,
        parameters=dict(min_area_mm2=args.min_area_mm2, marker_separation=args.marker_separation, max_pixels=args.max_pixels),
        instance_count=len(records), per_class={str(cid): sum(r['class_id'] == cid for r in records) for cid in (args.leaf_id, args.flower_id)},
        unresolved_vertices=int(np.sum(arrays['vertex_instance_id'] < 0)),
        source_geometry_and_semantics_preserved=True, instances=records,
        limitations=['Geometry proposals are not verified biological organs.',
        'Overlapping or fused surfaces can merge organs or split a flower into petal patches.',
        'Incorrect semantic classes are preserved, not corrected.',
        'Center is projected silhouette depth, not a measured vein/flower anatomical center.',
        'Size is reconstructed surface area; missing/back surfaces and reconstruction thickness affect it.'])
    (out / 'organ_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    exported = [dict(vertex_index=r['seed_vertex_index'], class_id=r['class_id']) for r in records]
    (out / 'seed_markers.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
    print(f'ORGAN_PROPOSALS_WRITTEN: {out}; review required ({len(records)} proposals)', flush=True)


if __name__ == '__main__':
    main()
