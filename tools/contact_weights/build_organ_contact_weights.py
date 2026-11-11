#!/usr/bin/env python3
"""Apply generic per-organ center and leaf-area factors to semantic meshes.

Input paths are required: no plant-specific coordinates, counts or geometry.
The semantic policy defines the class IDs and hard constraints.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plant_organ_instances'))
from split_plant_organs import read_mesh, write_fields, fingerprint
from weight_palette import risk_colors, palette_description


def center_multiplier(center_score, gain=1., power=1.):
    return 1. + gain * np.clip(np.asarray(center_score, float), 0., 1.) ** power


def leaf_size_multiplier(area_mm2, reference_area_mm2=2000., gain=1.):
    if not np.isfinite(reference_area_mm2) or not np.isfinite(gain) or reference_area_mm2 <= 0 or gain < 0:
        raise ValueError('Reference area must be positive and gain nonnegative.')
    area = np.maximum(np.asarray(area_mm2, float), 0.)
    return 1. + gain * area / (area + reference_area_mm2)


def uncertainty_factors(labels,uncertain,ids,organs,class_multiplier,scope,leaf,flower):
    """Keep metadata local, but spread uncertainty cost over each complete organ.

    Pointwise flags at every small triangle created checkerboard weights on one
    leaf. A constant area-weighted fraction preserves their evidence without
    introducing a new set of artificial high-risk islands.
    """
    result=np.where(uncertain,class_multiplier,1.).astype(np.float32)
    if scope=='vertex':return result
    if scope!='organ_area_fraction':raise ValueError('Unknown uncertainty_scope: '+scope)
    area=np.asarray(organs['vertex_surface_area_mm2'],float)
    if area.shape!=ids.shape or not np.isfinite(area).all()or np.any(area<0):
        raise ValueError('Matching finite per-vertex surface areas required')
    resolved=(ids>0)&np.isin(labels,[leaf,flower])
    members=np.flatnonzero(resolved)
    unique,inverse=np.unique(ids[members],return_inverse=True)
    mass=np.maximum(area[members],1e-12)
    fraction=np.bincount(inverse,weights=mass*uncertain[members],minlength=len(unique))/np.maximum(
        np.bincount(inverse,weights=mass,minlength=len(unique)),1e-12)
    result[members]=1+(class_multiplier[members]-1)*fraction[inverse]
    return result


def build_arrays(labels, uncertain, faces, face_labels, face_uncertain, organs, policy, geometry_policy):
    classes = {int(k): c for k, c in policy['classes'].items()}
    missing = set(map(int, np.unique(labels))) | set(map(int, np.unique(face_labels)))
    missing -= set(classes)
    if missing:
        raise ValueError(f'Policy has no entries for semantic IDs: {sorted(missing)}')
    leaf = int(geometry_policy['leaf_class_id'])
    flower = int(geometry_policy['flower_class_id'])
    if leaf == flower or leaf not in classes or flower not in classes:
        raise ValueError('Leaf/flower class IDs must be distinct and present in policy.')
    if classes[leaf]['hard_collision'] or classes[flower]['hard_collision']:
        raise ValueError('The requested leaf/flower classes must be soft in the semantic policy.')
    # Class-specific center gains let a flower centre and a large leaf centre
    # carry different planning costs. The legacy ``center_gain`` remains a
    # fallback for older policy files.
    center_gain = float(geometry_policy.get('center_gain', 1.0))
    leaf_center_gain = float(geometry_policy.get('leaf_center_gain', center_gain))
    flower_center_gain = float(geometry_policy.get('flower_center_gain', center_gain))
    leaf_size_gain = float(geometry_policy.get('leaf_size_gain', 1.0))
    center_power = float(geometry_policy.get('center_power', 1.0))
    for name, value in (('center_gain', center_gain), ('leaf_center_gain', leaf_center_gain),
                        ('flower_center_gain', flower_center_gain), ('leaf_size_gain', leaf_size_gain)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {name}')
    if not np.isfinite(center_power) or center_power <= 0:
        raise ValueError('center_power must be positive.')

    def class_arrays(ids):
        base = np.zeros(len(ids), np.float32)
        hard = np.zeros(len(ids), bool)
        target = np.zeros(len(ids), bool)
        multiplier = np.ones(len(ids), np.float32)
        for cid, cls in classes.items():
            m = ids == cid
            hard[m] = cls['hard_collision']
            target[m] = cls['watering_target']
            base[m] = float(cls['weight'] or 0.)
            multiplier[m] = float(cls.get('uncertainty_multiplier', policy['uncertainty']['default_multiplier']))
        if not np.isfinite(base).all() or np.any(base < 0):
            raise ValueError('Base costs must be finite and nonnegative. Use hard flags, not infinity.')
        return base, hard, target, multiplier

    vb, vh, vt, um = class_arrays(labels)
    fb, fh, ft, fum = class_arrays(face_labels)
    ids = np.asarray(organs['vertex_instance_id'], np.int32)
    score = np.asarray(organs['vertex_center_score'], np.float32)
    area = np.asarray(organs['vertex_organ_area_mm2'], np.float32)
    if not (len(ids) == len(score) == len(area) == len(labels)):
        raise ValueError('Instance/mesh length mismatch.')
    if not np.isfinite(score).all() or not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError('Invalid geometry factor data.')
    requested = np.isin(labels, (leaf, flower))
    if np.any(ids[requested] == 0) or np.any(ids[~requested] != 0):
        raise ValueError('Instance classes do not match geometry policy; configure leaf/flower IDs.')
    cm = np.ones(len(labels), np.float32)
    sm = np.ones(len(labels), np.float32)
    cm[labels == leaf] = center_multiplier(score[labels == leaf], leaf_center_gain, center_power)
    cm[labels == flower] = center_multiplier(score[labels == flower], flower_center_gain, center_power)
    sm[labels == leaf] = leaf_size_multiplier(area[labels == leaf], geometry_policy['leaf_reference_surface_area_mm2'], leaf_size_gain)
    unresolved = requested & (ids < 0)
    cm[unresolved & (labels == leaf)] = 1 + leaf_center_gain
    cm[unresolved & (labels == flower)] = 1 + flower_center_gain
    sm[unresolved & (labels == leaf)] = 1 + leaf_size_gain
    uncertainty=uncertainty_factors(labels,uncertain,ids,organs,um,
        geometry_policy.get('uncertainty_scope','vertex'),leaf,flower)
    ve = vb * cm * sm * uncertainty
    ve[vh] = 0. # A hard flag is the validity constraint, zero is not a permission.

    # Class-consistent conservative face bound. Never average a leaf with an
    # adjacent stem or hard vertex: hard if face OR ANY corner is forbidden.
    fh = fh | np.any(vh[faces], axis=1)
    corner_weights = ve[faces]
    matching = labels[faces] == face_labels[:, None]
    local = np.max(np.where(matching, corner_weights, 0), axis=1)
    fallback = fb * np.where(face_uncertain, fum, 1.)
    fe = np.maximum(local, fallback).astype(np.float32)
    face_organ = np.isin(face_labels, (leaf, flower))
    fcm = np.max(np.where(matching, cm[faces], 1.), axis=1)
    fsm = np.max(np.where(matching, sm[faces], 1.), axis=1)
    # Face-only labels with no matching corner have no reliable instance:
    # use a maximum multiplier fallback instead of a cheap unweighted patch.
    no_match = face_organ & ~np.any(matching, axis=1)
    fcm[no_match & (face_labels == leaf)] = 1 + leaf_center_gain
    fcm[no_match & (face_labels == flower)] = 1 + flower_center_gain
    fsm[no_match & (face_labels == leaf)] = 1 + leaf_size_gain
    bound=fb*fcm*fsm*np.where(face_uncertain,fum,1.)
    # Matching organ corners already contain the organ-wide uncertainty factor.
    # Do not multiply random face flags on top and reintroduce checkerboards.
    spatial_match=face_organ&np.any(matching,axis=1)
    fe[spatial_match]=local[spatial_match]
    fe[~spatial_match]=np.maximum(fe[~spatial_match],bound[~spatial_match])
    fe[fh] = 0.
    corners = ids[faces]
    face_id = np.where(np.all(corners == corners[:, :1], axis=1), corners[:, 0], -2).astype('i4')
    return dict(vertex_base_weight=vb, vertex_effective_weight=ve,
        vertex_hard_collision=vh, vertex_watering_target=vt, vertex_uncertain=uncertain,
        vertex_center_multiplier=cm, vertex_size_multiplier=sm,
        vertex_center_score=score, vertex_organ_area_mm2=area, vertex_instance_id=ids,
        vertex_uncertainty_multiplier=uncertainty,
        vertex_instance_review=np.asarray(organs['vertex_instance_review'], bool),
        vertex_unresolved_instance=unresolved,
        face_base_weight=fb, face_effective_weight=fe, face_hard_collision=fh,
        face_watering_target=ft, face_uncertain=face_uncertain, face_instance_id=face_id,
        vertex_labels=labels, face_labels=face_labels)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--labels', type=Path, help='Matching semantic NPZ (preferred for uncertainty and face labels).')
    ap.add_argument('--label-field', default='label')
    ap.add_argument('--instances-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--policy', type=Path, default=here/'contact_policy.json')
    ap.add_argument('--geometry-policy', type=Path, default=here/'organ_weight_policy.json')
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out in (args.mesh.resolve().parent, args.instances_dir.resolve()):
        ap.error('Use a new output directory, distinct from mesh and instance source.')
    if (out/'contact_weights.npz').exists():
        ap.error('Output already exists. Choose a new directory.')
    policy = json.loads(args.policy.read_text(encoding='utf-8-sig'))
    gp = json.loads(args.geometry_policy.read_text(encoding='utf-8-sig'))
    policy['spatial_weighting'] = dict(require_local_weight=True,
        class_ids=[int(gp['leaf_class_id']), int(gp['flower_class_id'])],
        source='contact_weights.npz or PLY contact_weight; already includes uncertainty')
    organs = np.load(args.instances_dir/'organ_instances.npz', allow_pickle=False)
    scale = float(organs['mm_per_unit'])
    print('Reading semantic mesh and checking instance fingerprint', flush=True)
    ply, xyz, faces, labels = read_mesh(args.mesh, args.labels, args.label_field, scale)
    if str(organs['mesh_fingerprint']) != fingerprint(xyz, faces, labels):
        raise ValueError('Instances belong to a different geometry/vertex order/labels/scale; rerun splitter.')
    if args.labels:
        semantic = np.load(args.labels, allow_pickle=False)
        uncertain = np.asarray(semantic['uncertain'], bool) if 'uncertain' in semantic else np.zeros(len(labels), bool)
        contact_uncertain = np.asarray(semantic['contact_uncertain'], bool) if 'contact_uncertain' in semantic else np.zeros(len(labels), bool)
        fl = np.asarray(semantic['face_labels'], np.int32)
        fu = np.asarray(semantic['face_uncertain'], bool) if 'face_uncertain' in semantic else np.any((uncertain | contact_uncertain)[faces], axis=1)
        uncertain = uncertain | contact_uncertain
    else:
        v, f = ply['vertex'].data, ply['face'].data
        uncertain = v['uncertain'].astype(bool) if 'uncertain' in v.dtype.names else np.zeros(len(v), bool)
        contact_uncertain = np.zeros(len(v), bool)
        if args.label_field in f.dtype.names:
            fl = np.asarray(f[args.label_field], np.int32)
        else:
            ap.error('Supply labels.npz with face_labels or a PLY face label field; do not infer hard constraints by majority vote.')
        fu = f['uncertain'].astype(bool) if 'uncertain' in f.dtype.names else np.any(uncertain[faces], axis=1)
    if len(fl) != len(faces) or len(fu) != len(faces) or len(uncertain) != len(labels):
        raise ValueError('Semantic metadata length mismatch.')
    arrays = build_arrays(labels, uncertain, faces, fl, fu, organs, policy, gp)
    arrays['vertex_contact_uncertain'] = contact_uncertain
    arrays['mesh_fingerprint'] = organs['mesh_fingerprint']
    arrays['mm_per_unit'] = np.array(scale)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'contact_weights.npz', **arrays)
    fields = dict(contact_weight=arrays['vertex_effective_weight'], hard_collision=arrays['vertex_hard_collision'].astype('u1'),
        watering_target=arrays['vertex_watering_target'].astype('u1'), organ_id=arrays['vertex_instance_id'],
        center_score=arrays['vertex_center_score'], center_factor=arrays['vertex_center_multiplier'],
        size_factor=arrays['vertex_size_multiplier'], organ_area_mm2=arrays['vertex_organ_area_mm2'],
        organ_review=arrays['vertex_instance_review'].astype('u1'),
        uncertainty_factor=arrays['vertex_uncertainty_multiplier'])
    ff = dict(contact_weight=arrays['face_effective_weight'], hard_collision=arrays['face_hard_collision'].astype('u1'),
        watering_target=arrays['face_watering_target'].astype('u1'), organ_id=arrays['face_instance_id'])
    print('Writing complete meshes; original positions, topology and semantic fields retained', flush=True)
    write_fields(out/'contact_weighted_mesh.ply', ply, fields, face_fields=ff)
    rgb = risk_colors(arrays['vertex_effective_weight'], arrays['vertex_hard_collision'])
    write_fields(out/'contact_risk_heatmap.ply', ply, fields, rgb, ff)
    (out/'contact_policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'organ_weight_policy.json').write_text(json.dumps(gp, ensure_ascii=False, indent=2), encoding='utf-8')
    stats = {}
    for cid in np.unique(labels):
        w = arrays['vertex_effective_weight'][labels == cid]
        cls = policy['classes'][str(cid)]
        stats[str(cid)] = dict(name=cls['name'], count=len(w), hard_collision=cls['hard_collision'],
                              finite_weight_range=[float(w.min()), float(w.max())])
    proposal_report = json.loads((args.instances_dir/'organ_report.json').read_text(encoding='utf-8'))
    report = dict(version='organ-contact-weights-0.3.0', status='REVIEW_REQUIRED', data_export='PASS',
        mesh=str(args.mesh.resolve()), instances=str(args.instances_dir.resolve()),
        mesh_fingerprint=str(organs['mesh_fingerprint']), coordinate_unit='source model unit; mm_per_unit applies',
        mm_per_unit=scale, vertex_count=len(labels), face_count=len(faces), class_statistics=stats,
        geometry_policy=gp, instance_proposals=proposal_report['instance_count'],
        unresolved_vertices=int(arrays['vertex_unresolved_instance'].sum()),
        face_hard_rule='original face hard OR any original corner hard; stricter at class boundaries',
        source_geometry_and_semantics_preserved=True, visualization=palette_description(),
        limitations=['Instances and their centers/areas require review.',
        'Leaf area is reconstructed surface area; size multiplier is a planning preference, not stiffness measurement.',
        'Score trace must supply local_effective_weight to use these spatial factors.',
        'No robot registration, motion plan or force/damage validation is provided.'])
    (out/'contact_weights_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'CONTACT_WEIGHTS_EXPORTED_REVIEW_REQUIRED').write_text('Export complete; inspect organ proposals before planning use.\n', encoding='utf-8')
    print(json.dumps({'output':str(out), 'status':report['status'], 'class_statistics':stats}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
