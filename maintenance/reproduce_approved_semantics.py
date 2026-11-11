"""Reproduce the accepted organ postprocessing from its exact inputs/evidence.

This is an explicit reproduction, not a new segmentation algorithm. Frozen
sources are isolated from current experimental defaults. An original RGB mesh,
matching labels and complete parent-camera evidence are required. The source
and every accepted result remain read-only; outputs go to a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT/'maintenance/reproduction/approved_20261107'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def verify_inputs(source, evidence, snapshot=SNAPSHOT):
    manifest = read(snapshot/'snapshot_manifest.json')
    for name, expected in manifest['files'].items():
        if digest(snapshot/name) != expected:
            raise ValueError('Approved source snapshot changed: '+name)
    config = read(evidence/'review_config.json')
    report = read(evidence/'report.json')
    for file, key in [('refine_organ_semantics.py','code_sha256'),
                      ('organ_shape_rules.py','rules_sha256'),
                      ('organ_surface_consistency.py','consistency_sha256')]:
        if digest(snapshot/'tools/plant_semantics'/file) != config[key]:
            raise ValueError('Source does not match accepted review: '+file)
    if digest(source/'labels.npz') != config['source_labels_sha256']:
        raise ValueError('Use the exact accepted original labels, not a newly reconstructed mesh')
    if digest(source/'labeled_original_rgb_mm.ply') != report['source_mesh_sha256']:
        raise ValueError('Use the exact accepted original RGB geometry')
    provenance_path = evidence/'evidence_provenance.json'
    if provenance_path.exists():
        provenance = read(provenance_path)
        if digest(evidence/'review_evidence_checkpoint.npz') != provenance['checkpoint_sha256']:
            raise ValueError('Accepted multiview evidence checkpoint changed')
    with np.load(evidence/'review_evidence_checkpoint.npz', allow_pickle=False) as a:
        if int(a['completed_cameras']) != 24:
            raise ValueError('Accepted 24-camera evidence is incomplete')
    return config, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-result', type=Path,
                        help='Compare every label/evidence array with the accepted output')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    source, evidence, out = args.source.resolve(), args.evidence.resolve(), args.output.resolve()
    if (out.exists() or source == out or evidence == out or out in source.parents
            or out in evidence.parents or args.threads < 1):
        raise ValueError('Choose a fresh independent output directory and positive threads')
    started = time.monotonic()
    snapshot = args.snapshot.resolve()
    config, previous_report = verify_inputs(source, evidence, snapshot)
    # Import only the old implementation. These are its unchanged original
    # files; its PROJECT is the isolated snapshot root.
    for relative in ('tools/plant_semantics_with_car','tools/plant_organ_instances',
                     'tools/plant_semantics'):
        sys.path.insert(0, str(snapshot/relative))
    import refine_organ_semantics as approved
    out.mkdir(parents=True)
    try:
        ply, xyz, tri = approved.mesh(source/'labeled_original_rgb_mm.ply')
        with np.load(source/'labels.npz', allow_pickle=False) as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        old = data['vertex_labels'].copy()
        if (not np.array_equal(ply['vertex'].data['label'], old)
                or not np.array_equal(ply['face'].data['label'], data['face_labels'])):
            raise ValueError('Accepted RGB mesh and label arrays disagree')
        coordinates = read(source/'coordinate_system.json')
        # Deliberately preserve the accepted frame convention and every old
        # threshold. Reproduction must not quietly incorporate later fixes.
        frame = np.asarray(coordinates['board_frame_axes'], float)
        upright = (xyz-np.asarray(coordinates['board_center_mm'])) @ frame
        plant_faces = tri[~np.any(old[tri] == 6, axis=1)]
        plant_ids = np.unique(plant_faces)
        remap = np.full(len(xyz), -1, np.int32)
        remap[plant_ids] = np.arange(len(plant_ids))
        positions = upright[plant_ids].astype(np.float32)
        faces = remap[plant_faces]
        rgb = np.column_stack([ply['vertex'].data[k] for k in ('red','green','blue')])[plant_ids].astype(np.float32)/255
        with np.load(evidence/'review_evidence_checkpoint.npz', allow_pickle=False) as a:
            total, votes, observed = a['total'].copy(), a['votes'].copy(), a['observed'].copy()
        if (total.shape != (len(positions),6) or votes.shape != total.shape
                or observed.shape != (len(positions),)):
            raise ValueError('Accepted evidence/geometry mapping differs')
        print('APPROVED_REPRODUCTION_FEATURES',len(plant_ids),flush=True)
        features, scale = approved.geometric_features(positions,args.threads)
        reviewed, _, means, agreement = approved.decide_relabels(
            old[plant_ids],data['class_scores'][plant_ids],total,votes,observed)
        reviewed, extension = approved.extend_surface_consensus(positions,faces,rgb,old[plant_ids],
            total,votes,observed,features,scale,reviewed)
        data['surface_consensus_extended'] = np.zeros(len(xyz),bool)
        data['surface_consensus_extended'][plant_ids] = extension
        print('APPROVED_REPRODUCTION_SURFACE',flush=True)
        consistent, consistency_report = approved.regularize_surface(
            positions,faces,rgb,reviewed,total,votes,features,scale)
        data['organ_consistency_changed'] = np.zeros(len(xyz),bool)
        data['organ_consistency_changed'][plant_ids] = consistent != reviewed
        reviewed = consistent
        print('APPROVED_REPRODUCTION_RIMS',flush=True)
        rim, rim_report = approved.refine_stem_rims(positions,faces,rgb,reviewed,total,votes,features,scale)
        data['stem_rim_changed'] = np.zeros(len(xyz),bool)
        data['stem_rim_changed'][plant_ids] = rim != reviewed
        new = old.copy(); new[plant_ids] = rim
        full_means = np.zeros((len(xyz),6),np.float32);full_means[plant_ids] = means
        full_agreement = np.zeros(len(xyz),np.float32);full_agreement[plant_ids] = agreement
        # Preserve the old export convention exactly, including uncertainty.
        approved.export(source,out,data,old,new,full_means,full_agreement,
            previous_report['cameras'],xyz,tri,ply,coordinates)
        write(out/'surface_consistency_report.json',consistency_report)
        write(out/'stem_rim_report.json',rim_report)
        write(out/'review_config.json',config)
        checks = {}
        if args.expected_result:
            with np.load(args.expected_result/'labels.npz',allow_pickle=False) as reference, \
                    np.load(out/'labels.npz',allow_pickle=False) as actual:
                if set(reference.files) != set(actual.files):
                    raise ValueError('Reproduction and accepted arrays have different fields')
                checks = {key:bool(np.array_equal(reference[key],actual[key],equal_nan=True))
                          for key in reference.files}
            if not all(checks.values()):
                raise ValueError('Reproduction differs from accepted result: '+str([k for k,v in checks.items()if not v]))
        write(out/'reproduction_report.json',dict(status='APPROVED_POSTPROCESS_REPRODUCED',
            source=str(source),evidence=str(evidence),snapshot=str(snapshot),
            source_rgb_sha256=digest(source/'labeled_original_rgb_mm.ply'),
            source_labels_sha256=digest(source/'labels.npz'),
            evidence_sha256=digest(evidence/'review_evidence_checkpoint.npz'),
            geometry_unchanged=True,all_reference_arrays_equal=all(checks.values()) if checks else None,
            reference_array_checks=checks,elapsed_s=time.monotonic()-started,
            note='Reproduces saved evidence and the accepted method; does not rerun photo reconstruction or neural inference'))
        (out/'APPROVED_POSTPROCESS_REPRODUCED').write_text('Accepted method reproduced\n',encoding='ascii')
        print('APPROVED_POSTPROCESS_REPRODUCED',out,flush=True)
    except BaseException as error:
        write(out/'REPRODUCTION_FAILED.json',dict(error=str(error)))
        raise


if __name__ == '__main__':
    main()
