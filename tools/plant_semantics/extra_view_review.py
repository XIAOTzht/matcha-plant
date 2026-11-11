"""Add independent RGB viewpoints to review uncertainty in an existing v3 mesh.

Uses the same GroundingDINO/CLIPSeg/MobileSAM weights as v3. Final semantic
labels and scores remain frozen. New views can confirm or flag disagreement,
but cannot silently relabel organs or erase historical review metadata.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import plantseg as p

REVISION = 1


def source_data(source):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if p.sha256(manifest['input']) != manifest['input_sha256']:
        raise ValueError('Source mesh identity changed')
    return manifest, dict(np.load(source/'labels.npz'))


def select_views(visibility, weights, cameras, budget=12, min_angle=25):
    """Greedy surface-area coverage with diminishing returns and angular spread."""
    counts = np.zeros(visibility.shape[1], np.int16)
    selected = []
    threshold = np.cos(np.deg2rad(min_angle))
    for _ in range(min(budget, len(cameras))):
        gain = np.array([(weights*seen/(1+counts)**1.5).sum() for seen in visibility])
        for i, cam in enumerate(cameras):
            if i in selected or any(np.dot(cam['direction'], cameras[j]['direction']) > threshold for j in selected):
                gain[i] = -1
        if gain.max() <= 0:
            break
        chosen = int(gain.argmax())
        selected.append(chosen)
        counts += visibility[chosen]
    return selected


def prepare(args):
    from PIL import Image, ImageDraw
    manifest, data = source_data(args.source)
    signature = dict(revision=REVISION, input_sha256=manifest['input_sha256'],
                     source_labels_sha256=p.sha256(args.source/'labels.npz'),
                     source=str(args.source), size=args.size, budget=args.views,
                     min_angular_separation_degrees=25, class_coverage_weights=[1.5, 1, 1, .5, 2.5],
                     required_high_view_elevations=65)
    configfile = args.output/'selection_config.json'
    if configfile.exists():
        if json.loads(configfile.read_text(encoding='utf-8')) != signature:
            raise ValueError('Selection recipe changed; use a new output directory')
        if (args.output/'manifest.json').exists():
            print('Extra views already prepared', flush=True)
            return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'views').mkdir(exist_ok=True)
    p.save_json(configfile, signature)
    xyz, tri, rgb = p.mesh_data(manifest['input'])
    scene = p.scene_from_arrays(xyz, tri)
    ids = np.flatnonzero(data['uncertain'])
    cameras = []
    visibility = []
    up_axis = manifest['views'][0]['up_axis']
    for elevation in [-50, -15, 30, 65]:
        for azimuth in range(30, 360, 60):
            _, _, _, camera = p.render_view(scene, xyz, tri, rgb, azimuth, elevation, 8, up_axis)
            _, seen = p.sample_visible_vertices(scene, xyz, camera,
                                               np.zeros((args.size, args.size, 6), np.float32))
            cameras.append(camera)
            visibility.append(seen[ids])
    visibility = np.array(visibility)
    area = np.linalg.norm(np.cross(xyz[tri[:, 1]]-xyz[tri[:, 0]], xyz[tri[:, 2]]-xyz[tri[:, 0]]), axis=1)*.5
    va = np.zeros(len(xyz), np.float64)
    for j in range(3):
        np.add.at(va, tri[:, j], area/3)
    weight = va[ids].copy()
    for cls, priority in enumerate(signature['class_coverage_weights'], 1):
        mask = data['vertex_labels'][ids] == cls
        if mask.any():
            weight[mask] *= priority/max(weight[mask].sum(), 1e-12)
    selected = select_views(visibility, weight, cameras, args.views)
    # Three spaced high views add context for the watering target, which has a
    # small area and could otherwise be outvoted by large flower backs.
    if args.views >= 6 and np.any(data['vertex_labels'][ids] == 5):
        for i, camera in enumerate(cameras):
            if camera['elevation'] == 65 and camera['azimuth'] in [30, 150, 270] and i not in selected:
                selected.append(i)
    views, thumbs = [], []
    for number, index in enumerate(selected):
        camera = cameras[index]
        image, fid, bary, camera = p.render_view(scene, xyz, tri, rgb, camera['azimuth'],
                                                camera['elevation'], args.size, up_axis)
        name = f'extra_{number:03d}'
        camera.update(name=name, independent_camera=name)
        Image.fromarray(image).save(args.output/'views'/f'{name}.png')
        np.savez_compressed(args.output/'views'/f'{name}_rays.npz', face_id=fid, barycentric=bary)
        views.append(camera)
        thumb = Image.fromarray(image).resize((280, 280))
        ImageDraw.Draw(thumb).text((6, 6), f'{name} az={camera["azimuth"]} el={camera["elevation"]}', fill='black')
        thumbs.append(thumb)
    if thumbs:
        sheet = Image.new('RGB', (280*4, 280*((len(thumbs)+3)//4)), 'white')
        for i, im in enumerate(thumbs):
            sheet.paste(im, (280*(i%4), 280*(i//4)))
        sheet.save(args.output/'render_contact_sheet.jpg', quality=92)
    counts = visibility[selected].sum(axis=0) if selected else np.zeros(len(ids))
    p.save_json(args.output/'coverage_report.json', dict(candidate_cameras=len(cameras),
                selected_cameras=len(selected), old_uncertain_vertices=len(ids),
                visible_in_at_least_three_new_views=int((counts >= 3).sum()),
                note='Visibility opportunity only; no semantic confirmation yet.'))
    p.save_json(args.output/'manifest.json', dict(manifest, source_directory=str(args.source),
                source_labels_sha256=signature['source_labels_sha256'], views=views, size=args.size,
                status='additional_views_rendered', independent_parent_cameras=True))
    print('Prepared', len(views), 'independent extra cameras', flush=True)


def predict(args):
    # Reuse the unchanged default v3 inference recipe; it does not import SAM3.
    p.predict(SimpleNamespace(output=args.output, detector=args.detector, dense=args.dense,
              sam=args.sam, threads=args.threads, device=args.device, box_threshold=.22,
              repredict=False, limit=0))


def collect_evidence(args, xyz, tri, manifest):
    """Count each new camera once, using direct depth-tested vertex evidence."""
    config = json.loads((args.output/'inference_config.json').read_text(encoding='utf-8'))
    source = json.loads((Path(manifest['base_directory'])/'inference_config.json').read_text(encoding='utf-8'))
    for key in ['detector_sha256', 'dense_sha256', 'sam_sha256', 'phrases', 'dense_phrases', 'recipe_revision', 'box_threshold']:
        if config[key] != source[key]:
            raise ValueError(f'Extra-view inference differs from v3 source: {key}')
    if p.sha256(args.source/'labels.npz') != manifest['source_labels_sha256']:
        raise ValueError('Source labels changed after selecting new cameras')
    cache = args.output/'extra_evidence.npz'
    cache_key = p.sha256(args.output/'manifest.json')+config['cache_key']
    if cache.exists():
        data = dict(np.load(cache))
        if str(data.pop('cache_key')) != cache_key:
            raise ValueError('Stale extra-view projection')
        return data
    n = len(xyz)
    count = np.zeros(n, np.uint16)
    votes = np.zeros((n, 6), np.uint16)
    total = np.zeros((n, 6), np.float32)
    scene = p.scene_from_arrays(xyz, tri)
    cameras = manifest['views']
    directions = np.array([camera['direction'] for camera in cameras])
    if len(cameras) > 1:
        similarity = directions@directions.T
        np.fill_diagonal(similarity, -1)
        if similarity.max() > np.cos(np.deg2rad(20)):
            raise ValueError('New cameras too close to count independently')
    for camera in cameras:
        path = args.output/'views'/f'{camera["name"]}_scores.npz'
        with np.load(path) as prediction:
            if str(prediction['cache_key']) != config['cache_key']:
                raise ValueError('Stale new-view prediction')
            values, seen = p.sample_visible_vertices(scene, xyz, camera, prediction['scores'])
        # Demand a clear per-camera winner; all observed cameras remain in the
        # denominator, including abstentions and competing labels.
        winner, _, _ = p.evidence_labels(values, minimum=.20, margin=.035)
        ids = np.flatnonzero(seen)
        count += seen
        votes[ids, winner[ids]] += 1
        total += values
        print('Extra projection', camera['name'], 'visible', int(seen.sum()), flush=True)
    result = dict(view_count=count, votes=votes, total=total)
    np.savez_compressed(cache, **result, cache_key=cache_key)
    return result


def review_flags(data, extra):
    """Require at least three new winning cameras to confirm frozen semantics.

    A strong independent disagreement creates a new review flag. Pure scores,
    similarity to neighbors, and geometry-only growth cannot clear flags.
    """
    labels = np.asarray(data['vertex_labels'])
    n = len(labels)
    scores = np.asarray(data['class_scores'])
    if (labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer) or
            np.any((labels < 1) | (labels > 5)) or scores.shape != (n, 5) or
            np.asarray(data['uncertain']).shape != (n,) or not np.isfinite(scores).all()):
        raise ValueError('Expected finite five-class scores and closed-set source labels')
    for key in ['view_count', 'votes']:
        value = np.asarray(extra[key])
        if (not np.issubdtype(value.dtype, np.integer) or
                np.any(value < 0) or np.any(value > np.iinfo(np.uint16).max)):
            raise ValueError('Invalid independent camera vote counts')
    count = np.asarray(extra['view_count'], np.int32)
    votes = np.asarray(extra['votes'], np.int32)
    total = np.asarray(extra['total'])
    if votes.shape != (n, 6) or total.shape != (n, 6) or count.shape != (n,):
        raise ValueError('Extra-view evidence does not match source mesh')
    if (count < 0).any() or (votes < 0).any() or not np.array_equal(votes.sum(axis=1), count):
        raise ValueError('Invalid independent camera vote counts')
    if not np.isfinite(total).all() or (total < 0).any():
        raise ValueError('Invalid extra-view semantic evidence')
    row = np.arange(n)
    rank = np.sort(scores, axis=1)
    final_good = ((scores.argmax(axis=1)+1 == labels) & (scores[row, labels-1] >= .65) &
                  (rank[:, -1]-rank[:, -2] >= .30))
    selected = votes[row, labels]
    other = votes[:, 1:].copy()
    other[row, labels-1] = 0
    competitor = other.max(axis=1)
    agreement = selected/np.maximum(count, 1)
    lead = (selected-competitor)/np.maximum(count, 1)
    mean = total[row, labels]/np.maximum(count, 1)
    independent = ((count >= 3) & (selected >= 3) & (agreement >= .70) &
                   (lead >= .30) & (mean >= .20))
    confirmed = independent & final_good
    contradiction = ((count >= 3) & (competitor >= 3) &
                     (competitor/np.maximum(count, 1) >= .70) &
                     ((competitor-selected)/np.maximum(count, 1) >= .30))
    old = np.asarray(data['uncertain'], bool)
    released = old & confirmed
    new = (old & ~released) | contradiction
    return dict(uncertain=new, uncertain_before_extra_views=old.copy(),
                extra_view_released=released, extra_view_conflict=contradiction,
                extra_view_confirmed=confirmed,
                extra_view_count=count.astype(np.uint16),
                extra_view_assigned_votes=selected.astype(np.uint16),
                extra_view_agreement=agreement.astype(np.float32))


def apply(args):
    import shutil
    manifest = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    original_manifest, data = source_data(args.source)
    if 'uncertain_before_extra_views' in data:
        raise ValueError('Already reviewed with extra views; use original reviewed v3 source')
    xyz, tri, rgb = p.mesh_data(original_manifest['input'])
    extra = collect_evidence(args, xyz, tri, manifest)
    result = review_flags(data, extra)
    before = dict(data)
    data.update(result)
    final = args.output/'complete'
    final.mkdir(exist_ok=True)
    np.savez_compressed(final/'labels.npz', **data)
    for name, semantic in [('semantic_mesh.ply', True), ('labeled_original_rgb.ply', False)]:
        p.write_labeled_ply(final/name, xyz, tri, rgb, data['vertex_labels'], data['evidence'], semantic,
                            uncertain=data['uncertain'], inferred=data['inferred'], face_labels=data['face_labels'])
    old, current = result['uncertain_before_extra_views'], result['uncertain']
    summary = dict(revision=REVISION, new_independent_cameras=len(manifest['views']),
                   before=int(old.sum()), released=int(result['extra_view_released'].sum()),
                   newly_flagged=int((current & ~old).sum()), after=int(current.sum()),
                   before_fraction=float(old.mean()), after_fraction=float(current.mean()),
                   by_class=[], semantic_labels_changed=0, semantic_accuracy_measured=False,
                   note='Additional rendered views confirm frozen labels; fewer flags do not measure accuracy.')
    for cls in p.CLASSES[1:]:
        mask = data['vertex_labels'] == cls['id']
        summary['by_class'].append(dict(class_id=cls['id'], name=cls['name'], total_vertices=int(mask.sum()),
                    before=int((old & mask).sum()), after=int((current & mask).sum()),
                    released=int((result['extra_view_released'] & mask).sum()),
                    newly_flagged=int((current & ~old & mask).sum())))
    p.save_json(final/'extra_view_report.json', summary)
    policy = dict(revision=REVISION, min_new_winning_views=3, min_agreement=.70, min_vote_lead=.30,
                  min_mean_evidence=.20, min_final_score=.65, min_final_margin=.30,
                  source_directory=str(args.source), source_labels_sha256=p.sha256(args.source/'labels.npz'),
                  selection_config_sha256=p.sha256(args.output/'selection_config.json'),
                  inference_config_sha256=p.sha256(args.output/'inference_config.json'),
                  extra_evidence_sha256=p.sha256(args.output/'extra_evidence.npz'),
                  code_sha256=p.sha256(Path(__file__)), new_conflicts_retained=True)
    p.save_json(final/'extra_view_config.json', policy)
    p.save_json(final/'manifest.json', dict(original_manifest, extra_view_directory=str(args.output),
                extra_view_source=str(args.source), extra_view_source_labels_sha256=policy['source_labels_sha256']))
    report = json.loads((args.source/'report.json').read_text(encoding='utf-8'))
    report.update(uncertain_vertex_fraction=float(current.mean()), additional_view_review=summary)
    p.save_json(final/'report.json', report)
    p.save_json(final/'classes.json', p.CLASSES[1:])
    p.save_json(final/'inference_config.json', dict(recipe='v3_extra_view_review', revision=REVISION,
                config_sha256=p.sha256(final/'extra_view_config.json'), semantic_labels_unchanged=True,
                complete_closed_set=True, manual_annotations=False, sam3_used=False))
    audit = {key: bool(np.array_equal(value, data[key])) for key, value in before.items() if key != 'uncertain'}
    audit['previous_flags_preserved'] = bool(np.array_equal(old, before['uncertain']))
    if not all(audit.values()):
        raise RuntimeError('Non-uncertainty source fields changed')
    p.save_json(final/'extra_view_audit.json', audit)
    for name in ['uncertainty_config.json', 'uncertainty_report.json', 'uncertainty_audit.json', 'rim_config.json',
                 'boundary_config.json', 'leaf_edge_side_comparison.jpg', 'leaf_rim_compact.jpg']:
        if (args.source/name).exists():
            shutil.copy2(args.source/name, final/name)
    from reassess_uncertainty import uncertainty_preview
    uncertainty_preview(final, original_manifest, tri, data['vertex_labels'], old, current)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'predict', 'apply', 'run'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=896)
    parser.add_argument('--views', type=int, default=12)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--detector', type=Path, default=p.ROOT/'models/grounding-dino-tiny')
    parser.add_argument('--dense', type=Path, default=p.ROOT/'models/clipseg-rd64-refined')
    parser.add_argument('--sam', type=Path, default=p.ROOT/'models/mobile_sam_parallel.pt')
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output == args.source:
        raise ValueError('Use a separate output directory')
    if not 1 <= args.views <= 24 or args.size < 32:
        raise ValueError('Expected 1..24 requested views and resolution >=32')
    if args.command in ['prepare', 'run']:
        prepare(args)
    if args.command in ['predict', 'run']:
        predict(args)
    if args.command in ['apply', 'run']:
        apply(args)


if __name__ == '__main__':
    main()
