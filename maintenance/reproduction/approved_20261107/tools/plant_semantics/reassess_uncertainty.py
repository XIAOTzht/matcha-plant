"""Reassess legacy v3 review flags using existing independent camera evidence.

Changes metadata only. No models are loaded and no geometry, semantic labels,
scores, or inferred/propagated history are changed. Evidence thresholds are
heuristics, not calibrated correctness probabilities or robot safety checks.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import plantseg as p

REVISION = 1
POLICY = dict(min_views=3, min_support_fraction=.70, min_support_lead=.20,
              min_instance_mean=.15, min_final_score=.65, min_final_margin=.30,
              min_leaf_views=3, min_leaf_agreement=.70)


def reassess(data, instances):
    """Clear historical flags only when the assigned class passes final review.

    reason bits: 1 = unreviewed historical flag, 2 = final score conflict,
    4 = fewer than three observed cameras, 8 = insufficient instance consensus,
    16 = rim changed by geometry without independent leaf-mask confirmation.
    Masks can confirm a leaf even when old organ detectors disagreed. All mask
    counts must represent distinct parent cameras, not duplicate image crops.
    """
    label = np.asarray(data['vertex_labels'])
    n = len(label)
    if not np.isin(label, np.arange(1, 6)).all():
        raise ValueError('Expected complete five-class labels')
    scores = np.asarray(data['class_scores'])
    count = np.asarray(instances['view_count']).astype(np.int32)
    support = np.asarray(instances['support_count']).astype(np.int32)
    total = np.asarray(instances['total'])
    if scores.shape != (n, 5) or support.shape != (n, 6) or total.shape != (n, 6) or count.shape != (n,):
        raise ValueError('Evidence dimensions do not match the final labels')
    if (not np.isfinite(scores).all() or not np.isfinite(total).all() or
            (count < 0).any() or (support < 0).any() or (support > count[:, None]).any()):
        raise ValueError('Invalid camera evidence')
    if not np.array_equal(count, data['view_count']):
        raise ValueError('Instance evidence and source observation counts differ')
    row = np.arange(n)
    selected = scores[row, label-1]
    ranking = np.sort(scores, axis=1)
    margin = ranking[:, -1] - ranking[:, -2]
    score_good = ((scores.argmax(axis=1)+1 == label) &
                  (selected >= POLICY['min_final_score']) &
                  (margin >= POLICY['min_final_margin']))
    selected_support = support[row, label]
    rival = support[:, 1:].copy()
    rival[row, label-1] = 0
    fraction = selected_support / np.maximum(count, 1)
    # Cast counts before subtracting to prevent unsigned wraparound.
    lead = (selected_support - rival.max(axis=1)) / np.maximum(count, 1)
    mean = total[row, label] / np.maximum(selected_support, 1)
    consensus = ((count >= POLICY['min_views']) &
                 (selected_support >= POLICY['min_views']) &
                 (fraction >= POLICY['min_support_fraction']) &
                 (lead >= POLICY['min_support_lead']) &
                 (mean >= POLICY['min_instance_mean']))
    positive = np.asarray(data.get('leaf_mask_positive_views', np.zeros(n)), np.int32)
    observed = np.asarray(data.get('leaf_mask_observed_views', np.zeros(n)), np.int32)
    if positive.shape != (n,) or observed.shape != (n,) or (positive < 0).any() or (positive > observed).any():
        raise ValueError('Invalid independent leaf-mask camera counts')
    protected = np.asarray(data.get('bud_protected', np.zeros(n, bool)), bool)
    leaf_confirmed = ((label == 2) & ~protected &
                      (positive >= POLICY['min_leaf_views']) &
                      (positive / np.maximum(observed, 1) >= POLICY['min_leaf_agreement']))
    rim = np.asarray(data.get('leaf_rim_refined', np.zeros(n, bool)), bool)
    semantic_confirmed = consensus & (~rim | leaf_confirmed)
    historical = np.asarray(data['uncertain'], bool)
    released = historical & score_good & (semantic_confirmed | leaf_confirmed)
    current = historical & ~released
    reason = current.astype(np.uint8)
    reason |= (current & ~score_good).astype(np.uint8)*2
    reason |= (current & (count < POLICY['min_views'])).astype(np.uint8)*4
    reason |= (current & ~consensus).astype(np.uint8)*8
    reason |= (current & rim & ~leaf_confirmed).astype(np.uint8)*16
    return dict(uncertain=current, uncertain_before_reassessment=historical.copy(),
                uncertainty_released=released, uncertainty_reason=reason,
                uncertainty_semantic_confirmed=historical & score_good & semantic_confirmed,
                uncertainty_leaf_confirmed=historical & score_good & leaf_confirmed)


def uncertainty_preview(output, manifest, tri, label, old, new):
    from PIL import Image, ImageDraw, ImageFont
    base = Path(manifest['base_directory'])
    fontfile = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(fontfile), 22) if fontfile.exists() else ImageFont.load_default(size=22)
    width = 560
    cameras = manifest['views']
    chosen = [cameras[i] for i in [0, 2, 4, 6, 13] if i < len(cameras)]
    sheet = Image.new('RGB', (width*4, 65+(width+30)*len(chosen)), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(['原始 RGB', '复核前：橙色为需复核', '复核后：橙色为需复核', '语义分类保持不变']):
        draw.text((col*width+8, 7), title, fill='black', font=font)
    for row, camera in enumerate(chosen):
        name = camera['name']
        original = np.asarray(Image.open(base/'views'/f'{name}.png').convert('RGB'))
        ray = np.load(base/'views'/f'{name}_rays.npz')
        fid, bary = ray['face_id'], ray['barycentric']
        valid = fid >= 0
        corners = tri[fid[valid]]
        vertex = corners[np.arange(len(corners)), bary[valid].argmax(axis=1)]
        images = [original]
        for flags in [old, new]:
            im = original.copy()
            im[valid] = np.where(flags[vertex, None], [245, 140, 35], [175, 185, 190])
            images.append(im)
        semantic = original.copy()
        semantic[valid] = p.COLORS[label[vertex]]
        images.append(semantic)
        y = 65 + row*(width+30)
        for col, im in enumerate(images):
            sheet.paste(Image.fromarray(im).resize((width, width)), (col*width, y+30))
        draw.text((6, y), name, font=font, fill='black')
    sheet.save(output/'uncertainty_comparison.jpg', quality=94)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--instances', type=Path, required=True,
                        help='Legacy complete_five directory containing instance_vertices.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, evidence, output = (x.resolve() for x in (args.source, args.instances, args.output))
    if output in (source, evidence):
        raise ValueError('Use a separate output directory; preserve original flags')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((evidence/'instance_config.json').read_text(encoding='utf-8'))
    if manifest['input_sha256'] != p.sha256(manifest['input']) or config['input_sha256'] != manifest['input_sha256']:
        raise ValueError('Input/evidence mesh identity mismatch')
    if [str(Path(manifest[key]).resolve()) for key in ('base_directory', 'detail_directory')] != [str(Path(x).resolve()) for x in config['sources']]:
        raise ValueError('Instance evidence uses different source cameras')
    data = dict(np.load(source/'labels.npz'))
    if 'uncertain_before_reassessment' in data:
        raise ValueError('Already reassessed; use the original v3 result as source')
    instances = dict(np.load(evidence/'instance_vertices.npz'))
    if str(instances.pop('cache_key')) != config['cache_key']:
        raise ValueError('Stale projected instance evidence')
    expected = json.loads((evidence/'inference_config.json').read_text(encoding='utf-8'))
    if expected.get('instance_config_sha256') != p.sha256(evidence/'instance_config.json'):
        raise ValueError('Instance configuration identity changed')
    if (output/'manifest.json').exists():
        previous = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if previous['input_sha256'] != manifest['input_sha256']:
            raise ValueError('Output belongs to a different mesh')
    result = reassess(data, instances)
    data.update(result)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'labels.npz', **data)
    xyz, tri, rgb = p.mesh_data(manifest['input'])
    # Write the exact same labels, scores and geometry, with the reviewed flag.
    for name, semantic in [('semantic_mesh.ply', True), ('labeled_original_rgb.ply', False)]:
        p.write_labeled_ply(output/name, xyz, tri, rgb, data['vertex_labels'], data['evidence'], semantic,
                            uncertain=data['uncertain'], inferred=data['inferred'], face_labels=data['face_labels'])
    old, new, released = result['uncertain_before_reassessment'], result['uncertain'], result['uncertainty_released']
    counts = []
    for cls in p.CLASSES[1:]:
        mask = data['vertex_labels'] == cls['id']
        counts.append(dict(class_id=cls['id'], name=cls['name'], total_vertices=int(mask.sum()),
                           before=int((old & mask).sum()), released=int((released & mask).sum()),
                           after=int((new & mask).sum())))
    summary = dict(revision=REVISION, total_vertices=len(old), before=int(old.sum()),
                   released=int(released.sum()), after=int(new.sum()),
                   before_fraction=float(old.mean()), after_fraction=float(new.mean()),
                   semantic_confirmation=int(result['uncertainty_semantic_confirmed'].sum()),
                   leaf_confirmation=int(result['uncertainty_leaf_confirmed'].sum()),
                   by_class=counts, semantic_labels_changed=0, semantic_accuracy_measured=False,
                   note='Evidence metadata reassessment, not improved segmentation accuracy or robot safety certification.')
    p.save_json(output/'uncertainty_report.json', summary)
    policy = dict(revision=REVISION, thresholds=POLICY, code_sha256=p.sha256(Path(__file__)),
                  source_directory=str(source), source_labels_sha256=p.sha256(source/'labels.npz'),
                  instances_directory=str(evidence), instances_sha256=p.sha256(evidence/'instance_vertices.npz'),
                  instance_config_sha256=p.sha256(evidence/'instance_config.json'),
                  boundary_config_sha256=p.sha256(source/'boundary_config.json'),
                  baseline_flags_preserved=True, robot_collision_constraints_unchanged=True,
                  uncertainty_reason_bits={'1': 'historical flag pending review', '2': 'final score conflict',
                                           '4': 'fewer than 3 cameras', '8': 'insufficient instance consensus',
                                           '16': 'rim repair lacks independent leaf masks'})
    p.save_json(output/'uncertainty_config.json', policy)
    report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    report.update(uncertain_vertex_fraction=float(new.mean()), uncertainty_reassessment=summary,
                  uncertainty_source=str(source))
    p.save_json(output/'report.json', report)
    p.save_json(output/'manifest.json', dict(manifest, uncertainty_source=str(source),
                uncertainty_source_labels_sha256=policy['source_labels_sha256']))
    p.save_json(output/'inference_config.json', dict(recipe='v3_uncertainty_reassessment', revision=REVISION,
                uncertainty_config_sha256=p.sha256(output/'uncertainty_config.json'),
                source_inference_config_sha256=p.sha256(source/'inference_config.json'),
                semantic_labels_unchanged=True, complete_closed_set=True, manual_annotations=False))
    p.save_json(output/'classes.json', p.CLASSES[1:])
    # These still describe the same semantic result and its source stages.
    for name in ['rim_config.json', 'rim_diagnostics.npz', 'boundary_config.json',
                 'boundary_records.json', 'boundary_votes.npz', 'comparison_preview.jpg',
                 'comparison_compact.jpg', 'leaf_boundary_comparison.jpg',
                 'leaf_rim_comparison.jpg', 'leaf_rim_compact.jpg', 'leaf_edge_side_comparison.jpg']:
        if (source/name).exists():
            shutil.copy2(source/name, output/name)
    # Verify all original NPZ fields, allowing only the reviewed uncertain field.
    original = dict(np.load(source/'labels.npz'))
    audit = {key: bool(np.array_equal(original[key], data[key])) for key in original if key != 'uncertain'}
    audit.update(original_flags_preserved=bool(np.array_equal(old, original['uncertain'])),
                 no_new_flags=bool(~(new & ~old).any()),
                 changes_exact=bool(np.array_equal(old & ~new, released)))
    if not all(audit.values()):
        raise RuntimeError(f'Metadata-only audit failed: {audit}')
    p.save_json(output/'uncertainty_audit.json', audit)
    uncertainty_preview(output, manifest, tri, data['vertex_labels'], old, new)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
