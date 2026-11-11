"""Export a small, read-only evidence bundle for a failed plant/cart crop."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import zipfile

import numpy as np
from plyfile import PlyData

PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def audit(points, rgb, rng):
    """Mirror the deployed detector, recording every rejection, without cropping."""
    from auto_crop_with_car import masks, fit_blue_plane
    from auto_crop_mesh import voxel_clusters
    blue, botanical = masks(rgb)
    extent = float(np.linalg.norm(np.diff(np.percentile(points, [5, 95], axis=0), axis=0)[0]))
    report = dict(samples=len(points), blue_samples=int(blue.sum()),
                  botanical_samples=int(botanical.sum()), scene_extent=extent,
                  voxel_ratio=.006, candidates=[], status='NO_ACCEPTED_CANDIDATE')
    frames = []
    if blue.sum() < 100 or botanical.sum() < 50 or extent <= 0:
        report['reason'] = 'Insufficient color evidence or degenerate scene extent'
        return report, frames
    bp, plants = points[blue], points[botanical]
    labels, counts, _, _ = voxel_clusters(bp, extent*.006)
    report['blue_components'] = len(counts)
    for rank, k in enumerate(np.argsort(counts)[-10:][::-1], 1):
        row = dict(rank=rank, cluster=int(k), blue_samples=int(counts[k]))
        report['candidates'].append(row)
        if counts[k] < 100:
            row['rejected_at'] = 'fewer_than_100_blue_samples'; continue
        p = bp[labels == k]; center = np.median(p, axis=0)
        span = float(np.linalg.norm(np.diff(np.percentile(p, [1, 99], axis=0), axis=0)[0]))
        near = plants[np.linalg.norm(plants-center, axis=1) < span*1.35]
        row.update(center=center.tolist(), span=span, nearby_botanical_samples=len(near))
        if len(near) < 50:
            row['rejected_at'] = 'fewer_than_50_nearby_botanical_samples'; continue
        try:
            origin, up, tolerance = fit_blue_plane(p, near, rng)
        except ValueError as exc:
            row.update(rejected_at='blue_plane_fit', reason=str(exc)); continue
        heights = (near-origin)@up
        row.update(plane_origin=origin.tolist(), up=up.tolist(), tolerance=float(tolerance),
                   botanical_height_percentiles=np.percentile(heights, [0, 5, 50, 95, 100]).tolist())
        canopy = near[(heights > span*.16) & (heights < span*2)]
        row['canopy_before_footprint'] = len(canopy)
        u = np.eye(3)[np.argmin(abs(up))]; u = np.cross(up, u); u /= np.linalg.norm(u)
        w = np.cross(u, up); basis = np.column_stack((u, w))
        _, ev = np.linalg.eigh(np.cov(((p-origin)@basis).T)); u = basis@ev[:, -1]
        w = np.cross(u, up); frame = np.column_stack((u, up, w))
        frames.append((rank, origin, frame, center, span))
        if len(canopy) < 50:
            row['rejected_at'] = 'fewer_than_50_canopy_samples'; continue
        q = (p-origin)@frame
        lo, hi = np.percentile(q, [.2, 99.8], axis=0); length = max((hi-lo)[[0, 2]])
        cq = (canopy-origin)@frame
        centered = ((cq[:, [0, 2]] >= lo[[0, 2]]-length*.12) &
                    (cq[:, [0, 2]] <= hi[[0, 2]]+length*.12)).all(axis=1)
        row.update(canopy_inside_footprint=int(centered.sum()), blue_bounds=[lo.tolist(), hi.tolist()])
        if centered.sum() < 50:
            row['rejected_at'] = 'fewer_than_50_canopy_samples_inside_footprint'; continue
        row['accepted'] = True
        row['score'] = float(np.sqrt(len(p)*centered.sum()))
        report['status'] = 'DETECTOR_CANDIDATE_FOUND'
    return report, frames


def export(root, run, samples=300000):
    root = Path(root).resolve()
    sys.path.insert(0, str(root/'tools/plant_extract_with_car'))
    from auto_crop_with_car import masks
    from auto_crop_mesh import draw_points
    if not 10000 <= samples <= 1000000:
        raise ValueError('samples must be 10000..1000000')
    state_file = root/'outputs/pipeline'/run/'pipeline_state.json'
    state = json.loads(state_file.read_text(encoding='utf-8'))
    stage = state['stages'].get('04_mesh', {})
    if stage.get('status') != 'SUCCESS':
        raise ValueError('Mesh extraction has not succeeded')
    source = Path(stage['artifacts']['mesh']['path']).resolve()
    source.relative_to(root)
    # Triangular face lists are mmap-compatible; only selected vertices are
    # copied, rather than copying the entire multi-million-face mesh into RAM.
    print('CROP_DIAGNOSTICS_LOADING', source, flush=True)
    ply = PlyData.read(str(source), known_list_len={'face': {'vertex_indices': 3, 'vertex_index': 3}})
    vertices = ply['vertex'].data
    if not all(key in vertices.dtype.names for key in ('x', 'y', 'z', 'red', 'green', 'blue')):
        raise ValueError('Need XYZ and RGB vertex properties')
    rng = np.random.default_rng(42)
    indices = rng.choice(len(vertices), min(samples, len(vertices)), replace=False)
    points = np.column_stack([vertices[key][indices] for key in ('x', 'y', 'z')]).astype(np.float64)
    rgb = np.column_stack([vertices[key][indices] for key in ('red', 'green', 'blue')])
    if not np.isfinite(points).all() or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError('Nonfinite XYZ or RGB out of range')
    if rgb.dtype.kind == 'f' and rgb.max() <= 1:
        raise ValueError('Detector expects RGB in 0..255')
    tag = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    destination = root/'exports/crop_diagnostics'/run/tag
    destination.mkdir(parents=True, exist_ok=False)
    # Exact sampled XYZ retained as float64 for local reproduction; these are
    # points for diagnosis only, not a replacement mesh or downstream result.
    np.savez_compressed(destination/'mesh_sample.npz', xyz=points, rgb=rgb, source_vertex_indices=indices)
    print('CROP_DIAGNOSTICS_ANALYZING', len(points), 'samples', flush=True)
    report, frames = audit(points, rgb, rng)
    report.update(input=str(source), vertices=len(vertices), faces=len(ply['face'].data),
                  sample_seed=42, rgb_percentiles=np.percentile(rgb, [0, 1, 50, 99, 100], axis=0).tolist(),
                  coordinate_percentiles=np.percentile(points, [0, 1, 50, 99, 100], axis=0).tolist(),
                  note='Read-only diagnostics of the original crop rules. No model, label, policy, stage signature or state was changed.')
    (destination/'diagnostic_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (destination/'pipeline_state.json').write_bytes(state_file.read_bytes())
    failure = state_file.parent/'05_crop'/('attempt_'+str(state['stages'].get('05_crop', {}).get('attempt', 1)).zfill(3))/'console.log'
    if failure.is_file():
        (destination/'crop_console.log').write_text(failure.read_text(encoding='utf-8', errors='replace')[-40000:], encoding='utf-8')
    ids = np.random.default_rng(91).choice(len(points), min(120000, len(points)), replace=False)
    center = np.median(points, axis=0)
    draw_points(points[ids], rgb[ids], destination/'scene_world_axes.jpg', 'Raw scene RGB', np.eye(3), center)
    low, high = np.percentile(points, [2, 98], axis=0)
    inner = points[((points >= low) & (points <= high)).all(axis=1)]
    _, basis = np.linalg.eigh(np.cov((inner-center).T))
    draw_points(points[ids], rgb[ids], destination/'scene_pca_axes.jpg', 'Raw scene PCA RGB', basis, center)
    blue, plant = masks(rgb[ids])
    colors = np.full((len(ids), 3), 85, dtype=np.uint8)
    colors[blue] = [30, 120, 255]; colors[plant] = [70, 240, 80]
    draw_points(points[ids], colors, destination/'detector_colors.jpg', 'Blue=cart seeds / green=plant seeds', basis, center)
    for rank, origin, frame, cluster_center, span in frames[:5]:
        selected = np.linalg.norm(points[ids]-cluster_center, axis=1) < span*2.5
        if selected.sum() < 100:
            continue
        draw_points(points[ids][selected], rgb[ids][selected], destination/f'candidate_{rank:02d}.jpg',
                    'Candidate '+str(rank)+' RGB', frame, origin)
    archive = destination/'crop_diagnostics.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.iterdir()):
            if path != archive and path.is_file():
                z.write(path, path.name)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n', encoding='ascii')
    latest = destination.parent/'latest.json'
    temp = latest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(dict(status='CROP_DIAGNOSTICS_READY', archive=str(archive),
                                    sha256=sha(archive), report=report['status'])), encoding='utf-8')
    temp.replace(latest)
    print('CROP_DIAGNOSTICS_READY', archive, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return archive


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=PROJECT)
    ap.add_argument('--run', required=True)
    ap.add_argument('--samples', type=int, default=300000)
    args = ap.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run):
        ap.error('Invalid run name')
    export(args.root, args.run, args.samples)
