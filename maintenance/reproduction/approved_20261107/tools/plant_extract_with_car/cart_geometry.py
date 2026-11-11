"""Locate a fixed blue cart, then grow the object above its calibrated board.

Only cart localization uses color. Plant/pot proposals use geometry regardless
of leaf/flower color; they are not semantic class predictions.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from auto_crop_mesh import voxel_clusters
from calibrate_upper_plate import measure_upper


def fit_chassis(points, rng, trials=400):
    span = np.linalg.norm(np.diff(np.percentile(points, [1, 99], axis=0), axis=0)[0])
    tolerance = span*.006
    fit = points if len(points) <= 12000 else points[rng.choice(len(points), 12000, replace=False)]
    best, count = None, 0
    for _ in range(trials):
        tri = fit[rng.choice(len(fit), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        near = np.abs((fit-tri[0])@normal) < tolerance
        if near.sum() > count:
            best, count = near, int(near.sum())
    if best is None or count < len(fit)*.20:
        raise ValueError('Blue cluster has no broad planar chassis')
    patch = fit[best]; origin = patch.mean(0)
    _, values, axes = np.linalg.svd(patch-origin, full_matrices=False)
    if values[1] < values[0]*.18:
        raise ValueError('Blue chassis plane is narrow/degenerate')
    return origin, axes[-1], tolerance


def cart_frame(blue_points, origin, up):
    u = np.eye(3)[np.argmin(abs(up))]; u = np.cross(up, u); u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(u, up)))
    _, eigenvectors = np.linalg.eigh(np.cov(((blue_points-origin)@basis).T))
    u = basis@eigenvectors[:, -1]
    return np.column_stack((u, up, np.cross(u, up)))


def grow_above_board(points, board_center, board_frame, board_size, cart_length,
                     height_limit=4., radius_limit=2.):
    local = (points-board_center)@board_frame
    radius = np.linalg.norm(local[:, [0, 2]], axis=1)
    roi = ((local[:, 1] > -.035*cart_length) & (local[:, 1] < height_limit*cart_length) &
           (radius < radius_limit*cart_length))
    indices = np.flatnonzero(roi); cloud = points[indices]; q = local[indices]
    if len(cloud) < 100:
        raise ValueError('Not enough geometry above the upper board')
    voxel = .018*cart_length
    labels, counts, _, _ = voxel_clusters(cloud, voxel)
    footprint = np.asarray(board_size)/2 + .025*cart_length
    seeds = ((np.abs(q[:, [0, 2]]) < footprint).all(axis=1) &
             (q[:, 1] > -.025*cart_length) & (q[:, 1] < .15*cart_length))
    selected = np.unique(labels[seeds])
    selected = selected[counts[selected] >= 8]
    if not len(selected):
        raise ValueError('No substantial geometry is anchored near the upper board')
    keep = np.isin(labels, selected)
    # Sampled/thin leaves may form separate components. Attach nearby fragments
    # by geometric distance, never by hue or by assuming the biggest component.
    attached = 0
    for _ in range(3):
        distances = cKDTree(cloud[keep]).query(cloud[~keep], workers=1)[0]
        close = np.unique(labels[~keep][distances < .04*cart_length])
        close = close[counts[close] >= 3]
        if not len(close):
            break
        attached += len(close)
        keep |= np.isin(labels, close)
    grown = cloud[keep]; heights = q[keep, 1]
    if len(grown) < 100 or np.percentile(heights, 99) < .15*cart_length:
        raise ValueError('Upper-board geometry does not extend far enough above the board')
    near_edge = ((heights > .97*height_limit*cart_length) |
                 (np.linalg.norm(q[keep][:, [0, 2]], axis=1) > .97*radius_limit*cart_length))
    if near_edge.sum() >= max(8, len(grown)*.0005):
        raise ValueError('Object reaches the geometric search envelope; enlarge search limits or inspect background connections')
    return grown, dict(method='upper-board anchored 26-neighbor voxel growth and nearby fragments; no plant color',
                       samples=len(grown), seed_samples=int(seeds.sum()), attached_components=attached,
                       voxel_per_cart_length=.018, attachment_per_cart_length=.04,
                       max_height_cart_lengths=height_limit, max_radius_cart_lengths=radius_limit,
                       geometry_height_above_board=float(np.max(heights)), color_filter_used=False)


def detect_geometry(points, rgb, rng, voxel_ratio=.006, candidate_rank=1,
                    plate_mm=(135., 110.), max_height=4., max_radius=2.):
    r, g, b = np.asarray(rgb, dtype=float).T
    blue = (b > r*1.4) & (b > g*1.15) & (b > 60)
    if blue.sum() < 100:
        raise ValueError('Need a reconstructed blue cart; plant color is unrestricted')
    extent = float(np.linalg.norm(np.diff(np.percentile(points, [5, 95], axis=0), axis=0)[0]))
    if extent <= 0:
        raise ValueError('Degenerate scene')
    bp = points[blue]
    labels, counts, _, _ = voxel_clusters(bp, extent*voxel_ratio)
    options, rejected = [], []
    for cluster in np.argsort(counts)[-10:][::-1]:
        if counts[cluster] < 100:
            continue
        panel = bp[labels == cluster]; center = np.median(panel, axis=0)
        try:
            origin, normal, tolerance = fit_chassis(panel, rng)
        except ValueError as exc:
            rejected.append(str(exc)); continue
        for sign in (1, -1):
            frame = cart_frame(panel, origin, normal*sign)
            lo, hi = np.percentile((panel-origin)@frame, [.2, 99.8], axis=0)
            length = float(max((hi-lo)[[0, 2]]))
            if length <= 0:
                continue
            near = np.linalg.norm(points-center, axis=1) < 2*length
            crop = dict(frame_origin=origin.tolist(), frame_axes=frame.tolist(),
                        crop_lower=lo.tolist(), crop_upper=hi.tolist())
            try:
                plate, _, board_center, board_frame = measure_upper(points[near], rgb[near], crop, plate_mm)
                measured = np.asarray(plate['measured_plate_model_units'])
                # The upper board must be materially smaller than the lower
                # chassis, not a large blue tabletop with a similar aspect ratio.
                if not .12*length < measured[0] < .80*length:
                    raise ValueError('Upper plate is not smaller than the chassis')
                canopy, growth = grow_above_board(points, board_center, board_frame, measured, length,
                                                  max_height, max_radius)
            except ValueError as exc:
                rejected.append(str(exc)); continue
            score = float(len(panel)/(1 + plate['scale_disagreement_fraction']))
            options.append(dict(blue=panel, canopy=canopy, origin=origin, frame=frame,
                                lo=lo, hi=hi, length=length, score=score,
                                upper_plate=plate, growth=growth, tolerance=tolerance))
    options.sort(key=lambda item: item['score'], reverse=True)
    if len(options) < candidate_rank:
        reasons = '; '.join(dict.fromkeys(rejected)) or 'No sufficiently large blue cart component'
        raise ValueError('No cart with a verified upper plate and anchored object: '+reasons)
    chosen = options[candidate_rank-1]
    chosen['candidates'] = [dict(score=item['score'], blue_samples=len(item['blue']),
                                 plant_samples=len(item['canopy']), center=item['origin'].tolist(),
                                 scale_disagreement_fraction=item['upper_plate']['scale_disagreement_fraction'],
                                 selection='fixed blue cart + upper rectangle + all-color geometry') for item in options]
    chosen['rejected_candidates'] = list(dict.fromkeys(rejected))
    return chosen
