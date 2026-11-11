#!/usr/bin/env python3
"""Estimate a uniform millimetre scale from the UPPER blue plant support plate.

Requires the unchanged crop PLY and its matching car-crop report.json. The
larger chassis is explicitly excluded. Never stretches axes independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from plyfile import PlyData, PlyElement

from auto_extract_plant import load_mesh
from auto_crop_mesh import draw_points


def rectangle(points, trim=0.5):
    """Minimum-area trimmed rectangle, with no imposed target aspect ratio."""
    sample = points
    if len(sample) > 16000:
        sample = sample[np.random.default_rng(91).choice(len(sample), 16000, replace=False)]

    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def area(angle):
        bounds = np.percentile(sample @ rotation(angle), [trim, 100-trim], axis=0)
        return float(np.prod(bounds[1]-bounds[0]))

    angles = np.linspace(0, np.pi/2, 181, endpoint=False)
    angle = min(angles, key=area)
    result = minimize_scalar(area, bounds=(angle-.02, angle+.02), method='bounded')
    axes = rotation(result.x)
    uv = points @ axes
    low, high = np.percentile(uv, [trim, 100-trim], axis=0)
    if high[1]-low[1] > high[0]-low[0]:
        axes = axes[:, [1, 0]]
        uv = uv[:, [1, 0]]
        low, high = low[[1, 0]], high[[1, 0]]
    size = high-low
    if np.any(size <= 0):
        raise ValueError('Degenerate plate rectangle')
    coverage = []
    for axis in (0, 1):
        other = 1-axis
        for boundary in (low[axis], high[axis]):
            edge = np.abs(uv[:, axis]-boundary) < size[axis]*.025
            bins, _ = np.histogram(uv[edge, other], bins=np.linspace(low[other], high[other], 11))
            coverage.append(float((bins >= 3).mean()))
    return axes, low, high, coverage


def measure_upper(xyz, rgb, crop, known_mm=(135., 110.)):
    known = np.array(known_mm, dtype=float)
    if not np.isfinite(known).all() or not known[0] >= known[1] > 0:
        raise ValueError('Provide finite positive plate length >= width')
    origin = np.asarray(crop['frame_origin'], dtype=float)
    frame = np.asarray(crop['frame_axes'], dtype=float)
    if frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5):
        raise ValueError('Invalid crop coordinate frame')
    q = (xyz-origin) @ frame
    r, g, b = rgb.astype(float).T
    blue = (b > 1.4*r) & (b > 1.15*g) & (b > 60)
    chassis_span = float(np.max(np.asarray(crop['crop_upper'])[[0, 2]] -
                                np.asarray(crop['crop_lower'])[[0, 2]]))
    if chassis_span <= 0:
        raise ValueError('Invalid crop extent')
    # Only elevated blue surfaces: the large lower chassis at height zero
    # cannot be chosen even if it has many more blue samples.
    elevated = blue & (q[:, 1] > .06*chassis_span) & (q[:, 1] < .65*chassis_span)
    if elevated.sum() < 500:
        raise ValueError('Upper blue support plate not sufficiently visible')
    edges = np.linspace(.06*chassis_span, .65*chassis_span, 296)
    hist, edges = np.histogram(q[elevated, 1], bins=edges)
    smooth = gaussian_filter1d(hist.astype(float), 1.5)
    peaks, _ = find_peaks(smooth, prominence=max(10, smooth.max()*.04))
    if not len(peaks):
        raise ValueError('No distinct elevated blue plate')
    peak = peaks[np.argmax(smooth[peaks])]
    height = float((edges[peak]+edges[peak+1])/2)
    select = elevated & (np.abs(q[:, 1]-height) < .013*chassis_span)
    plate = xyz[select]
    center = plate.mean(axis=0)
    _, sv, vh = np.linalg.svd(plate-center, full_matrices=False)
    up = vh[-1]
    if up @ frame[:, 1] < 0:
        up = -up
    if up @ frame[:, 1] < .97 or sv[-1]/sv[1] > .08:
        raise ValueError('Upper blue patch is not a sufficiently flat horizontal plate')
    u = frame[:, 0] - (frame[:, 0] @ up)*up
    u /= np.linalg.norm(u)
    w = np.cross(u, up)
    basis = np.column_stack((u, w))
    planar = (plate-center) @ basis
    axes, low, high, coverage = rectangle(planar)
    measured = high-low
    separate_scales = known/measured
    disagreement = float(abs(np.diff(separate_scales)[0])/separate_scales.mean())
    if disagreement > .05:
        raise ValueError(f'Length/width scale disagreement {disagreement:.1%} > 5%; inspect plate, do not force calibration')
    if min(coverage) < .6:
        raise ValueError(f'Not enough visible boundary on all four plate edges: {coverage}')
    scale = float(measured @ known / (measured @ measured))
    fit_mm = measured*scale
    board_u, board_w = (basis @ axes).T
    board_frame = np.column_stack((board_u, up, board_w))
    board_center = center + basis @ axes @ ((low+high)/2)
    corners_2d = np.array([[low[0], low[1]], [high[0], low[1]],
                           [high[0], high[1]], [low[0], high[1]]])
    corners = center + corners_2d @ axes.T @ basis.T
    # Report trimming sensitivity, not a claimed real-world error bound.
    sensitivity = []
    for trim in (.1, .5, 1.):
        lo, hi = np.percentile(planar @ axes, [trim, 100-trim], axis=0)
        length = hi-lo
        sensitivity.append(float(length @ known/(length @ length)))
    result = dict(known_plate_mm=known.tolist(), measured_plate_model_units=measured.tolist(),
                  scale_mm_per_model_unit=scale, length_width_separate_scales=separate_scales.tolist(),
                  scale_disagreement_fraction=disagreement, fitted_plate_mm=fit_mm.tolist(),
                  plate_residual_mm=(fit_mm-known).tolist(), edge_coverage=coverage,
                  boundary_trim_scale_sensitivity=sensitivity, upper_plate_height_in_crop_frame=height,
                  selected_blue_vertices=int(select.sum()), board_center_original=board_center.tolist(),
                  board_frame_axes=board_frame.tolist(), estimated_corners_original=corners.tolist())
    return result, select, board_center, board_frame


def run(args):
    src, out = args.input.resolve(), args.output_dir.resolve()
    if src.is_relative_to(out) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a separate, empty output directory')
    print('Loading cropped mesh and matching crop report', flush=True)
    ply, xyz, faces = load_mesh(src)
    crop = json.loads(args.crop_report.read_text(encoding='utf-8'))
    if crop['output_vertices'] != len(xyz) or crop['output_faces'] != len(faces):
        raise ValueError('Crop report does not match input mesh counts')
    rgb = np.column_stack([ply['vertex'].data[k] for k in ('red', 'green', 'blue')])
    print('Detecting UPPER blue plate above the chassis', flush=True)
    report, mask, center, frame = measure_upper(xyz, rgb, crop, (args.length_mm, args.width_mm))
    scale = report['scale_mm_per_model_unit']
    out.mkdir(parents=True, exist_ok=True)
    vertices = ply['vertex'].data.copy()
    for axis in ('x', 'y', 'z'):
        vertices[axis] *= scale
    elements = [PlyElement.describe(vertices, 'vertex') if e.name == 'vertex' else e for e in ply.elements]
    target = out/'plant_car_mm.ply'
    print(f'Uniform scale: {scale:.6f} mm / model unit; writing mesh', flush=True)
    PlyData(elements, text=False, byte_order=ply.byte_order,
            comments=list(ply.comments)+['coordinate_unit: millimeter', 'uniform scale; original origin and orientation retained'],
            obj_info=list(ply.obj_info)).write(str(target))
    local = (xyz-center) @ frame * scale
    dims = np.ptp(local, axis=0)
    robust_low, robust_high = np.percentile(local, [.1, 99.9], axis=0)
    above = local[:, 1] > 3.0
    report.update(input=str(src), output=str(target), units='mm',
                  vertices=len(vertices), faces=len(faces),
                  object_extent_in_board_axes_mm=dict(length_axis=float(dims[0]), height=float(dims[1]), width_axis=float(dims[2])),
                  object_robust_extent_0p1_99p9_mm=(robust_high-robust_low).tolist(),
                  top_above_support_plate_mm=float(local[:, 1].max()),
                  above_plate_object_extent_mm=np.ptp(local[above], axis=0).tolist() if above.any() else None,
                  measurement_note='Above-plate object includes pot and plant. Plant-only height requires pot/soil boundary or semantic labels.',
                  accuracy_note='Scale agreement and edge trimming sensitivity are internal checks, not absolute accuracy or confidence bounds.',
                  transform_original_to_mm=np.diag([scale, scale, scale, 1.]).tolist(),
                  inverse_transform_mm_to_original=np.diag([1/scale, 1/scale, 1/scale, 1.]).tolist(),
                  projection_note='Keep original mesh for original camera poses. For mm geometry, camera centres/translations must use the same scale.',
                  method='Elevated blue height peak, planar SVD, unconstrained trimmed minimum-area rectangle, single least-squares uniform scale')
    np.save(out/'upper_plate_vertex_indices.npy', np.flatnonzero(mask))
    rng = np.random.default_rng(17)
    ids = rng.choice(len(xyz), min(180000, len(xyz)), replace=False)
    highlight = np.full((len(ids), 3), 105, dtype=np.uint8)
    highlight[mask[ids]] = [255, 205, 30]
    draw_points(xyz[ids], highlight, out/'plate_selection.jpg', 'UPPER reference plate in yellow', frame, center)
    board_ids = np.flatnonzero(mask)
    draw_points(xyz[board_ids], rgb[board_ids], out/'upper_plate.jpg', 'Upper plate: pot occludes centre', frame, center)
    (out/'scale_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'units.txt').write_text('plant_car_mm.ply coordinate unit: millimeter (mm)\nOriginal vertex order and faces retained.\nSee scale_report.json for transform and measurement limitations.\n', encoding='utf-8')
    print(f'SCALE_CALIBRATION_SUCCESS: {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--crop-report', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--length-mm', type=float, default=135.)
    parser.add_argument('--width-mm', type=float, default=110.)
    run(parser.parse_args())
