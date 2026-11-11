"""Shared input checks and transforms for metric semantic plant meshes."""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import numpy as np
from plyfile import PlyData


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def effective_policy(root, config):
    """Apply scene-only target heights without altering source weights/policy."""
    policy = read_json(root / config["source_root"] / config["policy"])
    override = config.get("watering_target_override", {})
    if set(override) - {"approach_height_range_mm", "default_height_mm"}:
        raise ValueError("Only watering target heights can be overridden by a scene")
    policy = copy.deepcopy(policy)
    policy["watering_target"].update(override)
    target = policy["watering_target"]
    bounds = np.asarray(target["approach_height_range_mm"], dtype=float)
    height = float(target["default_height_mm"])
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] < 0
            or bounds[1] < bounds[0] or not bounds[0] <= height <= bounds[1]):
        raise ValueError("Invalid watering target height policy")
    return policy


def fingerprint(xyz, faces, labels):
    h = hashlib.sha256()
    for a, dtype in ((xyz, "<f8"), (faces, "<i4"), (labels, "<i4")):
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
    return h.hexdigest()


def board_rotation(coordinate_system):
    """Source columns are long-axis, up-axis, transverse-axis; output is XYZ-up."""
    frame = np.array(coordinate_system["board_frame_axes"], dtype=float)
    center = np.array(coordinate_system["board_center_mm"], dtype=float)
    if frame.shape != (3, 3) or center.shape != (3,) or not np.isfinite(frame).all() or not np.isfinite(center).all():
        raise ValueError("Invalid board frame/center")
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5):
        raise ValueError("Board frame must be orthonormal")
    rotation = frame[:, [0, 2, 1]].copy()
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] *= -1
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("Board frame did not produce a proper rotation")
    return center, rotation


def make_transform(center_mm, board_axes, local_min_z_m, layout):
    yaw = np.deg2rad(float(layout["yaw_deg"]))
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    board_world = np.array([*layout["board_center_xy_m"], float(layout["ground_clearance_mm"]) / 1000 - local_min_z_m])
    rotation = yaw_rotation @ board_axes.T
    transform = np.eye(4)
    transform[:3, :3] = rotation / 1000
    transform[:3, 3] = board_world - rotation @ center_mm / 1000
    return transform, board_world, yaw_rotation


def transform_points(xyz, transform):
    return np.asarray(xyz) @ transform[:3, :3].T + transform[:3, 3]


def load_inputs(root, config):
    base = root / config["source_root"]
    cs = read_json(base / config["coordinate_system"])
    if cs.get("units") != "mm":
        raise ValueError("This importer requires metric source coordinates in mm")
    ply = PlyData.read(str(base / config["semantic_mesh"]), known_list_len={"face": {"vertex_indices": 3}})
    vertices, face_data = ply["vertex"].data, ply["face"].data
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(float)
    faces = np.asarray(face_data["vertex_indices"], dtype=np.int32)
    labels = np.asarray(vertices["label"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Invalid triangle geometry")
    with np.load(base / config["semantic_labels"], allow_pickle=False) as z:
        if not np.array_equal(z["vertex_labels"], labels) or not np.array_equal(z["face_labels"], face_data["label"]):
            raise ValueError("Semantic PLY and labels NPZ do not match")
    with np.load(base / config["weights"], allow_pickle=False) as z:
        weights = {k: z[k] for k in z.files}
    with np.load(base / config["organs"], allow_pickle=False) as z:
        organs = {k: z[k] for k in z.files}
    fp = fingerprint(xyz, faces, labels)
    for kind, arrays in (("weights", weights), ("organs", organs)):
        if str(arrays["mesh_fingerprint"]) != fp or float(arrays["mm_per_unit"]) != 1:
            raise ValueError(f"Wrong geometry fingerprint or unit in {kind}")
        if not np.array_equal(arrays["vertex_labels"], labels):
            raise ValueError(f"Wrong vertex semantics in {kind}")
    if not np.array_equal(weights["vertex_instance_id"], organs["vertex_instance_id"]):
        raise ValueError("Weights and organ instances disagree")
    if not np.array_equal(weights["face_labels"], face_data["label"]):
        raise ValueError("Weight and semantic face labels disagree")
    policy = effective_policy(root, config)
    hard_ids = np.array(policy["hard_collision"]["classes"])
    expected_hard = np.isin(labels, hard_ids)
    if not np.array_equal(expected_hard, weights["vertex_hard_collision"]):
        raise ValueError("Hard vertex flags disagree with contact policy")
    expected_face_hard = np.isin(weights["face_labels"], hard_ids) | np.any(expected_hard[faces], axis=1)
    if not np.array_equal(expected_face_hard, weights["face_hard_collision"]):
        raise ValueError("Hard face flags disagree with boundary rule")
    for name in ("vertex_effective_weight", "face_effective_weight"):
        if not np.isfinite(weights[name]).all() or np.any(weights[name] < 0):
            raise ValueError(f"Invalid weights: {name}")
    center, rotation = board_rotation(cs)
    local_m = (xyz - center) @ rotation / 1000
    report = dict(status="INPUTS_MATCH", mesh_fingerprint=fp, vertex_count=len(xyz), face_count=len(faces), source_units="mm",
                  world_units="m", hard_vertex_count=int(expected_hard.sum()), hard_face_count=int(expected_face_hard.sum()),
                  organ_review_required=True, source_coordinates_unchanged=True)
    assumption_path = (base / config["semantic_mesh"]).parent / 'virtual_substrate.json'
    if assumption_path.is_file():
        assumption = read_json(assumption_path)
        with np.load(base / config["semantic_labels"], allow_pickle=False) as z:
            vm, fm = z['virtual_substrate'], z['face_virtual_substrate']
        start, count = int(assumption['vertex_start']), int(assumption['vertex_count'])
        face_start, face_count = int(assumption['face_start']), int(assumption['face_count'])
        if (assumption.get('surface_source') != 'assumed_virtual_substrate'
                or assumption.get('measured') is not False
                or vm.shape != labels.shape or fm.shape != weights['face_labels'].shape
                or start < 0 or count <= 0 or face_start < 0 or face_count <= 0
                or start+count != len(labels) or face_start+face_count != len(faces)
                or not np.array_equal(vm, np.arange(len(labels)) >= start)
                or not np.array_equal(fm, np.arange(len(faces)) >= face_start)
                or not np.all(labels[vm] == 5) or not np.all(weights['face_labels'][fm] == 5)
                or not np.all(expected_hard[vm]) or not np.all(expected_face_hard[fm])):
            raise ValueError('Virtual substrate provenance/labels/hard flags disagree')
        # The sidecar must describe the actual appended surface, not just its IDs.
        frame = np.asarray(cs['board_frame_axes'], float)
        appended = (xyz[vm]-np.asarray(cs['board_center_mm']))@frame
        height = float(assumption['assumed_height_mm'])
        polygon = np.asarray(assumption['footprint_xz_mm'], float)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(polygon)
        if (not np.isfinite(height) or not np.allclose(appended[:, 1], height, atol=.001, rtol=0)
                or not np.all(appended[:, [0, 2]]@hull.equations[:, :2].T+hull.equations[:, 2] <= .001)
                or not np.all(vm[faces[fm]]) or np.any(vm[faces[~fm]])):
            raise ValueError('Virtual substrate geometry disagrees with its assumption')
        report.update(watering_surface_source='assumed_virtual_substrate', measured_soil_available=False,
                      virtual_substrate=assumption, virtual_substrate_sha256=hashlib.sha256(assumption_path.read_bytes()).hexdigest(),
                      real_robot_ready=False)
    else:
        with np.load(base / config["semantic_labels"], allow_pickle=False) as z:
            if any(k in z and np.any(z[k]) for k in ('virtual_substrate', 'face_virtual_substrate')):
                raise ValueError('Virtual substrate geometry is missing its assumption sidecar')
        report.update(watering_surface_source='semantic_mesh_substrate', measured_soil_available=None)
    return xyz, faces, labels, weights, local_m, center, rotation, policy, report


def soil_candidates(local_m, faces, weights, class_id, settings):
    indices = np.flatnonzero(weights["face_labels"] == class_id)
    if not len(indices):
        raise ValueError("No soil triangles; cannot create watering candidates")
    triangles = local_m[faces[indices]]
    centers = triangles.mean(axis=1)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    good = areas > 1e-12
    indices, centers, areas = indices[good], centers[good], areas[good]
    if not len(indices):
        raise ValueError("No non-degenerate soil triangles")
    mean = np.average(centers, axis=0, weights=areas)
    chosen = [int(np.argmin(np.linalg.norm(centers[:, :2] - mean[:2], axis=1)))]
    for _ in range(int(settings["count"]) - 1):
        distance = np.min(np.linalg.norm(centers[:, None, :2] - centers[chosen][None, :, :2], axis=2), axis=1)
        nxt = int(np.argmax(distance))
        if distance[nxt] < float(settings["separation_mm"]) / 1000:
            break
        chosen.append(nxt)
    surface = centers[chosen]
    targets = surface.copy()
    targets[:, 2] += float(settings["height_above_soil_mm"]) / 1000
    return targets, surface, indices[chosen]
