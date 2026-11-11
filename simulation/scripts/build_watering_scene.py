"""Build a layout preview; plant display meshes are never planning colliders."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import fast_simplification
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from plant_scene import load_inputs, make_transform, read_json, soil_candidates, transform_points, write_json


def vector(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def _nozzle_point(direction, distance_mm):
    """Return a nozzle-local point at ``distance_mm`` along its axis."""
    return np.asarray(direction, dtype=float) / np.linalg.norm(direction) * float(distance_mm) / 1000.0


def _add_nozzle_geom(parent, name, *, fromto, diameter_mm, rgba, mass_g=0.0,
                     contype="1", conaffinity="1", group="0"):
    ET.SubElement(parent, "geom", name=name, type="cylinder", fromto=vector(fromto),
                  size=f"{float(diameter_mm) / 2000:.9g}",
                  mass=f"{float(mass_g) / 1000:.9g}", rgba=rgba,
                  contype=str(contype), conaffinity=str(conaffinity), group=str(group))


def _lathe_mesh(asset, mount, section, direction):
    """Code-native revolved metal surface; ridges fit inside collision envelopes."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    u = np.cross(direction, [0, 1, 0])
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(direction, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    count = 160
    angles = np.arange(count) * 2 * np.pi / count
    ring = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    vertices = []
    for station in section["profile_mm"]:
        z, radius = station[:2]
        depth = station[2] if len(station) > 2 else 0.0
        # Forty shallow axial ribs make the reference's machined grip visible.
        radii = radius - depth * (.5 + .5 * np.cos(40 * angles))
        vertices.extend(direction * z / 1000 + ring * radii[:, None] / 1000)
    faces = []
    for k in range(len(section["profile_mm"]) - 1):
        for j in range(count):
            a, b = k * count + j, k * count + (j + 1) % count
            faces.extend([[a, b, a + count], [b, b + count, a + count]])
    name = section["name"]
    ET.SubElement(asset, "mesh", name=name, vertex=vector(np.asarray(vertices).ravel()),
                  face=" ".join(str(i) for i in np.asarray(faces).ravel()), inertia="shell")
    ET.SubElement(mount, "geom", name=name + "_visual", type="mesh", mesh=name,
                  rgba=section.get("rgba", "0.73 0.77 0.81 1"),
                  material="nozzle_brushed_metal", group="2", mass="0",
                  contype="0", conaffinity="0")


def _build_stepped_nozzle(asset, mount, nozzle, direction):
    """Build coaxial rear grasp, extension and detailed front spray head."""
    grip = nozzle.get("rear_grip", {})
    _add_nozzle_geom(mount, "nozzle_grasp_handle",
                     fromto=[*_nozzle_point(direction, grip["start_mm"]),
                             *_nozzle_point(direction, grip["end_mm"])],
                     diameter_mm=float(grip.get("diameter_mm", 9.3)),
                     rgba=grip.get("rgba", "0.26 0.29 0.32 1"),
                     mass_g=float(grip.get("mass_g", 5.0)), group="3")
    ends = []
    for segment in nozzle.get("segments", []):
        start_mm, end_mm = float(segment["start_mm"]), float(segment["end_mm"])
        if not (0.0 <= start_mm < end_mm):
            raise ValueError(f"Invalid stepped nozzle segment: {segment}")
        p0, p1 = _nozzle_point(direction, start_mm), _nozzle_point(direction, end_mm)
        _add_nozzle_geom(mount, segment["name"], fromto=[*p0, *p1],
                         diameter_mm=float(segment["diameter_mm"]),
                         rgba=segment.get("rgba", "0.58 0.61 0.65 1"),
                         mass_g=float(segment.get("mass_g", 0.5)), group="3")
        ends.append(end_mm)
    tip_mm = float(nozzle.get("tip_end_mm", max(ends, default=0.0)))
    ET.SubElement(mount, "site", name="nozzle_tip", pos=vector(_nozzle_point(direction, tip_mm)),
                  size="0.002", rgba="0 0.8 1 1", group="0")
    ET.SubElement(asset, "material", name="nozzle_brushed_metal", specular="0.7",
                  shininess="0.55", reflectance="0.12")
    for section in nozzle["visual_sections"]:
        _lathe_mesh(asset, mount, section, direction)
    return tip_mm


def relative(source, scene_dir):
    return Path(os.path.relpath(source, scene_dir)).as_posix()


def write_stl(path, points, faces):
    triangles = np.asarray(points[faces], dtype=np.float32)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    keep = norm > 1e-15
    triangles, normal, norm = triangles[keep], normal[keep], norm[keep]
    if not len(triangles):
        return 0
    # MuJoCo deduplicates STL vertices and requires at least four. A color bin
    # can contain only one triangle. Split it in its own plane so the displayed
    # surface, orientation and color stay the same; never add fake thickness.
    if len(np.unique(triangles.reshape(-1, 3), axis=0)) == 3:
        a, b, c = triangles[0].astype(np.float64)
        center = ((a+b+c)/3).astype(np.float32)
        if any(np.array_equal(center, p) for p in (a, b, c)):
            raise ValueError('Display triangle is too small for a distinct subdivision vertex')
        triangles = np.asarray([[a, b, center], [b, c, center], [c, a, center]], dtype=np.float32)
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        norm = np.linalg.norm(normal, axis=1)
    normal /= norm[:, None]
    records = np.zeros(len(triangles), dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records["normal"], records["vertices"] = normal, triangles
    with path.open("wb") as f:
        f.write(b"DISPLAY ONLY: NOT A PLANNING COLLIDER".ljust(80, b"\0"))
        f.write(np.uint32(len(records)).astype("<u4").tobytes())
        records.tofile(f)
    return len(records)


def display_meshes(out, local, faces, labels, weights, config, palette):
    directory = out / "display_meshes"
    directory.mkdir()
    records = []
    anchors = np.array([a["weight"] for a in palette["soft_anchors"]])
    colors = np.array([a["rgb"] for a in palette["soft_anchors"]])
    face_classes = weights["face_labels"]
    for cid in np.unique(face_classes):
        face_ids = np.flatnonzero(face_classes == cid)
        used, remapped = np.unique(faces[face_ids].reshape(-1), return_inverse=True)
        source_points = local[used]
        source_faces = remapped.reshape(-1, 3).astype(np.int32)
        budget = int(config["max_faces_by_class"].get(str(cid), config["default_max_faces"]))
        print(f"DISPLAY class={cid}: {len(source_faces):,} -> at most {budget:,} faces", flush=True)
        if len(source_faces) > budget:
            points, triangles = fast_simplification.simplify(source_points, source_faces, target_count=budget, agg=5.)
        else:
            points, triangles = source_points, source_faces
        # Nearest source attributes are used ONLY to color the simplified display.
        nearest = cKDTree(source_points).query(points, workers=1)[1]
        nearest_global = used[nearest]
        hard = weights["vertex_hard_collision"][nearest_global]
        face_hard = np.any(hard[triangles], axis=1) | bool(np.all(weights["face_hard_collision"][face_ids]))
        soft_weight = weights["vertex_effective_weight"][nearest_global][triangles].mean(axis=1)
        nbin = int(config["soft_color_bins_per_class"])
        bins = np.full(len(triangles), -1, dtype=int)
        soft = ~face_hard
        if soft.any():
            lo, hi = float(soft_weight[soft].min()), float(soft_weight[soft].max())
            bins[soft] = np.minimum(nbin - 1, ((soft_weight[soft] - lo) / max(hi - lo, 1e-9) * nbin).astype(int))
        for b in np.unique(bins):
            mask = bins == b
            if b < 0:
                rgb = np.array(palette["hard_collision_rgb"])
                level = None
            else:
                level = float(soft_weight[mask].mean())
                rgb = np.array([np.interp(level, anchors, colors[:, c]) for c in range(3)])
            name = f"plant_class_{cid}_color_{int(b) + 1}"
            path = directory / f"{name}.stl"
            count = write_stl(path, points, triangles[mask])
            if count:
                records.append(dict(name=name, file=path, rgb=np.rint(rgb).astype(int).tolist(), faces=count, class_id=int(cid), display_weight=level))
    return records


def build_xml(root, out, config, meshes, board_world, yaw, targets_world):
    tree = ET.parse(root / config["robot_xml"])
    model = tree.getroot()
    model.set("model", "so101_watering_layout_preview")
    compiler = model.find("compiler")
    compiler.set("meshdir", ".")
    asset = model.find("asset")
    robot_assets = (root / config["robot_xml"]).parent / "assets"
    for mesh in asset.findall("mesh"):
        mesh.set("file", relative(robot_assets / mesh.attrib["file"], out))
    world = model.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 0.02", rgba="0.86 0.88 0.9 1", friction="0.8 0.01 0.001")
    ET.SubElement(model, "statistic", center="0.23 0.10 0.17", extent="0.95")
    visual = model.find("visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960", azimuth="135", elevation="-25")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.6 0.6 0.6", specular="0.1 0.1 0.1")
    plant = ET.SubElement(world, "body", name="plant_display_only", pos=vector(board_world), euler=f"0 0 {np.deg2rad(yaw):.9g}")
    for entry in meshes:
        ET.SubElement(asset, "mesh", name=entry["name"], file=relative(entry["file"], out), inertia="shell", maxhullvert="64")
        rgba = vector([*(np.array(entry["rgb"]) / 255), 1])
        ET.SubElement(plant, "geom", name=entry["name"], type="mesh", mesh=entry["name"], contype="0", conaffinity="0", mass="0", group="1", rgba=rgba)
    nozzle = config["nozzle"]
    tip_mm = float(nozzle.get("length_mm", 0.0))
    if nozzle["enabled"]:
        direction = np.array(nozzle["direction_in_mount_frame"], dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError("Invalid nozzle direction")
        matches = [(body, site) for body in world.iter("body") for site in body.findall("site") if site.get("name") == nozzle["mount_site"]]
        if len(matches) != 1:
            raise ValueError("Nozzle mount site must exist exactly once")
        parent, site = matches[0]
        mount = ET.SubElement(parent, "body", name="placeholder_nozzle",
                              pos=vector(nozzle["body_position_in_gripper_m"]) if "body_position_in_gripper_m" in nozzle else site.get("pos", "0 0 0"),
                              quat=vector(nozzle["body_quaternion_in_gripper"]) if "body_quaternion_in_gripper" in nozzle else site.get("quat", "1 0 0 0"))
        profile = nozzle.get("profile", "legacy_cylinder")
        if profile == "stepped_nozzle":
            tip_mm = _build_stepped_nozzle(asset, mount, nozzle, direction)
        else:
            diameter, length, mass = float(nozzle["diameter_mm"]), float(nozzle["length_mm"]), float(nozzle["mass_g"])
            if min(diameter, length, mass) <= 0:
                raise ValueError("Nozzle dimensions and mass must be positive")
            end = direction / np.linalg.norm(direction) * length / 1000
            _add_nozzle_geom(mount, "placeholder_nozzle_cylinder", fromto=[0, 0, 0, *end],
                             diameter_mm=diameter, mass_g=mass, rgba="0.12 0.58 0.86 1")
        if nozzle.get("grasp_handle", False) and profile != "stepped_nozzle":
            # A short handle lies between the two finger pads. The elbow below
            # the fingers joins it to the projecting spray tube. Only the handle
            # may touch the two grasp pads; the spray tube remains collision checked.
            ET.SubElement(mount, "geom", name="nozzle_grasp_handle", type="cylinder",
                          fromto=vector(nozzle.get("handle_fromto_m", [-.013, 0, -.0002, -.004, 0, -.0002])),
                          size=f"{float(nozzle.get('handle_diameter_mm', 9.3))/2000:.9g}", mass="0.001",
                          rgba="0.18 0.22 0.27 1", contype="1", conaffinity="1")
            contact = model.find("contact")
            if contact is None:
                contact = ET.SubElement(model, "contact")
            ET.SubElement(contact, "exclude", body1="placeholder_nozzle", body2="moving_jaw_so101_v1")
        if profile != "stepped_nozzle":
            ET.SubElement(mount, "site", name="nozzle_tip", pos=vector(end), size="0.002", rgba="0 0.8 1 1", group="0")
        elif nozzle.get("grasp_handle", False):
            contact = model.find("contact")
            if contact is None:
                contact = ET.SubElement(model, "contact")
            ET.SubElement(contact, "exclude", body1="placeholder_nozzle", body2="moving_jaw_so101_v1")
    bounds = config["workspace"]
    lo, hi = np.array(bounds["min_m"], dtype=float), np.array(bounds["max_m"], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(hi > lo):
        raise ValueError("Invalid workspace limits")
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                ET.SubElement(world, "site", name=f"workspace_edge_{i}_{j}", type="capsule", fromto=vector([*corners[i], *corners[j]]), size="0.0007", rgba="0.15 0.45 0.65 0.22", group="5")
    for i, p in enumerate(targets_world):
        ET.SubElement(world, "site", name=f"soil_target_{i}", type="sphere", pos=vector(p), size="0.0025", rgba="0.1 0.8 1 1", group="0")
    ET.indent(tree, space="  ")
    tree.write(out / "scene.xml", encoding="utf-8", xml_declaration=True)
    return tip_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/scene_preview.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-promote", action="store_true", help="Do not replace the default scene pointer during diagnostics")
    args = parser.parse_args()
    config = read_json(args.config)
    out = args.output_dir or root / "scenes" / ("watering_preview_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory; existing scenes are preserved")
    out.mkdir(parents=True, exist_ok=True)
    print("Checking full-resolution geometry, labels, weights, organs and coordinate frame", flush=True)
    xyz, faces, labels, weights, local, center, rotation, policy, report = load_inputs(root, config)
    print(f"INPUTS_MATCH: {len(xyz):,} vertices; {len(faces):,} faces", flush=True)
    transform, board_world, yaw_rotation = make_transform(center, rotation, float(local[:, 2].min()), config["plant_layout"])
    world = transform_points(xyz, transform)
    if not np.allclose(world, local @ yaw_rotation.T + board_world, atol=1e-10):
        raise ValueError("Coordinate transformation consistency failed")
    target_class = int(policy["watering_target"]["class_id"])
    targets, surfaces, face_ids = soil_candidates(local, faces, weights, target_class, config["target_candidates"])
    target_world = targets @ yaw_rotation.T + board_world
    height = float(config["target_candidates"]["height_above_soil_mm"])
    allowed = policy["watering_target"]["approach_height_range_mm"]
    if not allowed[0] <= height <= allowed[1]:
        raise ValueError("Candidate height is outside the configured policy range")
    palette = read_json(root / config["source_root"] / config["palette"])["palette"]
    meshes = display_meshes(out, local, faces, labels, weights, config["visual_mesh"], palette)
    tip_mm = build_xml(root, out, config, meshes, board_world, config["plant_layout"]["yaw_deg"], target_world)
    print("Loading combined robot, nozzle and plant display scene", flush=True)
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    data = mujoco.MjData(model)
    if config["nozzle"].get("grasp_handle", False):
        angle = float(config["nozzle"].get("gripper_rad", -0.08))
        data.qpos[model.joint("gripper").qposadr] = angle
        data.ctrl[model.actuator("gripper").id] = angle
    mujoco.mj_forward(model, data)
    if config["nozzle"]["enabled"]:
        profile = config["nozzle"].get("profile", "legacy_cylinder")
        if profile == "stepped_nozzle":
            expected = ["nozzle_grasp_handle"] + [s["name"] for s in config["nozzle"].get("segments", [])]
            missing = [name for name in expected if not any(model.geom(i).name == name for i in range(model.ngeom))]
            if missing:
                raise ValueError(f"Compiled stepped nozzle is missing geometries: {missing}")
        else:
            nozzle_geom = model.geom("placeholder_nozzle_cylinder")
            if not np.allclose(nozzle_geom.size[:2], [config["nozzle"]["diameter_mm"] / 2000, config["nozzle"]["length_mm"] / 2000]):
                raise ValueError("Compiled nozzle size disagrees with configuration")
    for i in range(model.ngeom):
        if model.geom(i).name.startswith("plant_class_") and (model.geom_contype[i] or model.geom_conaffinity[i]):
            raise ValueError("Display mesh was accidentally enabled as a plant collider")
    for _ in range(1000):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
        raise ValueError("Combined scene simulation produced an instability or warning")
    provenance = []
    # Hash declared inputs only: source_root may be the complete server project.
    # Recursing there would scan environments, checkpoints and historical runs.
    input_keys = ("semantic_mesh", "semantic_labels", "weights", "organs", "coordinate_system", "policy", "palette")
    if config.get('virtual_substrate'):
        input_keys += ('virtual_substrate',)
    for p in sorted({(root / config["source_root"] / config[k]).resolve() for k in input_keys}):
        if p.is_file():
            with p.open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            provenance.append(dict(path=os.path.relpath(p, root).replace(os.sep, "/"), bytes=p.stat().st_size, sha256=sha))
    report.update(stage="LAYOUT_PREVIEW_ONLY", plant_collision_ready=False, path_planning_ready=False,
                  robot_registration_available=False,
                  nozzle_is_placeholder=True,
                  nozzle_dimensions_physically_measured=False,
                  nozzle_profile=config["nozzle"].get("profile", "legacy_cylinder"),
                  nozzle_tip_length_mm=tip_mm, palette=palette["version"],
                  display_meshes=len(meshes), display_triangles=sum(m["faces"] for m in meshes),
                  world_bounds_m=[world.min(axis=0).tolist(), world.max(axis=0).tolist()],
                  ground_clearance_m=float(world[:, 2].min()), source_to_world_m=transform.tolist(),
                  world_m_to_source_mm=np.linalg.inv(transform).tolist(), board_center_world_m=board_world.tolist(),
                  soil_candidate_count=len(targets), simulation_steps=1000, simulation_warnings=data.warning.number.tolist(),
                  notes=["Full-resolution data is preserved; simplified STL meshes are display only.",
                         "Plant collision geometry, inverse kinematics, goal clearance and path planning are not implemented in this stage.",
                         f"Blue markers are candidates {height:g} mm above substrate face centroids, not approved reachable goals.",
                         "Substrate source: " + report.get('watering_surface_source', 'semantic_mesh_substrate')],
                  effective_watering_target=policy["watering_target"])
    write_json(out / "scene_report.json", report)
    write_json(out / "input_manifest.json", provenance)
    write_json(out / "config_snapshot.json", config)
    write_json(out / "watering_candidates.json", dict(status="UNCHECKED_CANDIDATES", units="m",
        watering_surface_source=report.get('watering_surface_source', 'semantic_mesh_substrate'),
        entries=[dict(id=i, source_face_index=int(f), soil_surface_world_m=(s @ yaw_rotation.T + board_world).tolist(), nozzle_tip_candidate_world_m=t.tolist(), height_above_soil_mm=height) for i, (f, s, t) in enumerate(zip(face_ids, surfaces, target_world))]))
    np.savez_compressed(out / "full_resolution_world_geometry.npz", xyz_m=world.astype("<f8"), faces=faces, vertex_labels=labels,
                        mesh_fingerprint=np.array(report["mesh_fingerprint"]), source_to_world_m=transform,
                        note=np.array("Numeric weights reference the unchanged source vertex/face order"))
    (out / "LAYOUT_PREVIEW_SUCCESS").write_text("Scene loads and steps; plant collisions and path planning remain pending.\n", encoding="utf-8")
    if not args.no_promote:
        write_json(root / "configs/latest_scene.json", dict(scene=relative(out / "scene.xml", root), report=relative(out / "scene_report.json", root)))
    print(f"LAYOUT_PREVIEW_SUCCESS: {out / 'scene.xml'}", flush=True)


if __name__ == "__main__":
    main()
