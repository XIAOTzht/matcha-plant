#!/usr/bin/env python3
"""Build per-vertex/per-face contact planning weights from the final semantic PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from weight_palette import risk_colors, palette_description


CLASS_NAMES = {0: "unknown", 1: "stem", 2: "leaf", 3: "flower", 4: "pot", 5: "substrate", 6: "cart"}


def _scalar_dtype(token: str) -> str:
    return {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}[token]


def _element_properties(lines: list[str], element: str) -> list[str]:
    start = next(i for i, x in enumerate(lines) if x.split()[:2] == ["element", element])
    result = []
    for line in lines[start + 1:]:
        p = line.split()
        if not p or p[0] == "element" or p[0] == "end_header":
            break
        if p[0] == "property":
            result.append(line)
    return result


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Incomplete PLY header")
            text = line.decode("ascii", errors="replace").rstrip("\r\n")
            header_lines.append(text)
            if text == "end_header":
                break
        header = "\n".join(header_lines)
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only binary_little_endian PLY is supported")
        counts = {}
        for line in header_lines:
            p = line.split()
            if len(p) == 3 and p[0] == "element":
                counts[p[1]] = int(p[2])

        vp = _element_properties(header_lines, "vertex")
        vd = []
        for line in vp:
            p = line.split()
            if len(p) != 3:
                raise ValueError(f"Unsupported vertex property: {line}")
            vd.append((p[2], _scalar_dtype(p[1])))
        vertices = np.fromfile(f, dtype=np.dtype(vd), count=counts["vertex"])

        fp = _element_properties(header_lines, "face")
        fd = [("vertex_count", "u1"), ("vertex_indices", "<i4", (3,))]
        for line in fp:
            p = line.split()
            if p[0] != "property":
                continue
            if p[1] == "list":
                if p[-1] != "vertex_indices":
                    raise ValueError(f"Unsupported face list property: {line}")
                continue
            fd.append((p[2], _scalar_dtype(p[1])))
        raw_faces = np.fromfile(f, dtype=np.dtype(fd), count=counts["face"])
        faces = np.asarray(raw_faces["vertex_indices"], dtype=np.int32)
        if len(vertices) != counts["vertex"] or len(faces) != counts["face"]:
            raise ValueError("PLY ended before all elements were read")
        if np.any(raw_faces["vertex_count"] != 3):
            raise ValueError("Only triangle faces are supported")
        return vertices, faces


def _field(a: np.ndarray, name: str, default=0) -> np.ndarray:
    return np.asarray(a[name]) if name in a.dtype.names else np.full(len(a), default)


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray,
                vertex_weight: np.ndarray, vertex_hard: np.ndarray, vertex_target: np.ndarray,
                face_weight: np.ndarray, face_hard: np.ndarray, face_target: np.ndarray,
                face_uncertain: np.ndarray, comment: str) -> None:
    vd = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
                   ("label", "u1"), ("uncertain", "u1"), ("inferred", "u1"),
                   ("label_source", "u1"), ("contact_weight", "<f4"),
                   ("hard_collision", "u1"), ("watering_target", "u1")])
    vo = np.empty(len(vertices), dtype=vd)
    for name in ("x", "y", "z"):
        vo[name] = vertices[name]
    vo["red"], vo["green"], vo["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name in ("alpha", "label", "uncertain", "inferred", "label_source"):
        vo[name] = _field(vertices, name, 255 if name == "alpha" else 0)
    vo["contact_weight"] = vertex_weight
    vo["hard_collision"] = vertex_hard
    vo["watering_target"] = vertex_target

    fd = np.dtype([("vertex_count", "u1"), ("vertex_indices", "<i4", (3,)),
                   ("contact_weight", "<f4"), ("hard_collision", "u1"),
                   ("watering_target", "u1"), ("uncertain", "u1")])
    fo = np.empty(len(faces), dtype=fd)
    fo["vertex_count"] = 3
    fo["vertex_indices"] = faces
    fo["contact_weight"] = face_weight
    fo["hard_collision"] = face_hard
    fo["watering_target"] = face_target
    fo["uncertain"] = face_uncertain
    header = ("ply\nformat binary_little_endian 1.0\ncomment " + comment +
              "\ncomment coordinate_unit: millimeter\n" +
              f"element vertex {len(vo)}\n" +
              "property float x\nproperty float y\nproperty float z\n" +
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n" +
              "property uchar label\nproperty uchar uncertain\nproperty uchar inferred\nproperty uchar label_source\n" +
              "property float contact_weight\nproperty uchar hard_collision\nproperty uchar watering_target\n" +
              f"element face {len(fo)}\nproperty list uchar int vertex_indices\n" +
              "property float contact_weight\nproperty uchar hard_collision\nproperty uchar watering_target\nproperty uchar uncertain\nend_header\n")
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vo.tofile(f)
        fo.tofile(f)


def _heat_colors(base_weight: np.ndarray, effective_weight: np.ndarray,
                 hard: np.ndarray, uncertain: np.ndarray,
                 labels: np.ndarray) -> np.ndarray:
    """Use the same palette as the standalone colorbar; hard surfaces are uniform."""
    return risk_colors(effective_weight, hard)


def main() -> int:
    here = Path(__file__).resolve().parent
    default_input = here.parent.parent / "outputs" / "with_car" / "complete_hardware_v2"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=default_input)
    ap.add_argument("--output-dir", type=Path, default=default_input.parent / "contact_weights_v1")
    ap.add_argument("--policy", type=Path, default=here / "contact_policy.json")
    args = ap.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path, labels_path = input_dir / "semantic_mesh_mm.ply", input_dir / "labels.npz"
    vertices, faces = _read_ply(mesh_path)
    labels = np.load(labels_path, allow_pickle=False)
    vl = np.asarray(labels["vertex_labels"], dtype=np.uint8)
    fl = np.asarray(labels["face_labels"], dtype=np.uint8)
    vu = np.asarray(labels["uncertain"], dtype=bool)
    fu = np.asarray(labels["face_uncertain"], dtype=bool)
    cu = np.asarray(labels["contact_uncertain"], dtype=bool)
    if len(vertices) != len(vl) or len(faces) != len(fl):
        raise ValueError("PLY and labels.npz sizes do not match")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    classes = {int(k): v for k, v in policy["classes"].items()}
    um = float(policy["uncertainty"]["default_multiplier"])

    def class_arrays(ids: np.ndarray):
        weight = np.zeros(len(ids), dtype=np.float32)
        hard = np.zeros(len(ids), dtype=bool)
        target = np.zeros(len(ids), dtype=bool)
        for cid in np.unique(ids):
            c = classes.get(int(cid), classes[0])
            m = ids == cid
            weight[m] = float(c["weight"] or 0.0)
            hard[m] = bool(c["hard_collision"])
            target[m] = bool(c["watering_target"])
        return weight, hard, target

    vb, vh, vt = class_arrays(vl)
    fb, fh, ft = class_arrays(fl)
    ve = vb.copy()
    ve[(vu | cu) & ~vh] *= um
    fe = fb.copy()
    fe[fu & ~fh] *= um
    np.savez_compressed(output_dir / "contact_weights.npz",
                        vertex_base_weight=vb, vertex_effective_weight=ve,
                        vertex_hard_collision=vh, vertex_watering_target=vt,
                        vertex_uncertain=vu, vertex_contact_uncertain=cu,
                        face_base_weight=fb, face_effective_weight=fe,
                        face_hard_collision=fh, face_watering_target=ft,
                        face_uncertain=fu, class_ids=np.arange(7, dtype=np.uint8))
    (output_dir / "contact_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    original_rgb = np.stack([_field(vertices, "red", 0), _field(vertices, "green", 0), _field(vertices, "blue", 0)], axis=1).astype(np.uint8)
    _write_mesh(output_dir / "contact_weighted_mesh.ply", vertices, faces, original_rgb, ve, vh, vt, fe, fh, ft, fu.astype(np.uint8), "contact weights are planning costs, not forces")
    heat_rgb = _heat_colors(vb, ve, vh, (vu | cu), vl)
    _write_mesh(output_dir / "contact_risk_heatmap.ply", vertices, faces, heat_rgb, ve, vh, vt, fe, fh, ft, fu.astype(np.uint8), "contact risk heatmap: green -> yellow -> pale red; stem20=light red; stem24=slightly deeper; ALL hard classes=bright deep red")
    report = {
        "status": "PASS", "version": "contact-weights-0.2.0", "units": "mm",
        "input_mesh": str(mesh_path), "input_labels": str(labels_path),
        "vertex_count": int(len(vertices)), "face_count": int(len(faces)),
        "vertex_class_counts": {CLASS_NAMES.get(int(i), str(int(i))): int(np.sum(vl == i)) for i in np.unique(vl)},
        "face_class_counts": {CLASS_NAMES.get(int(i), str(int(i))): int(np.sum(fl == i)) for i in np.unique(fl)},
        "unknown_vertices": int(np.sum(vl == 0)), "uncertain_vertices": int(np.sum(vu)),
        "contact_uncertain_vertices": int(np.sum(cu)), "hard_collision_vertices": int(np.sum(vh)),
        "hard_collision_faces": int(np.sum(fh)), "watering_target_vertices": int(np.sum(vt)),
        "watering_target_faces": int(np.sum(ft)),
        "effective_soft_weight_min": float(np.min(ve[~vh])) if np.any(~vh) else None,
        "effective_soft_weight_max": float(np.max(ve[~vh])) if np.any(~vh) else None,
        "visualization": palette_description(),
        "files": {"arrays": str(output_dir / "contact_weights.npz"), "policy": str(output_dir / "contact_policy.json"), "weighted_mesh": str(output_dir / "contact_weighted_mesh.ply"), "heatmap_mesh": str(output_dir / "contact_risk_heatmap.ply")},
        "note": "Hard classes are validity constraints; substrate is a target but not a permitted nozzle intersection."
    }
    (output_dir / "contact_weights_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "CONTACT_WEIGHTS_SUCCESS").write_text("source mesh unchanged\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
