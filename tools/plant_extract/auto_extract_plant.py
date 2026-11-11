#!/usr/bin/env python3
"""Extract a plant AND its pot using foreground masks and matching COLMAP poses.

Uses CPU only. Input must be the original (untransformed) MAtCha triangle mesh.
No fixed bounding box, largest-component assumption, or green-color threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from plyfile import PlyData, PlyElement


@dataclass
class View:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # world-to-camera; COLMAP/OpenCV: +Z forward, +Y down
    t: np.ndarray
    mask: np.ndarray | None = None


def qvec_to_rotation(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def unpack(f, fmt):
    n = struct.calcsize("<" + fmt)
    raw = f.read(n)
    if len(raw) != n:
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack("<" + fmt, raw)


def read_colmap(sfm: Path):
    candidates = [sfm / "sparse" / "0", sfm / "sparse", sfm]
    folder = next((p for p in candidates if
                   (p / "cameras.bin").exists() and (p / "images.bin").exists()
                   or (p / "cameras.txt").exists() and (p / "images.txt").exists()), None)
    if folder is None:
        raise FileNotFoundError(f"No COLMAP cameras/images pair under {sfm}")
    cameras = {}
    poses = []
    if (folder / "cameras.bin").exists() and (folder / "images.bin").exists():
        with (folder / "cameras.bin").open("rb") as f:
            for _ in range(unpack(f, "Q")[0]):
                cid, model_id, width, height = unpack(f, "iiQQ")
                if model_id not in (0, 1):
                    raise ValueError("Use undistorted PINHOLE/SIMPLE_PINHOLE cameras from mast3r_sfm; "
                                     f"camera {cid} has model id {model_id}")
                params = unpack(f, "d" * (3 if model_id == 0 else 4))
                intr = (params[0], params[0], *params[1:]) if model_id == 0 else params
                cameras[cid] = (int(width), int(height), *intr)
        with (folder / "images.bin").open("rb") as f:
            for _ in range(unpack(f, "Q")[0]):
                vals = unpack(f, "idddddddi")
                name_bytes = bytearray()
                while True:
                    c = f.read(1)
                    if not c:
                        raise ValueError("Unterminated COLMAP image name")
                    if c == b"\0":
                        break
                    name_bytes.extend(c)
                name = name_bytes.decode("utf-8")
                n_points = unpack(f, "Q")[0]
                # Seek does not detect truncated files, so check the actual end position.
                target = f.tell() + 24 * n_points
                if target > (folder / "images.bin").stat().st_size:
                    raise ValueError("Truncated COLMAP observations")
                f.seek(target)
                poses.append((name, vals[8], vals[1:5], vals[5:8]))
    else:
        for line in (folder / "cameras.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            cid, model, w, h = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
            params = tuple(map(float, fields[4:]))
            if model == "SIMPLE_PINHOLE" and len(params) == 3:
                intr = (params[0], params[0], *params[1:])
            elif model == "PINHOLE" and len(params) == 4:
                intr = params
            else:
                raise ValueError(f"Unsupported camera {model}; use undistorted SfM images")
            cameras[cid] = (w, h, *intr)
        with (folder / "images.txt").open(encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.strip().split(maxsplit=9)
                if len(fields) != 10:
                    raise ValueError("Malformed COLMAP image pose")
                poses.append((fields[9], int(fields[8]), tuple(map(float, fields[1:5])),
                              tuple(map(float, fields[5:8]))))
                # Each pose is followed by exactly one observations line, possibly empty.
                if f.readline() == "":
                    raise ValueError("Missing COLMAP observations line")
    views = [View(name, *cameras[cid], qvec_to_rotation(q), np.asarray(t))
             for name, cid, q, t in sorted(poses)]
    names = [v.name for v in views]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names in COLMAP model")
    return views, folder


def image_file(folder: Path, name: str, is_mask=False):
    # Exact names first; rembg outputs use <stem>_mask.png or <stem>.png.
    normalized = name.replace("\\", "/")
    rel = Path(normalized)
    stem = rel.stem
    choices = ([rel.with_name(stem + "_mask.png"), rel.with_suffix(".png"), rel]
               if is_mask else [rel])
    if rel.name != str(rel):
        choices += [Path(p.name) for p in choices]
    for p in dict.fromkeys(choices):
        path = (folder / p).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Image name escapes the input folder")
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot match {name} in {folder}")


def prepare_masks(views, args):
    mask_dir = args.masks_dir
    if mask_dir is None:
        if args.images_dir is None:
            raise ValueError("Supply --masks-dir, or --images-dir to generate masks with rembg")
        # Kept in the output directory for inspection; never infer foreground from white pixels.
        mask_dir = args.output_dir / "generated_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        try:
            from rembg import new_session, remove
        except ImportError as e:
            raise RuntimeError('Automatic masks need: pip install "rembg[cpu]"') from e
        session = new_session(args.rembg_model, providers=["CPUExecutionProvider"])
        for v in views:
            src = image_file(args.images_dir, v.name)
            dst = mask_dir / (Path(v.name.replace("\\", "/")).stem + "_mask.png")
            if dst.exists():
                raise FileExistsError(f"Stale generated mask exists: {dst}; use a fresh output directory")
            with Image.open(src) as im:
                if abs((im.width / im.height) / (v.width / v.height) - 1) > 0.02:
                    raise ValueError(f"Image aspect ratio does not match camera: {src}")
                print(f"Generating mask: {v.name}", flush=True)
                result = remove(im.convert("RGB"), session=session)
                result.getchannel("A").save(dst)
    for v in views:
        src = image_file(mask_dir, v.name, is_mask=True)
        with Image.open(src) as im:
            if abs((im.width / im.height) / (v.width / v.height) - 1) > 0.02:
                raise ValueError(f"Mask aspect ratio differs from COLMAP: {src}")
            if "A" in im.getbands():
                im = im.getchannel("A")
            else:
                im = im.convert("L")
            im.thumbnail((args.mask_size, args.mask_size), Image.Resampling.LANCZOS)
            mask = np.array(im) >= args.mask_threshold
        foreground_fraction = float(mask.mean())
        if foreground_fraction < 0.0001 or foreground_fraction > 0.98:
            raise ValueError(f"Suspicious empty/full foreground mask {src}: {foreground_fraction:.3%}")
        # A small expansion protects thin leaves from segmentation/pose pixel errors.
        if args.dilate > 0:
            mask = maximum_filter(mask, size=2 * args.dilate + 1)
        v.mask = mask
    return mask_dir


def project(points, v):
    cam = points @ v.R.T + v.t
    z = cam[:, 2]
    safe_z = np.where(z > 1e-8, z, 1.0)
    # Pixel-center-aware scaling from COLMAP dimensions to mask dimensions.
    mh, mw = v.mask.shape
    u = ((v.fx * cam[:, 0] / safe_z + v.cx + 0.5) * mw / v.width) - 0.5
    y = ((v.fy * cam[:, 1] / safe_z + v.cy + 0.5) * mh / v.height) - 0.5
    inside = np.isfinite(u) & np.isfinite(y) & (z > 1e-8)
    inside &= (u >= -0.5) & (u < mw - 0.5) & (y >= -0.5) & (y < mh - 0.5)
    return u, y, inside


def score_points(points, views, min_views, min_ratio, chunk_size=200000):
    keep = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        p = points[start:start + chunk_size]
        visible = np.zeros(len(p), dtype=np.int32)
        hits = np.zeros(len(p), dtype=np.int32)
        for view in views:
            u, y, inside = project(p, view)
            ids = np.flatnonzero(inside)
            visible[ids] += 1
            ix = np.clip(np.floor(u[ids]+0.5).astype(np.int64), 0, view.mask.shape[1]-1)
            iy = np.clip(np.floor(y[ids]+0.5).astype(np.int64), 0, view.mask.shape[0]-1)
            hits[ids] += view.mask[iy, ix]
        keep[start:start + len(p)] = (hits >= min_views) & (hits >= min_ratio * visible)
        if len(points) > chunk_size:
            print(f"  Classified {start + len(p):,}/{len(points):,} samples", flush=True)
    return keep


def load_mesh(path):
    # MAtCha exports triangles. Known list length lets plyfile memory-map large meshes.
    ply = PlyData.read(str(path), known_list_len={"face": {"vertex_indices": 3, "vertex_index": 3}})
    if "vertex" not in ply or "face" not in ply:
        raise ValueError("Input must be a triangle mesh PLY, not an SfM/Gaussian point cloud")
    vertices = ply["vertex"].data
    face_data = ply["face"].data
    key = next((k for k in ("vertex_indices", "vertex_index") if k in face_data.dtype.names), None)
    if key is None or not len(face_data):
        raise ValueError("Mesh has no face indices")
    index_data = face_data[key]
    if index_data.ndim == 2 and index_data.shape[1] == 3:
        faces = np.asarray(index_data, dtype=np.int64)
    else:
        if any(len(row) != 3 for row in index_data):
            raise ValueError("Only triangle faces are supported")
        faces = np.stack(index_data).astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh has invalid vertex indices")
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Mesh contains nonfinite coordinates")
    return ply, xyz, faces


def component_filter(faces, min_faces):
    # Do not select only the largest component: detached reconstructed leaves matter.
    if min_faces <= 1:
        return np.ones(len(faces), dtype=bool), None
    used, inv = np.unique(faces.reshape(-1), return_inverse=True)
    tri = inv.reshape(-1, 3)
    edges = np.concatenate((tri[:, :2], tri[:, 1:], tri[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(used), len(used))).tocsr()
    n, labels = connected_components(graph, directed=False)
    face_labels = labels[tri[:, 0]]
    sizes = np.bincount(face_labels, minlength=n)
    return sizes[face_labels] >= min_faces, {
        "components_before": int(n), "components_after": int((sizes >= min_faces).sum()),
        "largest_components_faces": sorted(sizes.tolist(), reverse=True)[:20],
    }


def write_mesh(src_ply, faces, path):
    used = np.unique(faces)
    remap = np.full(len(src_ply["vertex"].data), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices = np.array(src_ply["vertex"].data[used])
    outfaces = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    outfaces["vertex_indices"] = remap[faces].astype(np.int32)
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(outfaces, "face")],
            text=False, byte_order="<", comments=["Foreground-mask crop; original world coordinates"]).write(str(path))
    return used


def previews(xyz, vertices, selected, views, output):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(xyz), size=min(60000, len(xyz)), replace=False)
    p = xyz[indices]
    kept = selected[indices]
    chosen = np.linspace(0, len(views)-1, min(4, len(views))).astype(int)
    cell_w, cell_h = 440, 500
    canvas = Image.new("RGB", (cell_w*len(chosen), cell_h*2), "#252a32")
    draw = ImageDraw.Draw(canvas)
    has_rgb = all(k in vertices.dtype.names for k in ("red", "green", "blue"))
    rgb = np.column_stack([vertices[k][indices] for k in ("red", "green", "blue")]) if has_rgb else None
    for col, vid in enumerate(chosen):
        v = views[vid]
        u, y, inside = project(p, v)
        mh, mw = v.mask.shape
        scale = min((cell_w-20)/mw, (cell_h-50)/mh)
        sw, sh = max(1, int(mw*scale)), max(1, int(mh*scale))
        bg = np.zeros((mh, mw, 3), dtype=np.uint8) + 30
        bg[v.mask] = [90, 90, 90]
        for row in range(2):
            tile = Image.fromarray(bg).resize((sw, sh), Image.Resampling.NEAREST)
            td = ImageDraw.Draw(tile)
            ids = np.flatnonzero(inside & (kept if row else True))
            for j in ids:
                color = tuple(int(k) for k in rgb[j]) if row and rgb is not None else ((65, 230, 120) if kept[j] else (240, 90, 75))
                px, py = int((u[j]+0.5)*scale), int((y[j]+0.5)*scale)
                td.point((px, py), fill=color)
            ox, oy = col*cell_w+(cell_w-sw)//2, row*cell_h+35
            canvas.paste(tile, (ox, oy))
            label = ("Before: green=keep, red=remove" if row == 0 else "After: original vertex colors")
            draw.text((col*cell_w+10, row*cell_h+6), label, fill="white")
            draw.text((col*cell_w+10, (row+1)*cell_h-15), Path(v.name).name, fill="white")
    canvas.save(output)


def run(args):
    started = time.monotonic()
    args.mesh = args.mesh.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir == args.mesh.parent or args.mesh.is_relative_to(args.output_dir):
        raise ValueError("Use a separate output directory to protect the source mesh")
    for name in ("plant.ply", "plant_preview.jpg", "report.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Output already exists: {name}; use a new --output-dir")
    if not (0 < args.min_ratio <= 1) or args.min_views < 2:
        raise ValueError("--min-ratio must be in (0,1]; --min-views must be >= 2")
    if args.chunk_size < 1 or args.mask_size < 32 or args.dilate < 0 or args.min_component_faces < 1:
        raise ValueError("Invalid size/component/dilation parameters")
    if not 1 <= args.mask_threshold <= 255:
        raise ValueError("--mask-threshold must be in [1,255]")
    views, colmap_dir = read_colmap(args.sfm_dir)
    if len(views) < args.min_views:
        raise ValueError(f"Only {len(views)} poses, but --min-views={args.min_views}")
    if args.images_dir is not None:
        args.images_dir = args.images_dir.resolve()
    if args.masks_dir is not None:
        args.masks_dir = args.masks_dir.resolve()
    print(f"Loading {len(views)} matched views from {colmap_dir}", flush=True)
    masks = prepare_masks(views, args)
    ply, xyz, faces = load_mesh(args.mesh)
    print(f"Input: {len(xyz):,} vertices, {len(faces):,} triangles", flush=True)
    print("Projecting vertices into foreground masks...", flush=True)
    selected = score_points(xyz, views, args.min_views, args.min_ratio, args.chunk_size)
    # Require all three vertices, then the centroid. This also removes many long
    # triangles bridging a background surface to the foreground.
    face_keep = selected[faces].all(axis=1)
    candidates = np.flatnonzero(face_keep)
    print(f"Checking {len(candidates):,} candidate triangle centers...", flush=True)
    for start in range(0, len(candidates), args.chunk_size):
        ids = candidates[start:start+args.chunk_size]
        centers = xyz[faces[ids]].mean(axis=1)
        face_keep[ids] &= score_points(centers, views, args.min_views, args.min_ratio, args.chunk_size)
    filtered = faces[face_keep]
    if not len(filtered):
        raise RuntimeError("No triangles survived. Check mask/pose/mesh correspondence; "
                           "use the original mesh in the matching SfM coordinate system.")
    component_keep, components = component_filter(filtered, args.min_component_faces)
    filtered = filtered[component_keep]
    if not len(filtered):
        raise RuntimeError("Component filter removed everything; try --min-component-faces 1")
    output = args.output_dir / "plant.ply"
    used = write_mesh(ply, filtered, output)
    final_selected = np.zeros(len(xyz), dtype=bool)
    final_selected[used] = True
    previews(xyz, ply["vertex"].data, final_selected, views, args.output_dir / "plant_preview.jpg")
    warnings = []
    if len(filtered) / len(faces) < 0.001:
        warnings.append("Fewer than 0.1% of input faces survived: inspect pose/mask alignment and preview")
    if len(filtered) / len(faces) > 0.95:
        warnings.append("More than 95% survived: input may already be isolated, or masks may include environment")
    if len(views) < 6:
        warnings.append("Few views: unrelated surfaces inside the silhouettes may survive")
    report = {
        "input_mesh": str(args.mesh), "sfm_dir": str(colmap_dir.resolve()),
        "masks_dir": str(masks.resolve()), "output_mesh": str(output),
        "views": len(views), "view_names": [v.name for v in views],
        "input_vertices": len(xyz), "input_faces": len(faces),
        "output_vertices": len(used), "output_faces": len(filtered),
        "kept_face_ratio": len(filtered)/len(faces), "components": components,
        "parameters": {k: getattr(args, k) for k in ["min_views", "min_ratio", "mask_threshold",
                                                     "mask_size", "dilate", "min_component_faces"]},
        "elapsed_seconds": round(time.monotonic()-started, 2), "warnings": warnings,
        "method": "Multi-view silhouette crop; all triangle vertices and centroid must pass. "
                  "Ratio uses views where point is in front and within image bounds. "
                  "No occlusion test, hole filling, geometry synthesis or coordinate transform. "
                  "Vertex properties/colors are preserved; face properties/textures are not exported.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for warning in warnings:
        print("WARNING:", warning, flush=True)
    print(f"PLANT_EXTRACTION_SUCCESS: {output}\n"
          f"Retained {len(used):,} vertices / {len(filtered):,} triangles\n"
          f"Time: {report['elapsed_seconds']:.1f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True, help="Original untransformed MAtCha mesh PLY")
    p.add_argument("--sfm-dir", type=Path, required=True, help="Matching mast3r_sfm directory")
    p.add_argument("--masks-dir", type=Path, help="White foreground / black background masks, or RGBA cutouts")
    p.add_argument("--images-dir", type=Path, help="Original images, used to generate masks if --masks-dir is absent")
    p.add_argument("--output-dir", type=Path, required=True, help="New directory; outputs will not be overwritten")
    p.add_argument("--rembg-model", default="u2netp")
    p.add_argument("--min-views", type=int, default=3, help="At least this many foreground votes")
    p.add_argument("--min-ratio", type=float, default=0.7, help="Foreground votes / in-frame, positive-depth views")
    p.add_argument("--mask-size", type=int, default=1024)
    p.add_argument("--mask-threshold", type=int, default=100)
    p.add_argument("--dilate", type=int, default=3, help="Mask dilation radius after mask resizing, in pixels")
    p.add_argument("--min-component-faces", type=int, default=1,
                   help="1 preserves disconnected leaves; raise to e.g. 20 only to remove tiny components")
    p.add_argument("--chunk-size", type=int, default=200000)
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"PLANT_EXTRACTION_FAILED: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
