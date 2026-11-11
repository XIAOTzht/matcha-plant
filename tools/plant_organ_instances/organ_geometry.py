"""Reusable geometry-only organ proposals; never change semantic labels.

Projected silhouettes suggest markers. Assignment uses ORIGINAL mesh edges,
so adjacent disconnected leaves cannot be joined by a Euclidean shortcut.
These are proposals, not verified botanical instances.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components, dijkstra


def surface_areas(xyz, faces):
    area = .5 * np.linalg.norm(np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]],
                                        xyz[faces[:, 2]] - xyz[faces[:, 0]]), axis=1)
    return np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=len(xyz))


def mesh_edges(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)


def principal_projection(xyz, area):
    w = np.maximum(area, 1e-12)
    center = np.average(xyz, weights=w, axis=0)
    p = xyz - center
    eigen, vectors = np.linalg.eigh((p * w[:, None]).T @ p / w.sum())
    frame = vectors[:, ::-1]
    return p @ frame, center, frame, float(np.sqrt(max(eigen[0], 0) / max(eigen[1], 1e-12)))


def silhouette(projected, triangles, max_pixels=600):
    """Rasterize triangle support, rather than count vertices/density."""
    extent = np.ptp(projected[:, :2], axis=0)
    pixel = max(float(extent.max()) / max_pixels, 1e-6)
    uv = (projected[:, :2] - projected[:, :2].min(axis=0)) / pixel + 5
    dims = np.maximum(np.ceil(uv.max(axis=0)).astype(int) + 6, 12)
    im = Image.new("1", tuple(dims), 0)
    draw = ImageDraw.Draw(im)
    for tri in triangles:
        draw.polygon([tuple(p) for p in uv[tri]], fill=1)
    pix = np.rint(uv).astype(int)
    mask = np.array(im, dtype=bool)
    mask[pix[:, 1], pix[:, 0]] = True
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    distance = ndimage.distance_transform_edt(mask) * pixel
    return mask, distance, pix, pixel


def marker_candidates(distance, pixel, separation=1.0, min_peak_fraction=.24):
    smooth = ndimage.gaussian_filter(distance, sigma=1.2)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=5)) & (smooth >= smooth.max() * min_peak_fraction)
    labels, n = ndimage.label(peaks)
    candidates = []
    for k in range(1, n + 1):
        y, x = np.nonzero(labels == k)
        i = np.argmax(smooth[y, x])
        candidates.append((float(smooth[y[i], x[i]]), int(y[i]), int(x[i])))
    chosen = []
    for radius, y, x in sorted(candidates, reverse=True):
        if all(np.hypot(y - q[1], x - q[2]) * pixel >= separation * (radius + q[0]) for q in chosen):
            chosen.append((radius, y, x))
    if not chosen:
        y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
        chosen = [(float(smooth[y, x]), int(y), int(x))]
    return chosen


def smooth_center_field(projected, triangles, max_pixels=600):
    """Smooth distance from a filled organ contour, sampled continuously.

    Interior triangulation holes and wrinkles must not act as extra leaf rims.
    A modest scale-relative contour filter suppresses raster/jagged boundaries;
    a smoothstep remap preserves a bounded, gradual edge-to-interior increase.
    """
    mask,distance,pix,pixel=silhouette(projected,triangles,max_pixels)
    sigma=max(.8,float(distance.max()/pixel)*.035)
    contour=ndimage.gaussian_filter(mask.astype(float),sigma)>=.5
    contour=ndimage.binary_fill_holes(contour)
    dist=ndimage.distance_transform_edt(contour)*pixel
    dist=ndimage.gaussian_filter(dist,max(.8,float(dist.max()/pixel)*.025))
    uv=(projected[:,:2]-projected[:,:2].min(0))/pixel+5
    d=ndimage.map_coordinates(dist,[uv[:,1],uv[:,0]],order=1,mode='nearest')
    d=np.maximum(d-pixel,0)
    c=np.clip(d/max(float(d.max()),1e-9),0,1)
    c=c*c*(3-2*c)
    return c,mask,pixel


def split_organs(xyz, faces, labels, class_ids=(2, 3), min_area_mm2=10.,
                 marker_separation=1.0, max_pixels=600, explicit_seeds=None):
    """Return vertex IDs, center scores, areas and reviewable region records.

    0 = not a requested organ class; -1 = unresolved small fragment.
    explicit_seeds: list of dicts with vertex_index; optional class checked by CLI.
    """
    xyz = np.asarray(xyz, dtype=float)
    area = surface_areas(xyz, faces)
    edges = mesh_edges(faces)
    instance = np.zeros(len(xyz), np.int32)
    centrality = np.zeros(len(xyz), np.float32)
    organ_area = np.zeros(len(xyz), np.float32)
    review = np.zeros(len(xyz), bool)
    records = []
    next_id = 1
    for cid in class_ids:
        take = np.flatnonzero(labels == cid)
        if not len(take):
            continue
        instance[take] = -1
        review[take] = True
        inverse = np.full(len(xyz), -1, np.int32)
        inverse[take] = np.arange(len(take))
        keep = (labels[edges[:, 0]] == cid) & (labels[edges[:, 1]] == cid)
        e = inverse[edges[keep]]
        length = np.linalg.norm(xyz[take[e[:, 0]]] - xyz[take[e[:, 1]]], axis=1).clip(1e-9)
        graph = sparse.coo_matrix((np.r_[length, length], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                                  shape=(len(take), len(take))).tocsr()
        _, components = connected_components(graph, directed=False)
        component_area = np.bincount(components, weights=area[take])
        component_triangles = faces[np.all(labels[faces] == cid, axis=1)]
        tri_component = components[inverse[component_triangles[:, 0]]]
        for comp in np.flatnonzero(component_area >= min_area_mm2):
            local = np.flatnonzero(components == comp)
            members = take[local]
            if len(members) < 3:
                continue
            lookup = np.full(len(xyz), -1, np.int32)
            lookup[members] = np.arange(len(members))
            tris = lookup[component_triangles[tri_component == comp]]
            projected, _, _, nonplanarity = principal_projection(xyz[members], area[members])
            _, distance, pix, pixel = silhouette(projected, tris, max_pixels)
            candidates = marker_candidates(distance, pixel, marker_separation)
            supplied = [lookup[int(s['vertex_index'])] for s in (explicit_seeds or []) if lookup[int(s['vertex_index'])] >= 0]
            if supplied:
                seeds = np.unique(supplied)
                method = 'user_markers_surface_geodesic'
            else:
                # Pick an actual surface point nearest each silhouette maximum.
                seeds = np.unique([np.argmin((pix[:, 0] - x) ** 2 + (pix[:, 1] - y) ** 2) for _, y, x in candidates])
                method = 'silhouette_markers_surface_geodesic'
                # Sparse fragments and tiny silhouettes do not justify multiple
                # anatomical parts; avoid promoting raster speckles to organs.
                if component_area[comp] < 10 * min_area_mm2 or len(members) < 300:
                    seeds = np.array([np.argmax(distance[pix[:, 1], pix[:, 0]])])
            subgraph = graph[local][:, local]
            _, _, sources = dijkstra(subgraph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
            if not supplied and len(seeds) > 1:
                region_areas = np.array([area[members[sources == s]].sum() for s in seeds])
                kept = seeds[region_areas >= min_area_mm2]
                if not len(kept):
                    kept = seeds[[np.argmax(region_areas)]]
                if len(kept) != len(seeds):
                    seeds = kept
                    _, _, sources = dijkstra(subgraph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
            for seed in seeds:
                region = np.flatnonzero(sources == seed)
                if not len(region):
                    continue
                selected = members[region]
                # Recompute its own silhouette: centrality is per proposed organ,
                # not per whole plant or whole semantic class.
                region_map = np.full(len(members), -1, np.int32)
                region_map[region] = np.arange(len(region))
                rt = region_map[tris]
                rt = rt[np.all(rt >= 0, axis=1)]
                rp, center, frame, ratio = principal_projection(xyz[selected], area[selected])
                c,mask,rpx=smooth_center_field(rp,rt,max_pixels)
                ar = float(area[selected].sum())
                instance[selected] = next_id
                centrality[selected] = c
                organ_area[selected] = ar
                # Every automatic proposal needs biological review. Severe
                # overlap/nonplanarity is separately exposed in the manifest.
                review[selected] = True
                records.append(dict(instance_id=next_id, class_id=int(cid), vertex_count=len(selected),
                    reconstructed_surface_area_mm2=ar, projected_silhouette_area_mm2=float(mask.sum() * rpx ** 2),
                    center_xyz_mm=center.tolist(), plane_axes=frame.tolist(),
                    center_vertex_index=int(selected[np.argmax(c)]), seed_vertex_index=int(members[seed]),
                    component_id=int(comp), component_marker_count=len(seeds), method=method,
                    nonplanarity_ratio=ratio, parent_nonplanarity_ratio=nonplanarity,
                    overlap_review=bool(ratio > .30 or nonplanarity > .45),
                    reviewed=False, status='PROVISIONAL',
                    center_method='smoothed filled silhouette distance; continuous bilinear sampling; smoothstep normalized per organ'))
                next_id += 1
            print(f'Class {cid}, component {comp}: {len(members):,} vertices -> {len(seeds)} part proposals', flush=True)
    return dict(vertex_instance_id=instance, vertex_center_score=centrality,
                vertex_organ_area_mm2=organ_area, vertex_instance_review=review,
                vertex_surface_area_mm2=area.astype(np.float32)), records
