"""Certified lower distance bounds for native robot volumes versus triangles.

Each retained local box is enclosed by a sphere. Triangle distance is 1-Lipschitz,
so d(center)-radius bounds the distance of every point inside that box. Only
ambiguous boxes are subdivided. Unresolved boxes fail conservatively. Meshes use
their convex hull, matching MuJoCo's native mesh collision representation.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


class AdaptiveVolumeQuery:
    def __init__(self, model, geom_ids, settings):
        self.tolerance = float(settings.get("cell_tolerance_mm", 0.1)) / 1000
        self.max_cells = int(settings.get("max_cells", 16384))
        if self.tolerance <= 0 or self.max_cells < 2:
            raise ValueError("Invalid adaptive volume query limits")
        self.shapes = {}
        for gid in geom_ids:
            kind = int(model.geom_type[gid]); size = model.geom_size[gid].copy()
            planes = None
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(model.geom_dataid[gid]); start = model.mesh_vertadr[mesh]
                vertices = np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[mesh]], dtype=float)
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                planes = ConvexHull(vertices).equations
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                lo, hi = -size, size
            elif kind in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                half = np.array([size[0], size[0], size[1]])
                if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                    half[2] += size[0]
                lo, hi = -half, half
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                lo, hi = -np.full(3, size[0]), np.full(3, size[0])
            else:
                raise ValueError(f"Unsupported native collision volume: {kind}")
            self.shapes[int(gid)] = (kind, size, (lo + hi) / 2, (hi - lo) / 2, planes)

    @staticmethod
    def keep_cells(centers, halves, shape):
        kind, size, _, _, planes = shape
        if planes is not None:
            # A cell entirely outside any hull plane contains no collision mesh.
            return np.all(centers @ planes[:, :3].T + planes[:, 3]
                          - halves @ np.abs(planes[:, :3]).T <= 1e-12, axis=1)
        if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radial = np.maximum(abs(centers[:, :2]) - halves[:, :2], 0)
            return np.sum(radial ** 2, axis=1) <= size[0] ** 2 + 1e-18
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            distance = np.maximum(abs(centers) - halves - [0, 0, size[1]], 0)
            return np.sum(distance ** 2, axis=1) <= size[0] ** 2 + 1e-18
        return np.ones(len(centers), dtype=bool)

    def clearance(self, data, gid, triangle_distance, margin_m):
        shape = self.shapes[int(gid)]
        centers = np.array([shape[2]]); halves = np.array([shape[3]])
        rotation = data.geom_xmat[gid].reshape(3, 3)
        position = data.geom_xpos[gid]
        certified = float("inf"); visited = 0
        while len(centers):
            world = centers @ rotation.T + position
            lower = triangle_distance(world) - np.linalg.norm(halves, axis=1)
            visited += len(centers)
            safe = lower > margin_m
            if safe.any():
                certified = min(certified, float(lower[safe].min()))
            if safe.all():
                return certified
            centers, halves, lower = centers[~safe], halves[~safe], lower[~safe]
            if visited >= self.max_cells or np.any(np.linalg.norm(halves, axis=1) <= self.tolerance):
                return min(certified, float(lower.min()))
            # Bisect the longest local dimension; the two children cover parent.
            axis = np.argmax(halves, axis=1)
            halves[np.arange(len(halves)), axis] *= 0.5
            delta = np.zeros_like(centers)
            delta[np.arange(len(delta)), axis] = halves[np.arange(len(halves)), axis]
            centers = np.concatenate([centers - delta, centers + delta])
            halves = np.concatenate([halves, halves])
            keep = self.keep_cells(centers, halves, shape)
            centers, halves = centers[keep], halves[keep]
        return certified
