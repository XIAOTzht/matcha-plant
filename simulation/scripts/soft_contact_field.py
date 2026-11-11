"""Small triangle covers for spatially weighted, per-organ contact exposure.

Every soft source triangle is contained by a sphere. Nearby triangle spheres
are grouped into millimetre cells without losing their covering radius or
maximum vertex weight. Distances are to native robot volumes, rather than one
large bounding sphere around an entire robot link. This remains a conservative
geometric proxy, not contact force, tissue damage, or deformable plant physics.
"""
from __future__ import annotations

import igl
import mujoco
import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from trajectory_tools import distance_intensity


class SoftContactField:
    def __init__(self, xyz, faces, labels, hard, instances, weights, model,
                 geom_ids, cell_mm=1.0, surface_geom_ids=()):
        ids = np.flatnonzero(~hard)
        triangles = xyz[faces[ids]]
        centers = triangles.mean(axis=1)
        radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
        maximum = weights[faces[ids]].max(axis=1)
        cell = float(cell_mm) / 1000
        if cell <= 0:
            raise ValueError("Soft contact cell size must be positive")
        keys = np.c_[labels[ids], instances[ids], np.floor(centers / cell).astype(np.int64)]
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        lo = np.full((len(unique), 3), np.inf); hi = -lo.copy()
        np.minimum.at(lo, inverse, centers); np.maximum.at(hi, inverse, centers)
        self.centers = (lo + hi) / 2
        self.radii = np.zeros(len(unique)); self.weights = np.zeros(len(unique))
        np.maximum.at(self.radii, inverse, radii + np.linalg.norm(centers - self.centers[inverse], axis=1))
        np.maximum.at(self.weights, inverse, maximum)
        self.face_ids = np.full(len(unique), -1, dtype=np.int64)
        self.face_ids[inverse] = ids
        self.pairs, self.channel = np.unique(unique[:, :2], axis=0, return_inverse=True)
        self.tree = cKDTree(self.centers)
        self.max_radius = float(self.radii.max())
        self.model = model
        self.surface_geom_ids = set(int(i) for i in surface_geom_ids)
        self.geom_ids = np.unique(np.r_[geom_ids,list(self.surface_geom_ids)]).astype(int)
        self.shapes = {}
        self.local_bounds = {}
        for gid in self.geom_ids:
            kind = int(model.geom_type[gid]); size = model.geom_size[gid].copy()
            hull_data = None
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[gid]); start = model.mesh_vertadr[mid]
                v = np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[mid]], dtype=float)
                if int(gid) in self.surface_geom_ids:
                    adr=int(model.mesh_faceadr[mid]);n=int(model.mesh_facenum[mid])
                    f=np.ascontiguousarray(model.mesh_face[adr:adr+n],dtype=np.int64)
                    planes=None
                else:
                    hull = ConvexHull(v)
                    f = np.ascontiguousarray(hull.simplices, dtype=np.int64)
                    planes=hull.equations
                tree = igl.AABB(); tree.init(v, f)
                hull_data = (v, f, tree, planes)
            if hull_data is not None:
                self.local_bounds[int(gid)] = (v.min(axis=0), v.max(axis=0))
            else:
                half = size.copy()
                if kind in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                    half = np.array([size[0], size[0], size[1]])
                    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE): half[2] += size[0]
                if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE): half[:] = size[0]
                self.local_bounds[int(gid)] = (-half, half)
            self.shapes[int(gid)] = (kind, size, hull_data)

    @staticmethod
    def volume_distance(points, shape):
        kind, size, hull = shape
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            return np.linalg.norm(np.maximum(np.abs(points) - size, 0), axis=1)
        if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            outside = np.c_[np.linalg.norm(points[:, :2], axis=1) - size[0],
                            np.abs(points[:, 2]) - size[1]]
            return np.linalg.norm(np.maximum(outside, 0), axis=1)
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            offset = points.copy(); offset[:, 2] -= np.clip(offset[:, 2], -size[1], size[1])
            return np.maximum(np.linalg.norm(offset, axis=1) - size[0], 0)
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return np.maximum(np.linalg.norm(points, axis=1) - size[0], 0)
        if hull is not None:
            v, f, tree, planes = hull
            sq, face, closest = tree.squared_distance(v, f, np.ascontiguousarray(points))
            if planes is None:
                # Original visual triangles catch finger/camera surfaces omitted
                # by native proxies without filling the open jaws with a hull.
                return np.sqrt(np.maximum(sq,0))
            # On a convex hull the closest facet's outward normal separates
            # exterior points. Interior points have nonpositive signed offset.
            signed = np.einsum("ni,ni->n", points - closest, planes[face, :3])
            return np.where(signed > 1e-10, np.sqrt(np.maximum(sq, 0)), 0.)
        raise ValueError(f"Unsupported soft query robot geometry: {kind}")

    def trace(self, data, margin_mm=5., power=2.):
        count = len(self.pairs)
        best = np.zeros(count); gap = np.full(count, 1.0)
        weight = np.zeros(count); face = np.full(count, -1, dtype=np.int64)
        responsible_geom = np.full(count, -1, dtype=np.int32)
        for gid in self.geom_ids:
            center = data.geom_xpos[gid]
            ids = np.asarray(self.tree.query_ball_point(
                center, self.model.geom_rbound[gid] + self.max_radius + margin_mm / 1000), dtype=int)
            if not len(ids):
                continue
            points = (self.centers[ids] - center) @ data.geom_xmat[gid].reshape(3, 3)
            lo, hi = self.local_bounds[int(gid)]
            outside = np.linalg.norm(np.maximum(np.maximum(lo-points, points-hi), 0), axis=1)
            keep = outside <= self.radii[ids] + margin_mm / 1000
            ids, points = ids[keep], points[keep]
            if not len(ids): continue
            clear = self.volume_distance(points, self.shapes[int(gid)]) - self.radii[ids]
            values = self.weights[ids] * distance_intensity(clear, margin_mm, power=power)
            for channel in np.unique(self.channel[ids][values > 0]):
                selected = np.flatnonzero(self.channel[ids] == channel)
                k = selected[np.argmax(values[selected])]
                if values[k] > best[channel]:
                    best[channel] = values[k]; gap[channel] = clear[k]
                    weight[channel] = self.weights[ids[k]]; face[channel] = self.face_ids[ids[k]]
                    responsible_geom[channel] = gid
        rows = []
        for cid in np.unique(self.pairs[:, 0]):
            mask = self.pairs[:, 0] == cid
            rows.append((int(cid), np.where(mask, face, -1), np.where(mask, gap, 1.),
                         np.where(mask, weight, 0.), np.where(mask, self.pairs[:, 1], -999)))
        self.last_responsible_geom = responsible_geom
        return rows, float(best.sum())

    def report(self):
        return dict(method="native robot volumes vs conservative millimetre source-triangle covers",
                    source_triangles_covered=True, cell_count=len(self.centers),
                    organ_channels=len(self.pairs), maximum_cover_radius_mm=1000*self.max_radius,
                    median_cover_radius_mm=float(1000*np.median(self.radii)),
                    aggregation="maximum local weighted proximity per class/organ, summed across organs",
                    original_robot_visual_surface_geoms=len(self.surface_geom_ids),
                    limitation="Conservative geometric exposure; no measured force or plant deformation")
