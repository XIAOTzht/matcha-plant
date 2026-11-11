"""Plan a continuous SO-101 watering path with risk-aware RRT*.

This is a geometry planner, not a learned model. Pot, soil and cart are hard
collision classes. Stems, flowers and leaves remain traversable soft-risk
classes: proximity/contact raises the path cost, but does not automatically
reject the path. The result is a simulation trajectory only; it is not a
command sequence for a real robot until the base registration and nozzle are
calibrated.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import csv
import json
from pathlib import Path

import igl
import mujoco
import numpy as np

from plant_scene import effective_policy, read_json, write_json
from robot_volume_query import AdaptiveVolumeQuery
from trajectory_tools import distance_intensity, score_exposure, time_parameterize
from soft_contact_field import SoftContactField
import time


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def allowed_grasp_contact(model, gid1, gid2, distance_m, gripper_rad, nozzle_config):
    """Allow only named handle/pad proxy contacts at the fixed grasp angle.

    The fingertip spheres protrude beyond the original visual finger surfaces.
    Their small overlap with the held handle is an intentional grasp, not a
    license for the spray tube or another robot link to intersect the robot.
    """
    grasp = nozzle_config.get("grasp_contact", {})
    if not nozzle_config.get("grasp_handle", False) or not grasp.get("enabled", False):
        return False
    names = {model.geom(int(gid1)).name, model.geom(int(gid2)).name}
    pads = set(grasp.get("pad_geoms", []))
    handles = set(grasp.get("handle_geoms", ["nozzle_grasp_handle"]))
    if len(names) != 2 or len(names & handles) != 1 or len(names & pads) != 1:
        return False
    target = float(nozzle_config["gripper_rad"])
    return bool(abs(float(gripper_rad) - target) <= float(grasp.get("angle_tolerance_rad", .002))
                and distance_m >= -float(grasp.get("max_proxy_overlap_mm", .6)) / 1000)


class CollisionChecker:
    def __init__(self, root, scene, margin_m=0.0005):
        self.root = root
        self.scene = scene
        self.scene_dir = scene.parent
        g = np.load(self.scene_dir / "full_resolution_world_geometry.npz", allow_pickle=False)
        self.xyz = np.asarray(g["xyz_m"], dtype=np.float64)
        self.faces = np.asarray(g["faces"], dtype=np.int64)
        self.config = read_json(self.scene_dir / "config_snapshot.json")
        source = root / self.config["source_root"]
        weights = np.load(source / self.config["weights"], allow_pickle=False)
        expected_fp = read_json(self.scene_dir / "scene_report.json")["mesh_fingerprint"]
        if (str(g["mesh_fingerprint"]) != expected_fp or str(weights["mesh_fingerprint"]) != expected_fp
                or len(weights["vertex_labels"]) != len(self.xyz)
                or not np.array_equal(weights["vertex_labels"], g["vertex_labels"])
                or len(weights["face_labels"]) != len(self.faces)):
            raise ValueError("Scene geometry and contact weights do not match")
        self.vertex_weight = np.asarray(weights["vertex_effective_weight"], dtype=float)
        self.face_instance = np.asarray(weights["face_instance_id"], dtype=np.int32)
        self.policy = effective_policy(root, self.config)
        self.face_labels = np.asarray(weights["face_labels"], dtype=np.int32)
        self.face_hard = np.asarray(weights["face_hard_collision"], dtype=bool)
        self.face_weight = np.asarray(weights["face_effective_weight"], dtype=float)
        self.aabb = igl.AABB(); self.aabb.init(self.xyz, self.faces)
        ids = np.flatnonzero(self.face_hard)
        if not len(ids):
            raise ValueError("No hard plant faces")
        self.hard_faces = self.faces[ids]
        self.hard_aabb = igl.AABB(); self.hard_aabb.init(self.xyz, self.hard_faces)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.geom_ids = []
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name == "floor" or name.startswith("plant_class_"):
                continue
            if int(self.model.geom_contype[i]) or int(self.model.geom_conaffinity[i]):
                self.geom_ids.append(i)
        self.active_geom_ids = set(self.geom_ids)
        # The official fixed base has visual meshes only. Include them in plant
        # distance checks, while allowing its intentional mounting-plane contact.
        base_id = int(self.model.body("base").id)
        self.geom_ids += [i for i in range(self.model.ngeom)
                          if self.model.geom_bodyid[i] == base_id and i not in self.geom_ids]
        self.geom_ids = np.asarray(self.geom_ids, dtype=np.int32)
        self.moving_mask = self.model.geom_bodyid[self.geom_ids] != base_id
        self.radii = self.model.geom_rbound[self.geom_ids].astype(float)
        refinement = self.config.get("hard_collision_refinement", {})
        self.volume_query = AdaptiveVolumeQuery(self.model, self.geom_ids, refinement) if refinement.get("enabled") else None
        self.plant_volume_query = AdaptiveVolumeQuery(self.model, self.geom_ids, refinement) if refinement.get("enabled") else None
        self.plant_collision_enabled = bool(refinement.get("plant_surface_clearance_enabled", True))
        self.plant_margin_m = float(refinement.get("plant_margin_mm", margin_m * 1000)) / 1000
        nozzle_body = int(self.model.body("placeholder_nozzle").id)
        self.nozzle_indices = np.asarray([i for i, gid in enumerate(self.geom_ids)
                                          if self.model.geom_bodyid[gid] == nozzle_body], dtype=np.int32)
        if not len(self.nozzle_indices):
            raise ValueError("No nozzle collision geometry found")
        # Keep a primary index for backwards-compatible report fields.  For a
        # stepped nozzle the first axial segment is the conservative primary.
        self.nozzle_i = int(self.nozzle_indices[0])
        self.joint_ids = np.asarray([self.model.joint(n).id for n in ARM_JOINTS], dtype=np.int32)
        self.q_lo = np.asarray([self.model.jnt_range[j, 0] for j in self.joint_ids], dtype=float)
        self.q_hi = np.asarray([self.model.jnt_range[j, 1] for j in self.joint_ids], dtype=float)
        self.margin_m = float(margin_m)
        self.workspace = self.config["workspace"]
        self.geom_names = [self.model.geom(int(i)).name or
                           f"{self.model.body(int(self.model.geom_bodyid[i])).name}/geom_{i}"
                           for i in self.geom_ids]
        self.soft_visual_ids=[i for i in range(self.model.ngeom)
            if self.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH
            and not (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
            and not self.model.geom(i).name.startswith('plant_class_')
            and self.model.geom_bodyid[i]!=nozzle_body]
        self.contact_trees = None
        self.gripper_rad = float(self.config["nozzle"].get("gripper_rad", .35))
        self._trace_cache = None
        self._check_cache = {}
        # Native collision generation skips welded/adjacent bodies. Explicitly
        # check both tool volumes against all native robot collision shapes.
        tool_body = nozzle_body
        tools = [int(g) for g in self.geom_ids if self.model.geom_bodyid[g] == tool_body]
        self.tube_pairs = [(tool, int(g)) for tool in tools for g in self.geom_ids
                           if self.model.geom_bodyid[g] != tool_body]


    def contact_trace(self):
        return self._soft_state()[0]

    def _soft_state(self):
        if self.contact_trees is None:
            print("Building source-triangle soft contact field", flush=True)
            self.contact_trees = SoftContactField(
                self.xyz, self.faces, self.face_labels, self.face_hard,
                self.face_instance, self.vertex_weight, self.model, self.geom_ids,
                cell_mm=self.config.get("soft_contact_cell_mm", 1.0),surface_geom_ids=self.soft_visual_ids)
            print(self.contact_trees.report(), flush=True)
        distance = self.policy["distance_risk"]
        if getattr(self, "_trace_cache", None) is None:
            self._trace_cache = self.contact_trees.trace(
                self.data, distance["clearance_start_mm"], distance.get("power", 2.))
        return self._trace_cache

    def volume_workspace_ok(self):
        """Exact world AABB of each native moving collision primitive/mesh."""
        lo, hi = np.asarray(self.workspace["min_m"]), np.asarray(self.workspace["max_m"])
        violations = []
        for i in self.geom_ids[self.moving_mask]:
            rotation = self.data.geom_xmat[i].reshape(3, 3)
            center = self.data.geom_xpos[i]
            size = self.model.geom_size[i]
            kind = int(self.model.geom_type[i])
            if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                extent = np.full(3, size[0])
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                extent = np.abs(rotation) @ size
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                extent = np.abs(rotation[:, 2]) * size[1] + size[0]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                extent = np.abs(rotation[:, 2]) * size[1] + size[0] * np.sqrt(
                    np.maximum(0, 1 - rotation[:, 2] ** 2))
            elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(self.model.geom_dataid[i]); start = self.model.mesh_vertadr[mesh]
                vertices = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]]
                world = vertices @ rotation.T + center
                if np.any(world.min(axis=0) < lo - 1e-8) or np.any(world.max(axis=0) > hi + 1e-8):
                    violations.append(int(i))
                continue
            else:
                raise ValueError(f"Unsupported robot collision geom: {kind}")
            if np.any(center - extent < lo - 1e-8) or np.any(center + extent > hi + 1e-8):
                violations.append(int(i))
        return not violations, violations

    def set_q(self, q, gripper_rad=None):
        gripper_rad = self.gripper_rad if gripper_rad is None else gripper_rad
        self._trace_cache = None
        self.data.qpos[:] = 0
        for j, value in zip(self.joint_ids, q):
            self.data.qpos[self.model.jnt_qposadr[j]] = value
        self.data.qpos[self.model.joint("gripper").qposadr] = gripper_rad
        mujoco.mj_forward(self.model, self.data)

    def check(self, q, details=False, gripper_rad=None, evaluate_risk=True):
        gripper_rad = self.gripper_rad if gripper_rad is None else gripper_rad
        q = np.asarray(q, dtype=float)
        cache_key = (q.tobytes(), float(gripper_rad), bool(evaluate_risk))
        if not details and cache_key in self._check_cache:
            return self._check_cache[cache_key]
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("Expected five finite arm joint angles")
        joint_ok = bool(np.all(q >= self.q_lo - 1e-8) and np.all(q <= self.q_hi + 1e-8))
        if not np.isfinite(gripper_rad):
            raise ValueError("Gripper angle must be finite")
        gripper_range = self.model.joint("gripper").range
        joint_ok = joint_ok and gripper_range[0] - 1e-8 <= gripper_rad <= gripper_range[1] + 1e-8
        self.set_q(q, gripper_rad)
        centers = self.data.geom_xpos[self.geom_ids].copy()
        sq, tri, _ = self.aabb.squared_distance(self.xyz, self.faces, centers)
        sqh, trih, _ = self.hard_aabb.squared_distance(self.xyz, self.hard_faces, centers)
        dist = np.sqrt(np.maximum(sq, 0.0))
        hard_dist = np.sqrt(np.maximum(sqh, 0.0))
        hard_clear = hard_dist - self.radii
        if self.volume_query is not None:
            def triangle_distance(points):
                sq, _, _ = self.hard_aabb.squared_distance(self.xyz, self.hard_faces, points)
                return np.sqrt(np.maximum(sq, 0))
            for index in np.flatnonzero(hard_clear <= self.margin_m):
                hard_clear[index] = self.volume_query.clearance(
                    self.data, int(self.geom_ids[index]), triangle_distance, self.margin_m)
        hard_ok = bool(np.all(hard_clear > self.margin_m))
        soft_clear = dist - self.radii
        plant_clear = soft_clear.copy()
        if self.plant_collision_enabled and self.plant_volume_query is not None:
            def full_triangle_distance(points):
                sq, _, _ = self.aabb.squared_distance(self.xyz, self.faces, points)
                return np.sqrt(np.maximum(sq, 0))
            for index in np.flatnonzero(plant_clear <= self.margin_m):
                plant_clear[index] = self.plant_volume_query.clearance(
                    self.data, int(self.geom_ids[index]), full_triangle_distance, self.plant_margin_m)
        # Soft plant classes are traversable. Their local effective weight is
        # multiplied by the configured convex distance-risk curve; only hard
        # classes are handled by the feasibility check above.
        # Separate per-organ channels avoid class masking and duplicate link
        # penalties. The triangle covers retain the latest spatial weights.
        risk = self._soft_state()[1] if joint_ok and hard_ok and evaluate_risk else 0.0
        plant_ok = bool(np.all(plant_clear > self.plant_margin_m)) if self.plant_collision_enabled else True
        workspace_ok, workspace_violations = self.volume_workspace_ok()
        self_contacts = []
        grasp_contacts = []
        floor_contacts = []
        ids = self.active_geom_ids
        floor = int(self.model.geom("floor").id)
        for c in self.data.contact[:self.data.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if a in ids and b in ids:
                if allowed_grasp_contact(self.model, a, b, float(c.dist), gripper_rad, self.config["nozzle"]):
                    grasp_contacts.append((a, b, float(c.dist)))
                else:
                    self_contacts.append((a, b, float(c.dist)))
            elif ((a == floor and b in ids) or (b == floor and a in ids)) and c.dist <= 0:
                floor_contacts.append((a, b, float(c.dist)))
        for a, b in self.tube_pairs:
            distance = float(mujoco.mj_geomDistance(self.model, self.data, a, b, .002, None))
            if distance < -1e-7:
                if allowed_grasp_contact(self.model, a, b, distance, gripper_rad, self.config["nozzle"]):
                    grasp_contacts.append((a, b, distance))
                else:
                    self_contacts.append((a, b, distance))
        # Plant soft overlap is allowed and represented by risk. Hard classes,
        # robot self-collision, floor collision, limits and workspace violations
        # remain invalid.
        valid = joint_ok and hard_ok and workspace_ok and not self_contacts and not floor_contacts
        if not details:
            if len(self._check_cache) > 200000:
                self._check_cache.clear()
            self._check_cache[cache_key] = (valid, risk)
            return valid, risk
        return valid, {"hard_clearance_min_m": float(hard_clear.min()),
                       "nozzle_hard_clearance_m": float(np.min(hard_clear[self.nozzle_indices])),
                       "plant_clearance_min_m": float(plant_clear.min()),
                       "nozzle_plant_clearance_m": float(np.min(plant_clear[self.nozzle_indices])),
                       "soft_clearance_min_m": float(soft_clear.min()),
                       "risk_proxy": risk, "soft_risk_proxy": risk, "workspace_ok": workspace_ok,
                       "joint_limits_ok": bool(joint_ok), "floor_collision": bool(floor_contacts),
                       "plant_surface_clearance_ok": bool(plant_ok),
                       "soft_overlap_allowed": True,
                       "workspace_violation_geoms": workspace_violations,
                       "hard_clearance_by_geom_m": hard_clear.tolist(),
                       "self_collision": bool(self_contacts),
                       "intentional_grasp_contacts": [{"geom1": self.model.geom(a).name, "geom2": self.model.geom(b).name,
                                                       "proxy_overlap_mm": -1000*d} for a, b, d in grasp_contacts],
                       "self_contacts": [{"geom1": self.model.geom(a).name, "geom2": self.model.geom(b).name,
                                           "distance_m": d} for a, b, d in self_contacts]}

    def edge_cost(self, qa, qb, max_step_rad=0.025, risk_weight=1.0):
        """Return (feasible, spatial_cost, risk_integral_proxy, length).

        The edge is sampled in joint space. Soft plant classes are allowed,
        but their local effective weights are integrated into the cost. Hard
        classes, robot self-collision, floor and limits still reject the edge.
        This is the planner's spatial proxy; timed replay adds persistence.
        """
        delta = np.asarray(qb) - np.asarray(qa)
        # Keep hard feasibility sampling independent from the softer cost grid.
        # Thin gripper/tool clearances must not be missed by coarse cost samples.
        hard_count = max(1, int(np.ceil(np.max(np.abs(delta)) / .006)))
        length = float(np.linalg.norm(delta))
        for u in np.linspace(0, 1, hard_count + 1):
            if not self.check(np.asarray(qa)+u*delta, evaluate_risk=False)[0]:
                return False, float("inf"), float("inf"), length
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step_rad)))
        risks = []
        for u in np.linspace(0, 1, count + 1):
            ok, risk = self.check(np.asarray(qa)+u*delta)
            if not ok:
                return False, float("inf"), float("inf"), length
            risks.append(float(risk))
        risk_integral = float(np.trapezoid(risks, dx=1/count) * length)
        return True, length + float(risk_weight) * risk_integral, risk_integral, length

    def edge(self, qa, qb, max_step_rad=0.025):
        ok, _cost, risk, _length = self.edge_cost(qa, qb, max_step_rad=max_step_rad)
        return ok, (risk / max(float(np.linalg.norm(np.asarray(qb) - np.asarray(qa))), 1e-12)
                     if ok else float("inf"))


def nearest(nodes, q, scale):
    a = np.asarray(nodes)
    return int(np.argmin(np.sum(((a - q) / scale) ** 2, axis=1)))


def steer(q, target, step, lo, hi):
    d = target - q
    n = np.linalg.norm(d)
    if n <= step:
        return np.clip(target, lo, hi), True
    return np.clip(q + d / n * step, lo, hi), False


def _path_from_parent(nodes, parents, idx):
    path = []
    while idx >= 0:
        path.append(nodes[idx])
        idx = parents[idx]
    return np.asarray(path[::-1])


def _update_descendant_costs(parents, costs, parent_idx, delta):
    children = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)
    stack = list(children[parent_idx])
    while stack:
        child = stack.pop()
        costs[child] += delta
        stack.extend(children[child])


def rrt_connect(checker, q_start, q_goal, seed=20261023, iterations=1000):
    """Feasibility seed only; weight optimization follows in RRT*."""
    rng = np.random.default_rng(seed)
    trees = [([np.asarray(q_start).copy()], [-1]), ([np.asarray(q_goal).copy()], [-1])]
    swapped = False
    for it in range(iterations):
        nodes, parents = trees[0]; other, other_parents = trees[1]
        target = other[0] if rng.random() < .2 else rng.uniform(checker.q_lo, checker.q_hi)
        idx = nearest(nodes, target, checker.q_hi-checker.q_lo)
        q, _ = steer(nodes[idx], target, .22, checker.q_lo, checker.q_hi)
        ok, _ = checker.edge(nodes[idx], q, .04)
        if ok:
            nodes.append(q); parents.append(idx); own_idx = len(nodes)-1
            oi = nearest(other, q, checker.q_hi-checker.q_lo)
            while True:
                nxt, reached = steer(other[oi], q, .22, checker.q_lo, checker.q_hi)
                ok, _ = checker.edge(other[oi], nxt, .04)
                if not ok: break
                other.append(nxt); other_parents.append(oi); oi = len(other)-1
                if reached:
                    left = _path_from_parent(nodes, parents, own_idx)
                    right = _path_from_parent(other, other_parents, oi)[::-1]
                    path = np.vstack([left, right[1:]])
                    return (path[::-1] if swapped else path), dict(iterations=it+1)
        trees.reverse(); swapped = not swapped
    return None, dict(iterations=iterations)


def path_cost(checker, path, risk_weight=1., step=.04):
    costs = [checker.edge_cost(a, b, max_step_rad=step, risk_weight=risk_weight)
             for a, b in zip(path[:-1], path[1:])]
    if any(not row[0] for row in costs): return float("inf")
    return float(sum(row[1] for row in costs))


def rrt_star(checker, q_start, q_goal, seed=20261023, iterations=600,
             risk_weight=1., goal_bias=.12, initial_paths=(), max_seconds=240.):
    """Anytime five-joint RRT* with weighted local edge integration.

    Finite nearest-neighbor and run budgets make this a practical optimizer;
    no finite-run or global-optimality certificate is asserted.
    """
    rng = np.random.default_rng(seed)
    lo, hi = checker.q_lo, checker.q_hi
    scale = hi-lo; step = .28; edge_step = .045
    nodes = [np.asarray(q_start).copy()]; parents = [-1]; costs = [0.]
    goal_links = {}; improvements = []; candidates = []
    started = time.perf_counter()
    for path in initial_paths:
        parent = 0
        for q in path[1:]:
            ok, c, _, _ = checker.edge_cost(nodes[parent], q, max_step_rad=edge_step, risk_weight=risk_weight)
            if not ok: break
            nodes.append(np.asarray(q).copy()); parents.append(parent); costs.append(costs[parent]+c)
            parent = len(nodes)-1
        if np.linalg.norm(nodes[parent]-q_goal) < 1e-8:
            goal_links[parent] = 0.
    best_cost = min((costs[i]+c for i,c in goal_links.items()), default=float("inf"))
    completed = 0
    for it in range(iterations):
        if time.perf_counter()-started >= max_seconds: break
        completed = it+1
        draw = rng.random()
        if draw < goal_bias:
            target = q_goal
        elif goal_links and draw < .75:
            parent = min(goal_links, key=lambda i: costs[i]+goal_links[i])
            incumbent = _path_from_parent(nodes, parents, parent)
            if np.linalg.norm(incumbent[-1]-q_goal)>1e-8: incumbent=np.vstack([incumbent,q_goal])
            k = int(rng.integers(len(incumbent)-1)); u = rng.random()
            target = np.clip((1-u)*incumbent[k]+u*incumbent[k+1] + rng.normal(0,.3,5),lo,hi)
        else:
            target = rng.uniform(lo,hi)
        idx = nearest(nodes,target,scale)
        q_new,_ = steer(nodes[idx],target,step,lo,hi)
        if np.linalg.norm(q_new-nodes[idx])<1e-8: continue
        distances = np.linalg.norm((np.asarray(nodes)-q_new)/scale,axis=1)
        radius = min(.3, .65*(np.log(len(nodes)+2)/(len(nodes)+1))**.2)
        neighbors = np.flatnonzero(distances<=radius)
        neighbors = np.unique(np.r_[idx,neighbors[np.argsort(distances[neighbors])[:5]]])
        best = None; edge_results={}
        for parent in sorted(neighbors,key=lambda i:costs[i]+np.linalg.norm(nodes[i]-q_new)):
            parent=int(parent)
            if best and costs[parent]+np.linalg.norm(nodes[parent]-q_new)>=best[0]:continue
            ok,c,_,_=checker.edge_cost(nodes[parent],q_new,max_step_rad=edge_step,risk_weight=risk_weight)
            edge_results[parent]=(ok,c)
            if ok and (best is None or costs[parent]+c<best[0]):best=(costs[parent]+c,parent)
        if best is None:continue
        nodes.append(q_new);parents.append(best[1]);costs.append(best[0]);ni=len(nodes)-1
        ancestors=set();a=ni
        while a>=0:ancestors.add(a);a=parents[a]
        for other in neighbors:
            other=int(other)
            if other in ancestors or costs[ni]+np.linalg.norm(q_new-nodes[other])>=costs[other]:continue
            ok,c=edge_results.get(other,(None,None))
            if ok is None:ok,c,_,_=checker.edge_cost(q_new,nodes[other],max_step_rad=edge_step,risk_weight=risk_weight)
            if ok and costs[ni]+c+1e-9<costs[other]:
                delta=costs[ni]+c-costs[other];parents[other]=ni;costs[other]+=delta
                _update_descendant_costs(parents,costs,other,delta)
        if np.linalg.norm(q_new-q_goal)<.55 or it%15==0:
            ok,c,_,_=checker.edge_cost(q_new,q_goal,max_step_rad=edge_step,risk_weight=risk_weight)
            if ok:goal_links[ni]=c
        if goal_links:
            gi=min(goal_links,key=lambda i:costs[i]+goal_links[i]);value=costs[gi]+goal_links[gi]
            if value<best_cost-1e-7:
                best_cost=value
                path=_path_from_parent(nodes,parents,gi)
                if np.linalg.norm(path[-1]-q_goal)>1e-8:path=np.vstack([path,q_goal])
                candidates.append(path);improvements.append(dict(iteration=it,cost=value,elapsed_s=time.perf_counter()-started))
                print(f"RRTSTAR improvement iter={it} cost={value:.4f} nodes={len(nodes)}",flush=True)
        if it%20==0:print(f"RRTSTAR iter={it} nodes={len(nodes)} best={best_cost:.4f}",flush=True)
    if not goal_links:return None,dict(iterations=completed,nodes=len(nodes)),[]
    gi=min(goal_links,key=lambda i:costs[i]+goal_links[i]);path=_path_from_parent(nodes,parents,gi)
    if np.linalg.norm(path[-1]-q_goal)>1e-8:path=np.vstack([path,q_goal])
    return path,dict(iterations=completed,nodes=len(nodes),best_cost=costs[gi]+goal_links[gi],
                     elapsed_s=time.perf_counter()-started,improvements=improvements),candidates


def shortcut(checker, path, attempts=40, seed=20261023, risk_weight=1.):
    rng=np.random.default_rng(seed);result=np.asarray(path).copy()
    for _ in range(attempts):
        if len(result)<3:break
        i,j=sorted(rng.integers(0,len(result),2))
        if j<=i+1:continue
        old=path_cost(checker,result[i:j+1],risk_weight)
        new=path_cost(checker,result[[i,j]],risk_weight)
        if new<=old+1e-9:result=np.vstack([result[:i+1],result[j:]])
    return result


def timed_path_score(checker, path, config, dt=.04, timing_plan=None):
    trajectory=time_parameterize(path,config,dt,timing_plan=timing_plan)
    # Smoothing can leave the original straight joint-space edges. Check the
    # actual interpolated poses before treating their risk score as usable.
    if not all(checker.check(q, evaluate_risk=False)[0] for q in trajectory['q_ref']):
        return dict(persistence_cost=float('inf'), exposure_integral=float('inf'),
                    duration_s=float(trajectory['time_s'][-1]), objective=float('inf'))
    rows=[]
    for q in trajectory["q_ref"]:
        checker.set_q(q);rows.append(checker.contact_trace())
    classes=[r[0] for r in rows[0]]
    clear=np.asarray([[r[2] for r in trace] for trace in rows])
    weights=np.asarray([[r[3] for r in trace] for trace in rows])
    instances=np.asarray([[r[4] for r in trace] for trace in rows])
    exposure=score_exposure(trajectory["time_s"],classes,clear,weights,instances,checker.policy)
    return dict(persistence_cost=exposure["persistence_cost"],exposure_integral=exposure["exposure_integral"],
                duration_s=float(trajectory["time_s"][-1]),
                objective=float(exposure["persistence_cost"]+.05*trajectory["time_s"][-1]))


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screening",type=Path);ap.add_argument("--goal-id",type=int)
    ap.add_argument("--output-dir",type=Path);ap.add_argument("--iterations",type=int,default=600)
    ap.add_argument("--seconds",type=float,default=240.);ap.add_argument("--risk-weight",type=float,default=1.)
    ap.add_argument("--seed",type=int,default=20261023)
    ap.add_argument("--resume-candidates",type=Path)
    ap.add_argument("--optimize-goals",type=int,default=3)
    ap.add_argument('--config', type=Path)
    args=ap.parse_args()
    selected=args.screening or root/read_json(root/"configs/latest_screening.json")["report"]
    report=read_json(selected/"goal_screening_report.json" if selected.is_dir() else selected)
    scene=(root/report.get("scene_relative",report["scene"])).resolve()
    out=(args.output_dir or root/"outputs"/("weighted_path_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("Choose a fresh output directory")
    out.mkdir(parents=True,exist_ok=True)
    if args.seconds<=0 or args.iterations<1 or args.risk_weight<=0 or args.optimize_goals<1:
        raise ValueError("Positive run budget and risk weight required")
    checker=CollisionChecker(root,scene)
    start=np.asarray(checker.config.get("planning", {}).get("start_q_arm_rad", [0.,0.,0.,0.,0.]), dtype=float)
    if not checker.check(start)[0]:raise ValueError("Starting pose violates hard constraints")
    accepted=[e for e in report["entries"] if e["accepted_screening"] and (args.goal_id is None or e["id"]==args.goal_id)]
    config_path=(args.config or root/'configs/weighted_trajectory_validation.json').resolve()
    config=read_json(config_path)
    candidates=[]
    if args.resume_candidates:
        previous_report=read_json(args.resume_candidates/"path_report.json")
        if previous_report["scene_relative"] != scene.relative_to(root).as_posix():
            raise ValueError("Previous candidates belong to another scene")
        if previous_report["mesh_fingerprint"] != read_json(scene.parent/"scene_report.json")["mesh_fingerprint"]:
            raise ValueError("Previous candidate geometry does not match")
        prior=read_json(args.resume_candidates/"candidate_scores.json")
        with np.load(args.resume_candidates/"candidate_paths.npz") as z:
            for i,info in enumerate(prior):
                goal=next((e for e in accepted if int(e["id"])==int(info["goal_id"])),None)
                if goal is None:continue
                candidate_path=z[f"q_{i}"].copy()
                if not np.allclose(candidate_path[0],start,atol=1e-8):
                    raise ValueError("Previous candidate start does not match current scene")
                if not any(np.allclose(candidate_path[-1],b['q_arm_rad'],atol=1e-8)
                           for b in goal['checked_branches'] if b['accepted_screening']):
                    raise ValueError("Previous candidate endpoint is absent from current screening")
                cost=path_cost(checker,candidate_path,args.risk_weight)
                if np.isfinite(cost):
                    candidates.append(dict(path=candidate_path,goal=goal,spatial_cost=cost,
                        source="previous_search_candidate"))
    # Test all accepted soil targets/IK branches, never silently pick the first.
    for entry in accepted:
        branches=[b for b in entry["checked_branches"] if b["accepted_screening"]]
        unique=[]
        for branch in branches:
            q=np.asarray(branch["q_arm_rad"])
            if any(np.linalg.norm(q-p)<.05 for p in unique):continue
            unique.append(q)
            path=np.vstack([start,q]);cost=path_cost(checker,path,args.risk_weight)
            if not np.isfinite(cost):
                if any(c['goal']['id']==entry['id'] and np.linalg.norm(c['path'][-1]-q)<1e-5 for c in candidates):
                    continue
                path,_=rrt_connect(checker,start,q,seed=args.seed,iterations=200)
                if path is None:continue
                cost=path_cost(checker,path,args.risk_weight)
            candidates.append(dict(path=path,goal=entry,spatial_cost=cost,source="feasible_seed"))
            print(f"SEED goal={entry['id']} spatial_cost={cost:.3f}",flush=True)
    if not candidates:
        write_json(out/"path_report.json",dict(status="NO_PATH_FOUND"));return
    candidates.sort(key=lambda x:x["spatial_cost"])
    seed_candidates=list(candidates)
    stats=[]
    # Optimize more than the first soil point. Endpoint exposure and approach
    # corridors can differ sharply even when neighboring soil is reachable.
    # Rank goals by a lower bound dominated by their unavoidable final hold.
    goal_seeds=[]
    for entry in accepted:
        group=[c for c in seed_candidates if c["goal"]["id"]==entry["id"]]
        if group:
            candidate=min(group,key=lambda c:c["spatial_cost"])
            endpoint=checker.check(candidate["path"][-1])[1]
            candidate["goal_priority"]=endpoint*1.5+.2*candidate["spatial_cost"]
            goal_seeds.append(candidate)
    goal_seeds.sort(key=lambda c:c["goal_priority"])
    for gi,initial in enumerate(goal_seeds[:max(1,args.optimize_goals)]):
        q_goal=initial["path"][-1];goal=initial["goal"]
        print(f"OPTIMIZE soil_goal={goal['id']}",flush=True)
        matching=[c["path"] for c in candidates if c["goal"]["id"]==goal["id"] and np.linalg.norm(c["path"][-1]-q_goal)<1e-5]
        best,goal_stats,improved=rrt_star(checker,start,q_goal,seed=args.seed+gi,iterations=args.iterations,
              risk_weight=args.risk_weight,max_seconds=args.seconds,initial_paths=matching)
        stats.append(dict(goal_id=int(goal["id"]),**goal_stats))
        if best is not None:
            best=shortcut(checker,best,attempts=35,risk_weight=args.risk_weight)
            candidates.append(dict(path=best,goal=goal,source="RRTstar_weighted_shortcut"))
            for path in improved[-1:]:candidates.append(dict(path=path,goal=goal,source="RRTstar_incumbent"))
    # Include actual duration and continuous-contact episodes in final selection.
    scored=[]
    rejected=[]
    for i,candidate in enumerate(candidates):
        if not all(checker.edge_cost(a,b,max_step_rad=.008)[0] for a,b in zip(candidate["path"][:-1],candidate["path"][1:])):
            rejected.append(dict(candidate=i,goal_id=int(candidate["goal"]["id"]),reason="dense hard recheck"));continue
        score=timed_path_score(checker,candidate["path"],config)
        if not np.isfinite(score['objective']):
            rejected.append(dict(candidate=i, reason='interpolated pose check'));continue
        candidate.update(score);scored.append(candidate)
        print(f"TIMED candidate={i} goal={candidate['goal']['id']} persistence={score['persistence_cost']:.3f} total={score['objective']:.3f}",flush=True)
    write_json(out/"rejected_candidates.json",rejected)
    if not scored:
        write_json(out/'path_report.json',dict(status='NO_PATH_FOUND',reason='No valid timed candidates'))
        raise RuntimeError('No valid timed candidates; see rejected_candidates.json')
    winner=min(scored,key=lambda x:x["objective"]);path=winner["path"];goal=winner["goal"]
    np.savez_compressed(out/"candidate_paths.npz", **{f"q_{i}":c["path"] for i,c in enumerate(scored)})
    write_json(out/"candidate_scores.json", [dict(goal_id=int(c["goal"]["id"]),source=c["source"],**{k:c[k] for k in ("persistence_cost","objective")}) for c in scored])
    # Dense recheck before exporting the usable path.
    for a,b in zip(path[:-1],path[1:]):
        if not checker.edge_cost(a,b,max_step_rad=.008)[0]:raise ValueError("Dense hard-constraint recheck failed")
    np.savez_compressed(out/"trajectory_q.npz",q_arm_rad=path,q_gripper_rad=np.full(len(path),checker.gripper_rad),goal_id=np.int32(goal["id"]))
    with (out/"trajectory.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow([*ARM_JOINTS,"gripper"])
        for q in path:w.writerow([*q,checker.gripper_rad])
    trajectory=time_parameterize(path,config,.02);tips=[]
    for q in trajectory["q_ref"]:
        checker.set_q(q);tips.append(checker.data.site_xpos[checker.model.site("nozzle_tip").id].copy())
    np.savetxt(out/"nozzle_path_line.csv",np.c_[trajectory["time_s"],tips],delimiter=",",header="time_s,x_m,y_m,z_m",comments="")
    np.savez_compressed(out/"nozzle_path_line.npz",time_s=trajectory["time_s"],tip_m=np.asarray(tips),q_arm_rad=trajectory["q_ref"])
    report_out=dict(status="PATH_PLANNING_COMPLETE",planner="five-joint RRT* + weighted shortcut + persistence-aware candidate ranking",
        risk_optimized=True,global_optimum_certified=False,scene_relative=scene.relative_to(root).as_posix(),
        timing_config_relative=config_path.relative_to(root).as_posix(),timing_config=config,
        mesh_fingerprint=read_json(scene.parent/"scene_report.json")["mesh_fingerprint"],goal_id=int(goal["id"]),
        goal_target_m=goal["target_m"],start_q_rad=start.tolist(),goal_q_rad=path[-1].tolist(),waypoint_count=len(path),
        waypoints_q_rad=path.tolist(),gripper_rad=checker.gripper_rad,stats=stats,selected_score={k:winner[k] for k in ("persistence_cost","exposure_integral","duration_s","objective","source")},
        candidate_scores=[dict(goal_id=int(c["goal"]["id"]),source=c["source"],**{k:c[k] for k in ("persistence_cost","exposure_integral","duration_s","objective")}) for c in scored],
        objectives=dict(tree="joint length + risk_weight * weighted exposure integrated in joint distance",risk_weight=args.risk_weight,
                        final="time-integrated contact weights with per-organ persistence + 0.05 * duration_s"),
        soft_contact=checker.contact_trees.report(),real_robot_ready=False,
        collision_model="pot/soil/cart hard; stem/flower/leaf weighted soft; native robot self/tool/floor constraints",
        limitations=["Finite stochastic search finds the lowest evaluated candidate, not a proven global minimum.",
                     "Time/joint sampling is not continuous swept-volume certification.",
                     "Plant triangle covers are conservative proxies; plant deformation and contact forces are not simulated."])
    write_json(out/"path_report.json",report_out)
    (out/"PATH_PLANNING_SUCCESS").write_text("weighted simulation path; dense dynamics validation follows\n")
    write_json(root/"configs/latest_path.json",dict(output=out.relative_to(root).as_posix(),report=(out/"path_report.json").relative_to(root).as_posix(),status=report_out["status"]))
    print(json.dumps(dict(status=report_out["status"],output=str(out),goal_id=int(goal["id"]),score=report_out["selected_score"])),flush=True)


if __name__=="__main__":main()
