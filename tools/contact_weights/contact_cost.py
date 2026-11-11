#!/usr/bin/env python3
"""Score time-sampled nozzle/robot contact traces.

The collision engine should provide samples with:
time_s, class_id, clearance_mm, penetration_mm, contact_fraction, uncertain.
This is a planning cost, not a biological damage or force measurement.
"""
from __future__ import annotations
import argparse, csv, json, math
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ContactSample:
    time_s: float
    class_id: int
    clearance_mm: float = math.inf
    penetration_mm: float = 0.0
    contact_fraction: float = 0.0
    uncertain: bool = False
    local_effective_weight: float | None = None
    instance_id: int | None = None

def distance_factor(clearance_mm: float, penetration_mm: float, margin: float,
                    penetration_scale: float | None = None, power: float = 2.0) -> float:
    """Map proximity to a bounded nonlinear risk multiplier.

    Risk is zero at or beyond ``margin`` and exactly one at the surface. The
    curve is convex, so the increase accelerates as clearance approaches zero:

        clip((margin - clearance_mm) / margin, 0, 1) ** power

    Penetration is deliberately clamped to the same maximum value. The
    penetration arguments remain in the API for compatibility with old traces,
    but no extra penetration penalty is applied.
    """
    del penetration_mm, penetration_scale
    if not math.isfinite(clearance_mm) or not math.isfinite(margin) or not math.isfinite(power):
        return 0.0
    if margin <= 0 or power < 1:
        raise ValueError('distance-risk margin must be positive and power >= 1')
    x = min(1.0, max(0.0, (margin - clearance_mm) / margin))
    return x ** power

def instantaneous_cost(s: ContactSample, policy: dict) -> float:
    c = policy["classes"].get(str(s.class_id), policy["classes"]["0"])
    if c["hard_collision"] and s.contact_fraction > 0: return math.inf
    if c["weight"] is None: return 0.0
    if s.local_effective_weight is not None:
        w = float(s.local_effective_weight)
        if not math.isfinite(w) or w < 0:
            raise ValueError('local_effective_weight must be finite and nonnegative')
        # The queried mesh weight ALREADY includes uncertainty and geometry.
    else:
        spatial = policy.get('spatial_weighting', {})
        if spatial.get('require_local_weight') and s.class_id in spatial.get('class_ids', []):
            raise ValueError('Query the local mesh contact_weight; a class scalar loses organ center/area factors.')
        w = float(c["weight"])
        if s.uncertain: w *= float(c.get("uncertainty_multiplier", policy["uncertainty"]["default_multiplier"]))
    distance = policy["distance_risk"]
    return w * max(0.0, min(1.0, s.contact_fraction)) * distance_factor(
        s.clearance_mm, s.penetration_mm,
        float(distance["clearance_start_mm"]),
        float(distance.get("penetration_scale_mm", 2.0)),
        float(distance.get("power", 2.0)))

def score_trace(samples: list[ContactSample], policy: dict) -> dict:
    ss = sorted(samples, key=lambda s: s.time_s)
    if not ss: return {"total_cost": 0.0, "exposure_integral": 0.0, "episodes": []}
    vals = [instantaneous_cost(s, policy) for s in ss]
    if any(math.isinf(v) for v in vals): return {"total_cost": math.inf, "exposure_integral": math.inf, "hard_collision": True, "episodes": []}
    exposure = sum(0.5 * (a + b) * max(0.0, y.time_s - x.time_s) for x, y, a, b in zip(ss, ss[1:], vals, vals[1:]))
    gap = float(policy["continuous_contact"]["episode_gap_s"]); alpha = float(policy["continuous_contact"]["persistence_alpha"])
    tau = float(policy["continuous_contact"]["persistence_tau_s"]); cap = float(policy["continuous_contact"]["persistence_cap"])
    episodes = []; start = None
    for i, (s, cost) in enumerate(zip(ss, vals)):
        active = cost > 0
        new = active and (start is None or i == 0 or ss[i-1].class_id != s.class_id or ss[i-1].instance_id != s.instance_id or s.time_s - ss[i-1].time_s > gap)
        if new:
            if start is not None: episodes.append((start, i-1))
            start = i
        elif not active and start is not None:
            episodes.append((start, i-1)); start = None
    if start is not None: episodes.append((start, len(ss)-1))
    total = exposure; details = []
    for i0, i1 in episodes:
        duration = max(0.0, ss[i1].time_s - ss[i0].time_s)
        integral = sum(0.5*(vals[j]+vals[j+1])*max(0.0, ss[j+1].time_s-ss[j].time_s) for j in range(i0, i1))
        factor = 1.0 + alpha * min(duration / max(tau, 1e-9), cap)
        total += integral * (factor - 1.0)
        details.append({"class_id": ss[i0].class_id, "instance_id": ss[i0].instance_id, "start_s": ss[i0].time_s, "end_s": ss[i1].time_s, "duration_s": duration, "exposure_integral": integral, "persistence_factor": factor})
    return {"total_cost": total, "exposure_integral": exposure, "hard_collision": False, "episodes": details}

def read_trace(path: Path) -> list[ContactSample]:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return [ContactSample(float(r["time_s"]), int(r["class_id"]), float(r.get("clearance_mm", "inf")), float(r.get("penetration_mm", 0)), float(r.get("contact_fraction", 0)), str(r.get("uncertain", "false")).lower() in {"1", "true", "yes"},
        float(r['local_effective_weight']) if r.get('local_effective_weight') not in (None, '') else None,
        int(r['instance_id']) if r.get('instance_id') not in (None, '') else None) for r in rows]

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("trace", type=Path); p.add_argument("--policy", type=Path, required=True); a = p.parse_args()
    print(json.dumps(score_trace(read_trace(a.trace), json.loads(a.policy.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2, allow_nan=True))
