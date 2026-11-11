"""Report motion-style requirements separately from numerical validity.

Does not change candidate selection, risk budgets, endpoints, or the approved
coordinated-v2 algorithm. A valid fallback is still visibly a fallback.
"""
from __future__ import annotations

import numpy as np


def review_motion(report, elbow_min_deg=30., wrist_roll_min_deg=40.):
    motion = report.get('selected_motion', {})
    spans = np.asarray(motion.get('joint_span_deg', []), float)
    if spans.shape != (5,) or not np.isfinite(spans).all() or np.any(spans < 0):
        return dict(status='MOTION_STYLE_UNASSESSED', accepted=False,
                    reason='No finite five-joint motion-span report')
    source = report.get('selected_score', {}).get('source', 'unknown')
    checks = dict(elbow_span=float(spans[2]) >= elbow_min_deg,
                  wrist_roll_span=float(spans[4]) >= wrist_roll_min_deg,
                  coordination_not_fallback=source != 'unchanged_baseline')
    accepted = all(checks.values())
    return dict(status='MOTION_STYLE_TARGETS_MET' if accepted else 'MOTION_STYLE_NEEDS_REVIEW',
        accepted=accepted, checks=checks, selected_source=source,
        joint_span_deg=spans.tolist(), elbow_min_deg=elbow_min_deg,
        wrist_roll_min_deg=wrist_roll_min_deg,
        note='Style targets are separate from collision/dynamics validity; no extra contact is authorized')
