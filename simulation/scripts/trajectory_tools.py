"""Portable trajectory timing and sampled geometric exposure scoring."""
from __future__ import annotations

import numpy as np


def timed_quintic(waypoints, config, dt):
    q = np.asarray(waypoints, dtype=float)
    vmax = np.asarray(config["velocity_limit_rad_s"], dtype=float)
    amax = np.asarray(config["acceleration_limit_rad_s2"], dtype=float)
    factor = float(config["duration_safety_factor"])
    if (q.ndim != 2 or q.shape[1] != 5 or len(q) < 2 or not np.isfinite(q).all()
            or vmax.shape != (5,) or amax.shape != (5,)
            or np.any(vmax <= 0) or np.any(amax <= 0) or dt <= 0 or factor < 1):
        raise ValueError("Invalid waypoints, timestep, or simulation limits")
    times = [0.]; positions = [q[0].copy()]; velocities = [np.zeros(5)]; accelerations = [np.zeros(5)]
    phases = [0]; segments = []

    def hold(seconds, value, phase):
        if seconds < 0:
            raise ValueError("Hold durations cannot be negative")
        for _ in range(int(np.ceil(seconds / dt))):
            times.append(times[-1] + dt); positions.append(value.copy())
            velocities.append(np.zeros(5)); accelerations.append(np.zeros(5)); phases.append(phase)

    hold(float(config["initial_hold_s"]), q[0], 0)
    for index, (a, b) in enumerate(zip(q[:-1], q[1:])):
        delta = b - a
        # Exact extrema on [0,1]: max s'=15/8; max |s''|=10/sqrt(3).
        required = max(float(np.max(1.875 * abs(delta) / vmax)),
                       float(np.sqrt(np.max((10 / np.sqrt(3)) * abs(delta) / amax))),
                       float(config["minimum_segment_s"])) * factor
        count = max(1, int(np.ceil(required / dt))); duration = count * dt
        start = times[-1]
        for k in range(1, count + 1):
            u = k / count
            s = 10*u**3 - 15*u**4 + 6*u**5
            ds = (30*u**2 - 60*u**3 + 30*u**4) / duration
            dds = (60*u - 180*u**2 + 120*u**3) / duration**2
            times.append(start + k*dt); positions.append(a + delta*s)
            velocities.append(delta*ds); accelerations.append(delta*dds); phases.append(1)
        segments.append(dict(index=index, start_s=start, duration_s=duration,
                             analytic_peak_velocity_rad_s=(1.875*abs(delta)/duration).tolist(),
                             analytic_peak_acceleration_rad_s2=((10/np.sqrt(3))*abs(delta)/duration**2).tolist()))
    hold(float(config["target_hold_s"]), q[-1], 2)
    return dict(time_s=np.asarray(times), q_ref=np.asarray(positions), qd_ref=np.asarray(velocities),
                qdd_ref=np.asarray(accelerations), phase=np.asarray(phases, dtype=np.int8), segments=segments)


def _quintic_hermite_coefficients(a, b, va, vb, aa, ab, duration):
    """Return coefficients for a quintic Hermite segment in normalized time.

    The returned coefficients describe ``p(u)`` for ``u in [0, 1]``.  Endpoint
    position, velocity and acceleration are prescribed in physical seconds.
    Keeping this small helper independent of MuJoCo makes the timing profile
    testable and useful to both the validator and future real-robot exporters.
    """
    h = float(duration)
    c0 = a
    c1 = h * va
    c2 = 0.5 * h * h * aa
    c3 = (10.0 * (b - a) - 6.0 * h * va - 4.0 * h * vb
          - 1.5 * h * h * aa + 0.5 * h * h * ab)
    c4 = (-15.0 * (b - a) + 8.0 * h * va + 7.0 * h * vb
          + 1.5 * h * h * aa - h * h * ab)
    c5 = (6.0 * (b - a) - 3.0 * h * va - 3.0 * h * vb
          - 0.5 * h * h * aa + 0.5 * h * h * ab)
    return np.stack((c0, c1, c2, c3, c4, c5), axis=0)


def _evaluate_quintic(coefficients, duration, u):
    """Evaluate position, velocity and acceleration of normalized quintic."""
    u = np.asarray(u, dtype=float)
    powers = np.stack((np.ones_like(u), u, u**2, u**3, u**4, u**5), axis=0)
    d_powers = np.stack((np.zeros_like(u), np.ones_like(u), 2*u, 3*u**2,
                         4*u**3, 5*u**4), axis=0)
    dd_powers = np.stack((np.zeros_like(u), np.zeros_like(u), 2*np.ones_like(u),
                          6*u, 12*u**2, 20*u**3), axis=0)
    position = np.tensordot(coefficients.T, powers, axes=(1, 0)).T
    velocity = np.tensordot(coefficients.T, d_powers, axes=(1, 0)).T / duration
    acceleration = np.tensordot(coefficients.T, dd_powers, axes=(1, 0)).T / duration**2
    return position, velocity, acceleration


def timed_continuous_quintic(waypoints, config, dt):
    """Time-parameterize waypoints with shared non-zero knot velocities.

    ``timed_quintic`` deliberately stops at every waypoint.  This profile keeps
    the same endpoint and limit guarantees while using a quintic Hermite segment
    with a shared velocity and zero acceleration at every internal knot.  Thus
    position, velocity and acceleration are continuous across the route, while
    the first and last waypoint remain rest-to-rest.  The waypoint velocity is
    a tension-scaled centered finite difference; the duration is enlarged after
    analytical sampled extrema checks when needed.
    """
    q = np.asarray(waypoints, dtype=float)
    vmax = np.asarray(config["velocity_limit_rad_s"], dtype=float)
    amax = np.asarray(config["acceleration_limit_rad_s2"], dtype=float)
    factor = float(config["duration_safety_factor"])
    tension = float(config.get("waypoint_velocity_tension", 0.4))
    if (q.ndim != 2 or q.shape[1] != 5 or len(q) < 2 or not np.isfinite(q).all()
            or vmax.shape != (5,) or amax.shape != (5,)
            or np.any(vmax <= 0) or np.any(amax <= 0) or dt <= 0
            or factor < 1 or not 0 <= tension <= 1):
        raise ValueError("Invalid waypoints, timestep, simulation limits, or velocity tension")

    minimum = float(config["minimum_segment_s"])
    if minimum <= 0:
        raise ValueError("minimum_segment_s must be positive")
    # Start with the conservative rest-to-rest duration.  It is only a seed;
    # Hermite extrema below may lengthen individual segments further.
    durations = []
    for a, b in zip(q[:-1], q[1:]):
        delta = b - a
        required = max(float(np.max(1.875 * abs(delta) / vmax)),
                       float(np.sqrt(np.max((10 / np.sqrt(3)) * abs(delta) / amax))),
                       minimum) * factor
        durations.append(required)
    durations = np.asarray(durations, dtype=float)

    # Internal knot velocities are shared by adjacent segments.  The clipping
    # is intentionally below the hardware-like limit so a bend cannot create a
    # hidden velocity spike when the neighboring durations are later enlarged.
    velocity_cap = vmax * float(config.get("waypoint_velocity_cap", 0.65))
    velocities = np.zeros_like(q)
    for i in range(1, len(q) - 1):
        centered = (q[i + 1] - q[i - 1]) / (durations[i - 1] + durations[i])
        velocities[i] = np.clip(tension * centered, -velocity_cap, velocity_cap)
    accelerations = np.zeros_like(q)

    # Recompute segment duration until the sampled analytical extrema satisfy
    # both limits.  The knot velocities stay shared, preserving C1 continuity.
    u_check = np.linspace(0.0, 1.0, 401)
    for _ in range(12):
        changed = False
        for i in range(len(durations)):
            coeff = _quintic_hermite_coefficients(q[i], q[i + 1], velocities[i],
                                                   velocities[i + 1], accelerations[i],
                                                   accelerations[i + 1], durations[i])
            _, v, acc = _evaluate_quintic(coeff, durations[i], u_check)
            v_peak = np.max(np.abs(v), axis=0)
            a_peak = np.max(np.abs(acc), axis=0)
            scale = max(1.0, float(np.max(v_peak / vmax)),
                        float(np.sqrt(np.max(a_peak / amax)))) * 1.01
            if scale > 1.000001:
                durations[i] *= scale
                changed = True
        if not changed:
            break

    times = [0.0]; positions = [q[0].copy()]; velocities_out = [velocities[0].copy()]
    accelerations_out = [accelerations[0].copy()]; phases = [0]; segments = []

    def hold(seconds, value, phase):
        if seconds < 0:
            raise ValueError("Hold durations cannot be negative")
        for _ in range(int(np.ceil(seconds / dt))):
            times.append(times[-1] + dt); positions.append(value.copy())
            velocities_out.append(np.zeros(5)); accelerations_out.append(np.zeros(5)); phases.append(phase)

    hold(float(config["initial_hold_s"]), q[0], 0)
    for index, duration in enumerate(durations):
        count = max(1, int(np.ceil(duration / dt)))
        duration = count * dt
        # Re-evaluate with the quantized duration, then report the actual peaks.
        coeff = _quintic_hermite_coefficients(q[index], q[index + 1], velocities[index],
                                               velocities[index + 1], accelerations[index],
                                               accelerations[index + 1], duration)
        u = np.arange(1, count + 1, dtype=float) / count
        p, v, acc = _evaluate_quintic(coeff, duration, u)
        for k in range(count):
            times.append(times[-1] + dt); positions.append(p[k]); velocities_out.append(v[k])
            accelerations_out.append(acc[k]); phases.append(1)
        _, v_check, a_check = _evaluate_quintic(coeff, duration, u_check)
        segments.append(dict(index=index, start_s=float(times[-1] - duration), duration_s=float(duration),
                             start_velocity_rad_s=velocities[index].tolist(),
                             end_velocity_rad_s=velocities[index + 1].tolist(),
                             analytic_peak_velocity_rad_s=np.max(np.abs(v_check), axis=0).tolist(),
                             analytic_peak_acceleration_rad_s2=np.max(np.abs(a_check), axis=0).tolist()))
    hold(float(config["target_hold_s"]), q[-1], 2)
    return dict(time_s=np.asarray(times), q_ref=np.asarray(positions), qd_ref=np.asarray(velocities_out),
                qdd_ref=np.asarray(accelerations_out), phase=np.asarray(phases, dtype=np.int8), segments=segments,
                timing_profile="continuous_quintic")


def _septic_hermite_coefficients(a, b, va, vb, aa, ab, ja, jb, duration):
    """Return normalized-time coefficients for a seventh-order Hermite segment."""
    h = float(duration)
    c0 = a
    c1 = h * va
    c2 = 0.5 * h * h * aa
    c3 = (h ** 3) * ja / 6.0
    known_p = b - (c0 + c1 + c2 + c3)
    known_v = h * vb - (c1 + 2.0*c2 + 3.0*c3)
    known_a = h * h * ab - (2.0*c2 + 6.0*c3)
    known_j = h ** 3 * jb - (6.0*c3)
    matrix = np.array([[1., 1., 1., 1.],
                       [4., 5., 6., 7.],
                       [12., 20., 30., 42.],
                       [24., 60., 120., 210.]])
    tail = np.stack((known_p, known_v, known_a, known_j), axis=0)
    c4, c5, c6, c7 = np.linalg.solve(matrix, tail)
    return np.stack((c0, c1, c2, c3, c4, c5, c6, c7), axis=0)


def _evaluate_septic(coefficients, duration, u):
    """Evaluate position, velocity, acceleration and jerk of a septic segment."""
    u = np.asarray(u, dtype=float)
    powers = np.stack((np.ones_like(u), u, u**2, u**3, u**4, u**5, u**6, u**7), axis=0)
    d_powers = np.stack((np.zeros_like(u), np.ones_like(u), 2*u, 3*u**2,
                         4*u**3, 5*u**4, 6*u**5, 7*u**6), axis=0)
    dd_powers = np.stack((np.zeros_like(u), np.zeros_like(u), 2*np.ones_like(u),
                          6*u, 12*u**2, 20*u**3, 30*u**4, 42*u**5), axis=0)
    ddd_powers = np.stack((np.zeros_like(u), np.zeros_like(u), np.zeros_like(u),
                           6*np.ones_like(u), 24*u, 60*u**2, 120*u**3, 210*u**4), axis=0)
    position = np.tensordot(coefficients.T, powers, axes=(1, 0)).T
    velocity = np.tensordot(coefficients.T, d_powers, axes=(1, 0)).T / duration
    acceleration = np.tensordot(coefficients.T, dd_powers, axes=(1, 0)).T / duration**2
    jerk = np.tensordot(coefficients.T, ddd_powers, axes=(1, 0)).T / duration**3
    return position, velocity, acceleration, jerk


def _knot_velocities(q, durations, config, vmax):
    """Shared knot velocities; optionally suppress overshoot at joint reversals."""
    mode = config.get("waypoint_velocity_mode", "centered")
    tension = float(config.get("waypoint_velocity_tension", 0.4))
    cap = float(config.get("waypoint_velocity_cap", 0.65))
    if mode not in ("centered", "monotone_harmonic"):
        raise ValueError(f"Unknown waypoint_velocity_mode: {mode}")
    velocities = np.zeros_like(q)
    slopes = np.diff(q, axis=0) / durations[:, None]
    for i in range(1, len(q) - 1):
        if mode == "centered":
            value = (q[i + 1] - q[i - 1]) / (durations[i - 1] + durations[i])
        else:
            left, right = slopes[i - 1], slopes[i]
            same_direction = left * right > 0
            value = np.zeros(q.shape[1])
            # Weighted harmonic mean: a reversing/stationary joint reaches zero
            # speed at the knot, while every other joint continues moving.
            w1 = 2 * durations[i] + durations[i - 1]
            w2 = durations[i] + 2 * durations[i - 1]
            value[same_direction] = ((w1 + w2) /
                (w1 / left[same_direction] + w2 / right[same_direction]))
            bound = 1.5 * np.minimum(abs(left), abs(right))
            value = np.clip(value, -bound, bound)
        velocities[i] = np.clip(tension * value, -vmax * cap, vmax * cap)
    return velocities


def _polynomial_motion_peaks(coefficients, duration):
    """Exact extrema candidates from derivative roots on normalized [0, 1]."""
    peaks = []
    for order in (1, 2):
        result = []
        for joint in coefficients.T:
            derivative = np.polynomial.polynomial.polyder(joint, m=order)
            stationary = np.polynomial.polynomial.polyder(derivative)
            roots = np.polynomial.polynomial.polyroots(stationary)
            points = [0., 1.] + [float(r.real) for r in roots
                                      if abs(r.imag) < 1e-8 and 0 < r.real < 1]
            result.append(float(np.max(abs(np.polynomial.polynomial.polyval(points, derivative)))
                                / duration**order))
        peaks.append(np.asarray(result))
    return peaks


def timed_continuous_septic(waypoints, config, dt):
    """Time-parameterize a route with C3-continuous seventh-order segments.

    Internal waypoints share a non-zero velocity while acceleration and jerk are
    zero at every knot.  The result therefore avoids the small snap-like change
    that remains visible with piecewise quintics, without changing the spatial
    waypoints or allowing an endpoint to drift.  Durations are enlarged against
    sampled analytic velocity/acceleration extrema before samples are emitted.
    """
    q = np.asarray(waypoints, dtype=float)
    vmax = np.asarray(config["velocity_limit_rad_s"], dtype=float)
    amax = np.asarray(config["acceleration_limit_rad_s2"], dtype=float)
    factor = float(config["duration_safety_factor"])
    tension = float(config.get("waypoint_velocity_tension", 0.4))
    cap = float(config.get("waypoint_velocity_cap", 0.65))
    if (q.ndim != 2 or q.shape[1] != 5 or len(q) < 2 or not np.isfinite(q).all()
            or vmax.shape != (5,) or amax.shape != (5,) or np.any(vmax <= 0)
            or np.any(amax <= 0) or dt <= 0 or factor < 1 or not 0 <= tension <= 1
            or not 0 < cap <= 1):
        raise ValueError("Invalid waypoints, timestep, simulation limits, or velocity tension")
    minimum = float(config["minimum_segment_s"])
    durations = []
    for a, b in zip(q[:-1], q[1:]):
        delta = b - a
        required = max(float(np.max(1.875 * abs(delta) / vmax)),
                       float(np.sqrt(np.max((10 / np.sqrt(3)) * abs(delta) / amax))),
                       minimum) * factor
        durations.append(required)
    durations = np.asarray(durations, dtype=float)
    velocity_cap = vmax * cap
    velocities = _knot_velocities(q, durations, config, vmax)
    accelerations = np.zeros_like(q)
    jerks = np.zeros_like(q)
    u_check = np.linspace(0.0, 1.0, 601)
    for _ in range(12):
        changed = False
        for i in range(len(durations)):
            coeff = _septic_hermite_coefficients(q[i], q[i + 1], velocities[i], velocities[i + 1],
                                                  accelerations[i], accelerations[i + 1],
                                                  jerks[i], jerks[i + 1], durations[i])
            _, v, acc, _ = _evaluate_septic(coeff, durations[i], u_check)
            peak_v, peak_a = _polynomial_motion_peaks(coeff, durations[i])
            required_scale = max(float(np.max(peak_v / vmax)),
                                 float(np.sqrt(np.max(peak_a / amax))))
            # Legacy profiles retain their original duration for reproducibility.
            scale = (max(1., required_scale) * 1.01 if config.get("legacy_duration_padding", True)
                     else (required_scale * 1.01 if required_scale > 1. else 1.))
            if scale > 1.000001:
                durations[i] *= scale
                changed = True
        if not changed:
            break
    times = [0.0]; positions = [q[0].copy()]; velocities_out = [velocities[0].copy()]
    accelerations_out = [accelerations[0].copy()]; phases = [0]; segments = []

    def hold(seconds, value, phase):
        if seconds < 0:
            raise ValueError("Hold durations cannot be negative")
        for _ in range(int(np.ceil(seconds / dt))):
            times.append(times[-1] + dt); positions.append(value.copy())
            velocities_out.append(np.zeros(5)); accelerations_out.append(np.zeros(5)); phases.append(phase)

    hold(float(config["initial_hold_s"]), q[0], 0)
    for index, duration in enumerate(durations):
        count = max(1, int(np.ceil(duration / dt))); duration = count * dt
        coeff = _septic_hermite_coefficients(q[index], q[index + 1], velocities[index], velocities[index + 1],
                                              accelerations[index], accelerations[index + 1],
                                              jerks[index], jerks[index + 1], duration)
        u = np.arange(1, count + 1, dtype=float) / count
        p, v, acc, _ = _evaluate_septic(coeff, duration, u)
        for k in range(count):
            times.append(times[-1] + dt); positions.append(p[k]); velocities_out.append(v[k])
            accelerations_out.append(acc[k]); phases.append(1)
        peak_v, peak_a = _polynomial_motion_peaks(coeff, duration)
        segments.append(dict(index=index, start_s=float(times[-1] - duration), duration_s=float(duration),
                             start_velocity_rad_s=velocities[index].tolist(),
                             end_velocity_rad_s=velocities[index + 1].tolist(),
                             start_acceleration_rad_s2=accelerations[index].tolist(),
                             end_acceleration_rad_s2=accelerations[index + 1].tolist(),
                             analytic_peak_velocity_rad_s=peak_v.tolist(),
                             analytic_peak_acceleration_rad_s2=peak_a.tolist()))
    hold(float(config["target_hold_s"]), q[-1], 2)
    return dict(time_s=np.asarray(times), q_ref=np.asarray(positions), qd_ref=np.asarray(velocities_out),
                qdd_ref=np.asarray(accelerations_out), phase=np.asarray(phases, dtype=np.int8), segments=segments,
                timing_profile="continuous_septic")


def make_timing_plan(waypoints, config, durations=None, smooth=True):
    """Freeze a C3 curve independently of the later sampling step.

    Shared internal derivatives minimize integrated squared jerk in normalized
    time. A small snap term regularizes the solve, not a contact-cost change.
    Segment durations use the physics 5 ms grid, never the scoring grid.
    """
    q=np.asarray(waypoints,dtype=float)
    if q.ndim!=2 or q.shape[1]!=5 or len(q)<2 or not np.isfinite(q).all():
        raise ValueError('Finite five-joint waypoints required')
    if durations is None:
        legacy=timed_continuous_septic(q,config,.005)
        durations=[row['duration_s']for row in legacy['segments']]
    h=np.asarray(durations,dtype=float)
    if h.shape!=(len(q)-1,)or not np.isfinite(h).all()or np.any(h<=0):
        raise ValueError('One positive duration per segment required')
    h=np.ceil((h-1e-10)/.005)*.005
    for iteration in range(8):
        total=float(h.sum());hn=h/total
        v=np.zeros_like(q);a=np.zeros_like(q);j=np.zeros_like(q)
        if smooth and len(q)>2:
            # Linear least squares in shared v/a/j; endpoints are at rest.
            count=3*(len(q)-2);lhs=[];rhs=[]
            nodes,weights=np.polynomial.legendre.leggauss(8)
            u=(nodes+1)/2;weights=weights/2
            for i,segment in enumerate(hn):
                constant=_septic_hermite_coefficients(q[i],q[i+1],v[i],v[i+1],a[i],a[i+1],j[i],j[i+1],segment)
                basis=np.zeros((8,count))
                for end in (0,1):
                    knot=i+end
                    if not 0<knot<len(q)-1:continue
                    for order in range(3):
                        values=[np.zeros(1)for _ in range(8)]
                        values[2+2*order+end]=np.ones(1)
                        basis[:,3*(knot-1)+order]=_septic_hermite_coefficients(*values,segment)[:,0]
                for order,penalty in ((3,1.),(4,.0002)):
                    db=np.polynomial.polynomial.polyder(basis,m=order,axis=0)/segment**order
                    dc=np.polynomial.polynomial.polyder(constant,m=order,axis=0)/segment**order
                    evaluation=np.stack([u**k for k in range(len(db))],axis=1)
                    scale=np.sqrt(weights*segment*penalty)[:,None]
                    lhs.append(scale*(evaluation@db));rhs.append(-scale*(evaluation@dc))
            solved=np.linalg.lstsq(np.vstack(lhs),np.vstack(rhs),rcond=1e-10)[0]
            v[1:-1]=solved[0::3]/total;a[1:-1]=solved[1::3]/total**2;j[1:-1]=solved[2::3]/total**3
        elif not smooth:
            v=_knot_velocities(q,h,config,np.asarray(config['velocity_limit_rad_s']))
        peaks=[_polynomial_motion_peaks(_septic_hermite_coefficients(q[i],q[i+1],v[i],v[i+1],a[i],a[i+1],j[i],j[i+1],h[i]),h[i])for i in range(len(h))]
        scale=max(float(np.max(np.asarray([p[0]for p in peaks])/config['velocity_limit_rad_s'])),
                  float(np.sqrt(np.max(np.asarray([p[1]for p in peaks])/config['acceleration_limit_rad_s2']))))
        if scale<=1+1e-9:break
        h=np.ceil(h*scale*1.01/.005)*.005
    else:raise ValueError('Timing limits did not converge')
    return dict(version='fixed-septic-1',waypoints_q_rad=q.tolist(),segment_duration_s=h.tolist(),
        knot_velocity_rad_s=v.tolist(),knot_acceleration_rad_s2=a.tolist(),knot_jerk_rad_s3=j.tolist(),
        initial_hold_s=float(config['initial_hold_s']),target_hold_s=float(config['target_hold_s']),
        derivative_method='minimum_normalized_jerk_with_snap_regularization'if smooth else 'monotone_harmonic')


def sample_timing_plan(waypoints,config,dt,plan):
    q=np.asarray(waypoints,float);h=np.asarray(plan['segment_duration_s'],float)
    derivatives=[np.asarray(plan[k],float)for k in ('knot_velocity_rad_s','knot_acceleration_rad_s2','knot_jerk_rad_s3')]
    if (plan['version']!='fixed-septic-1'or dt<=0 or not np.isfinite(dt)
        or not np.array_equal(q,np.asarray(plan['waypoints_q_rad']))or h.shape!=(len(q)-1,)
        or np.any(h<=0)or not np.isfinite(h).all()
        or any(d.shape!=q.shape or not np.isfinite(d).all()for d in derivatives)):
        raise ValueError('Invalid fixed timing or waypoint mismatch')
    if any(not np.allclose(d[[0,-1]],0,atol=1e-10)for d in derivatives):
        raise ValueError('Fixed timing endpoints must join holds at rest through jerk')
    hold,endhold=plan['initial_hold_s'],plan['target_hold_s']
    if not np.isfinite([hold,endhold]).all()or min(hold,endhold)<0:raise ValueError('Invalid holds')
    if not np.allclose([hold,endhold],[config['initial_hold_s'],config['target_hold_s']]):
        raise ValueError('Saved timing hold/config mismatch')
    total=float(hold+h.sum()+endhold);t=np.arange(int(np.floor(total/dt+1e-9))+1)*dt
    if total-t[-1]>1e-9:t=np.r_[t,total]
    else:t[-1]=total
    positions=np.tile(q[-1],(len(t),1));vel=np.zeros_like(positions);acc=vel.copy();jerk=vel.copy()
    positions[t<=hold]=q[0];phase=np.where(t<=hold,0,np.where(t<hold+h.sum()-1e-10,1,2)).astype(np.int8)
    v,a,j=derivatives;segments=[];start=float(hold)
    for i,duration in enumerate(h):
        c=_septic_hermite_coefficients(q[i],q[i+1],v[i],v[i+1],a[i],a[i+1],j[i],j[i+1],duration)
        ids=(t>start)&(t<=start+duration+1e-10)
        p,d1,d2,d3=_evaluate_septic(c,duration,np.clip((t[ids]-start)/duration,0,1))
        positions[ids]=p;vel[ids]=d1;acc[ids]=d2;jerk[ids]=d3
        peak_v,peak_a=_polynomial_motion_peaks(c,duration)
        if np.any(peak_v>np.asarray(config['velocity_limit_rad_s'])*(1+1e-7))or np.any(peak_a>np.asarray(config['acceleration_limit_rad_s2'])*(1+1e-7)):
            raise ValueError('Saved timing exceeds analytic limits')
        segments.append(dict(index=i,start_s=start,duration_s=float(duration),
            start_velocity_rad_s=v[i].tolist(),end_velocity_rad_s=v[i+1].tolist(),
            start_acceleration_rad_s2=a[i].tolist(),end_acceleration_rad_s2=a[i+1].tolist(),
            analytic_peak_velocity_rad_s=peak_v.tolist(),analytic_peak_acceleration_rad_s2=peak_a.tolist()))
        start+=duration
    return dict(time_s=t,q_ref=positions,qd_ref=vel,qdd_ref=acc,qddd_ref=jerk,phase=phase,
                segments=segments,timing_profile='fixed_septic')


def time_parameterize(waypoints, config, dt, timing_plan=None):
    """Use the same timing in candidate ranking, export and dynamics validation."""
    if timing_plan is not None:return sample_timing_plan(waypoints,config,dt,timing_plan)
    profile = config.get('timing_profile', 'rest_to_rest_quintic')
    functions = {'continuous_septic': timed_continuous_septic,
                 'continuous_quintic': timed_continuous_quintic,
                 'rest_to_rest_quintic': timed_quintic, 'quintic': timed_quintic}
    if profile not in functions:
        raise ValueError(f'Unknown timing profile: {profile}')
    return functions[profile](waypoints, config, dt)


def actuator_command(model,data,config,q,qd,qdd,qadr,vadr,actuator_ids,inverse_data=None):
    """Position commands with explicit, model-specific feedforward.

    Legacy runs retain gravity and position lead. New natural profiles can use
    inverse dynamics for inertia, passive damping and model constraint friction.
    The actual actuator state is always simulated and checked independently.
    """
    import mujoco
    q=np.asarray(q,float);qd=np.asarray(qd,float);qdd=np.asarray(qdd,float)
    kp=model.actuator_gainprm[actuator_ids,0]
    if inverse_data is not None:
        inverse_data.qpos[qadr]=q;inverse_data.qvel[vadr]=qd;inverse_data.qacc[vadr]=qdd
        mujoco.mj_inverse(model,inverse_data)
        kv=-model.actuator_biasprm[actuator_ids,2]
        return q+(inverse_data.qfrc_inverse[vadr]+kv*qd)/kp
    command=q.copy();lead=float(config.get('position_lead_s',0))
    if lead<0:raise ValueError('Negative position lead')
    command[:5]+=lead*qd[:5]
    if config['bias_feedforward']:command+=data.qfrc_bias[vadr]/kp
    return command


def distance_intensity(clearance_m, margin_mm=5., penetration_scale_mm=2.,
                       floor_intensity=0.0, amplitude=1.0,
                       reference_step_mm=5., power=2.0):
    """Return a bounded convex proximity multiplier.

    Risk is zero at or beyond ``margin_mm`` and reaches exactly one when the
    robot reaches the surface (0 mm clearance).  Between those points it is
    a normalized convex power law.  Clearances below zero are clamped at one;
    this model deliberately has no separate penetration score.

    ``penetration_scale_mm`` and the older shape parameters remain accepted so
    older callers do not break, but the normalized rule is controlled by
    ``margin_mm`` and ``power``.
    """
    del penetration_scale_mm, floor_intensity, amplitude, reference_step_mm
    c = np.asarray(clearance_m, dtype=float) * 1000.0
    margin_mm = float(margin_mm)
    power = float(power)
    if (not np.isfinite([margin_mm, power]).all()
            or margin_mm <= 0 or power < 1):
        raise ValueError("Invalid nonlinear distance-risk parameters")
    x = np.clip((margin_mm - c) / margin_mm, 0.0, 1.0)
    return np.power(x, power)


def score_exposure(times, classes, clearances, weights, instances, policy):
    """Integrate per class/instance max exposure, avoiding duplicate geom covers.

    Arrays are [time, soft class, robot geom]. Persistence is applied to each
    episode's integral as in the existing contact policy; no physical area or
    penetration measurement is implied by the geometric covers.
    """
    times = np.asarray(times, dtype=float)
    clearances, weights = np.asarray(clearances), np.asarray(weights)
    instances = np.asarray(instances)
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or clearances.shape != weights.shape or weights.shape != instances.shape
            or clearances.ndim != 3 or clearances.shape[:2] != (len(times), len(classes))
            or not np.isfinite(clearances).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("Invalid geometric exposure time series")
    distance = policy["distance_risk"]
    values = weights * distance_intensity(
        clearances,
        distance["clearance_start_mm"],
        distance.get("penetration_scale_mm", 2.0),
        power=distance.get("power", 2.0),
    )
    total_series = np.zeros(len(times)); duration_series = np.zeros(len(times))
    exposure_series = np.zeros(len(times))
    continuity = policy["continuous_contact"]
    total, exposure, episodes = 0., 0., []
    for ci, class_id in enumerate(classes):
        # Follow the original class/organ identity. Mesh patch changes do not reset
        # an episode if their organ instance remains the same.
        active_ids = np.unique(instances[:, ci, :][values[:, ci, :] > 0])
        for organ_id in active_ids:
            mask = instances[:, ci, :] == organ_id
            v = np.where(mask, values[:, ci, :], 0).max(axis=1)
            exposure_series += v
            active = v > 0
            base = float(np.trapezoid(v, times)); exposure += base; total += base
            weighted = v.copy()
            indices = np.flatnonzero(active)
            cuts = np.flatnonzero(np.diff(times[indices]) > continuity["episode_gap_s"] + 1e-12) + 1
            for episode in np.split(indices, cuts):
                if not len(episode): continue
                begin, end = int(episode[0]), int(episode[-1])
                duration = float(times[end] - times[begin])
                integral = float(np.trapezoid(v[begin:end+1], times[begin:end+1]))
                factor = 1 + continuity["persistence_alpha"] * min(
                    duration / continuity["persistence_tau_s"], continuity["persistence_cap"])
                total += integral * (factor - 1)
                weighted[begin:end+1] *= factor
                duration_series[begin:end+1] = np.maximum(duration_series[begin:end+1], times[begin:end+1] - times[begin])
                episodes.append(dict(class_id=int(class_id), instance_id=int(organ_id),
                                     start_s=float(times[begin]), end_s=float(times[end]), duration_s=duration,
                                     exposure_integral=integral, persistence_factor=float(factor),
                                     proxy_overlap=bool(np.any(clearances[begin:end+1, ci, :][mask[begin:end+1]] <= 0))))
            total_series += weighted
    return dict(exposure_integral=exposure, persistence_cost=total,
                episodes=episodes, instantaneous_exposure=exposure_series,
                episode_weighted_series=total_series, active_episode_age_s=duration_series,
                overlap_any=np.any(clearances <= 0, axis=(1, 2)),
                proximity_any=np.any(values > 0, axis=(1, 2)))
