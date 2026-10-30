"""Lower bounds and feasibility estimates for rest-to-rest attitude acquisition.

Two kinds of quantity live here and must not be confused.

**Proven lower bounds** (hold for any controller, given the stated assumptions):

* ``momentum_lower_bound_s``: the time to move the initial body momentum into
  the wheels, ``t >= |J w0| / tau``. See ``docs/system_description.md`` section 6.1
  for the derivation. It needs ``h_w(0) = 0`` and bounds the *delivered* body
  torque norm: ``tau = tau_max`` when the body command is norm-limited and the
  wheel lag cannot overshoot it (version a), ``tau = sqrt(3) tau_max`` from the
  per-wheel clips alone (version b).

**Approximations** (not bounds):

* ``slew_time_rest_to_rest_s`` is the bang-bang eigenaxis time
  ``2 sqrt(theta J_eff / tau_max)`` from rest. It ignores gyroscopic coupling for
  non-principal axes and any initial rate.
* ``eigenaxis_feasibility_estimate_s`` is the time-optimal time of a single-axis
  double integrator ``J_eff theta'' = u, |u| <= tau_max`` that starts at the true
  angle and the projection of the initial rate on the correction axis, and may
  finish at any equivalent target (angle ``theta`` or ``theta - 2 pi``). It
  ignores the rate perpendicular to the axis, the coupling between axes, the
  motor lag, and the 2% settling band, so it can be above or below what a real
  controller achieves. It is a *feasibility estimate*, not a bound.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

NUM_WHEELS_DEFAULT = 3


def momentum_lower_bound_s(inertia: np.ndarray, omega0: np.ndarray, tau_max: float,
                           per_wheel_clips: bool = False,
                           num_wheels: int = NUM_WHEELS_DEFAULT) -> float:
    """Shortest time to stop the body given zero initial wheel momentum.

    ``per_wheel_clips=False`` (version a): body command norm <= ``tau_max``.
    ``per_wheel_clips=True`` (version b): each wheel limited to ``tau_max``, so the
    delivered body torque norm is at most ``sqrt(num_wheels) tau_max`` for
    orthogonal wheels.
    """
    if tau_max <= 0.0:
        raise ValueError("tau_max must be positive")
    authority = tau_max * (np.sqrt(num_wheels) if per_wheel_clips else 1.0)
    h0 = np.asarray(inertia, dtype=float) @ np.asarray(omega0, dtype=float)
    return float(np.linalg.norm(h0) / authority)


def slew_time_rest_to_rest_s(angle_rad: float, j_eff: float, tau_max: float) -> float:
    """Bang-bang rest-to-rest time about one axis: ``2 sqrt(theta J_eff / tau_max)``."""
    if angle_rad < 0.0 or j_eff <= 0.0 or tau_max <= 0.0:
        raise ValueError("angle must be >= 0 and J_eff, tau_max positive")
    return float(2.0 * np.sqrt(angle_rad * j_eff / tau_max))


def double_integrator_min_time(x0: float, v0: float, accel: float) -> float:
    """Minimum time to drive ``(x, v)`` to the origin with ``|x''| <= accel``."""
    if accel <= 0.0:
        raise ValueError("accel must be positive")
    switch = x0 + v0 * abs(v0) / (2.0 * accel)
    if abs(switch) < 1e-15:
        return abs(v0) / accel
    if switch > 0.0:  # brake first (u = -accel), then accelerate back
        return (v0 + 2.0 * np.sqrt(0.5 * v0 * v0 + accel * x0)) / accel
    return (-v0 + 2.0 * np.sqrt(0.5 * v0 * v0 - accel * x0)) / accel


def eigenaxis_feasibility_estimate_s(angle_rad: float, rate_along_axis: float,
                                     j_eff: float, tau_max: float) -> float:
    """Single-axis time-optimal estimate, allowing the 2 pi equivalent target.

    ``angle_rad`` in [0, pi] is the rotation angle from the desired attitude to
    the body about ``axis``; ``rate_along_axis = omega . axis`` (positive moves
    the body further from the desired attitude).
    """
    accel = tau_max / j_eff
    return float(min(double_integrator_min_time(angle_rad, rate_along_axis, accel),
                     double_integrator_min_time(angle_rad - 2.0 * np.pi, rate_along_axis,
                                                accel)))


def linear_closed_loop(j: float, k_q: float, k_omega: float,
                       control_type: str = "quaternion_feedback") -> Dict[str, float]:
    """Poles of ``J th'' + K_w th' + k th = 0`` with the effective stiffness ``k``.

    ``k = K_q / 2`` for ``quaternion_feedback`` (``q_e,v ~ theta / 2``) and ``K_q``
    for ``pd`` (where the controller uses ``2 K_q q_e,v``), matching
    ``compute_control_gains`` and ``docs/conventions.md``.
    """
    if control_type not in ("quaternion_feedback", "pd"):
        raise ValueError("linear analysis covers quaternion_feedback and pd only")
    stiffness = 0.5 * k_q if control_type == "quaternion_feedback" else k_q
    wn = np.sqrt(stiffness / j)
    zeta = k_omega / (2.0 * np.sqrt(stiffness * j))
    disc = zeta * zeta - 1.0
    if disc >= 0.0:
        slow = -wn * (zeta - np.sqrt(disc))
        fast = -wn * (zeta + np.sqrt(disc))
        slow_c = fast_c = None
    else:
        slow = fast = -zeta * wn
        slow_c = wn * np.sqrt(-disc)
        fast_c = -slow_c
    return {"stiffness": float(stiffness), "natural_frequency_rad_s": float(wn),
            "damping_ratio": float(zeta), "slow_pole": float(slow), "fast_pole": float(fast),
            "pole_imaginary_part": None if slow_c is None else float(slow_c)}


def step_settling_time_s(j: float, k_q: float, k_omega: float, theta0: float = 1.0,
                         omega0: float = 0.0, band: float = 0.02,
                         control_type: str = "quaternion_feedback",
                         horizon_s: float = 2000.0, step_s: float = 0.05) -> float:
    """Last time the linear response leaves ``band * |theta0|``; ``inf`` if beyond the horizon.

    Solves the linear second-order response exactly (no saturation, no coupling).
    """
    stiffness = 0.5 * k_q if control_type == "quaternion_feedback" else k_q
    t = np.arange(0.0, horizon_s + step_s, step_s)
    roots = np.roots([j, k_omega, stiffness]).astype(complex)
    r1, r2 = roots
    if abs(r1 - r2) < 1e-14:
        c2 = omega0 - r1 * theta0
        theta = ((theta0 + c2 * t) * np.exp(r1 * t)).real
    else:
        c1 = (omega0 - r2 * theta0) / (r1 - r2)
        c2 = theta0 - c1
        theta = (c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)).real
    outside = np.nonzero(np.abs(theta) > band * abs(theta0))[0]
    if outside.size == 0:
        return 0.0
    last = outside[-1]
    return float("inf") if last >= len(t) - 1 else float(t[last + 1])


def saturated_single_axis_settling_s(angle_rad: float, rate_along_axis: float, j_eff: float,
                                     k_q: float, k_omega: float, tau_max: float,
                                     band: float = 0.02, horizon_s: float = 600.0,
                                     dt: float = 0.05) -> float:
    """Settling time of the shipped ``quaternion_feedback`` law on one axis, with saturation.

    Integrates ``J_eff th'' = clip(-K_q sin(th / 2) - K_w th', +/- tau_max)`` with the
    angle wrapped to (-pi, pi] (the controller takes the shortest path), and
    returns the last time ``|th|`` exceeds ``band`` times the initial angle, or
    ``inf`` if that is beyond the horizon. Single axis only: it ignores the rate
    perpendicular to the axis, the coupling between axes, the motor lag, and the
    estimator. It keeps the two things the unsaturated linear model leaves out:
    the torque limit and the sin(theta / 2) nonlinearity.
    """
    theta, omega = float(angle_rad), float(rate_along_axis)
    threshold = band * abs(angle_rad)
    last_outside = 0.0
    steps = int(horizon_s / dt)
    for step in range(1, steps + 1):
        wrapped = (theta + np.pi) % (2.0 * np.pi) - np.pi
        torque = -k_q * np.sin(0.5 * wrapped) - k_omega * omega
        torque = max(-tau_max, min(tau_max, torque))
        omega += torque / j_eff * dt
        theta += omega * dt
        wrapped = (theta + np.pi) % (2.0 * np.pi) - np.pi
        if abs(wrapped) > threshold:
            last_outside = step * dt
    return float("inf") if last_outside >= horizon_s - dt else float(last_outside)


def stop_then_slew_plan(inertia: np.ndarray, q0: np.ndarray, omega0: np.ndarray, tau_max: float,
                        dt: float = 0.05, rate_tol: float = 1e-4,
                        max_time_s: float = 600.0) -> Dict[str, float]:
    """A constructive, open-loop plan: stop the body, then slew from rest.

    Phase 1 applies the full torque ``tau_max`` against the body momentum
    ``J w`` in the full rigid-body-plus-wheel model (wheels start at zero
    momentum) until the rate is below ``rate_tol``; the attitude drifts while it
    brakes. Phase 2 is the rest-to-rest bang-bang slew from the attitude reached.
    Not optimal (a real controller can start turning before it has stopped), so it
    is a time that is achievable in this idealisation, not a lower bound; the
    momentum bound is the matching lower bound for phase 1.
    """
    from src.dynamics.quaternion import error_quat_to_vec, right_error_quat
    from src.dynamics.rigid_body import RigidBody

    inertia = np.asarray(inertia, dtype=float)
    body = RigidBody(inertia, q0, omega0)
    wheel = np.zeros(3)
    elapsed = 0.0
    while np.linalg.norm(body.omega) > rate_tol and elapsed < max_time_s:
        momentum = inertia @ body.omega
        norm = np.linalg.norm(momentum)
        torque = -min(tau_max, norm / dt) * momentum / norm
        body.step(torque, dt, wheel_momentum=wheel, wheel_momentum_rate=-torque)
        wheel = wheel - torque * dt
        elapsed += dt
    rotation = error_quat_to_vec(right_error_quat(body.q, np.array([0.0, 0.0, 0.0, 1.0])))
    angle = float(np.linalg.norm(rotation))
    j_eff = float((rotation / angle) @ inertia @ (rotation / angle)) if angle > 1e-12 else 1.0
    slew = slew_time_rest_to_rest_s(angle, j_eff, tau_max)
    return {"stop_s": float(elapsed), "slew_s": slew, "total_s": float(elapsed + slew),
            "angle_after_stop_deg": float(np.rad2deg(angle))}


def slow_pole_settling_estimate_s(slow_pole: float, band: float = 0.02) -> float:
    """``ln(1 / band) / |slow pole|``, the usual single-exponential 2% estimate."""
    return float(np.log(1.0 / band) / abs(slow_pole))


def principal_axis_linear_summary(inertia: np.ndarray, k_q: float, k_omega: float,
                                  control_type: str = "quaternion_feedback"
                                  ) -> Sequence[Dict[str, float]]:
    """Linear closed-loop summary for each principal axis (diagonal inertia assumed)."""
    rows = []
    for index, j in enumerate(np.diag(np.asarray(inertia, dtype=float))):
        poles = linear_closed_loop(j, k_q, k_omega, control_type)
        rows.append({"axis": "xyz"[index], "inertia_kg_m2": float(j), **poles,
                     "settling_slow_pole_s": slow_pole_settling_estimate_s(poles["slow_pole"]),
                     "settling_exact_from_rest_s": step_settling_time_s(
                         j, k_q, k_omega, control_type=control_type)})
    return rows
