"""Independent sampled-linear reference calculations for MEKF validation."""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.sensors.gyro import GyroParams
from src.sensors.star_tracker import StarTrackerParams


def periodic_gyro_driven_riccati(
    gyro: GyroParams,
    tracker: StarTrackerParams,
    dt: float,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the posterior/prior covariance and cycle transition at tracker epochs.

    This independently assembles the zero-rate continuous error model and
    solves the sampled periodic DARE over the integer gyro/tracker cycle.
    """
    if dt <= 0.0 or q_scale <= 0.0 or r_scale <= 0.0:
        raise ValueError("dt and covariance scales must be positive")
    gyro_steps = int(round(tracker.dt / dt))
    if gyro_steps < 1 or not np.isclose(gyro_steps * dt, tracker.dt):
        raise ValueError("tracker period must be an integer multiple of dt")

    f = np.zeros((6, 6))
    f[:3, 3:] = -np.eye(3)
    qc = np.diag(
        [gyro.sigma_v**2 * q_scale] * 3
        + [gyro.sigma_u**2 * q_scale] * 3
    )
    van_loan = np.zeros((12, 12))
    van_loan[:6, :6] = -f
    van_loan[:6, 6:] = qc
    van_loan[6:, 6:] = f.T
    exponential = expm(van_loan * dt)
    phi_step = exponential[6:, 6:].T
    q_step = phi_step @ exponential[:6, 6:]
    q_step = (q_step + q_step.T) * 0.5

    phi_cycle = np.eye(6)
    q_cycle = np.zeros((6, 6))
    for _ in range(gyro_steps):
        q_cycle = phi_step @ q_cycle @ phi_step.T + q_step
        phi_cycle = phi_step @ phi_cycle

    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    r = np.eye(3) * tracker.sigma_theta**2 * r_scale
    prior = solve_discrete_are(phi_cycle.T, h.T, q_cycle, r)
    prior = (prior + prior.T) * 0.5
    gain = np.linalg.solve(h @ prior @ h.T + r, h @ prior).T
    residual = np.eye(6) - gain @ h
    posterior = residual @ prior @ residual.T + gain @ r @ gain.T
    posterior = (posterior + posterior.T) * 0.5
    return posterior, prior, phi_cycle


def inertia_mismatch_torque(
    delta_inertia: np.ndarray, omega: np.ndarray, omega_dot: np.ndarray
) -> np.ndarray:
    """Compute the first-order equivalent torque from inertia mismatch."""
    return delta_inertia @ omega_dot + np.cross(omega, delta_inertia @ omega)


def lyapunov_pointing_ultimate_bound(
    inertia: np.ndarray,
    proportional_gain: float,
    derivative_gain: float,
    attitude_error_bound: float,
    rate_error_bound: float,
    disturbance_torque_bound: float,
) -> float:
    """Bound small-angle pointing norm for an unsaturated linear PD hold."""
    from scipy.linalg import solve_continuous_lyapunov

    if min(
        proportional_gain,
        derivative_gain,
        attitude_error_bound,
        rate_error_bound,
        disturbance_torque_bound,
    ) < 0.0:
        raise ValueError("gains and error/disturbance bounds must be non-negative")
    if proportional_gain == 0.0 or derivative_gain == 0.0:
        raise ValueError("both PD gains must be positive")

    inertia_matrix = np.asarray(inertia, dtype=float)
    inertia_values = np.linalg.eigvalsh(inertia_matrix)
    if inertia_matrix.shape != (3, 3) or np.any(inertia_values <= 0.0):
        raise ValueError("inertia must be positive definite")

    inertia_inverse = np.linalg.inv(inertia_matrix)
    a = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [
            -proportional_gain * inertia_inverse,
            -derivative_gain * inertia_inverse,
        ],
    ])
    p = solve_continuous_lyapunov(a.T, -np.eye(6))
    p = (p + p.T) * 0.5
    disturbance_acceleration = np.linalg.norm(inertia_inverse, 2) * (
        disturbance_torque_bound
        + proportional_gain * attitude_error_bound
        + derivative_gain * rate_error_bound
    )
    return float(
        2.0 * np.linalg.norm(p, 2) * disturbance_acceleration
    )
