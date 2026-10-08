import numpy as np

from src.estimation.linear_reference import (
    inertia_mismatch_torque,
    lyapunov_pointing_ultimate_bound,
    periodic_gyro_driven_riccati,
)
from src.sensors.gyro import GyroParams
from src.sensors.star_tracker import StarTrackerParams


def test_periodic_riccati_is_positive_and_satisfies_cycle():
    gyro = GyroParams(rate=100.0)
    tracker = StarTrackerParams(rate=1.0)
    posterior, prior, phi = periodic_gyro_driven_riccati(
        gyro, tracker, dt=0.01
    )
    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    r = np.eye(3) * tracker.sigma_theta**2
    gain = np.linalg.solve(h @ prior @ h.T + r, h @ prior).T
    update = np.eye(6) - gain @ h
    expected_posterior = update @ prior @ update.T + gain @ r @ gain.T

    assert np.all(np.linalg.eigvalsh(posterior) > 0.0)
    assert np.allclose(posterior, expected_posterior, rtol=1e-10, atol=1e-18)
    assert np.all(np.linalg.eigvalsh(prior) > 0.0)
    assert np.all(np.isfinite(phi))


def test_inertia_mismatch_torque_matches_first_order_equation():
    delta_i = np.diag([10.0, -8.0, 6.0])
    omega = np.array([0.1, -0.2, 0.05])
    omega_dot = np.array([0.01, 0.02, -0.03])

    expected = delta_i @ omega_dot + np.cross(omega, delta_i @ omega)

    assert np.allclose(inertia_mismatch_torque(delta_i, omega, omega_dot), expected)


def test_lyapunov_pointing_bound_increases_with_estimation_error():
    inertia = np.diag([100.0, 80.0, 60.0])
    baseline = lyapunov_pointing_ultimate_bound(
        inertia, 2.0, 20.0, 1e-4, 1e-4, 0.0
    )
    degraded = lyapunov_pointing_ultimate_bound(
        inertia, 2.0, 20.0, 1e-3, 1e-3, 0.0
    )

    assert baseline > 0.0
    assert degraded > baseline
