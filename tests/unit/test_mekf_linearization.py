"""Finite-difference and discretization checks for the MEKF error model."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from src.dynamics.quaternion import (
    error_quat_to_vec,
    right_box_plus,
    right_error_quat,
    vec_to_error_quat,
)
from src.estimation.mekf import create_mekf_nominal


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def test_gyro_driven_error_dynamics_match_finite_difference():
    mekf = create_mekf_nominal("gyro_driven")
    omega = np.array([0.17, -0.09, 0.12])
    dt = 2e-5
    F, _ = mekf._compute_F_G(omega)
    q_hat = right_box_plus(np.array([0.2, -0.1, 0.3, 0.9]), np.zeros(3))

    def error_flow(error):
        dtheta, dbias = error[:3], error[3:]
        q_true = right_box_plus(q_hat, dtheta)
        q_hat_next = right_box_plus(q_hat, omega * dt)
        q_true_next = right_box_plus(q_true, (omega - dbias) * dt)
        error_next = np.concatenate([
            error_quat_to_vec(right_error_quat(q_true_next, q_hat_next)),
            dbias,
        ])
        return (error_next - error) / dt

    numerical = np.zeros_like(F)
    step = 2e-4
    for column in range(6):
        direction = np.eye(6)[column] * step
        numerical[:, column] = (
            error_flow(direction) - error_flow(-direction)
        ) / (2.0 * step)

    assert np.allclose(numerical, F, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "wheel_momentum",
    [np.zeros(3), np.array([4.0, -2.5, 7.0])],
    ids=["no_wheel_momentum", "with_wheel_momentum"],
)
def test_model_aided_full_error_dynamics_match_finite_difference(wheel_momentum):
    inertia = np.array([[100.0, 2.0, -1.0], [2.0, 80.0, 1.5], [-1.0, 1.5, 60.0]])
    mekf = create_mekf_nominal("model_aided", inertia=inertia)
    omega = np.array([0.17, -0.09, 0.12])
    torque = np.array([0.01, -0.02, 0.015])
    dt = 2e-5
    mekf._propagation_torque = torque
    mekf._propagation_wheel_momentum = wheel_momentum
    F, _ = mekf._compute_F_G_model_aided(omega)
    q_hat = right_box_plus(np.array([0.2, -0.1, 0.3, 0.9]), np.zeros(3))
    omega_dot_hat = np.linalg.solve(
        inertia, torque - np.cross(omega, inertia @ omega + wheel_momentum)
    )
    q_hat_next = right_box_plus(q_hat, omega * dt)
    omega_hat_next = omega + omega_dot_hat * dt

    def error_flow(error):
        dtheta = error[:3]
        domega = error[3:6]
        dinertia = error[9:15]
        inertia_true = inertia.copy()
        for value, scale, basis in zip(
            dinertia, mekf._inertia_scales, mekf._inertia_basis
        ):
            inertia_true += value * scale * basis
        omega_true = omega + domega
        q_true = right_box_plus(q_hat, dtheta)
        q_true_next = right_box_plus(q_true, omega_true * dt)
        omega_dot_true = np.linalg.solve(
            inertia_true,
            torque - np.cross(omega_true, inertia_true @ omega_true + wheel_momentum),
        )
        omega_true_next = omega_true + omega_dot_true * dt
        inertia_error_next = np.array([
            (inertia_true - inertia)[0, 0],
            (inertia_true - inertia)[1, 1],
            (inertia_true - inertia)[2, 2],
            (inertia_true - inertia)[0, 1],
            (inertia_true - inertia)[0, 2],
            (inertia_true - inertia)[1, 2],
        ]) / mekf._inertia_scales
        error_next = np.concatenate([
            error_quat_to_vec(right_error_quat(q_true_next, q_hat_next)),
            omega_true_next - omega_hat_next,
            error[6:9],
            inertia_error_next,
        ])
        return (error_next - error) / dt

    numerical = np.zeros_like(F)
    step = 2e-4
    for column in range(15):
        direction = np.eye(15)[column] * step
        numerical[:, column] = (
            error_flow(direction) - error_flow(-direction)
        ) / (2.0 * step)

    assert np.allclose(numerical, F, rtol=3e-5, atol=3e-6)


def test_attitude_reset_jacobian_matches_finite_difference():
    for variant, inertia in [
        ("gyro_driven", None),
        ("model_aided", np.diag([100.0, 80.0, 60.0])),
    ]:
        mekf = create_mekf_nominal(variant, inertia=inertia)
        correction = np.array([0.28, -0.19, 0.11])
        expected = np.zeros((3, 3))
        epsilon = 1e-7
        for column in range(3):
            perturbation = np.eye(3)[column] * epsilon
            q_hat = np.array([0.0, 0.0, 0.0, 1.0])
            q_true_plus = right_box_plus(q_hat, correction + perturbation)
            q_true_minus = right_box_plus(q_hat, correction - perturbation)
            q_hat_updated = right_box_plus(q_hat, correction)
            expected[:, column] = (
                error_quat_to_vec(right_error_quat(q_true_plus, q_hat_updated))
                - error_quat_to_vec(right_error_quat(q_true_minus, q_hat_updated))
            ) / (2.0 * epsilon)

        mekf.P = np.eye(mekf.state_dim)
        mekf._reset_attitude_error_covariance(correction)
        assert np.allclose(mekf.P[:3, :3], expected @ expected.T, atol=2e-8)


def test_star_tracker_right_error_measurement_jacobian_is_identity():
    mekf = create_mekf_nominal("gyro_driven")
    q_hat = right_box_plus(np.array([0.1, 0.2, -0.3, 0.9]), np.zeros(3))
    q_hat /= np.linalg.norm(q_hat)
    epsilon = 1e-6
    H_numeric = np.column_stack([
        error_quat_to_vec(right_error_quat(
            right_box_plus(q_hat, np.eye(3)[i] * epsilon), q_hat
        )) / epsilon
        for i in range(3)
    ])
    assert np.allclose(H_numeric, np.eye(3), atol=1e-9)
    assert mekf.R.shape == (3, 3)


def test_van_loan_covariance_matches_independent_quadrature():
    mekf = create_mekf_nominal("gyro_driven")
    F = np.array([
        [-0.03, 0.02, 0.01, -1.0, 0.0, 0.0],
        [-0.02, -0.01, 0.03, 0.0, -1.0, 0.0],
        [0.01, -0.04, 0.02, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, -0.004, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.003, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.002],
    ])
    G = np.eye(6)
    Qc = np.diag([1e-8, 2e-8, 3e-8, 1e-12, 2e-12, 3e-12])
    dt = 0.37
    _, Qd = mekf._van_loan_discretization(F, G, Qc, dt)

    nodes, weights = leggauss(48)
    times = 0.5 * dt * (nodes + 1.0)
    integral = np.zeros_like(Qd)
    for time, weight in zip(times, weights):
        transition = expm(F * time)
        integral += 0.5 * dt * weight * transition @ G @ Qc @ G.T @ transition.T

    assert np.allclose(Qd, integral, rtol=2e-10, atol=1e-16)


def test_model_aided_rate_process_noise_enters_rate_covariance():
    mekf = create_mekf_nominal(
        "model_aided",
        inertia=np.diag([100.0, 80.0, 60.0]),
        sigma_omega_process=1e-7,
    )
    F, G = mekf._compute_F_G_model_aided(np.array([0.01, -0.02, 0.005]))
    _, Qd = mekf._van_loan_discretization(F, G, mekf.Q, 0.01)

    assert np.allclose(G[3:6, 3:6], np.eye(3))
    assert np.allclose(G[6:9, 6:9], np.eye(3))
    assert np.all(np.diag(Qd)[3:6] > 0.0)


def test_model_aided_rate_noise_is_a_continuous_spectral_density():
    sigma_omega = 2e-5
    dt = 0.25
    mekf = create_mekf_nominal(
        "model_aided",
        inertia=np.diag([100.0, 80.0, 60.0]),
        sigma_omega_process=sigma_omega,
    )

    assert np.allclose(
        mekf.Q[3:6, 3:6], sigma_omega**2 * np.eye(3)
    )
    assert np.allclose(mekf.Q[9:15, 9:15], 0.0)

    _, Qd = mekf._van_loan_discretization(
        np.zeros((mekf.state_dim, mekf.state_dim)),
        np.eye(mekf.state_dim),
        mekf.Q,
        dt,
    )
    assert np.allclose(Qd[3:6, 3:6], sigma_omega**2 * dt * np.eye(3))
    assert np.allclose(Qd[9:15, 9:15], 0.0)


def test_small_angle_quaternion_exponential_matches_rotation_vector():
    delta = np.array([1e-9, -2e-9, 3e-9])
    quaternion = vec_to_error_quat(delta)
    assert np.allclose(error_quat_to_vec(quaternion), delta, atol=1e-15)
