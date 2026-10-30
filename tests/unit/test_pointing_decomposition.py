"""Truth-feedback option, pointing-error decomposition, and steady-state metrics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.run_experiment import (
    compute_pointing_decomposition_metrics,
    load_config,
    run_simulation,
)
from src.control.attitude_controller import AttitudeController, ControllerParams
from src.dynamics.quaternion import (
    error_quat_to_vec,
    inverse,
    multiply,
    normalize,
    right_box_plus,
    right_error_quat,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = PROJECT_ROOT / "tests" / "data" / "golden_v1_0_0_closed_loop.npz"
GOLDEN_KEYS = (
    "q_true", "omega_true", "q_hat", "omega_hat", "tau_cmd", "tau_rw",
    "wheel_momentum", "NEES", "pointing_error",
)


def _closed_loop(variant="gyro_driven", duration=10.0, feedback=None):
    config = load_config(str(PROJECT_ROOT / "config" / "closed_loop.yaml"))
    controller = dict(config.controller)
    if feedback is not None:
        controller["state_feedback"] = feedback
    return replace(
        config,
        duration=duration,
        filter={**config.filter, "variant": variant},
        controller=controller,
    )


def _random_unit_quaternions(rng, n):
    return normalize(rng.standard_normal((n, 4)))


def _same_rotation(q_a, q_b, tol):
    """q and -q are the same attitude."""
    return min(np.linalg.norm(q_a - q_b), np.linalg.norm(q_a + q_b)) < tol


# --- default path is unchanged (regression against release 1.0.0) -------------

@pytest.mark.parametrize("variant", ["gyro_driven", "model_aided"])
@pytest.mark.parametrize("feedback", [None, "estimated"])
def test_default_feedback_reproduces_release_1_0_0_bit_for_bit(variant, feedback):
    golden = np.load(GOLDEN)
    results = run_simulation(_closed_loop(variant, feedback=feedback))
    for key in GOLDEN_KEYS:
        assert np.array_equal(results[key], golden[f"{variant}/{key}"]), key


def test_invalid_state_feedback_is_rejected():
    with pytest.raises(ValueError, match="state_feedback"):
        run_simulation(_closed_loop(feedback="perfect"))


# --- truth feedback: same random draws, different control input --------------

def test_truth_feedback_feeds_controller_the_true_state():
    config = _closed_loop(feedback="truth", duration=2.0)
    results = run_simulation(config)
    controller = AttitudeController(ControllerParams(
        K_q=config.controller["K_q"], K_omega=config.controller["K_omega"],
        max_torque=config.controller["max_torque"],
        control_type=config.controller["control_type"],
    ))
    controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
    for i in (0, 50, 199):
        expected = controller.compute_control(
            results["q_true"][i], results["omega_true"][i]
        )
        assert np.array_equal(results["tau_cmd"][i], expected)


@pytest.mark.parametrize("variant", ["gyro_driven", "model_aided"])
def test_truth_and_estimated_runs_share_the_same_noise_draws(variant):
    estimated = run_simulation(_closed_loop(variant, feedback="estimated"))
    truth = run_simulation(_closed_loop(variant, feedback="truth"))
    # Gyro bias walk depends only on the gyro stream, not on the trajectory.
    assert np.array_equal(estimated["b_true"], truth["b_true"])
    # Step 0 sees the same truth state and the same initialization draw.
    assert np.array_equal(estimated["q_hat"][0], truth["q_hat"][0])
    # The closed-loop trajectories differ once the control input differs.
    assert not np.array_equal(estimated["tau_cmd"], truth["tau_cmd"])


# --- decomposition identity ---------------------------------------------------

def test_decomposition_identity_exact_for_random_rotations():
    rng = np.random.default_rng(0)
    q_d, q_hat, q_true = (_random_unit_quaternions(rng, 200) for _ in range(3))
    for qd, qh, qt in zip(q_d, q_hat, q_true):
        dq_p = multiply(inverse(qd), qt)
        dq_c = multiply(inverse(qd), qh)
        dq_e = multiply(inverse(qh), qt)
        assert _same_rotation(multiply(dq_c, dq_e), dq_p, 1e-14)


@pytest.mark.parametrize("feedback", ["estimated", "truth"])
def test_logged_error_vectors_satisfy_identity_in_simulation(feedback):
    results = run_simulation(_closed_loop(feedback=feedback, duration=20.0))
    q_d = results["q_desired"]
    for i in range(0, len(results["time"]), 25):
        dq_c = multiply(inverse(q_d), results["q_hat"][i])
        dq_e = multiply(inverse(results["q_hat"][i]), results["q_true"][i])
        dq_p = multiply(inverse(q_d), results["q_true"][i])
        assert _same_rotation(multiply(dq_c, dq_e), dq_p, 1e-12)
        # Logged vectors agree with the quaternion definitions.
        assert np.allclose(results["control_error_vector"][i],
                           error_quat_to_vec(dq_c), atol=1e-12)
        assert np.allclose(results["pointing_error_vector"][i],
                           error_quat_to_vec(dq_p), atol=1e-12)
        assert np.allclose(results["attitude_estimation_error_vector"][i],
                           error_quat_to_vec(dq_e), atol=1e-12)


def test_small_angle_vectors_add_to_second_order():
    rng = np.random.default_rng(1)
    q_d = _random_unit_quaternions(rng, 1)[0]
    direction_c = rng.standard_normal(3)
    direction_e = rng.standard_normal(3)
    scales = [1e-1, 1e-2, 1e-3]
    residuals, after_second_order = [], []
    for eps in scales:
        theta_c, theta_e = eps * direction_c, eps * direction_e
        q_hat = right_box_plus(q_d, theta_c)       # q_hat = q_d (x) Exp(theta_c)
        q_true = right_box_plus(q_hat, theta_e)    # q_true = q_hat (x) Exp(theta_e)
        theta_p = error_quat_to_vec(right_error_quat(q_true, q_d))
        residuals.append(np.linalg.norm(theta_p - theta_c - theta_e))
        # BCH: the leading residual is 1/2 theta_c x theta_e.
        after_second_order.append(np.linalg.norm(
            theta_p - theta_c - theta_e - 0.5 * np.cross(theta_c, theta_e)))
    slope = np.polyfit(np.log(scales), np.log(residuals), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)
    slope_next = np.polyfit(np.log(scales[:2]), np.log(after_second_order[:2]), 1)[0]
    assert slope_next == pytest.approx(3.0, abs=0.15)
    assert residuals[-1] < 1e-5


# --- steady-state metrics -----------------------------------------------------

def _synthetic(n=3001, dt=0.1, theta=None, omega_error=None):
    time = np.arange(n) * dt
    vec = np.zeros((n, 3)) if theta is None else theta
    omega_true = np.zeros((n, 3))
    omega_hat = omega_true + (0.0 if omega_error is None else omega_error)
    return {
        "time": time,
        "pointing_error": np.linalg.norm(vec, axis=1),
        "pointing_error_vector": vec,
        "control_error_vector": 0.6 * vec,
        "attitude_estimation_error_vector": 0.4 * vec,
        "omega_true": omega_true,
        "omega_hat": omega_hat,
        "tau_cmd": np.zeros((n, 3)),
        "tau_rw": np.zeros((n, 3)),
        "max_torque": 0.1,
    }


def test_metrics_constant_error_window_and_fraction():
    theta = np.zeros((3001, 3))
    theta[:, 0] = np.deg2rad(0.05)
    theta[:50, 0] = np.deg2rad(10.0)  # acquisition transient, settles at t = 5 s
    results = _synthetic(theta=theta)
    metrics = compute_pointing_decomposition_metrics(results)
    assert metrics["window_source"] == "settling_2pct"
    assert metrics["window_start_s"] == pytest.approx(5.0)
    assert metrics["pointing"]["ape"]["rms_rad"] == pytest.approx(np.deg2rad(0.05))
    assert metrics["pointing"]["ape"]["p99_rad"] == pytest.approx(np.deg2rad(0.05))
    assert metrics["pointing"]["rpe"]["rms_rad"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["control"]["ape"]["rms_rad"] == pytest.approx(0.6 * np.deg2rad(0.05))
    assert metrics["estimation"]["ape"]["rms_rad"] == pytest.approx(0.4 * np.deg2rad(0.05))
    assert metrics["fraction_within_requirement"] == 1.0
    tight = compute_pointing_decomposition_metrics(results, pointing_requirement_deg=0.04)
    assert tight["fraction_within_requirement"] == 0.0


def test_metrics_rpe_of_linear_ramp_and_p99():
    n, dt = 3001, 0.1
    theta = np.zeros((n, 3))
    theta[:, 1] = 1e-4 + 1e-6 * np.arange(n) * dt  # 1e-6 rad/s drift
    results = _synthetic(n, dt, theta=theta)
    # The error grows, so the run never settles to 2% of its initial value and
    # the configured window start applies.
    metrics = compute_pointing_decomposition_metrics(
        results, steady_state_start_s=50.0, rpe_window_s=10.0)
    assert metrics["window_source"] == "config"
    assert metrics["pointing"]["rpe"]["rms_rad"] == pytest.approx(1e-5, rel=1e-9)
    assert metrics["pointing"]["ape"]["p99_rad"] == pytest.approx(
        np.percentile(theta[500:, 1], 99.0))


def test_metrics_requires_a_window_when_run_never_settles():
    results = _synthetic(theta=np.zeros((3001, 3)))  # zero initial error
    with pytest.raises(ValueError, match="steady_state_start_s"):
        compute_pointing_decomposition_metrics(results)
    metrics = compute_pointing_decomposition_metrics(results, steady_state_start_s=10.0)
    assert metrics["window_source"] == "config"
    assert metrics["window_start_s"] == 10.0


def test_metrics_rpe_none_when_window_shorter_than_interval():
    results = _synthetic(n=101, dt=0.1, theta=np.zeros((101, 3)))
    metrics = compute_pointing_decomposition_metrics(
        results, steady_state_start_s=5.0, rpe_window_s=10.0)
    assert metrics["pointing"]["rpe"] is None


def test_metrics_rate_estimation_error_rms():
    omega_error = np.tile([3e-4, 4e-4, 0.0], (3001, 1))
    results = _synthetic(theta=np.zeros((3001, 3)), omega_error=omega_error)
    metrics = compute_pointing_decomposition_metrics(results, steady_state_start_s=1.0)
    assert metrics["rate_estimation_error"]["rms_rad_s"] == pytest.approx(5e-4)
    assert metrics["rate_estimation_error"]["rms_per_axis_rad_s"] == pytest.approx(
        [3e-4, 4e-4, 0.0])
