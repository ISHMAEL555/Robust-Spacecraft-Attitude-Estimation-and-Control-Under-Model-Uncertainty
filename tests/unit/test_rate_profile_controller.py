"""Rate-profile controller and gyroscopic feedforward option."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.run_experiment import load_config, run_simulation
from src.control.attitude_controller import AttitudeController, ControllerParams

J = np.diag([100.0, 80.0, 60.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _controller(**overrides):
    params = dict(K_q=1.0, K_omega=20.0, max_torque=0.1, control_type="rate_profile",
                  inertia=J, profile_accel_fraction=0.7, profile_rate_gain=0.05)
    params.update(overrides)
    return AttitudeController(ControllerParams(**params))


def _about(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def _commanded_rate(controller, angle, axis=(0.0, 0.0, 1.0)):
    """Recover |omega_cmd| from the unsaturated torque at zero body rate."""
    q = _about(axis, -angle)  # body is -angle away, so the correction is +angle
    return np.linalg.norm(controller._feedback(q, np.zeros(3))) / controller.K_omega


def test_requires_nominal_inertia_and_valid_parameters():
    with pytest.raises(ValueError):
        AttitudeController(ControllerParams(control_type="rate_profile"))
    with pytest.raises(ValueError):
        _controller(profile_accel_fraction=0.0)
    with pytest.raises(ValueError):
        _controller(profile_accel_fraction=1.5)
    with pytest.raises(ValueError):
        _controller(profile_rate_gain=-0.1)


def test_zero_error_and_zero_rate_command_zero_torque():
    assert np.allclose(_controller().compute_control(IDENTITY, np.zeros(3)), 0.0)


def test_linear_region_commands_k_p_times_the_angle():
    controller = _controller()
    angle = 0.05  # well inside 2a/k_p^2
    assert _commanded_rate(controller, angle) == pytest.approx(0.05 * angle, rel=1e-9)


def test_far_region_commands_the_constant_deceleration_profile():
    controller = _controller()
    angle = 1.0
    decel = 0.7 * 0.1 / 60.0  # rotation about z, J_zz = 60
    assert _commanded_rate(controller, angle) == pytest.approx(
        np.sqrt(2.0 * decel * angle), rel=1e-9)


def test_profile_is_continuous_at_the_switching_angle():
    controller = _controller()
    decel = 0.7 * 0.1 / 60.0
    switch = 2.0 * decel / 0.05 ** 2
    below = _commanded_rate(controller, switch * (1 - 1e-6))
    above = _commanded_rate(controller, switch * (1 + 1e-6))
    assert below == pytest.approx(above, rel=1e-4)


def test_torque_sign_matches_quaternion_feedback():
    """At rest, the rate-profile torque points the same way as the PD torque."""
    q = _about([1.0, 2.0, 3.0], 0.4)
    reference = AttitudeController(ControllerParams(K_q=1.0, K_omega=20.0, max_torque=1e9))
    rate_profile = _controller(max_torque=1e9)
    a = reference.compute_control(q, np.zeros(3))
    b = rate_profile.compute_control(q, np.zeros(3))
    assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) == pytest.approx(1.0, abs=1e-9)


def test_rate_loop_opposes_a_rate_error():
    controller = _controller()
    torque = controller.compute_control(IDENTITY, np.array([0.01, 0.0, 0.0]))
    assert torque[0] < 0.0 and np.allclose(torque[1:], 0.0)


def test_command_is_norm_limited():
    controller = _controller()
    torque = controller.compute_control(_about([0, 0, 1], -2.0), np.zeros(3))
    assert np.linalg.norm(torque) == pytest.approx(0.1)


def test_default_controller_type_is_unchanged():
    controller = AttitudeController(ControllerParams())
    assert controller.params.control_type == "quaternion_feedback"


def test_feedforward_option_is_off_by_default_and_changes_the_run_when_on():
    base = load_config("config/closed_loop.yaml")
    base = replace(base, duration=3.0)
    off = run_simulation(base)
    explicit_off = run_simulation(replace(
        base, controller={**base.controller, "gyroscopic_feedforward": False}))
    on = run_simulation(replace(
        base, controller={**base.controller, "gyroscopic_feedforward": True}))
    assert np.array_equal(off["tau_cmd"], explicit_off["tau_cmd"])
    assert not np.array_equal(off["tau_cmd"], on["tau_cmd"])
