"""Two-phase mode manager: detumble torque, hysteresis, and the switch on the estimated rate."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.run_e6_monte_carlo import build_config, sample_initial_conditions
from scripts.run_experiment import load_config, run_simulation
from src.control.attitude_controller import AttitudeController, ControllerParams

J = np.diag([100.0, 80.0, 60.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _manager(**overrides):
    params = dict(K_omega=20.0, max_torque=0.1, control_type="two_phase", inertia=J,
                  detumble_entry_rate=0.01, detumble_exit_rate=0.15,
                  detumble_time_constant_s=20.0)
    params.update(overrides)
    return AttitudeController(ControllerParams(**params))


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        _manager(detumble_entry_rate=0.2, detumble_exit_rate=0.15)
    with pytest.raises(ValueError):
        _manager(detumble_entry_rate=0.0)
    with pytest.raises(ValueError):
        _manager(detumble_time_constant_s=0.0)
    with pytest.raises(ValueError):
        _manager(profile_max_rate=0.2)  # would let acquisition re-trigger detumble


def test_detumble_torque_is_within_the_limit_and_opposes_the_momentum():
    rng = np.random.default_rng(0)
    for _ in range(500):
        omega = rng.standard_normal(3) * rng.choice([1e-3, 0.03, 0.1, 1.0])
        controller = _manager(detumble_entry_rate=1e-9, detumble_exit_rate=1e9)
        torque = controller.compute_control(IDENTITY, omega)
        assert controller.phase == "detumble"
        assert np.linalg.norm(torque) <= 0.1 + 1e-15
        momentum = J @ omega
        cosine = torque @ momentum / (np.linalg.norm(torque) * np.linalg.norm(momentum))
        assert cosine == pytest.approx(-1.0, abs=1e-12)


def test_detumble_is_at_the_limit_far_out_and_proportional_near_zero():
    controller = _manager(detumble_entry_rate=1e-9, detumble_exit_rate=1e9)
    far = controller.compute_control(IDENTITY, np.array([0.1, 0.0, 0.0]))  # |Jw| = 10
    assert np.linalg.norm(far) == pytest.approx(0.1)
    near = controller.compute_control(IDENTITY, np.array([0.0, 0.0, 0.001]))  # |Jw| = 0.06
    assert near == pytest.approx(-J @ np.array([0.0, 0.0, 0.001]) / 20.0)


def test_detumble_ignores_the_attitude():
    controller = _manager(detumble_entry_rate=1e-9, detumble_exit_rate=1e9)
    omega = np.array([0.05, -0.02, 0.03])
    q = np.array([0.5, 0.0, 0.0, np.sqrt(0.75)])
    assert np.array_equal(controller.compute_control(q, omega),
                          controller.compute_control(IDENTITY, omega))


def test_switch_happens_once_at_the_entry_rate():
    controller = _manager()
    for rate in (0.1, 0.05, 0.0101):
        controller.compute_control(IDENTITY, np.array([rate, 0.0, 0.0]))
        assert controller.phase == "detumble"
    controller.compute_control(IDENTITY, np.array([0.0099, 0.0, 0.0]))
    assert controller.phase == "acquisition" and controller.mode_switches == 1


def test_noisy_rate_inside_the_hysteresis_band_cannot_chatter():
    """A rate that wanders between the thresholds, as gyro noise does, switches at most once."""
    rng = np.random.default_rng(1)
    controller = _manager()
    for step in range(20000):
        rate = 0.01 + 0.004 * rng.standard_normal()  # around w_entry, far below w_exit
        controller.compute_control(IDENTITY, np.array([abs(rate), 0.0, 0.0]))
    assert controller.mode_switches == 1
    # A slew at the largest rate acquisition can command stays in acquisition.
    controller.compute_control(_slewed(), np.array([0.12, 0.0, 0.0]))
    assert controller.phase == "acquisition" and controller.mode_switches == 1


def _slewed():
    return np.array([np.sin(1.0), 0.0, 0.0, np.cos(1.0)])


def test_a_real_tumble_returns_to_detumble_and_back():
    controller = _manager()
    sequence = [0.005, 0.2, 0.05, 0.005]  # enter acquisition, tumble, damp, re-enter
    phases = []
    for rate in sequence:
        controller.compute_control(IDENTITY, np.array([0.0, rate, 0.0]))
        phases.append(controller.phase)
    assert phases == ["acquisition", "detumble", "detumble", "acquisition"]
    assert controller.mode_switches == 3


def test_acquisition_uses_the_rate_profile_law():
    manager = _manager()
    profile = AttitudeController(ControllerParams(
        K_omega=20.0, max_torque=0.1, control_type="rate_profile", inertia=J))
    q = np.array([0.3, -0.2, 0.1, np.sqrt(1 - 0.14)])
    omega = np.array([0.001, 0.002, -0.001])
    manager.compute_control(IDENTITY, np.zeros(3))  # enter acquisition
    assert np.allclose(manager.compute_control(q, omega), profile.compute_control(q, omega))


def test_feedforward_is_not_added_while_detumbling():
    controller = _manager(detumble_entry_rate=1e-9, detumble_exit_rate=1e9)
    omega = np.array([0.05, 0.0, 0.03])
    h_w = np.array([0.0, 5.0, 0.0])
    with_ff = controller.compute_control_with_feedforward(IDENTITY, omega, J, wheel_momentum=h_w)
    assert np.allclose(with_ff, controller.compute_control(IDENTITY, omega))


def test_switch_fires_on_the_estimated_rate_in_the_closed_loop():
    """The logged switch is the first step whose estimated rate is below w_entry."""
    base = load_config("config/closed_loop.yaml")
    seed = 7042
    conditions = sample_initial_conditions(seed, rate_rad_s=0.012)
    config = build_config(base, seed, conditions, ("attitude", "rate", "inertia"), 0.05)
    config = replace(config, duration=60.0, controller={
        **config.controller, "control_type": "two_phase", "detumble_entry_rate": 0.01,
        "detumble_exit_rate": 0.15, "detumble_time_constant_s": 20.0})
    results = run_simulation(config)
    phase = results["control_phase"]
    switch = int(np.argmax(phase))
    assert phase[switch] == 1 and np.all(phase[switch:] == 1)
    rate_hat = np.linalg.norm(results["omega_hat"], axis=1)
    rate_true = np.linalg.norm(results["omega_true"], axis=1)
    assert rate_hat[switch] <= 0.01 and np.all(rate_hat[:switch] > 0.01)
    # The true rate crosses at a different step: the decision uses the estimate.
    true_cross = int(np.argmax(rate_true <= 0.01))
    assert true_cross != switch


def test_low_pass_switching_ignores_single_noisy_samples():
    """One sample far below the threshold does not switch; a sustained low rate does."""
    controller = _manager(switch_filter_time_constant_s=1.0, detumble_entry_rate=0.009)
    for _ in range(300):
        controller.compute_control(IDENTITY, np.array([0.012, 0.0, 0.0]))
    controller.compute_control(IDENTITY, np.array([0.0, 0.0, 0.0]))  # one outlier sample
    assert controller.phase == "detumble"
    for step in range(1000):
        controller.compute_control(IDENTITY, np.array([0.008, 0.0, 0.0]))
        if controller.phase == "acquisition":
            break
    assert controller.phase == "acquisition" and controller.mode_switches == 1
    # First-order response from 0.012 toward 0.008 crosses 0.009 after about T_f ln 4 = 1.39 s.
    assert step + 1 == pytest.approx(139, abs=3)


def test_filtered_switching_signal_noise_matches_the_margin_calculation():
    """sigma_f = sigma sqrt(a / (2 - a)), a = dt / (T_f + dt): about 7.0e-5 for sigma 1e-3, T_f 1 s."""
    rng = np.random.default_rng(4)
    controller = _manager(switch_filter_time_constant_s=1.0, detumble_entry_rate=1e-9)
    values = []
    for k in range(60000):
        noise = 1e-3 * rng.standard_normal(3)
        controller.compute_control(IDENTITY, np.array([0.02, 0.0, 0.0]) + noise)
        if k > 1000:
            values.append(controller.switch_signal)
    alpha = 0.01 / 1.01
    expected = 1e-3 * np.sqrt(alpha / (2 - alpha))
    assert np.std(values) == pytest.approx(expected, rel=0.15)
    # Norm bias of a noisy vector: E|w + n| - |w| ~ sigma^2 (n - 1) / (2 |w|) = 5e-5 at 0.02.
    assert np.mean(values) - 0.02 == pytest.approx(1e-6 * 2 / (2 * 0.02), rel=0.3)


def test_acquisition_never_reaches_the_exit_threshold_near_j_min():
    """Worst case for the exit hysteresis: light inertia, a 179 deg slew about the minor
    axis, and the entry rate along that axis. The law must not return to detumble."""
    base = load_config("config/closed_loop.yaml")
    angle = np.deg2rad(179.0)
    light = 0.95 * np.diag([100.0, 80.0, 60.0])  # J_min = 57 kg m^2 about z
    config = replace(base, duration=200.0, spacecraft={
        **base.spacecraft, "initial_attitude": [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)],
        "initial_omega": [0.0, 0.0, 0.01]}, controller={
        **base.controller, "control_type": "two_phase", "K_omega": 10.0,
        "profile_rate_gain": 0.08, "profile_accel_fraction": 0.9,
        "detumble_entry_rate": 0.009, "detumble_exit_rate": 0.15,
        "detumble_time_constant_s": 20.0, "switch_filter_time_constant_s": 1.0})
    results = run_simulation(config, true_inertia=light)
    phase = results["control_phase"]
    changes = np.count_nonzero(np.diff(np.concatenate([[0], phase.astype(int)])))
    assert changes == 1
    peak_true = float(np.max(np.linalg.norm(results["omega_true"], axis=1)))
    assert peak_true < 0.15 and float(np.max(results["switch_signal"])) < 0.15
    # And it does slew fast, so the check is not vacuous.
    assert peak_true > 0.05
