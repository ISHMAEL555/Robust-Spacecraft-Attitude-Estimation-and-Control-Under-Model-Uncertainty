"""E6 feasibility: momentum bound, slew estimate, linear analysis, and their use on the 200 runs."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_e6_feasibility import analyse_run, run_inputs, summarise
from scripts.run_e6_monte_carlo import build_config, sample_initial_conditions
from scripts.run_experiment import load_config, run_simulation
from src.control.attitude_controller import compute_control_gains
from src.control.feasibility import (
    double_integrator_min_time,
    eigenaxis_feasibility_estimate_s,
    linear_closed_loop,
    momentum_lower_bound_s,
    principal_axis_linear_summary,
    slew_time_rest_to_rest_s,
    slow_pole_settling_estimate_s,
    step_settling_time_s,
)

ROOT = Path(__file__).resolve().parents[2]
J = np.diag([100.0, 80.0, 60.0])


# --- momentum bound -------------------------------------------------------------

def test_momentum_bound_values_and_the_per_wheel_version():
    omega = np.array([0.1, 0.0, 0.0])
    assert momentum_lower_bound_s(J, omega, 0.1) == pytest.approx(100.0)
    assert momentum_lower_bound_s(J, omega, 0.1, per_wheel_clips=True) == pytest.approx(
        100.0 / np.sqrt(3.0))
    assert momentum_lower_bound_s(J, np.zeros(3), 0.1) == 0.0


def test_momentum_bound_uses_the_momentum_not_the_rate():
    omega = np.array([0.05, 0.05, 0.05])
    expected = np.linalg.norm(J @ omega) / 0.1
    assert momentum_lower_bound_s(J, omega, 0.1) == pytest.approx(expected)


def _momentum_check(config, h_true):
    results = run_simulation(config)
    inertia = np.asarray(results["inertia_true_sampled"]
                         if "inertia_true_sampled" in results
                         else config.spacecraft["inertia"], dtype=float)
    omega = results["omega_true"]
    time = results["time"]
    # omega_true[i] is the state at the start of step i (time i dt); the true initial
    # momentum comes from the configured initial rate.
    body_momentum = np.linalg.norm(omega @ inertia.T, axis=1)
    h0 = np.linalg.norm(inertia @ np.asarray(config.spacecraft["initial_omega"], dtype=float))
    # The premise of version (a): the delivered body torque never exceeds tau_max.
    assert np.max(np.linalg.norm(results["tau_rw"], axis=1)) <= 0.1 + 1e-12
    return time, body_momentum, h0, results


@pytest.mark.parametrize("seed", [42, 1042, 2042, 5042])
def test_no_simulated_run_beats_the_momentum_bound(seed):
    """|J w(t)| >= |J w0| - tau t, for the proven (a) and the conservative (b) forms."""
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    conditions = sample_initial_conditions(seed)
    config = build_config(base, seed, conditions, ("attitude", "rate", "inertia"), 0.05)
    config = replace(config, duration=40.0)
    time, body_momentum, h0, _ = _momentum_check(config, None)
    for authority in (0.1, 0.1 * np.sqrt(3.0)):  # version (a), version (b)
        assert np.all(body_momentum + authority * time >= h0 - 1e-9)


def test_an_aggressive_braking_controller_still_cannot_beat_the_bound():
    """Truth feedback with a huge rate gain keeps the command saturated against the rate."""
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    seed = 3042
    conditions = sample_initial_conditions(seed)
    config = build_config(base, seed, conditions, ("attitude", "rate", "inertia"), 0.05)
    config = replace(config, duration=60.0, controller={
        **config.controller, "state_feedback": "truth", "K_omega": 1e6, "K_q": 1e-6})
    time, body_momentum, h0, results = _momentum_check(config, None)
    assert np.all(body_momentum + 0.1 * time >= h0 - 1e-9)
    # It does brake: the body momentum falls, and the wheels hold the difference.
    assert body_momentum[-1] < 0.7 * h0
    # Total momentum is conserved: |J w + h_w| stays at |J w0|.
    inertia = np.asarray(results["inertia_true_sampled"], dtype=float)
    # wheel_momentum[i] is logged at the end of step i, omega_true[i] at its start:
    # pair omega[i + 1] with wheel_momentum[i] (the same instant).
    total = np.linalg.norm(results["omega_true"][1:] @ inertia.T
                           + results["wheel_momentum"][:-1], axis=1)
    assert np.allclose(total, h0, rtol=1e-9)


# --- slew time and double integrator ---------------------------------------------

def test_rest_to_rest_slew_time_closed_form():
    assert slew_time_rest_to_rest_s(np.deg2rad(60.0), 100.0, 0.1) == pytest.approx(
        2.0 * np.sqrt(np.deg2rad(60.0) * 1000.0))
    assert slew_time_rest_to_rest_s(0.0, 100.0, 0.1) == 0.0


def _bang_bang_time(x0, v0, accel, dt=1e-3):
    """Integrate the switching-curve feedback until the state is within a small ball."""
    x, v, t = x0, v0, 0.0
    while abs(x) > 1e-4 * max(1.0, abs(x0)) or abs(v) > 1e-4 * accel * 100:
        s = x + v * abs(v) / (2.0 * accel)
        u = -accel * np.sign(s) if abs(s) > 1e-9 else -accel * np.sign(v)
        v += u * dt
        x += v * dt
        t += dt
        if t > 5000.0:
            raise AssertionError("did not converge")
    return t


@pytest.mark.parametrize("x0,v0", [(1.0, 0.0), (1.0, 0.1), (-2.0, 0.3), (1.0, -0.4), (5.0, -0.5)])
def test_double_integrator_min_time_matches_a_switching_curve_simulation(x0, v0):
    accel = 0.05
    formula = double_integrator_min_time(x0, v0, accel)
    assert _bang_bang_time(x0, v0, accel, dt=2e-3) == pytest.approx(formula, rel=0.02, abs=0.5)


def test_feasibility_estimate_takes_the_wrap_around_target_when_it_is_nearer():
    theta, rate = np.deg2rad(60.0), 0.1
    accel = 0.1 / 100.0
    direct = double_integrator_min_time(theta, rate, accel)
    wrapped = double_integrator_min_time(theta - 2.0 * np.pi, rate, accel)
    assert wrapped < direct
    assert eigenaxis_feasibility_estimate_s(theta, rate, 100.0, 0.1) == pytest.approx(wrapped)
    # The same rate in the other direction cannot use the shortcut.
    assert eigenaxis_feasibility_estimate_s(theta, -rate, 100.0, 0.1) > direct * 0.5
    assert eigenaxis_feasibility_estimate_s(theta, 0.0, 100.0, 0.1) == pytest.approx(
        slew_time_rest_to_rest_s(theta, 100.0, 0.1))


def test_estimate_is_consistent_with_the_stored_benchmark_trajectory():
    """The benchmark (rate +x along the +60 deg error) goes the long way round in about 102 s;
    the shipped linear gains take 174.5 s, which is above the estimate, as it must be."""
    estimate = eigenaxis_feasibility_estimate_s(np.deg2rad(60.0), 0.1, 100.0, 0.1)
    assert estimate == pytest.approx(102.35, abs=0.05)
    benchmark = json.loads((ROOT / "docs" / "e6_project_benchmark.json").read_text())
    assert benchmark["benchmark"]["checks"]["settling_time_2pct_s"]["measured"] > estimate


# --- linear closed loop ------------------------------------------------------------

def test_linear_analysis_of_the_e6_gains_on_the_x_axis():
    poles = linear_closed_loop(100.0, 1.0, 20.0)
    assert poles["stiffness"] == pytest.approx(0.5)  # K_q / 2 for quaternion_feedback
    assert poles["damping_ratio"] == pytest.approx(20.0 / (2.0 * np.sqrt(0.5 * 100.0)))
    assert poles["slow_pole"] == pytest.approx(-0.02929, abs=1e-5)
    assert poles["fast_pole"] == pytest.approx(-0.17071, abs=1e-5)
    # The poles satisfy the characteristic equation J s^2 + K_w s + k = 0.
    for pole in (poles["slow_pole"], poles["fast_pole"]):
        assert 100.0 * pole**2 + 20.0 * pole + 0.5 == pytest.approx(0.0, abs=1e-9)
    assert slow_pole_settling_estimate_s(poles["slow_pole"]) == pytest.approx(133.56, abs=0.01)


@pytest.mark.parametrize("control_type", ["quaternion_feedback", "pd"])
@pytest.mark.parametrize("zeta,settling", [(0.8, 60.0), (1.0, 100.0)])
def test_designed_gains_reproduce_the_requested_damping_and_frequency(control_type, zeta, settling):
    """Checks the factor of 2 against compute_control_gains for both control laws."""
    j = np.mean(np.diag(J))
    k_q, k_omega = compute_control_gains(settling, zeta, J, control_type)
    poles = linear_closed_loop(j, k_q, k_omega, control_type)
    assert poles["damping_ratio"] == pytest.approx(zeta)
    assert poles["natural_frequency_rad_s"] == pytest.approx(4.0 / (zeta * settling))


def test_exact_step_settling_agrees_with_a_numerical_simulation():
    j, k_q, k_omega, theta0 = 100.0, 1.0, 20.0, 1.0
    exact = step_settling_time_s(j, k_q, k_omega, theta0=theta0)
    dt = 0.01
    theta, rate, last_outside = theta0, 0.0, 0.0
    for step in range(int(400 / dt)):
        acc = (-k_omega * rate - 0.5 * k_q * theta) / j
        rate += acc * dt
        theta += rate * dt
        if abs(theta) > 0.02 * theta0:
            last_outside = (step + 1) * dt
    assert exact == pytest.approx(last_outside, abs=0.3)
    assert exact > slow_pole_settling_estimate_s(linear_closed_loop(j, k_q, k_omega)["slow_pole"])


def test_principal_axis_summary_covers_all_three_axes():
    rows = principal_axis_linear_summary(J, 1.0, 20.0)
    assert [r["axis"] for r in rows] == ["x", "y", "z"]
    assert rows[0]["damping_ratio"] == pytest.approx(1.4142, abs=1e-4)
    assert all(r["damping_ratio"] > 1.0 for r in rows)  # all overdamped
    assert rows[0]["settling_exact_from_rest_s"] == pytest.approx(140.0, abs=0.2)


# --- application to the 200 runs -----------------------------------------------------

def test_rebuilt_dispersion_matches_the_simulations_own_draw():
    """analyse_run must use the very inertia and initial conditions the Monte Carlo used."""
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    seed = 4042
    inputs = run_inputs(seed, np.array(base.spacecraft["inertia"], dtype=float))
    config = build_config(base, seed, inputs["conditions"], ("attitude", "rate", "inertia"), 0.05)
    results = run_simulation(replace(config, duration=1.0))
    assert np.array_equal(results["inertia_true_sampled"], inputs["inertia"])
    assert np.array_equal(results["q_true"][0], inputs["conditions"]["quaternion"])
    assert np.array_equal(results["omega_true"][0], inputs["conditions"]["rate"])


def test_benchmark_like_run_reproduces_the_hand_numbers():
    """With the benchmark's own initial condition the analysis returns the closed forms."""
    from scripts import analyze_e6_feasibility as module
    base = np.diag([100.0, 80.0, 60.0])
    row = module.analyse_run(42, base, 0.1, 1.0, 20.0)
    assert 0.0 < row["feasibility_estimate_s"] < 240.0
    assert row["momentum_bound_b_s"] == pytest.approx(row["momentum_bound_a_s"] / np.sqrt(3.0))
    assert row["slew_rest_to_rest_s"] == pytest.approx(
        2.0 * np.sqrt(np.deg2rad(row["angle_deg"]) * row["j_eff_kg_m2"] / 0.1))


def test_summary_counts_are_consistent():
    rows = [
        {"momentum_bound_a_s": 50.0, "momentum_bound_b_s": 29.0, "slew_rest_to_rest_s": 60.0,
         "feasibility_estimate_s": est, "linear_unsaturated_settling_s": lin,
         "saturated_single_axis_settling_s": lin, "stop_then_slew_plan_s": est + 30.0}
        for est, lin in ((100.0, 150.0), (100.0, 250.0), (200.0, 250.0), (200.0, None))
    ]
    summary = summarise(rows, [True, False, False, False])
    assert summary["feasibility_estimate_above_limit"] == 2
    assert summary["feasible_by_estimate_but_linear_gains_exceed_limit"] == 1
    assert summary["feasible_by_estimate_and_linear_gains_within_limit"] == 1
    assert summary["infeasible_by_estimate_and_linear_gains_exceed_limit"] == 2
    assert summary["monte_carlo_passes_among_feasible_by_estimate"] == 1
    assert summary["momentum_bound_a_above_limit"] == 0


# --- saturated single-axis model and the constructive plan -------------------------------

def test_saturated_model_reduces_to_the_linear_response_when_nothing_saturates():
    from src.control.feasibility import saturated_single_axis_settling_s
    # A 0.05 rad step needs about K_q/2 * 0.05 = 0.025 N m, far below a 10 N m limit,
    # and sin(theta / 2) is linear to 1e-4 there.
    linear = step_settling_time_s(100.0, 1.0, 20.0, theta0=0.05)
    saturated = saturated_single_axis_settling_s(0.05, 0.0, 100.0, 1.0, 20.0, tau_max=10.0)
    assert saturated == pytest.approx(linear, abs=1.0)


def test_saturation_lengthens_a_large_slew():
    from src.control.feasibility import saturated_single_axis_settling_s
    linear = step_settling_time_s(100.0, 1.0, 20.0, theta0=np.deg2rad(60.0))
    saturated = saturated_single_axis_settling_s(np.deg2rad(60.0), 0.0, 100.0, 1.0, 20.0, 0.1)
    assert saturated > linear + 5.0  # 140.0 s linear against about 148.8 s saturated, from rest


def test_stop_then_slew_plan_is_tight_against_the_momentum_bound_and_consistent():
    from src.control.feasibility import stop_then_slew_plan
    omega = np.array([0.04, -0.07, 0.05])
    q0 = np.array([0.5, 0.0, 0.0, np.sqrt(0.75)])
    plan = stop_then_slew_plan(J, q0, omega, 0.1)
    bound = momentum_lower_bound_s(J, omega, 0.1)
    # The plan stops at rate_tol, so it may finish up to |J| rate_tol / tau early.
    assert plan["stop_s"] >= bound - 100.0 * 1e-4 / 0.1 - 1e-9
    assert plan["stop_s"] <= 1.15 * bound
    assert plan["total_s"] == pytest.approx(plan["stop_s"] + plan["slew_s"])


def test_stop_then_slew_plan_without_a_rate_is_the_rest_to_rest_slew():
    from src.control.feasibility import stop_then_slew_plan
    q0 = np.array([0.5, 0.0, 0.0, np.sqrt(0.75)])
    plan = stop_then_slew_plan(J, q0, np.zeros(3), 0.1)
    assert plan["stop_s"] == 0.0
    assert plan["total_s"] == pytest.approx(slew_time_rest_to_rest_s(np.deg2rad(60.0), 100.0, 0.1))


def test_demonstration_selection_is_unique_ordered_and_only_estimate_feasible():
    from scripts.analyze_e6_feasibility import demonstration_indices
    rows = [{"feasibility_estimate_s": 100.0 + (50.0 if i % 7 == 0 else 0.0) * (i > 30),
             "stop_then_slew_plan_s": 120.0 + i} for i in range(60)]
    rows[0]["feasibility_estimate_s"] = 300.0  # infeasible by the estimate
    picks = demonstration_indices(rows, 10)
    assert len(picks) == len(set(picks)) == 10
    assert all(rows[i]["feasibility_estimate_s"] <= 180.0 for i in picks)
    assert [rows[i]["stop_then_slew_plan_s"] for i in picks] == sorted(
        rows[i]["stop_then_slew_plan_s"] for i in picks)
