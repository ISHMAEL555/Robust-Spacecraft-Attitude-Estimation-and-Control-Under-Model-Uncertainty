"""E6 Monte Carlo: dispersion samplers, binomial intervals, aggregation, reproducibility."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from scripts.run_e6_monte_carlo import (
    aggregate,
    clopper_pearson,
    distribution_summary,
    run_one,
    sample_initial_conditions,
)
from src.dynamics.quaternion import error_quat_to_vec, right_error_quat

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- samplers -----------------------------------------------------------------

def test_attitude_has_exact_magnitude_and_unit_quaternion():
    for seed in range(0, 50000, 1000):
        c = sample_initial_conditions(seed)
        assert np.linalg.norm(c["quaternion"]) == pytest.approx(1.0, abs=1e-14)
        error = error_quat_to_vec(right_error_quat(c["quaternion"], np.array([0, 0, 0, 1.0])))
        assert np.linalg.norm(error) == pytest.approx(np.deg2rad(60.0), abs=1e-12)
        assert np.linalg.norm(c["attitude_axis"]) == pytest.approx(1.0)


def test_rate_has_exact_magnitude():
    for seed in range(0, 20000, 1000):
        assert np.linalg.norm(sample_initial_conditions(seed)["rate"]) == pytest.approx(0.1)


def test_axes_and_rate_directions_are_isotropic():
    axes, directions = [], []
    for i in range(4000):
        c = sample_initial_conditions(42 + 1000 * i)
        axes.append(c["attitude_axis"])
        directions.append(c["rate_direction"])
    for sample in (np.array(axes), np.array(directions)):
        # Uniform on the sphere: zero mean; E[v v^T] = I/3. Standard error of a
        # mean component is sqrt(1/3 / n) ~ 0.0091.
        assert np.all(np.abs(sample.mean(axis=0)) < 4 * 0.0091)
        assert np.allclose(sample.T @ sample / len(sample), np.eye(3) / 3, atol=0.02)


def test_attitude_axis_and_rate_direction_come_from_independent_streams():
    pairs = [sample_initial_conditions(42 + 1000 * i) for i in range(2000)]
    a = np.array([p["attitude_axis"][0] for p in pairs])
    r = np.array([p["rate_direction"][0] for p in pairs])
    assert abs(np.corrcoef(a, r)[0, 1]) < 4 / np.sqrt(len(pairs))


def test_dispersed_magnitude_mode_covers_the_requested_range():
    angles = np.array([
        sample_initial_conditions(42 + 1000 * i, attitude_range_deg=(30.0, 90.0))
        ["attitude_angle_rad"] for i in range(2000)])
    degrees = np.rad2deg(angles)
    assert degrees.min() >= 30.0 and degrees.max() <= 90.0
    assert degrees.mean() == pytest.approx(60.0, abs=2.0)
    assert degrees.std() == pytest.approx(60.0 / np.sqrt(12.0) * 1.0, abs=1.5)


def test_run_draws_are_deterministic_and_depend_only_on_the_run_seed():
    a = sample_initial_conditions(1042)
    b = sample_initial_conditions(1042)
    other = sample_initial_conditions(2042)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["quaternion"], other["quaternion"])
    # Not affected by how many runs precede it (no shared state).
    for seed in (42, 3042, 5042):
        sample_initial_conditions(seed)
    assert np.array_equal(sample_initial_conditions(1042)["rate"], a["rate"])


# --- Clopper-Pearson ----------------------------------------------------------

@pytest.mark.parametrize("successes,trials", [(0, 200), (200, 200), (1, 200), (150, 200),
                                              (37, 200), (5, 10)])
def test_clopper_pearson_matches_scipy_exact_interval(successes, trials):
    expected = stats.binomtest(successes, trials).proportion_ci(
        confidence_level=0.95, method="exact")
    low, high = clopper_pearson(successes, trials)
    assert low == pytest.approx(expected.low, abs=1e-12)
    assert high == pytest.approx(expected.high, abs=1e-12)


def test_clopper_pearson_boundary_closed_forms():
    assert clopper_pearson(0, 200)[0] == 0.0
    assert clopper_pearson(0, 200)[1] == pytest.approx(1 - 0.025 ** (1 / 200))
    assert clopper_pearson(200, 200)[1] == 1.0
    assert clopper_pearson(200, 200)[0] == pytest.approx(0.025 ** (1 / 200))


# --- aggregation --------------------------------------------------------------

def _run(i, failing=(), settling=100.0, rate_components=(0.5, 0.5, 0.7), cos=0.3):
    names = ["settling_time_2pct_s", "settling_hold_duration_s", "pointing_peak_final_20s_rad",
             "peak_wheel_momentum_utilization"]
    checks = {n: n not in failing for n in names}
    return {"index": i, "seed": i, "checks": checks, "measured": {**{n: 0.0 for n in names},
                         "final_angular_rate_norm_rad_s": 1e-4},
            "passed": all(checks.values()), "settling_time_s": settling,
            "peak_terminal_error_deg": 1.0, "peak_wheel_momentum_use": 0.5,
            "rate_component_on_principal_axes": list(rate_components),
            "rate_vs_attitude_axis_cos": cos, "rate_vs_attitude_axis_signed_cos": cos,
            "attitude_angle_deg": 60.0,
            "principal_moments": [60, 80, 100], "inertia_rejections": 0}


def test_aggregate_counts_and_most_frequent_failure():
    runs = ([_run(i, failing=("settling_time_2pct_s", "settling_hold_duration_s"),
                  settling=190.0) for i in range(30)]
            + [_run(30 + i, failing=("peak_wheel_momentum_utilization",)) for i in range(5)]
            + [_run(35 + i) for i in range(65)])
    report = aggregate(runs)
    assert report["overall"]["passes"] == 65 and report["overall"]["fails"] == 35
    assert report["criteria"]["settling_time_2pct_s"]["fails"] == 30
    assert report["criteria"]["peak_wheel_momentum_utilization"]["fails"] == 5
    assert set(report["most_frequent_failing_criteria"]) == {
        "settling_time_2pct_s", "settling_hold_duration_s"}
    low, high = report["overall"]["ci95_clopper_pearson"]
    assert low < 0.65 < high
    assert report["distributions"]["settling_time_s"]["median"] == pytest.approx(100.0)


def test_aggregate_with_no_failures_reports_none():
    report = aggregate([_run(i) for i in range(20)])
    assert report["most_frequent_failing_criteria"] == []
    assert report["overall"]["p_pass"] == 1.0
    assert report["overall"]["ci95_clopper_pearson"][1] == 1.0


def test_rate_direction_analysis_finds_a_planted_dependence():
    rng = np.random.default_rng(0)
    runs = []
    for i in range(200):
        component = float(rng.uniform())
        failing = ("settling_time_2pct_s",) if component > 0.6 else ()
        runs.append(_run(i, failing=failing, settling=150.0 + 60.0 * component,
                         rate_components=(component, 0.1, 0.1)))
    analysis = aggregate(runs)["rate_direction_analysis"]
    assert analysis["most_frequent_failing_criterion"] == "settling_time_2pct_s"
    targets = analysis["targets"]
    planted = targets["failure_of_settling_time_2pct_s"]["|rate . minor_axis|"]
    assert planted["spearman_rho"] > 0.7 and planted["p_value"] < 1e-10
    assert targets["settling_time"]["|rate . minor_axis|"]["spearman_rho"] > 0.95
    # A constant covariate has no rank correlation; a constant target neither.
    assert targets["overall_failure"]["|rate . attitude_axis|"]["spearman_rho"] is None
    assert targets["peak_terminal_error_deg"]["|rate . minor_axis|"]["spearman_rho"] is None


def test_unsettled_runs_sort_last_and_are_reported_as_none():
    values = np.array([100.0] * 80 + [float("inf")] * 20)
    summary = distribution_summary(values)
    assert summary["median"] == pytest.approx(100.0)
    assert summary["p95"] is None and summary["n_not_finite"] == 20


# --- reproducibility ----------------------------------------------------------

def _task(index, **overrides):
    task = {"index": index, "seed": 42 + 1000 * index, "attitude_deg": 60.0,
            "attitude_range_deg": None, "rate_rad_s": 0.1,
            "dispersions": ["attitude", "rate", "inertia"], "inertia_sigma": 0.05,
            "duration": 5.0}
    task.update(overrides)
    return task


def test_runs_do_not_depend_on_worker_count_or_order():
    tasks = [_task(i) for i in range(4)]
    serial = [run_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(run_one, tasks))
    reversed_order = [run_one(t) for t in reversed(tasks)][::-1]
    assert json.dumps(serial) == json.dumps(parallel) == json.dumps(reversed_order)


def test_with_dispersions_off_run_matches_the_e6_benchmark():
    """Same config, seed 42, no dispersions: identical to evaluate_e6_benchmark.py."""
    run = run_one(_task(0, dispersions=[], duration=None))
    benchmark = json.loads((PROJECT_ROOT / "docs" / "e6_project_benchmark.json").read_text())
    expected = benchmark["benchmark"]["checks"]
    assert run["passed"] and benchmark["benchmark"]["passed"]
    assert run["measured"]["settling_time_2pct_s"] == pytest.approx(
        expected["settling_time_2pct_s"]["measured"])
    assert run["measured"]["peak_wheel_momentum_utilization"] == pytest.approx(
        expected["peak_wheel_momentum_utilization"]["measured"])
