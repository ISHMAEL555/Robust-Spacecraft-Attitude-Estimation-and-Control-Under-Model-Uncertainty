"""B5 comparison: scenario initial states and aggregation."""

import numpy as np
import pytest

from scripts.run_b5_comparison import aggregate, scenario_initial_state
from scripts.run_e6_monte_carlo import sample_initial_conditions
from src.control.feasibility import stop_then_slew_plan
from src.dynamics.quaternion import error_quat_to_vec, right_error_quat

J = np.diag([100.0, 80.0, 60.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _angle_deg(q):
    return float(np.rad2deg(np.linalg.norm(error_quat_to_vec(right_error_quat(q, IDENTITY)))))


def test_envelope_matches_the_e6_monte_carlo_draw():
    state = scenario_initial_state("envelope", 20000042, J, 0.05, 0.01, 0.1)
    c = sample_initial_conditions(20000042, 60.0, None, 0.1)
    assert np.array_equal(state["quaternion"], c["quaternion"])
    assert np.array_equal(state["rate"], c["rate"])


def test_acquisition_starts_at_the_entry_rate_and_60_degrees():
    state = scenario_initial_state("acq_axis60", 20001042, J, 0.05, 0.01, 0.1)
    assert np.linalg.norm(state["rate"]) == pytest.approx(0.01)
    assert _angle_deg(state["quaternion"]) == pytest.approx(60.0)


def test_drifted_acquisition_starts_where_the_stopped_envelope_draw_ends():
    seed = 20002042
    state = scenario_initial_state("acq_drifted", seed, J, 0.05, 0.01, 0.1)
    c = sample_initial_conditions(seed, 60.0, None, 0.1)
    stop = stop_then_slew_plan(state["inertia"], c["quaternion"], c["rate"], 0.1)
    assert np.allclose(state["quaternion"], stop["q_after_stop"])
    assert np.allclose(state["rate"], 0.01 * c["rate_direction"])
    assert _angle_deg(state["quaternion"]) == pytest.approx(stop["angle_after_stop_deg"])


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        scenario_initial_state("other", 42, J, 0.05, 0.01, 0.1)


def _run(passed, det, settle, switches=1):
    checks = {"settling_time_2pct_s": passed, "peak_commanded_torque_nm": True,
              "peak_wheel_torque_nm": True, "peak_wheel_momentum_utilization": True}
    return {"passed": passed, "checks": checks, "settling_time_s": settle,
            "detumble_time_s": det, "mode_switch_time_s": det, "mode_switches": switches,
            "optimality_gap_ratio": None if settle is None else settle / 150.0,
            "plan_s": 150.0, "peak_wheel_momentum_use": 0.4, "peak_commanded_torque_nm": 0.1,
            "saturation_duty": 0.5, "peak_terminal_error_deg": 1.0}


def test_aggregate_counts_detumble_and_e6_passes_separately():
    runs = ([_run(True, 90.0, 160.0)] * 4 + [_run(False, 160.0, None)]
            + [_run(False, None, None, switches=0)])
    s = aggregate(runs, det_limit_s=150.0)
    assert s["e6_all_criteria"]["passes"] == 4 and s["e6_all_criteria"]["n"] == 6
    assert s["detumble_within_limit"]["passes"] == 4
    assert s["mode_switches"] == {"max": 1, "runs_with_more_than_one": 0, "runs_with_none": 1}
    assert s["settling_time_s"]["n_none"] == 2
    assert s["optimality_gap_ratio"]["median"] == pytest.approx(160.0 / 150.0)
