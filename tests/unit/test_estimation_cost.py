"""Statistics helpers of scripts/verify_estimation_cost.py."""

import numpy as np
import pytest
from scipy import stats

from scripts.run_experiment import decomposition_settings
from scripts.verify_estimation_cost import MODES, summarize, t_interval


def test_t_interval_matches_scipy():
    values = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    result = t_interval(values)
    expected = stats.t.interval(0.95, 4, loc=values.mean(), scale=stats.sem(values))
    assert result["mean"] == pytest.approx(3.0)
    assert result["ci95"] == pytest.approx(list(expected))
    assert t_interval(np.array([1.0]))["ci95"] is None


def _metrics(ape, fraction=1.0):
    block = {"ape": {"rms_rad": np.deg2rad(ape), "p99_rad": np.deg2rad(ape)},
             "rpe": {"rms_rad": np.deg2rad(ape), "p99_rad": np.deg2rad(ape)}}
    return {"window_source": "config", "window_start_s": 1.0,
            "pointing": block, "control": block, "estimation": block,
            "rate_estimation_error": {"rms_rad_s": 1e-3},
            "fraction_within_requirement": fraction}


def test_summarize_reports_paired_increment_per_seed():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.01, 0.02, 6)
    offset = rng.uniform(0.04, 0.05, 6)
    records = []
    for variant in ("gyro_driven", "model_aided"):
        for i in range(6):
            records.append({"name": "X", "variant": variant, "mode": "truth",
                            "seed_index": i, "metrics": _metrics(truth[i])})
            records.append({"name": "X", "variant": variant, "mode": "estimated",
                            "seed_index": i, "metrics": _metrics(truth[i] + offset[i])})
    increment = summarize(records, ["X"])["X"]["gyro_driven"][
        "increment_estimated_minus_truth"]["ape_rms_deg"]
    assert increment["mean"] == pytest.approx(offset.mean())
    assert increment["n"] == 6
    # Pairing matters: the interval is for the paired difference, not the
    # difference of two independent intervals.
    assert increment["ci95"][0] > 0.0


def test_decomposition_settings_defaults_and_overrides():
    assert decomposition_settings({}) == {
        "steady_state_start_s": None, "rpe_window_s": 10.0,
        "pointing_requirement_deg": 0.1}
    custom = decomposition_settings({"steady_state_start_s": 50, "rpe_window_s": 5,
                                     "pointing_requirement_deg": 0.2})
    assert custom["steady_state_start_s"] == 50 and custom["rpe_window_s"] == 5.0
    assert MODES == ("estimated", "truth")
