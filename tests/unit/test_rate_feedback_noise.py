"""The gyro-driven filter's rate estimate carries the raw per-sample gyro noise into the controller."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.run_experiment import load_config, run_simulation


@pytest.fixture(scope="module")
def steady_rate_error():
    config = load_config("config/nominal.yaml")
    assert config.filter["variant"] == "gyro_driven"
    results = run_simulation(replace(config, duration=60.0))
    error = results["omega_hat"] - results["omega_true"]
    return error[int(10.0 / config.dt):], config


def test_rate_error_rms_is_sigma_v_times_sqrt_3_over_dt(steady_rate_error):
    """sigma_v / sqrt(dt) = 1e-4 / sqrt(0.01) = 1e-3 rad/s per axis; times sqrt(3) for the norm."""
    error, config = steady_rate_error
    expected = config.gyroscope["sigma_v"] * np.sqrt(3.0 / config.dt)  # 1.732e-3 rad/s
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    # The samples are white (next test), so the RMS of 3N squared components has a
    # relative standard error of about 1 / sqrt(2 * 3N).
    relative_se = 1.0 / np.sqrt(2.0 * error.size)
    assert abs(rms / expected - 1.0) < 4.0 * relative_se
    assert error.std(axis=0) == pytest.approx(np.full(3, 1e-3), rel=0.05)


def test_rate_error_is_white_and_unbiased(steady_rate_error):
    """Nothing filters omega_hat: sample-to-sample correlation is zero, bias is negligible."""
    error, _ = steady_rate_error
    bound = 4.0 / np.sqrt(len(error))
    for axis in range(3):
        lag1 = np.corrcoef(error[:-1, axis], error[1:, axis])[0, 1]
        assert abs(lag1) < bound
    assert np.all(np.abs(error.mean(axis=0)) < 0.05 * 1e-3)
