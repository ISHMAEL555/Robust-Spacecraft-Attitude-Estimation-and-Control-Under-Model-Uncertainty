"""Per-seed truth-inertia sampling from the filter prior."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.run_experiment import load_config, run_monte_carlo, run_simulation
from src.dynamics.inertia_sampling import (
    inertia_element_scales,
    inertia_from_relative_errors,
    is_physically_realisable,
    sample_inertia_from_prior,
)
from src.estimation.mekf import MEKF, MEKFParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOMINAL = np.diag([100.0, 80.0, 60.0])


def _model_aided_config(mode="sampled_from_prior", sigma=0.05, duration=None, seed=42):
    config = load_config(str(PROJECT_ROOT / "config" / "model_uncertainty.yaml"))
    return replace(
        config,
        seed=seed,
        duration=config.dt if duration is None else duration,
        spacecraft={**config.spacecraft, "inertia_truth_mode": mode},
        filter={**config.filter, "variant": "model_aided",
                "inertia_relative_sigma": sigma},
    )


# --- realisability ------------------------------------------------------------

def test_realisability_checks():
    assert is_physically_realisable(NOMINAL)
    assert not is_physically_realisable(np.diag([10.0, 1.0, 1.0]))   # triangle violated
    assert not is_physically_realisable(np.diag([1.0, 1.0, -1.0]))   # not PD
    assert not is_physically_realisable(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 1.0]]))  # not symmetric
    assert is_physically_realisable(np.diag([2.0, 1.0, 1.0]))        # equality is allowed


def test_scales_and_basis_match_the_filter_parameterisation():
    params = MEKFParams(variant="model_aided", inertia_model=NOMINAL,
                        inertia_relative_sigma=0.05)
    mekf = MEKF(params)
    assert np.array_equal(inertia_element_scales(NOMINAL), mekf._inertia_scales)
    # A relative error vector maps to the same tensor the filter would inject.
    delta = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
    expected = NOMINAL + sum(
        d * s * basis for d, s, basis in
        zip(delta, mekf._inertia_scales, mekf._inertia_basis))
    assert np.allclose(inertia_from_relative_errors(NOMINAL, delta), expected)


# --- sampler ------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.05, 0.25, 0.5])
def test_samples_are_spd_and_satisfy_triangle_inequality(sigma):
    rng = np.random.default_rng(3)
    for _ in range(2000):
        inertia, _ = sample_inertia_from_prior(NOMINAL, sigma, rng)
        assert np.allclose(inertia, inertia.T)
        moments = np.linalg.eigvalsh(inertia)
        assert moments[0] > 0.0
        assert moments[0] + moments[1] >= moments[2]


def test_unrejected_samples_follow_the_prior():
    rng = np.random.default_rng(4)
    sigma, n = 0.05, 20000
    errors = np.empty((n, 6))
    rejections = 0
    scales = inertia_element_scales(NOMINAL)
    for k in range(n):
        inertia, rejected = sample_inertia_from_prior(NOMINAL, sigma, rng)
        rejections += rejected
        errors[k] = np.array([inertia[0, 0] - 100.0, inertia[1, 1] - 80.0,
                              inertia[2, 2] - 60.0, inertia[0, 1], inertia[0, 2],
                              inertia[1, 2]]) / scales
    assert rejections == 0  # 5% is far from the realisability boundary
    assert np.all(np.abs(errors.mean(axis=0)) < 4 * sigma / np.sqrt(n))
    assert np.allclose(errors.std(axis=0), sigma, rtol=0.04)


def test_large_sigma_rejects_and_reports_the_count():
    rng = np.random.default_rng(5)
    total = sum(sample_inertia_from_prior(NOMINAL, 0.5, rng)[1] for _ in range(500))
    assert total > 0
    rng = np.random.default_rng(5)
    accepted_sigma_small = [sample_inertia_from_prior(NOMINAL, 0.5, rng)[1]
                            for _ in range(5)]
    assert all(isinstance(count, int) for count in accepted_sigma_small)


def test_sampler_is_deterministic_and_streams_differ():
    first = sample_inertia_from_prior(NOMINAL, 0.1, np.random.default_rng(7))
    again = sample_inertia_from_prior(NOMINAL, 0.1, np.random.default_rng(7))
    other = sample_inertia_from_prior(NOMINAL, 0.1, np.random.default_rng(8))
    assert np.array_equal(first[0], again[0]) and first[1] == again[1]
    assert not np.array_equal(first[0], other[0])


def test_zero_sigma_returns_nominal_and_unrealisable_sigma_raises():
    inertia, rejections = sample_inertia_from_prior(NOMINAL, 0.0, np.random.default_rng(0))
    assert np.array_equal(inertia, NOMINAL) and rejections == 0
    with pytest.raises(RuntimeError):
        sample_inertia_from_prior(NOMINAL, 50.0, np.random.default_rng(0), max_attempts=20)
    with pytest.raises(ValueError):
        sample_inertia_from_prior(NOMINAL, -0.1, np.random.default_rng(0))


# --- runner integration -------------------------------------------------------

def test_fixed_mode_is_the_default_and_unchanged():
    default = load_config(str(PROJECT_ROOT / "config" / "model_uncertainty.yaml"))
    default = replace(default, duration=2.0,
                      filter={**default.filter, "variant": "model_aided"})
    explicit = replace(default, spacecraft={**default.spacecraft,
                                            "inertia_truth_mode": "fixed"})
    a, b = run_simulation(default), run_simulation(explicit)
    for key in ("q_true", "omega_true", "q_hat", "NEES", "inertia_hat"):
        assert np.array_equal(a[key], b[key]), key
    assert np.array_equal(a["inertia_true"], NOMINAL)
    assert "inertia_sampling" not in a


def test_invalid_mode_and_conflicting_true_inertia_are_rejected():
    with pytest.raises(ValueError, match="inertia_truth_mode"):
        run_simulation(_model_aided_config(mode="random"))
    with pytest.raises(ValueError, match="true_inertia"):
        run_simulation(_model_aided_config(), true_inertia=NOMINAL * 1.1)


def test_sampled_mode_changes_only_the_truth_inertia_stream():
    sampled = run_simulation(_model_aided_config(duration=0.5))
    fixed = run_simulation(_model_aided_config(mode="fixed", duration=0.5))
    # Gyro bias walk and the initial filter errors share their streams.
    assert np.array_equal(sampled["b_true"], fixed["b_true"])
    assert not np.array_equal(sampled["inertia_true"], fixed["inertia_true"])
    assert sampled["inertia_sampling"]["attempts"] == 1


def test_run_result_does_not_depend_on_other_runs_or_order():
    """No global state: one seed gives the same inertia alone or inside a campaign."""
    alone = run_simulation(_model_aided_config(seed=1042, duration=1.0))
    results = run_monte_carlo(_model_aided_config(seed=42, duration=1.0), 3)
    assert results["inertia_sampling"]["runs"] == 3
    again = run_simulation(_model_aided_config(seed=1042, duration=1.0))
    assert np.array_equal(alone["inertia_true"], again["inertia_true"])


def _worker(seed):
    return run_simulation(_model_aided_config(seed=seed, duration=0.5))["inertia_true"]


def test_sampled_inertia_is_independent_of_worker_count():
    seeds = [42 + 1000 * i for i in range(4)]
    serial = [_worker(seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(_worker, seeds))
    for a, b in zip(serial, parallel):
        assert np.array_equal(a, b)


# --- initial NEES: the point of the feature (extends EST-INIT-001) ------------

def _initial_nees_groups(mode, sigma=0.05, n=200):
    total, inertia_group = [], []
    for seed in range(n):
        r = run_simulation(_model_aided_config(mode=mode, sigma=sigma, seed=seed))
        total.append(r["NEES"][0])
        inertia_group.append(r["NEES_components"][0, 3])
    return np.array(total), np.array(inertia_group)


def test_initial_15_state_nees_equals_15_when_inertia_is_sampled():
    total, inertia_group = _initial_nees_groups("sampled_from_prior")
    # Mean of n chi-square(k) draws has standard error sqrt(2k/n); allow 4 SE.
    assert abs(total.mean() - 15.0) < 4 * np.sqrt(2 * 15 / 200)
    assert abs(inertia_group.mean() - 6.0) < 4 * np.sqrt(2 * 6 / 200)


def test_initial_15_state_nees_is_biased_low_when_inertia_is_fixed():
    """The defect the sampling mode removes: zero initial inertia error."""
    total, inertia_group = _initial_nees_groups("fixed")
    assert inertia_group.max() < 1e-12
    assert total.mean() < 15.0 - 4 * np.sqrt(2 * 15 / 200)
    assert abs(total.mean() - 9.0) < 4 * np.sqrt(2 * 9 / 200)
