from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats
import yaml

from scripts.analyze_results import generate_report, load_results, main as analyze_results_main
from scripts.run_experiment import (
    compute_performance_metrics,
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
    run_simulation,
    save_results,
)
from scripts.run_research_campaigns import build_campaign_cases

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_consistency_bounds_match_raw_nees_and_nis():
    results = {
        "NEES": np.array([6.0, 6.0]),
        "NIS": np.array([3.0, 3.0]),
        "innovation_valid": np.array([True, True]),
    }

    consistency = compute_consistency_stats(results)

    assert np.allclose(
        consistency["nees_bounds"],
        (stats.chi2.ppf(0.025, 6), stats.chi2.ppf(0.975, 6)),
    )
    assert np.allclose(
        consistency["nis_bounds"],
        (stats.chi2.ppf(0.025, 3), stats.chi2.ppf(0.975, 3)),
    )
    assert consistency["nees_in_bounds"] == 1.0
    assert consistency["nis_in_bounds"] == 1.0


def test_nis_reporting_confidence_is_independent_from_nees_confidence():
    result = {
        "NEES": np.array([6.0]),
        "NIS": np.array([3.0]),
        "measurement_available": np.array([True]),
    }

    consistency = compute_consistency_stats(result, confidence=0.95, nis_confidence=0.99)

    assert np.allclose(
        consistency["nees_bounds"],
        (stats.chi2.ppf(0.025, 6), stats.chi2.ppf(0.975, 6)),
    )
    assert np.allclose(
        consistency["nis_bounds"],
        (stats.chi2.ppf(0.005, 3), stats.chi2.ppf(0.995, 3)),
    )


def test_campaign_matrix_covers_e1_through_e7():
    cases = build_campaign_cases("all", duration=12.0)

    assert len(cases) == 27
    assert {case["campaign"] for case in cases} == {
        "E1", "E2", "E3", "E4", "E5", "E6", "E7"
    }
    assert all(case["config"].duration == 12.0 for case in cases)
    assert {case["nis_gate_enabled"] for case in cases if case["campaign"] == "E4"} == {
        False, True
    }
    outages = [case for case in cases if case["campaign"] == "E5"]
    assert [case["config"].star_tracker["outage_duration"] for case in outages] == [
        1, 10, 30
    ]


def test_monte_carlo_nis_bounds_follow_available_sample_count():
    results = {
        "NEES_mean": np.array([6.0, 6.0]),
        "all_runs": [
            {
                "NEES": np.array([6.0, 6.0]),
                "NIS": np.array([3.0, 0.0]),
                "measurement_available": np.array([True, False]),
            },
            {
                "NEES": np.array([6.0, 6.0]),
                "NIS": np.array([3.0, 3.0]),
                "measurement_available": np.array([True, True]),
            },
        ],
    }

    consistency = compute_consistency_stats(results)

    assert consistency["nis_bounds_mc"][0].shape == (2,)
    assert consistency["nis_bounds_mc"][1].shape == (2,)
    assert consistency["nis_bounds_mc"][0][0] < consistency["nis_bounds_mc"][1][0]
    assert not np.isnan(consistency["nis_bounds_mc"][0][1])


def test_run_experiment_script_entrypoint(tmp_path):
    config_path = PROJECT_ROOT / "config" / "nominal.yaml"
    config_data = yaml.safe_load(config_path.read_text())
    config_data["simulation"]["duration"] = 0.1
    short_config_path = tmp_path / "short.yaml"
    short_config_path.write_text(yaml.safe_dump(config_data))
    output_path = tmp_path / "smoke.npz"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
            "--config",
            str(short_config_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert output_path.exists()
    assert "Results saved" in completed.stdout


def test_simulation_routes_outliers_through_nis_gate(tmp_path):
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    short_config = replace(
        config,
        duration=1.1,
        star_tracker={**config.star_tracker, "outlier_prob": 1.0},
        filter={**config.filter, "nis_gate_enabled": True},
    )

    results = run_simulation(short_config)

    rejected = results["measurement_available"]
    assert np.any(rejected)
    assert np.all(~results["measurement_accepted"][rejected])
    assert np.all(~results["innovation_valid"][rejected])

    results["consistency"] = compute_consistency_stats(results)
    report_path = tmp_path / "outlier_report.txt"
    generate_report(results, str(report_path))

    attempted = int(np.sum(results["measurement_available"]))
    rejected_count = int(np.sum(
        results["measurement_available"] & ~results["measurement_accepted"]
    ))
    assert np.isfinite(results["consistency"]["nis_mean"])
    assert (
        f"Star-tracker updates: {attempted} attempted, "
        f"{rejected_count} rejected"
    ) in report_path.read_text()


def test_nominal_config_runs_and_analyzes(tmp_path):
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    short_config = replace(config, duration=1.1)

    results = run_simulation(short_config)

    assert results["time"].shape == (110,)
    assert results["b_true"].shape == (110, 3)
    assert np.any(results["innovation_valid"])

    results["consistency"] = compute_consistency_stats(results)
    output_path = tmp_path / "nominal.npz"
    report_path = tmp_path / "nominal_report.txt"
    save_results(results, str(output_path))

    loaded = load_results(str(output_path))
    generate_report(loaded, str(report_path))

    assert report_path.exists()
    assert "Consistency Statistics:" in report_path.read_text()


def test_simulation_separates_pointing_from_estimation_error():
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    results = run_simulation(replace(config, duration=0.1))

    assert results["pointing_error"][0] == 0.0
    assert results["attitude_estimation_error"][0] > 0.005
    metrics = compute_performance_metrics(results)
    assert metrics["pointing_peak_rad"] < metrics["attitude_estimation_rmse_rad"] * 2
    assert 0.0 <= metrics["attitude_3sigma_component_coverage"] <= 1.0


def test_scheduled_star_tracker_outage_recovers():
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    outage_config = replace(
        config,
        duration=4.1,
        star_tracker={
            **config.star_tracker,
            "outage_prob": 0.0,
            "outage_start_time": 1.0,
            "outage_duration": 2,
        },
    )

    results = run_simulation(outage_config)

    assert np.sum(results["star_tracker_outage"]) >= 190
    assert np.sum(results["measurement_available"]) == 2
    assert np.any(results["innovation_valid"][300:])


def test_monte_carlo_reports_run_level_performance(tmp_path):
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    results = run_monte_carlo(replace(config, duration=1.1), n_runs=2)
    results["consistency"] = compute_consistency_stats(results)
    output_path = tmp_path / "performance.npz"
    save_results(results, str(output_path))

    loaded = load_results(str(output_path))

    assert loaded["performance_summary"]["pointing_rms_rad"]["n"] == 2
    assert len(results["performance_by_run"]) == 2


def test_monte_carlo_results_can_be_summarized_and_saved(tmp_path, monkeypatch):
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    short_config = replace(config, duration=1.1)

    results = run_monte_carlo(short_config, n_runs=2)
    results["consistency"] = compute_consistency_stats(results)
    output_path = tmp_path / "monte_carlo.npz"
    save_results(results, str(output_path))

    loaded = load_results(str(output_path))

    assert loaded["n_runs"] == 2
    assert loaded["P_mean"].shape == (110, 6, 6)
    assert loaded["innovation_count"].max() == 2
    assert "consistency" in loaded

    output_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            str(output_path),
            "--all",
            "--output-dir",
            str(output_dir),
        ],
    )
    analyze_results_main()

    assert (output_dir / "monte_carlo_report.txt").exists()
    assert (output_dir / "monte_carlo_mc_consistency.png").exists()


def test_model_aided_simulation_uses_nine_state_consistency():
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    model_config = replace(
        config,
        duration=1.1,
        filter={**config.filter, "variant": "model_aided"},
    )

    results = run_simulation(model_config)
    consistency = compute_consistency_stats(results)

    assert results["P"].shape == (110, 9, 9)
    assert results["nees_dof"] == 9
    assert consistency["dof_nees"] == 9
    assert np.isfinite(consistency["nees_mean"])


def test_control_metrics_distinguish_commanded_from_applied_torque():
    results = {
        "time": np.array([0.0, 1.0]),
        "pointing_error": np.zeros(2),
        "tau_cmd": np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "tau_rw": np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        "max_torque": 2.0,
    }

    metrics = compute_performance_metrics(results)

    assert metrics["control_effort_nm2_s"] == 8.0
    assert metrics["achieved_control_effort_nm2_s"] == 1.25
    assert metrics["peak_commanded_torque_nm"] == 2.0
    assert metrics["peak_achieved_torque_nm"] == 1.0
