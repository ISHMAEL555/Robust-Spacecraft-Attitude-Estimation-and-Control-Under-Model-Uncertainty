import ast
import csv
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
from scripts.run_research_campaigns import (
    _augment_saved_measurement_masks,
    _json_safe,
    build_campaign_cases,
    _write_prediction_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_json_safe_preserves_structure_and_converts_nonfinite_values():
    value = {
        "metrics": np.array([1.0, np.nan, np.inf]),
        "interval": (np.float64(2.0), np.float64(-np.inf)),
        "passed": np.bool_(True),
    }

    assert _json_safe(value) == {
        "metrics": [1.0, None, None],
        "interval": [2.0, None],
        "passed": True,
    }


def test_saved_measurement_masks_reconstruct_only_deterministic_schedule():
    config = load_config(str(PROJECT_ROOT / "config" / "nominal.yaml"))
    results = {
        "innovation_count": np.array([0, 2, 0]),
        "measurement_rejection_count": np.zeros(3, dtype=int),
        "n_runs": 2,
    }

    _augment_saved_measurement_masks(
        {"campaign": "E0", "name": "matched_nominal", "config": config},
        results,
    )

    expected = np.array([[False, True, False], [False, True, False]])
    assert np.array_equal(results["measurement_available_runs"], expected)
    assert np.array_equal(results["measurement_accepted_runs"], expected)
    assert np.array_equal(results["innovation_valid_runs"], expected)


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

    assert len(cases) == 155
    assert {case["campaign"] for case in cases} == {
        "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"
    }
    assert all(case["config"].duration == 12.0 for case in cases)
    assert {case["nis_gate_enabled"] for case in cases if case["campaign"] == "E4"} == {
        True
    }
    assert any(
        case["name"] == "no_outlier_gate" for case in cases if case["campaign"] == "E4"
    )
    outages = [case for case in cases if case["campaign"] == "E5"]
    assert [case["config"].star_tracker["outage_duration"] for case in outages] == [
        60, 300, 900
    ]
    assert {case["inertia_mismatch"] for case in cases if case["campaign"] == "E1"} == {
        -0.25, -0.1, 0.0, 0.1, 0.25
    }
    assert {
        case["gyro_noise_scale"]
        for case in cases
        if case["campaign"] == "E1" and "gyro_noise_scale" in case
    } == {
        1e-5, 1e-4, 3e-4, 5e-4, 0.001, 0.01, 0.025, 0.05,
        0.075, 0.1, 1.0, 10.0, 100.0,
    }
    crossover_cases = [
        case for case in cases
        if case["campaign"] == "E1"
        and case.get("inertia_mismatch") == 0.1
        and "gyro_noise_scale" in case
    ]
    assert len(crossover_cases) == 12
    assert {case["gyro_noise_scale"] for case in crossover_cases} == {
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0
    }
    long_disturbance_cases = [
        case for case in build_campaign_cases("E0")
        if case["name"].endswith("_long")
    ]
    assert len(long_disturbance_cases) == 3
    assert all(case["config"].duration == 1000.0 for case in long_disturbance_cases)
    assert all(
        case["equilibrium_tail_window_s"] == [900.0, 1000.0]
        for case in long_disturbance_cases
    )
    e6_cases = [case for case in cases if case["campaign"] == "E6"]
    assert all(
        case["config"].reaction_wheels["max_torque"] == case["torque_limit_nm"]
        and case["config"].reaction_wheels["max_momentum"] == 100.0
        and np.allclose(case["config"].spacecraft["initial_omega"], 0.0)
        for case in e6_cases
    )
    assert build_campaign_cases("E6")[0]["config"].duration == 600.0
    assert all(
        case["config"].analysis["disperse"]
        for case in cases if case["campaign"] == "E7"
    )


def test_prediction_table_reports_applied_wheel_term_and_omits_empty_trends(tmp_path):
    rows = [
        {
            "campaign": "E0",
            "case": f"wheel_h0.001_gyro_{state}",
            "wheel_gyroscopic_coupling": enabled,
            "initial_wheel_momentum_nms": 0.001,
            "control_torque_limit_nm": 0.1,
            "wheel_gyroscopic_torque_rms_nm_mean": 1e-4,
            "pointing_rms_rad_mean": pointing,
        }
        for state, enabled, pointing in (
            ("off", False, 0.1),
            ("on", True, 0.101),
        )
    ]

    _write_prediction_table(rows, tmp_path)

    with (tmp_path / "predicted_vs_measured.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        table = list(csv.DictReader(stream))
    off_case = next(
        row for row in table if row["experiment"] == "P5 wheel_h0.001_gyro_off wheel-momentum term"
    )
    assert ast.literal_eval(off_case["measured"])["applied_torque_rms_nm"] == 0.0
    assert off_case["pass"] == "True"
    assert not any(row["experiment"].startswith("P3") for row in table)


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
    assert np.isnan(results["consistency"]["nis_mean"])
    assert np.isfinite(results["consistency"]["pre_gate_nis_mean"])
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
    short_config = replace(
        config,
        duration=1.1,
        analysis={
            **config.analysis,
            "steady_state_window_s": [0.0, 0.5],
        },
    )
    results = run_monte_carlo(short_config, n_runs=2)
    results["consistency"] = compute_consistency_stats(results)
    output_path = tmp_path / "performance.npz"
    save_results(results, str(output_path))

    loaded = load_results(str(output_path))

    assert loaded["performance_summary"]["pointing_rms_rad"]["n"] == 2
    assert loaded["performance_summary"]["pointing_rms_rad_full"]["n"] == 2
    assert loaded["performance_summary"]["pointing_final_rad_full"]["n"] == 2
    assert len(results["performance_by_run"]) == 2
    predicted_nees = np.mean(
        np.stack([run["predicted_NEES"] for run in results["all_runs"]]),
        axis=0,
    )
    assert np.allclose(results["predicted_NEES_mean"], predicted_nees)
    assert np.all(np.isfinite([
        run["pointing_rms_rad_full"] for run in results["performance_by_run"]
    ]))


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
    assert loaded["measurement_available_runs"].shape == (2, 110)
    assert loaded["measurement_accepted_runs"].shape == (2, 110)
    assert loaded["innovation_valid_runs"].shape == (2, 110)
    assert loaded["measurement_available_runs"].dtype == bool
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


def test_model_aided_inertia_bound_prevents_nees_inconsistency():
    config = load_config(str(PROJECT_ROOT / "config" / "model_uncertainty.yaml"))
    inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    true_inertia = inertia + 0.25 * np.diag(np.diag(inertia))
    matched_filter = replace(
        config,
        duration=12.0,
        filter={
            **config.filter,
            "variant": "model_aided",
            "model_error_relative_bound": 0.0,
        },
    )
    bounded_filter = replace(
        matched_filter,
        filter={**matched_filter.filter, "model_error_relative_bound": 0.25},
    )

    unbounded = run_simulation(matched_filter, true_inertia, inertia)
    bounded = run_simulation(bounded_filter, true_inertia, inertia)
    window = (unbounded["time"] >= 5.0) & (unbounded["time"] < 12.0)
    unbounded_nees = float(np.mean(unbounded["NEES"][window]))
    bounded_nees = float(np.mean(bounded["NEES"][window]))

    assert unbounded_nees > 100.0
    assert bounded_nees < 15.0
    assert bounded_nees < 0.01 * unbounded_nees


def test_e4_campaign_includes_no_outlier_gate_baseline():
    cases = build_campaign_cases("E4")

    baseline = next(case for case in cases if case["name"] == "no_outlier_gate")
    assert baseline["nis_ratio_to_gate"] == 0.0
    assert baseline["config"].star_tracker["outlier_prob"] == 0.0
    assert baseline["config"].filter["nis_gate_confidence"] == 0.95


def test_prediction_table_explains_failed_covariance_prediction(tmp_path):
    row = {
        "campaign": "E0",
        "case": "matched_nominal",
        "riccati_relative_frobenius_error": 0.2,
        "run_nees_mean_ci95": [5.0, 7.0],
        "run_nis_mean_ci95": [2.5, 3.5],
    }

    _write_prediction_table([row], tmp_path)

    with (tmp_path / "predicted_vs_measured.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        report = list(csv.DictReader(stream))
    covariance = next(
        item for item in report
        if item["experiment"].startswith("E0 periodic Riccati covariance")
    )
    assert covariance["pass"] == "False"
    assert covariance["miss_explanation"]


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
