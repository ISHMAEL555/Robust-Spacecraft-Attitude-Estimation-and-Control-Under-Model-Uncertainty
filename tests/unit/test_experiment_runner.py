from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats
import yaml

from scripts.analyze_results import generate_report, load_results, main as analyze_results_main
from scripts.run_experiment import (
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
    run_simulation,
    save_results,
)

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
