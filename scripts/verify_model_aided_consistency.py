"""Reproduce the matched-inertia 15-state model-aided consistency check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import (  # noqa: E402
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs" / "model_aided_consistency.json",
    )
    parser.add_argument(
        "--inertia-truth-mode", choices=("fixed", "sampled_from_prior"),
        default="fixed",
        help="'sampled_from_prior' draws each seed's true inertia from the filter prior",
    )
    args = parser.parse_args()
    if args.runs < 2 or args.duration <= 0.0:
        parser.error("--runs must be at least 2 and --duration must be positive")

    config = load_config(str(ROOT / "config" / "model_uncertainty.yaml"))
    config.duration = args.duration
    config.filter["variant"] = "model_aided"
    config.filter["inertia_relative_sigma"] = 0.05
    config.spacecraft = {**config.spacecraft, "inertia_truth_mode": args.inertia_truth_mode}
    results = run_monte_carlo(config, args.runs)
    consistency = compute_consistency_stats(
        results,
        config.analysis["nees_confidence"],
        config.analysis["nis_confidence"],
    )
    run_nees = np.asarray([
        np.mean(run["NEES"]) for run in results["all_runs"]
    ])
    run_nis = np.asarray([
        np.mean(run["NIS"][run["measurement_available"] & run["measurement_accepted"]])
        for run in results["all_runs"]
    ])
    mean_run_nees = float(np.mean(run_nees))
    nees_half_width = float(
        stats.t.ppf(0.975, args.runs - 1)
        * np.std(run_nees, ddof=1)
        / np.sqrt(args.runs)
    )
    mean_run_nis = float(np.mean(run_nis))
    nis_half_width = float(
        stats.t.ppf(0.975, args.runs - 1)
        * np.std(run_nis, ddof=1)
        / np.sqrt(args.runs)
    )
    report = {
        "scenario": "matched inertia, model-aided MEKF",
        "n_runs": args.runs,
        "duration_s": args.duration,
        "dt_s": config.dt,
        "seeds": [config.seed + 1000 * index for index in range(args.runs)],
        "inertia_relative_prior_sigma": config.filter["inertia_relative_sigma"],
        "sigma_omega_process": config.filter["sigma_omega_process"],
        "nees_dof": consistency["dof_nees"],
        "nis_dof": consistency["dof_nis"],
        "mean_nees": float(consistency["nees_mean"]),
        "run_mean_nees_ci95": [
            mean_run_nees - nees_half_width,
            mean_run_nees + nees_half_width,
        ],
        "mean_nis": float(consistency["nis_mean"]),
        "mean_run_nis": mean_run_nis,
        "run_mean_nis_ci95": [
            mean_run_nis - nis_half_width,
            mean_run_nis + nis_half_width,
        ],
        "run_mean_nis_ci_contains_3_dof": bool(
            mean_run_nis - nis_half_width <= 3.0 <= mean_run_nis + nis_half_width
        ),
        "nis_epoch_means_inside_pointwise_mc_bounds_fraction": float(
            consistency["nis_in_bounds"]
        ),
        "mean_nees_inside_pointwise_mc_bounds_fraction": float(
            consistency["nees_in_bounds"]
        ),
        "marginal_nees_by_state_group": consistency["nees_component_summary"],
        "interpretation": (
            "Exploratory matched-inertia ensemble check. Pointwise epoch fractions "
            "are descriptive because each run is temporally correlated."
        ),
    }
    if args.inertia_truth_mode != "fixed":
        report["scenario"] = "per-seed sampled inertia (from the filter prior), model-aided MEKF"
        report["inertia_truth_mode"] = args.inertia_truth_mode
        report["inertia_sampling"] = results["inertia_sampling"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved matched-inertia consistency report: {args.output}")
    print(
        f"Mean NEES={report['mean_nees']:.3f} (15 DOF); "
        f"run-mean 95% CI={report['run_mean_nees_ci95']}"
    )
    print(f"Mean NIS={report['mean_nis']:.3f} (3 DOF)")


if __name__ == "__main__":
    main()
