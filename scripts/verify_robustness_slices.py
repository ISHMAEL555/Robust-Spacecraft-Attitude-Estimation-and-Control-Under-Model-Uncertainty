"""Run a small, reproducible Monte Carlo slice across the main review risks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_config, run_monte_carlo  # noqa: E402


def _mean_ci(values: list[float]) -> list[float] | None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    mean = float(np.mean(data))
    if data.size < 2:
        return [mean, mean]
    half_width = float(
        stats.t.ppf(0.975, data.size - 1)
        * np.std(data, ddof=1)
        / np.sqrt(data.size)
    )
    return [mean - half_width, mean + half_width]


def _run_case(case: dict, runs: int) -> dict:
    config = case["config"]
    nominal_inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    mismatch = float(case.get("inertia_mismatch", 0.0))
    true_inertia = nominal_inertia + mismatch * np.diag(np.diag(nominal_inertia))
    filter_inertia = nominal_inertia if config.filter["variant"] == "model_aided" else None

    results = run_monte_carlo(config, runs, true_inertia, filter_inertia)
    per_run = results["performance_by_run"]
    nees_means = [float(np.mean(run["NEES"])) for run in results["all_runs"]]
    nis_means = []
    for run in results["all_runs"]:
        accepted = run["measurement_available"] & run["measurement_accepted"]
        nis_values = run["NIS"][accepted]
        nis_means.append(float(np.mean(nis_values)) if nis_values.size else float("nan"))

    def metric_summary(name: str) -> dict:
        values = [float(item[name]) for item in per_run if name in item]
        finite = [value for value in values if np.isfinite(value)]
        return {
            "mean": float(np.mean(finite)) if finite else None,
            "run_mean_ci95": _mean_ci(finite),
        }

    accepted_count = int(sum(
        np.sum(run["measurement_available"] & run["measurement_accepted"])
        for run in results["all_runs"]
    ))
    measurement_count = int(sum(
        np.sum(run["measurement_available"]) for run in results["all_runs"]
    ))
    rejected_count = measurement_count - accepted_count
    return {
        "case": case["name"],
        "category": case["category"],
        "assumptions": case["assumptions"],
        "duration_s": config.duration,
        "runs": runs,
        "seeds": [config.seed + 1000 * index for index in range(runs)],
        "nees_dof": int(results["nees_dof"]),
        "mean_nees": float(np.mean([np.mean(run["NEES"]) for run in results["all_runs"]])),
        "mean_nees_per_dof": float(
            np.mean([np.mean(run["NEES"]) for run in results["all_runs"]])
            / results["nees_dof"]
        ),
        "run_mean_nees_ci95": _mean_ci(nees_means),
        "mean_nis_accepted": float(np.nanmean(nis_means)),
        "run_mean_nis_accepted_ci95": _mean_ci(nis_means),
        "pointing_rms_rad": metric_summary("pointing_rms_rad"),
        "pointing_final_rad": metric_summary("pointing_final_rad"),
        "settling_time_2pct_s": metric_summary("settling_time_2pct_s"),
        "settling_hold_duration_s": metric_summary("settling_hold_duration_s"),
        "attitude_estimation_rmse_rad": metric_summary("attitude_estimation_rmse_rad"),
        "bias_estimation_rmse_rad_s": metric_summary("bias_estimation_rmse_rad_s"),
        "final_angular_rate_norm_rad_s": metric_summary("final_angular_rate_norm_rad_s"),
        "peak_wheel_momentum_utilization": metric_summary("peak_wheel_momentum_utilization"),
        "outage_recovery_time_s": metric_summary("outage_recovery_time_s"),
        "measurement_count": measurement_count,
        "rejected_measurements": rejected_count,
        "rejection_fraction": (
            rejected_count / measurement_count if measurement_count else None
        ),
    }


def _build_cases(duration: float) -> list[dict]:
    model = load_config(str(ROOT / "config" / "model_uncertainty.yaml"))
    model = replace(
        model,
        duration=duration,
        filter={**model.filter, "variant": "model_aided", "inertia_relative_sigma": 0.25},
    )
    sensor = load_config(str(ROOT / "config" / "sensor_degradation.yaml"))
    sensor = replace(sensor, duration=duration)

    cases = [{
        "name": "model_aided_inertia_mismatch_25pct",
        "category": "inertia mismatch",
        "config": model,
        "inertia_mismatch": 0.25,
        "assumptions": {
            "variant": "model_aided",
            "true_inertia_scale": 1.25,
            "filter_inertia_scale": 1.0,
            "inertia_prior_relative_sigma": 0.25,
            "sigma_omega_process": model.filter["sigma_omega_process"],
        },
    }]

    for gated in (False, True):
        outlier_config = replace(
            sensor,
            filter={**sensor.filter, "nis_gate_enabled": gated},
            star_tracker={
                **sensor.star_tracker,
                "outlier_prob": 0.05,
                "outlier_magnitude": 1.0,
                "outage_prob": 0.0,
            },
        )
        cases.append({
            "name": f"one_rad_outliers_gated_{str(gated).lower()}",
            "category": "star-tracker outliers",
            "config": outlier_config,
            "assumptions": {
                "outlier_probability": 0.05,
                "outlier_magnitude_rad": 1.0,
                "nis_gate_enabled": gated,
            },
        })

    outage_config = replace(
        sensor,
        star_tracker={
            **sensor.star_tracker,
            "outlier_prob": 0.0,
            "outage_prob": 0.0,
            "outage_start_time": 5.0,
            "outage_duration": 30,
        },
    )
    cases.append({
        "name": "scheduled_outage_30_updates",
        "category": "star-tracker outage",
        "config": outage_config,
        "assumptions": {"outage_start_time_s": 5.0, "outage_duration_updates": 30},
    })

    for name, overrides in (
        ("process_noise_scale_10", {"filter": {"Q_scale": 10.0}}),
        ("measurement_noise_scale_10", {"filter": {"R_scale": 10.0}}),
    ):
        noise_config = replace(
            model,
            filter={**model.filter, "variant": "gyro_driven", **overrides["filter"]},
        )
        cases.append({
            "name": name,
            "category": "filter noise mismatch",
            "config": noise_config,
            "assumptions": overrides["filter"],
        })

    closed_loop = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    for name, initial in (
        ("closed_loop_zero_initial_rate", [0.0, 0.0, 0.0]),
        ("closed_loop_near_180deg_initial_attitude", [0.0, 0.0, 0.996194698, 0.087155743]),
    ):
        spacecraft = {**closed_loop.spacecraft}
        if name.endswith("initial_rate"):
            spacecraft["initial_omega"] = initial
        else:
            spacecraft["initial_attitude"] = initial
        case_config = replace(
            closed_loop,
            duration=max(200.0, duration),
            spacecraft=spacecraft,
        )
        cases.append({
            "name": name,
            "category": "closed-loop initial condition",
            "config": case_config,
            "assumptions": {"initial_condition": initial},
        })
    return cases


def _write_plot(rows: list[dict], output: Path, runs: int) -> None:
    names = [
        f"{row['case'].replace('_', chr(10))}\n({row['duration_s']:g} s)"
        for row in rows
    ]
    x = np.arange(len(rows))
    nees = np.asarray([row["mean_nees_per_dof"] for row in rows])
    nees_ci = np.asarray([row["run_mean_nees_ci95"] for row in rows], dtype=float)
    dof = np.asarray([row["nees_dof"] for row in rows], dtype=float)
    nees_ci_per_dof = nees_ci / dof[:, None]
    # A t interval for a non-negative statistic can cross zero with only five
    # runs; truncate just the plotted lower whisker so the logarithmic axis is
    # well-defined. The untruncated interval remains in the JSON report.
    nees_ci_per_dof[:, 0] = np.maximum(nees_ci_per_dof[:, 0], 1e-3)
    nees_error = np.vstack((
        np.maximum(nees - nees_ci_per_dof[:, 0], 0.0),
        np.maximum(nees_ci_per_dof[:, 1] - nees, 0.0),
    ))
    pointing = np.asarray([row["pointing_rms_rad"]["mean"] for row in rows])
    point_ci = np.asarray([row["pointing_rms_rad"]["run_mean_ci95"] for row in rows], dtype=float)
    pointing_deg = np.rad2deg(pointing)
    pointing_error = np.rad2deg(np.vstack((pointing - point_ci[:, 0], point_ci[:, 1] - pointing)))

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    colors = ["#0072B2" if row["category"] != "star-tracker outliers" else "#D55E00" for row in rows]
    axes[0].bar(x, nees, color=colors, alpha=0.9)
    axes[0].errorbar(x, nees, yerr=nees_error, fmt="none", ecolor="#202124", capsize=3)
    axes[0].axhline(1.0, color="#009E73", linestyle="--", label="expected NEES / DOF = 1")
    axes[0].set_xticks(x, [])
    axes[0].set_ylabel("Full-state mean NEES / DOF")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", color="#D9DEE5", linewidth=0.7)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#009E73", linestyle="--", label="expected NEES / DOF = 1"),
            Patch(color="#0072B2", label="other selected conditions"),
            Patch(color="#D55E00", label="star-tracker outlier cases"),
        ],
        frameon=False,
        ncol=3,
    )

    axes[1].bar(x, pointing_deg, color=colors, alpha=0.9)
    axes[1].errorbar(x, pointing_deg, yerr=pointing_error, fmt="none", ecolor="#202124", capsize=3)
    axes[1].set_ylabel("True pointing RMS (deg)")
    axes[1].grid(True, axis="y", color="#D9DEE5", linewidth=0.7)
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_xlabel("Case (95% t intervals across independent runs)")
    figure.suptitle(
        f"Selected GNC robustness slices · exploratory {runs}-run ensembles",
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument(
        "--case", action="append",
        help="Run only this case name; repeat to select several. Existing report rows are retained.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate the figure from the existing JSON report without running cases.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs" / "robustness_slices.json",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=ROOT / "docs" / "figures" / "robustness_slices.png",
    )
    args = parser.parse_args()
    if args.plot_only:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        _write_plot(report["cases"], args.figure, int(report["runs_per_case"]))
        print(f"Saved robustness figure: {args.figure}")
        return
    if args.runs < 2 or args.duration <= 0.0:
        parser.error("--runs must be at least two and --duration must be positive")

    cases = _build_cases(args.duration)
    if args.case:
        selected_names = set(args.case)
        unknown = selected_names - {case["name"] for case in cases}
        if unknown:
            parser.error(f"unknown case name(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["name"] in selected_names]

    rows = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case['name']}: {args.runs} runs x "
            f"{case['config'].duration:g} s",
            flush=True,
        )
        rows.append(_run_case(case, args.runs))

    existing_rows = {}
    base_duration = args.duration
    if args.case and args.output.exists():
        old_report = json.loads(args.output.read_text(encoding="utf-8"))
        existing_rows = {row["case"]: row for row in old_report.get("cases", [])}
        base_duration = float(old_report.get("base_duration_s", args.duration))
    existing_rows.update({row["case"]: row for row in rows})
    report = {
        "scope": "Exploratory robustness slices; not qualification or mission evidence.",
        "runs_per_case": args.runs,
        "base_duration_s": base_duration,
        "base_seed": 42,
        "cases": list(existing_rows.values()),
        "interpretation": (
            "The matched-inertia 10-run result remains the statistical baseline. "
            "These selected cases show sensitivity to one-at-a-time mismatch and fault "
            "conditions; small-run confidence intervals are descriptive and do not "
            "establish a validated operating envelope."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_plot(report["cases"], args.figure, args.runs)
    print(f"Saved robustness report: {args.output}")
    print(f"Saved robustness figure: {args.figure}")


if __name__ == "__main__":
    main()
