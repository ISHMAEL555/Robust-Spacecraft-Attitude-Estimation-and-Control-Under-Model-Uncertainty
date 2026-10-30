"""E6 Monte Carlo: probability of passing the project benchmark under dispersions.

Reuses the E6 configuration (config/closed_loop.yaml) and the benchmark criteria of
``evaluate_e6_project_benchmark``; nothing is redefined here. Per run ``i`` with run
seed ``s_i = base + 1000 i``:

* initial attitude: 60 deg about a uniformly random axis (optionally also a
  dispersed magnitude), from its own SeedSequence child;
* initial rate: 0.1 rad/s magnitude, uniformly random direction, own child;
* truth inertia: sampled from the filter prior (``inertia_truth_mode:
  "sampled_from_prior"``), own child of the run seed;
* sensor noise and filter initialization: the run seed itself (new per run).

Dispersion streams come from ``SeedSequence(entropy=s_i, spawn_key=(1000,))``, which
cannot collide with the run's own children ``spawn_key=(0..3,)``. Results do not
depend on ``--jobs``. Thresholds are project-defined, not mission requirements.
"""

from __future__ import annotations

import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time as wall_clock
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.run_experiment import (
    compute_performance_metrics,
    evaluate_e6_project_benchmark,
    load_config,
    run_simulation,
)

SEED_STRIDE = 1000
DISPERSION_SPAWN_KEY = 1000
UNSETTLED = float("inf")


# --- dispersions ---------------------------------------------------------------

def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Uniform direction on the sphere."""
    vector = rng.standard_normal(3)
    return vector / np.linalg.norm(vector)


def sample_initial_conditions(
    run_seed: int,
    attitude_deg: float = 60.0,
    attitude_range_deg: Optional[Tuple[float, float]] = None,
    rate_rad_s: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Initial attitude and body rate for one run, each from its own stream.

    The attitude is a rotation of ``attitude_deg`` (or, with
    ``attitude_range_deg``, a magnitude uniform on that range) about a uniformly
    random axis, as a scalar-last quaternion. The rate has magnitude
    ``rate_rad_s`` in a uniformly random body direction.
    """
    root = np.random.SeedSequence(entropy=run_seed, spawn_key=(DISPERSION_SPAWN_KEY,))
    attitude_stream, rate_stream = (np.random.default_rng(c) for c in root.spawn(2))
    axis = _unit_vector(attitude_stream)
    if attitude_range_deg is None:
        angle = np.deg2rad(attitude_deg)
    else:
        low, high = attitude_range_deg
        angle = np.deg2rad(attitude_stream.uniform(low, high))
    quaternion = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
    direction = _unit_vector(rate_stream)
    return {
        "attitude_axis": axis,
        "attitude_angle_rad": np.array(angle),
        "quaternion": quaternion,
        "rate_direction": direction,
        "rate": rate_rad_s * direction,
    }


# --- statistics ----------------------------------------------------------------

def clopper_pearson(successes: int, trials: int, confidence: float = 0.95
                    ) -> Tuple[float, float]:
    """Exact two-sided binomial confidence interval."""
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def distribution_summary(values: np.ndarray, scale: float = 1.0) -> Dict[str, Any]:
    """
    Median, 5th and 95th percentiles. Infinite values (a run that never settled)
    sort above every finite value; a quantile that falls on or between an infinite
    value is reported as ``None``, meaning "not reached within the run".
    """
    values = np.asarray(values, dtype=float)
    finite_stand_in = np.where(np.isfinite(values), values, 1e18)
    result: Dict[str, Any] = {}
    for name, q in (("median", 50.0), ("p05", 5.0), ("p95", 95.0)):
        value = float(np.percentile(finite_stand_in, q))
        result[name] = value * scale if value < 1e6 else None
    result["n_not_finite"] = int(np.sum(~np.isfinite(values)))
    return result


def rank_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"spearman_rho": None, "p_value": None}
    rho, p = stats.spearmanr(x, y)
    return {"spearman_rho": float(rho), "p_value": float(p)}


# --- one run -------------------------------------------------------------------

def build_config(base_config, run_seed: int, conditions: Dict[str, np.ndarray],
                 dispersions: Tuple[str, ...], inertia_sigma: float):
    spacecraft = dict(base_config.spacecraft)
    if "attitude" in dispersions:
        spacecraft["initial_attitude"] = conditions["quaternion"].tolist()
    if "rate" in dispersions:
        spacecraft["initial_omega"] = conditions["rate"].tolist()
    if "inertia" in dispersions:
        spacecraft["inertia_truth_mode"] = "sampled_from_prior"
    return replace(
        base_config, seed=run_seed, spacecraft=spacecraft,
        filter={**base_config.filter, "inertia_relative_sigma": inertia_sigma},
    )


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    conditions = sample_initial_conditions(
        task["seed"], task["attitude_deg"], task["attitude_range_deg"], task["rate_rad_s"])
    config = build_config(base, task["seed"], conditions, tuple(task["dispersions"]),
                          task["inertia_sigma"])
    if task.get("duration") is not None:
        config = replace(config, duration=float(task["duration"]))
    if task.get("controller_overrides"):
        config = replace(config, controller={**config.controller,
                                             **task["controller_overrides"]})
    results = run_simulation(config)
    metrics = compute_performance_metrics(results)
    benchmark = evaluate_e6_project_benchmark(metrics)
    inertia = np.asarray(results["inertia_true_sampled"]
                         if "inertia_true_sampled" in results
                         else config.spacecraft["inertia"], dtype=float)
    moments, axes = np.linalg.eigh(inertia)  # ascending: minor, intermediate, major
    rate_direction = conditions["rate_direction"]
    settling = metrics["settling_time_2pct_s"]
    return {
        "index": task["index"],
        "seed": task["seed"],
        "checks": {name: bool(c["passed"]) for name, c in benchmark["checks"].items()},
        "measured": {name: c["measured"] for name, c in benchmark["checks"].items()},
        "passed": bool(benchmark["passed"]),
        "settling_time_s": float(settling) if np.isfinite(settling) else UNSETTLED,
        "peak_terminal_error_deg": float(np.rad2deg(metrics["pointing_peak_final_20s_rad"])),
        "peak_wheel_momentum_use": float(metrics["peak_wheel_momentum_utilization"]),
        "rate_component_on_principal_axes": [
            float(abs(rate_direction @ axes[:, k])) for k in range(3)],
        "rate_vs_attitude_axis_cos": float(abs(rate_direction @ conditions["attitude_axis"])),
        "rate_vs_attitude_axis_signed_cos": float(rate_direction @ conditions["attitude_axis"]),
        "attitude_angle_deg": float(np.rad2deg(conditions["attitude_angle_rad"])),
        "principal_moments": moments.tolist(),
        "inertia_rejections": int(results.get("inertia_sampling", {}).get("rejections", 0)),
    }


# --- aggregation ---------------------------------------------------------------

def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(runs)
    criteria = list(runs[0]["checks"])
    report: Dict[str, Any] = {"n_runs": n}
    passes = sum(r["passed"] for r in runs)
    low, high = clopper_pearson(passes, n)
    report["overall"] = {"passes": passes, "fails": n - passes,
                         "p_pass": passes / n, "ci95_clopper_pearson": [low, high]}
    report["criteria"] = {}
    for name in criteria:
        k = sum(r["checks"][name] for r in runs)
        low, high = clopper_pearson(k, n)
        report["criteria"][name] = {"passes": k, "fails": n - k, "p_pass": k / n,
                                    "ci95_clopper_pearson": [low, high]}
    fail_counts = {name: report["criteria"][name]["fails"] for name in criteria}
    worst = max(fail_counts.values())
    report["most_frequent_failing_criteria"] = (
        [name for name, c in fail_counts.items() if c == worst] if worst > 0 else [])
    report["failed_simultaneously_with_overall"] = {
        "runs_failing_any_criterion": n - passes,
        "runs_failing_only_one_criterion": sum(
            1 for r in runs if sum(not v for v in r["checks"].values()) == 1),
    }
    settling = np.array([r["settling_time_s"] for r in runs])
    report["distributions"] = {
        "settling_time_s": {**distribution_summary(settling),
                            "n_never_settled": int(np.sum(~np.isfinite(settling)))},
        "peak_terminal_error_deg": distribution_summary(
            np.array([r["peak_terminal_error_deg"] for r in runs])),
        "peak_wheel_momentum_use": distribution_summary(
            np.array([r["peak_wheel_momentum_use"] for r in runs])),
    }
    report["inertia_sampling"] = {
        "rejections": int(sum(r["inertia_rejections"] for r in runs)),
        "attempts": int(n + sum(r["inertia_rejections"] for r in runs)),
    }
    report["inertia_sampling"]["rejection_rate"] = (
        report["inertia_sampling"]["rejections"] / report["inertia_sampling"]["attempts"])
    report["rate_direction_analysis"] = rate_direction_analysis(runs, worst_criteria=report[
        "most_frequent_failing_criteria"])
    return report


def rate_direction_analysis(runs: List[Dict[str, Any]], worst_criteria: List[str]
                            ) -> Dict[str, Any]:
    """
    Exploratory rank correlations of outcomes with the initial-rate direction.

    Covariates: the rate's components on the truth inertia's principal axes, its
    absolute and signed alignment with the initial attitude axis. Targets: overall
    failure, the most frequent failing criterion, settling time, and the continuous
    peak terminal error and final rate norm. When nearly every run fails, the
    binary and settling-time targets are degenerate, which is why the continuous
    targets are reported as well.
    """
    names = ["minor_axis", "intermediate_axis", "major_axis"]
    covariates = {f"|rate . {n}|": np.array([r["rate_component_on_principal_axes"][k]
                                             for r in runs])
                  for k, n in enumerate(names)}
    covariates["|rate . attitude_axis|"] = np.array(
        [r["rate_vs_attitude_axis_cos"] for r in runs])
    covariates["rate . attitude_axis (signed)"] = np.array(
        [r["rate_vs_attitude_axis_signed_cos"] for r in runs])
    targets = {
        "overall_failure": np.array([not r["passed"] for r in runs], dtype=float),
        "settling_time": np.array([1e9 if not np.isfinite(r["settling_time_s"])
                                   else r["settling_time_s"] for r in runs]),
        "peak_terminal_error_deg": np.array([r["peak_terminal_error_deg"] for r in runs]),
        "final_angular_rate_norm_rad_s": np.array(
            [r["measured"]["final_angular_rate_norm_rad_s"] for r in runs]),
    }
    if worst_criteria:
        targets[f"failure_of_{worst_criteria[0]}"] = np.array(
            [not r["checks"][worst_criteria[0]] for r in runs], dtype=float)
    return {
        "note": ("exploratory Spearman rank correlations, not corrected for the "
                 f"{len(covariates) * len(targets)} covariate-target pairs tested"),
        "covariates": list(covariates),
        "most_frequent_failing_criterion": worst_criteria[0] if worst_criteria else None,
        "targets": {t: {k: rank_correlation(v, y) for k, v in covariates.items()}
                    for t, y in targets.items()},
    }


def make_figure(runs: List[Dict[str, Any]], report: Dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    settling = np.array([r["settling_time_s"] for r in runs])
    finite = settling[np.isfinite(settling)]
    ax = axes[0]
    if finite.size:
        bins = np.linspace(min(finite.min(), 120.0), max(finite.max(), 200.0), 31)
        ax.hist(finite, bins=bins, color="#0072B2", edgecolor="white")
    ax.axvline(180.0, color="#D55E00", linestyle="--", linewidth=1.5,
               label="180 s project limit")
    ax.set_xlabel("2% settling time (s)")
    ax.set_ylabel("runs")
    never = int(np.sum(~np.isfinite(settling)))
    ax.set_title(f"Settling time, {len(runs)} runs" + (f" ({never} never settled within the run)" if never else ""))
    ax.legend(frameon=False)
    ax = axes[1]
    terminal = np.array([r["peak_terminal_error_deg"] for r in runs])
    ax.hist(terminal, bins=np.linspace(0.0, max(terminal.max(), 2.5) * 1.02, 31),
            color="#0072B2", edgecolor="white")
    ax.axvline(2.0, color="#D55E00", linestyle="--", linewidth=1.5, label="2° project limit")
    ax.set_xlabel("peak pointing error in the final 20 s (deg)")
    ax.set_ylabel("runs")
    ax.set_title("Terminal-window pointing error")
    ax.legend(frameon=False)
    ax = axes[2]
    names = list(report["criteria"])
    probs = np.array([report["criteria"][n]["p_pass"] for n in names] + [report["overall"]["p_pass"]])
    lows = np.array([report["criteria"][n]["ci95_clopper_pearson"][0] for n in names]
                    + [report["overall"]["ci95_clopper_pearson"][0]])
    highs = np.array([report["criteria"][n]["ci95_clopper_pearson"][1] for n in names]
                     + [report["overall"]["ci95_clopper_pearson"][1]])
    labels = [n.replace("_", " ") for n in names] + ["ALL criteria"]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, probs, color=["#0072B2"] * len(names) + ["#009E73"])
    ax.errorbar(probs, y, xerr=[probs - lows, highs - probs], fmt="none",
                ecolor="#222222", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("P(pass), Clopper–Pearson 95% interval")
    ax.set_title("Pass probability per project criterion")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle("E6 Monte Carlo (project-defined criteria)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--attitude-deg", type=float, default=60.0)
    parser.add_argument("--attitude-range-deg", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="also disperse the attitude magnitude, uniform on this range")
    parser.add_argument("--rate-rad-s", type=float, default=0.1)
    parser.add_argument("--inertia-sigma", type=float, default=0.05,
                        help="relative 1-sigma of the inertia prior the truth is drawn from")
    parser.add_argument("--dispersions", nargs="*", default=["attitude", "rate", "inertia"],
                        choices=["attitude", "rate", "inertia"])
    parser.add_argument("--duration", type=float, default=None,
                        help="override the E6 run length (smoke tests only)")
    parser.add_argument("--controller", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override controller settings (YAML values), e.g. "
                             "K_omega=20 control_type=rate_profile gyroscopic_feedforward=true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "e6_monte_carlo.json")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "docs" / "figures" / "e6_monte_carlo.png")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    overrides = {}
    for item in args.controller:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error(f"--controller expects KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    tasks = [{"index": i, "seed": args.base_seed + SEED_STRIDE * i,
              "attitude_deg": args.attitude_deg,
              "attitude_range_deg": (tuple(args.attitude_range_deg)
                                     if args.attitude_range_deg else None),
              "rate_rad_s": args.rate_rad_s, "dispersions": args.dispersions,
              "inertia_sigma": args.inertia_sigma, "duration": args.duration,
              "controller_overrides": overrides}
             for i in range(args.runs)]
    started = wall_clock.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(run_one, tasks))
    else:
        runs = [run_one(t) for t in tasks]
    elapsed = wall_clock.time() - started

    report = aggregate(runs)
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    benchmark_limits = evaluate_e6_project_benchmark(
        {"max_torque_nm": 0.1, "max_wheel_torque_nm": 0.1})["checks"]
    report = {
        "label": "E6 Monte Carlo; criteria are project-defined, not mission requirements",
        "config": "config/closed_loop.yaml",
        "filter_variant": base.filter["variant"],
        "controller": {**base.controller, **overrides},
        "duration_s": args.duration if args.duration is not None else base.duration,
        "criteria_limits": {n: {"operator": c["operator"], "limit": c["limit"]}
                            for n, c in benchmark_limits.items()},
        "dispersions": {
            "enabled": args.dispersions,
            "attitude": ({"magnitude_deg_range": list(args.attitude_range_deg)}
                         if args.attitude_range_deg else {"magnitude_deg": args.attitude_deg}),
            "attitude_axis": "uniform on the sphere",
            "rate": {"magnitude_rad_s": args.rate_rad_s, "direction": "uniform on the sphere"},
            "inertia": {"mode": "sampled_from_prior", "relative_sigma": args.inertia_sigma,
                        "note": "the E6 filter is gyro-driven and does not use inertia"},
            "sensor_noise": "new SeedSequence streams per run (seed = base + 1000 i)",
        },
        "seeds": [t["seed"] for t in tasks],
        **report,
        "runs": [{"seed": r["seed"], "passed": r["passed"],
                  "settling_time_s": (r["settling_time_s"]
                                      if np.isfinite(r["settling_time_s"]) else None),
                  "peak_terminal_error_deg": r["peak_terminal_error_deg"],
                  "peak_wheel_momentum_use": r["peak_wheel_momentum_use"],
                  "failed_criteria": [n for n, ok in r["checks"].items() if not ok]}
                 for r in runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figure(runs, report, args.figure)

    o = report["overall"]
    print(f"{args.runs} runs in {elapsed:.0f} s (jobs={args.jobs})")
    print(f"overall: {o['passes']}/{args.runs} pass, P = {o['p_pass']:.3f} "
          f"[{o['ci95_clopper_pearson'][0]:.3f}, {o['ci95_clopper_pearson'][1]:.3f}]")
    for name, c in report["criteria"].items():
        print(f"  {name:34s} {c['passes']:4d}/{args.runs} P = {c['p_pass']:.3f} "
              f"[{c['ci95_clopper_pearson'][0]:.3f}, {c['ci95_clopper_pearson'][1]:.3f}]")
    for name, d in report["distributions"].items():
        print(f"  {name}: {d}")
    print("most frequent failing:", report["most_frequent_failing_criteria"])
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
