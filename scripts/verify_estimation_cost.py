"""Estimation cost: closed-loop pointing with estimated versus true-state feedback.

For E0, E5 and E6 and both filter variants, each seed is run twice, once with the
controller fed the filter estimate and once fed the truth. The filter, sensors and
every random stream run identically, so the two runs of a seed share the same
noise draws and the difference is a paired estimate of the pointing error the
estimator causes. Intervals are 95% Student-t over the per-seed values (n = number
of seeds); time samples inside a run are never counted as independent trials.

Thresholds (the 0.1 deg pointing requirement) are project-defined.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time as wall_clock
from typing import Any, Dict, List

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
    compute_pointing_decomposition_metrics,
    decomposition_settings,
    load_config,
    run_simulation,
)

FILTERS = ("gyro_driven", "model_aided")
MODES = ("estimated", "truth")
SEED_STRIDE = 1000  # same rule as run_monte_carlo: seed_i = base + 1000 i

# Scalar metrics reported per (scenario, filter, mode), as (label, extractor).
SCALARS = {
    "ape_rms_deg": lambda m: np.rad2deg(m["pointing"]["ape"]["rms_rad"]),
    "ape_p99_deg": lambda m: np.rad2deg(m["pointing"]["ape"]["p99_rad"]),
    "rpe_rms_deg": lambda m: np.rad2deg(m["pointing"]["rpe"]["rms_rad"]),
    "rpe_p99_deg": lambda m: np.rad2deg(m["pointing"]["rpe"]["p99_rad"]),
    "control_ape_rms_deg": lambda m: np.rad2deg(m["control"]["ape"]["rms_rad"]),
    "control_rpe_rms_deg": lambda m: np.rad2deg(m["control"]["rpe"]["rms_rad"]),
    "estimation_ape_rms_deg": lambda m: np.rad2deg(m["estimation"]["ape"]["rms_rad"]),
    "estimation_rpe_rms_deg": lambda m: np.rad2deg(m["estimation"]["rpe"]["rms_rad"]),
    "rate_error_rms_mrad_s": lambda m: 1e3 * m["rate_estimation_error"]["rms_rad_s"],
    "fraction_within_requirement": lambda m: m["fraction_within_requirement"],
}
INCREMENT_METRICS = ("ape_rms_deg", "ape_p99_deg", "rpe_rms_deg", "rpe_p99_deg",
                     "fraction_within_requirement")


def build_scenarios(duration_scale: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """Scenario definitions. ``steady_state_start_s`` is the fallback window start."""
    return {
        "E0": {
            "config": "nominal.yaml",
            "duration_s": 300.0 * duration_scale,
            "steady_state_start_s": 100.0 * duration_scale,
            "description": "nominal start at the target attitude, 0.01 rad/s initial rate",
            "overrides": {},
        },
        "E5": {
            "config": "closed_loop.yaml",
            "duration_s": 600.0 * duration_scale,
            "steady_state_start_s": 200.0 * duration_scale,
            "late_window_start_s": 400.0 * duration_scale,
            "description": ("E6 acquisition, then a 30-sample (30 s) tracker outage "
                            "from t = 450 s, inside both steady-state windows"),
            "overrides": {"star_tracker": {
                "outage_prob": 0.0, "outage_start_time": 450.0 * duration_scale,
                "outage_duration": 30}},
        },
        "E6": {
            "config": "closed_loop.yaml",
            "duration_s": 600.0 * duration_scale,
            "steady_state_start_s": 200.0 * duration_scale,
            "late_window_start_s": 400.0 * duration_scale,
            "description": "60 deg / 0.1 rad/s acquisition, extended to 600 s",
            "overrides": {},
        },
        "E6_200s": {
            "config": "closed_loop.yaml",
            "duration_s": 200.0 * duration_scale,
            "steady_state_start_s": 180.0 * duration_scale,  # the 180 s settling limit
            "description": "the 200 s benchmark run length used by evaluate_e6_benchmark.py",
            "overrides": {},
        },
    }


def _make_config(scenario: Dict[str, Any], variant: str, mode: str, seed: int,
                 rpe_window_s: float = 10.0, requirement_deg: float = 0.1):
    config = load_config(str(ROOT / "config" / scenario["config"]))
    overrides = scenario["overrides"]
    filter_cfg = {**config.filter, "variant": variant}
    if variant == "model_aided":
        # Matched inertia; rate-process noise as in the model-uncertainty campaigns (R-06).
        filter_cfg.setdefault("inertia_relative_sigma", 0.05)
        filter_cfg["sigma_omega_process"] = 1.0e-7
    return replace(
        config,
        duration=scenario["duration_s"],
        seed=seed,
        filter=filter_cfg,
        star_tracker={**config.star_tracker, **overrides.get("star_tracker", {})},
        controller={**config.controller, "state_feedback": mode},
        analysis={**config.analysis,
                  "steady_state_start_s": scenario["steady_state_start_s"],
                  "rpe_window_s": rpe_window_s,
                  "pointing_requirement_deg": requirement_deg},
    )


def run_one(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One (scenario, filter, seed): both feedback modes over a common window.

    The window start is the later of the two runs' 2% settling times, so the
    paired estimated-minus-truth difference is taken over identical samples.
    If either run never settles, the configured fallback start is used for both.
    A fixed late window (``late_window_start_s``) is evaluated as well.
    """
    scenario = task["scenario"]
    runs = {}
    for mode in MODES:
        config = _make_config(scenario, task["variant"], mode, task["seed"],
                              task["rpe_window_s"], task["requirement_deg"])
        runs[mode] = (config, run_simulation(config))
    settling = [compute_performance_metrics(r)["settling_time_2pct_s"]
                for _, r in runs.values()]
    if all(np.isfinite(settling)):
        common_start, source = float(max(settling)), "settling_2pct_common"
    else:
        common_start, source = float(scenario["steady_state_start_s"]), "config"
    windows = [("", common_start, source)]
    if "late_window_start_s" in scenario:
        windows.append(("_late", float(scenario["late_window_start_s"]), "fixed_late"))
    records = []
    for suffix, start, label in windows:
        for mode, (config, results) in runs.items():
            metrics = compute_pointing_decomposition_metrics(
                results, **{**decomposition_settings(config.analysis),
                            "forced_window_start_s": start,
                            "forced_window_source": label})
            records.append({"name": task["name"] + suffix, "variant": task["variant"],
                            "mode": mode, "seed_index": task["seed_index"],
                            "metrics": metrics})
    return records


def t_interval(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(np.mean(values))
    if n < 2:
        return {"mean": mean, "ci95": None, "n": int(n)}
    half = float(stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": int(n)}


def summarize(records: List[Dict[str, Any]], names: List[str]) -> Dict[str, Any]:
    """Per-mode run-level statistics and paired estimated-minus-truth increments."""
    out: Dict[str, Any] = {}
    for name in names:
        out[name] = {}
        for variant in FILTERS:
            by_mode: Dict[str, List[Dict[str, Any]]] = {}
            for mode in MODES:
                runs = sorted(
                    (r for r in records if (r["name"], r["variant"], r["mode"])
                     == (name, variant, mode)),
                    key=lambda r: r["seed_index"])
                by_mode[mode] = runs
            entry: Dict[str, Any] = {"modes": {}, "increment_estimated_minus_truth": {}}
            for mode, runs in by_mode.items():
                sources = {r["metrics"]["window_source"] for r in runs}
                starts = {r["metrics"]["window_start_s"] for r in runs}
                entry["modes"][mode] = {
                    "window_sources": sorted(sources),
                    "window_start_s": [min(starts), max(starts)],
                    **{key: t_interval([fn(r["metrics"]) for r in runs])
                       for key, fn in SCALARS.items()},
                }
            for key in INCREMENT_METRICS:
                est = np.array([SCALARS[key](r["metrics"]) for r in by_mode["estimated"]])
                tru = np.array([SCALARS[key](r["metrics"]) for r in by_mode["truth"]])
                entry["increment_estimated_minus_truth"][key] = t_interval(est - tru)
            out[name][variant] = entry
    return out


def make_figure(summary: Dict[str, Any], names: List[str], path: Path, seeds: int) -> None:
    colors = {"estimated": "#D55E00", "truth": "#0072B2"}
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names) + 1.5, 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.36
    for axis, name in zip(axes, names):
        for j, mode in enumerate(MODES):
            means, errs = [], []
            for variant in FILTERS:
                stat = summary[name][variant]["modes"][mode]["ape_rms_deg"]
                means.append(stat["mean"])
                ci = stat["ci95"]
                errs.append([stat["mean"] - ci[0], ci[1] - stat["mean"]] if ci else [0, 0])
            x = np.arange(len(FILTERS)) + (j - 0.5) * width
            axis.bar(x, means, width, color=colors[mode], label=f"{mode} feedback",
                     yerr=np.array(errs).T, capsize=3, ecolor="#333333")
        axis.axhline(0.1, color="#444444", linestyle="--", linewidth=1.0,
                     label="0.1° project-defined requirement")
        axis.set_yscale("log")
        axis.set_xticks(np.arange(len(FILTERS)))
        axis.set_xticklabels([v.replace("_", "-") for v in FILTERS])
        window = ("window: fixed 400 s to end" if name.endswith("_late")
                  else "window: from 2% settling" if name != "E0"
                  else "window: 100 s to end")
        axis.set_title(name + chr(10) + window, fontsize=10)
        axis.grid(True, axis="y", color="#D9DEE5", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Steady-state APE, RMS of ‖θ_p‖ (deg)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"Pointing error with estimated vs true-state feedback "
        f"({seeds} paired seeds, 95% t intervals)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["E0", "E5", "E6", "E6_200s"])
    parser.add_argument("--rpe-window-s", type=float, default=10.0)
    parser.add_argument("--requirement-deg", type=float, default=0.1,
                        help="project-defined pointing requirement")
    parser.add_argument("--duration-scale", type=float, default=1.0,
                        help="scale all durations/window starts (quick smoke tests only)")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "estimation_cost.json")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "docs" / "figures" / "estimation_cost.png")
    args = parser.parse_args()

    scenarios = build_scenarios(args.duration_scale)
    names = list(args.scenarios)
    tasks = [
        {"name": name, "scenario": scenarios[name], "variant": variant,
         "seed_index": i, "seed": args.base_seed + SEED_STRIDE * i,
         "rpe_window_s": args.rpe_window_s, "requirement_deg": args.requirement_deg}
        for name in names for variant in FILTERS for i in range(args.seeds)
    ]
    started = wall_clock.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            records = [r for batch in pool.map(run_one, tasks) for r in batch]
    else:
        records = [r for task in tasks for r in run_one(task)]
    elapsed = wall_clock.time() - started

    present = {r["name"] for r in records}
    names = [n + suffix for n in names for suffix in ("", "_late") if n + suffix in present]
    summary = summarize(records, names)
    report = {
        "label": "estimation cost: estimated vs true-state feedback; thresholds are project-defined",
        "seeds": [args.base_seed + SEED_STRIDE * i for i in range(args.seeds)],
        "pairing": ("same seed => same filter-init, gyro and star-tracker random draws in "
                    "both modes; realized measurements differ once trajectories diverge"),
        "interval": "95% Student-t over per-seed values (n = seeds)",
        "rpe_window_s": args.rpe_window_s,
        "pointing_requirement_deg": args.requirement_deg,
        "pointing_requirement_label": "project-defined; not a mission requirement",
        "model_aided_settings": {"inertia": "matched", "inertia_relative_sigma": 0.05,
                                 "sigma_omega_process": 1.0e-7},
        "windows": ("primary: starts at the later of the paired runs' 2% settling times "
                    "(config fallback if either never settles); *_late: fixed late window"),
        "scenarios": {k: v for k, v in scenarios.items() if k in args.scenarios},
        "results": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figure(summary, [n for n in names if n != "E6_200s"], args.figure, args.seeds)

    print(f"{len(tasks)} runs in {elapsed:.0f} s (jobs={args.jobs})")
    for name in names:
        for variant in FILTERS:
            e = summary[name][variant]
            est, tru = e["modes"]["estimated"], e["modes"]["truth"]
            inc = e["increment_estimated_minus_truth"]["ape_rms_deg"]
            print(f"{name:8s} {variant:12s} window={est['window_sources']} "
                  f"APE est {est['ape_rms_deg']['mean']:.4f} deg, "
                  f"truth {tru['ape_rms_deg']['mean']:.4f} deg, "
                  f"increment {inc['mean']:+.4f} {inc['ci95']}; "
                  f"within-req est {est['fraction_within_requirement']['mean']:.3f} "
                  f"truth {tru['fraction_within_requirement']['mean']:.3f}")
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
