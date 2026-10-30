"""Model-aided consistency: deterministic model-error study vs sampled-truth study.

Deterministic model-error study ("fixed" truth): the true inertia is the nominal
tensor scaled by 1 + m (m = 0, 10%, 25%) for every seed, with the filter prior
1-sigma = max(m, 5%), as in campaign E1. Only sensor noise varies across seeds.

Sampled-truth study ("sampled_from_prior"): each seed draws its own truth inertia
from the filter's inertia prior N(J_nom, sigma^2) (sigma = 5%, 10%, 25%), rejecting
non-realisable tensors (not SPD, or violating the triangle inequality). The
filter's initial inertia error is then a draw from its own prior, so the 15-state
NEES is not biased low by construction.

Intervals are 95% Student-t over run-level means (n = number of seeds).
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

from scripts.run_experiment import load_config, run_monte_carlo

GROUPS = (("attitude", 3), ("rate", 3), ("bias", 3), ("inertia", 6))
DETERMINISTIC_OFFSETS = (0.0, 0.10, 0.25)
SAMPLED_SIGMAS = (0.05, 0.10, 0.25)


def t_interval(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, values.size - 1) * np.std(values, ddof=1)
                 / np.sqrt(values.size))
    return {"mean": mean, "ci95": [mean - half, mean + half],
            "contains_1": bool(mean - half <= 1.0 <= mean + half)}


def build_cases() -> List[Dict[str, Any]]:
    cases = [{"study": "deterministic_model_error", "mode": "fixed",
              "offset": m, "prior_sigma": max(m, 0.05),
              "name": f"deterministic_offset_{m:.2f}"} for m in DETERMINISTIC_OFFSETS]
    cases += [{"study": "sampled_from_prior", "mode": "sampled_from_prior",
               "offset": None, "prior_sigma": s,
               "name": f"sampled_sigma_{s:.2f}"} for s in SAMPLED_SIGMAS]
    return cases


def run_case(task: Dict[str, Any]) -> Dict[str, Any]:
    case, runs, duration = task["case"], task["runs"], task["duration"]
    config = load_config(str(ROOT / "config" / "model_uncertainty.yaml"))
    config = replace(
        config, duration=duration,
        spacecraft={**config.spacecraft, "inertia_truth_mode": case["mode"]},
        filter={**config.filter, "variant": "model_aided",
                "inertia_relative_sigma": case["prior_sigma"]})
    nominal = np.asarray(config.spacecraft["inertia"], dtype=float)
    true_inertia = None
    if case["mode"] == "fixed":
        true_inertia = nominal + case["offset"] * np.diag(np.diag(nominal))
    results = run_monte_carlo(config, runs, true_inertia,
                              nominal if case["mode"] == "fixed" else None)
    runs_data = results["all_runs"]
    nees = np.array([np.mean(r["NEES"]) for r in runs_data]) / 15.0
    nis = np.array([np.mean(r["NIS"][r["measurement_available"]
                                     & r["measurement_accepted"]])
                    for r in runs_data]) / 3.0
    groups = {}
    for column, (name, dof) in enumerate(GROUPS):
        per_run = np.array([np.nanmean(r["NEES_components"][:, column])
                            for r in runs_data]) / dof
        groups[name] = t_interval(per_run)
    initial = np.array([r["NEES"][0] for r in runs_data])
    record = {
        **case,
        "runs": runs, "duration_s": duration,
        "seeds": [config.seed + 1000 * i for i in range(runs)],
        "nees_per_dof": t_interval(nees),
        "nees_per_dof_median": float(np.median(nees)),
        "nees_per_dof_by_run": [float(x) for x in nees],
        "nis_per_dof": t_interval(nis),
        "marginal_nees_per_dof": groups,
        "initial_nees_mean": float(np.mean(initial)),
        "initial_nees_dof": 15,
    }
    if "inertia_sampling" in results:
        record["inertia_sampling"] = results["inertia_sampling"]
    return record


def make_figure(records: List[Dict[str, Any]], path: Path, runs: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ["0 / 5%", "10%", "25%"]
    colors = {"deterministic_model_error": "#D55E00", "sampled_from_prior": "#0072B2"}
    names = {"deterministic_model_error": "deterministic model-error (fixed truth)",
             "sampled_from_prior": "sampled from prior"}
    for axis, key, title in (
        (axes[0], "nees_per_dof", "15-state NEES / DOF"),
        (axes[1], "nis_per_dof", "Accepted NIS / DOF"),
    ):
        for j, study in enumerate(colors):
            subset = [r for r in records if r["study"] == study]
            means = [r[key]["mean"] for r in subset]
            lows = [r[key]["mean"] - r[key]["ci95"][0] for r in subset]
            highs = [r[key]["ci95"][1] - r[key]["mean"] for r in subset]
            x = np.arange(len(subset)) + (j - 0.5) * 0.3
            axis.errorbar(x, means, yerr=[lows, highs], fmt="o", color=colors[study],
                          capsize=4, label=names[study])
        axis.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(labels)
        axis.set_xlabel("inertia offset (deterministic) / prior 1σ (sampled)")
        axis.set_title(title)
        axis.grid(True, axis="y", color="#D9DEE5", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Model-aided consistency: deterministic vs sampled truth inertia "
                 f"({runs} seeds, 95% t intervals)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "model_aided_inertia_sampling.json")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "docs" / "figures" / "inertia_sampling_consistency.png")
    args = parser.parse_args()
    if args.runs < 2 or args.duration <= 0.0:
        parser.error("--runs must be at least 2 and --duration must be positive")

    tasks = [{"case": c, "runs": args.runs, "duration": args.duration}
             for c in build_cases()]
    started = wall_clock.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            records = list(pool.map(run_case, tasks))
    else:
        records = [run_case(t) for t in tasks]
    elapsed = wall_clock.time() - started

    report = {
        "label": "model-aided consistency: deterministic model-error study vs sampled truth inertia",
        "filter": {"variant": "model_aided", "sigma_omega_process": 1.0e-7},
        "interval": "95% Student-t over run-level means (n = runs)",
        "truncation_note": ("Rejection of non-realisable inertia tensors (not SPD, or "
                            "violating the triangle inequality) truncates the prior "
                            "slightly; the rejection rate is reported per case."),
        "studies": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figure(records, args.figure, args.runs)

    print(f"{len(tasks)} cases x {args.runs} runs in {elapsed:.0f} s (jobs={args.jobs})")
    for r in records:
        n, i = r["nees_per_dof"], r["nis_per_dof"]
        rej = r.get("inertia_sampling", {}).get("rejection_rate")
        print(f"{r['name']:28s} NEES/DOF {n['mean']:.3f} [{n['ci95'][0]:.3f}, {n['ci95'][1]:.3f}] "
              f"median {r['nees_per_dof_median']:.2f} NIS/DOF {i['mean']:.3f} initial NEES {r['initial_nees_mean']:.2f}"
              + (f" rejection {rej:.3f}" if rej is not None else ""))
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
