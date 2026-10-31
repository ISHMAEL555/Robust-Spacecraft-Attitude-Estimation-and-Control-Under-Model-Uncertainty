"""E6 feasibility analysis: bounds and estimates for every run of the 200-run Monte Carlo.

Reads ``docs/e6_monte_carlo.json`` (unchanged) and, for each stored run, rebuilds
that run's own dispersion (initial attitude axis, initial rate direction and the
sampled truth inertia, from the same ``SeedSequence`` streams the Monte Carlo
used) and computes

* the proven momentum lower bound, in both versions (body command norm limit,
  and the conservative per-wheel-clip limit);
* the approximate rest-to-rest bang-bang slew time;
* the single-axis feasibility estimate (initial rate included, 2 pi wrap
  allowed): an estimate, not a bound;
* the linearised, unsaturated settling time of the configured gains for the
  run's own effective axis inertia.

``--demonstrate N`` additionally runs the full nonlinear simulation with the
``rate_profile`` controller (parameters fixed in advance, not tuned) on N runs the
estimate calls feasible, to show that a real controller settles in time.
Thresholds are project-defined, not mission requirements.
"""

from __future__ import annotations

import argparse
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

from scripts.parallel_runs import completeness, run_tasks
from scripts.run_e6_monte_carlo import run_one, sample_initial_conditions
from scripts.run_experiment import load_config
from src.control.feasibility import (
    eigenaxis_feasibility_estimate_s,
    momentum_lower_bound_s,
    principal_axis_linear_summary,
    saturated_single_axis_settling_s,
    slew_time_rest_to_rest_s,
    stop_then_slew_plan,
    step_settling_time_s,
)
from src.dynamics.inertia_sampling import sample_inertia_from_prior
from src.dynamics.quaternion import error_quat_to_vec, right_error_quat

SETTLING_LIMIT_S = 180.0
#: Fixed before any evaluation; not tuned. See docs/current_evidence.md.
DEMONSTRATION_CONTROLLER = {
    "control_type": "rate_profile", "K_omega": 20.0,
    "profile_rate_gain": 0.05, "profile_accel_fraction": 0.7,
}


def run_inputs(seed: int, nominal_inertia: np.ndarray, inertia_sigma: float = 0.05,
               attitude_deg: float = 60.0, rate_rad_s: float = 0.1) -> Dict[str, Any]:
    """Rebuild one Monte Carlo run's dispersion exactly as ``run_simulation`` draws it."""
    conditions = sample_initial_conditions(seed, attitude_deg, None, rate_rad_s)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
    inertia, _ = sample_inertia_from_prior(nominal_inertia, inertia_sigma, rng)
    return {"conditions": conditions, "inertia": inertia}


def analyse_run(seed: int, nominal_inertia: np.ndarray, tau_max: float, k_q: float,
                k_omega: float, inertia_sigma: float = 0.05) -> Dict[str, Any]:
    inputs = run_inputs(seed, nominal_inertia, inertia_sigma)
    inertia, c = inputs["inertia"], inputs["conditions"]
    rotation = error_quat_to_vec(right_error_quat(c["quaternion"], np.array([0, 0, 0, 1.0])))
    angle = float(np.linalg.norm(rotation))
    axis = rotation / angle
    omega = c["rate"]
    j_eff = float(axis @ inertia @ axis)
    along = float(omega @ axis)
    momentum_a = momentum_lower_bound_s(inertia, omega, tau_max)
    momentum_b = momentum_lower_bound_s(inertia, omega, tau_max, per_wheel_clips=True)
    slew = slew_time_rest_to_rest_s(angle, j_eff, tau_max)
    estimate = eigenaxis_feasibility_estimate_s(angle, along, j_eff, tau_max)
    linear = step_settling_time_s(j_eff, k_q, k_omega, theta0=angle, omega0=along)
    saturated = saturated_single_axis_settling_s(angle, along, j_eff, k_q, k_omega, tau_max)
    plan = stop_then_slew_plan(inertia, c["quaternion"], omega, tau_max)
    return {
        "seed": int(seed), "angle_deg": float(np.rad2deg(angle)), "j_eff_kg_m2": j_eff,
        "rate_along_axis_rad_s": along,
        "rate_perpendicular_rad_s": float(np.linalg.norm(omega - along * axis)),
        "momentum_bound_a_s": momentum_a, "momentum_bound_b_s": momentum_b,
        "slew_rest_to_rest_s": slew, "feasibility_estimate_s": estimate,
        "linear_unsaturated_settling_s": (linear if np.isfinite(linear) else None),
        "saturated_single_axis_settling_s": (saturated if np.isfinite(saturated) else None),
        "stop_then_slew_plan_s": plan["total_s"], "plan_stop_s": plan["stop_s"],
        "plan_slew_s": plan["slew_s"], "plan_angle_after_stop_deg": plan["angle_after_stop_deg"],
    }


def _count(rows: List[Dict[str, Any]], predicate) -> int:
    return int(sum(1 for r in rows if predicate(r)))


def summarise(rows: List[Dict[str, Any]], passed: List[bool]) -> Dict[str, Any]:
    est = np.array([r["feasibility_estimate_s"] for r in rows])
    lin = np.array([np.inf if r["linear_unsaturated_settling_s"] is None
                    else r["linear_unsaturated_settling_s"] for r in rows])
    sat = np.array([np.inf if r["saturated_single_axis_settling_s"] is None
                    else r["saturated_single_axis_settling_s"] for r in rows])
    plan = np.array([r["stop_then_slew_plan_s"] for r in rows])
    plan_ok = plan <= SETTLING_LIMIT_S
    feasible = est <= SETTLING_LIMIT_S
    linear_ok = lin <= SETTLING_LIMIT_S
    sat_ok = sat <= SETTLING_LIMIT_S
    passed = np.asarray(passed, dtype=bool)
    pct = lambda a: [float(np.percentile(a, q)) for q in (5, 50, 95)]
    return {
        "n_runs": len(rows), "limit_s": SETTLING_LIMIT_S,
        "momentum_bound_a_above_limit": _count(rows, lambda r: r["momentum_bound_a_s"] > SETTLING_LIMIT_S),
        "momentum_bound_b_above_limit": _count(rows, lambda r: r["momentum_bound_b_s"] > SETTLING_LIMIT_S),
        "slew_rest_to_rest_above_limit": _count(rows, lambda r: r["slew_rest_to_rest_s"] > SETTLING_LIMIT_S),
        "feasibility_estimate_above_limit": int((~feasible).sum()),
        "feasibility_estimate_at_or_below_limit": int(feasible.sum()),
        "linear_unsaturated_above_limit_or_never": int((~linear_ok).sum()),
        "feasible_by_estimate_but_linear_gains_exceed_limit": int((feasible & ~linear_ok).sum()),
        "feasible_by_estimate_and_linear_gains_within_limit": int((feasible & linear_ok).sum()),
        "infeasible_by_estimate_and_linear_gains_exceed_limit": int((~feasible & ~linear_ok).sum()),
        "infeasible_by_estimate_but_linear_within_limit": int((~feasible & linear_ok).sum()),
        "saturated_single_axis_above_limit_or_never": int((~sat_ok).sum()),
        "feasible_by_estimate_but_saturated_gains_exceed_limit": int((feasible & ~sat_ok).sum()),
        "feasible_by_estimate_and_saturated_gains_within_limit": int((feasible & sat_ok).sum()),
        "infeasible_by_estimate_and_saturated_gains_exceed_limit": int((~feasible & ~sat_ok).sum()),
        "infeasible_by_estimate_but_saturated_within_limit": int((~feasible & sat_ok).sum()),
        "monte_carlo_settled_within_limit": _count(
            rows, lambda r: r.get("monte_carlo_settling_time_s") is not None
            and r["monte_carlo_settling_time_s"] <= SETTLING_LIMIT_S),
        "stop_then_slew_plan_at_or_below_limit": int(plan_ok.sum()),
        "stop_then_slew_plan_above_limit": int((~plan_ok).sum()),
        "between_estimate_and_plan_undetermined": int((feasible & ~plan_ok).sum()),
        "monte_carlo_passes": int(passed.sum()),
        "monte_carlo_passes_among_feasible_by_estimate": int((passed & feasible).sum()),
        "monte_carlo_passes_among_infeasible_by_estimate": int((passed & ~feasible).sum()),
        "percentiles_5_50_95": {
            "momentum_bound_a_s": pct([r["momentum_bound_a_s"] for r in rows]),
            "slew_rest_to_rest_s": pct([r["slew_rest_to_rest_s"] for r in rows]),
            "feasibility_estimate_s": pct(est),
            "stop_then_slew_plan_s": pct(plan),
        },
        "estimate_max_s": float(est.max()),
    }


def demonstration_indices(rows: List[Dict[str, Any]], count: int) -> List[int]:
    """Evenly spaced runs, by increasing stop-then-slew plan time, among those the estimate
    calls feasible: spans runs the constructive plan already achieves and runs it does not."""
    order = [i for i in np.argsort([r["stop_then_slew_plan_s"] for r in rows])
             if rows[i]["feasibility_estimate_s"] <= SETTLING_LIMIT_S]
    if count >= len(order):
        return [int(i) for i in order]
    picks = np.linspace(0, len(order) - 1, count).round().astype(int)
    return [int(order[p]) for p in picks]


def make_figure(rows: List[Dict[str, Any]], passed: List[bool], path: Path) -> None:
    est = np.array([r["feasibility_estimate_s"] for r in rows])
    mom = np.array([r["momentum_bound_a_s"] for r in rows])
    order = np.argsort(est)
    passed = np.asarray(passed, dtype=bool)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(rows))
    ax.plot(x, mom[order], "-", color="0.55", lw=1.2, label="momentum bound (proven)")
    colors = np.where(passed[order], "tab:green", "tab:red")
    ax.scatter(x, est[order], c=colors, s=14, zorder=3)
    plan = np.array([r["stop_then_slew_plan_s"] for r in rows])
    ax.plot(x, plan[order], "-", color="tab:blue", lw=1.2, alpha=0.8,
            label="stop-then-slew plan (constructive, achievable in the idealisation)")
    ax.axhline(SETTLING_LIMIT_S, color="k", ls="--", lw=1, label="180 s criterion")
    ax.scatter([], [], c="tab:red", s=14, label="feasibility estimate, run failed E6")
    ax.scatter([], [], c="tab:green", s=14, label="feasibility estimate, run passed E6")
    ax.set_xlabel("run, sorted by feasibility estimate")
    ax.set_ylabel("time (s)")
    ax.set_title("E6 Monte Carlo: lower bound and feasibility estimate per run")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monte-carlo", type=Path, default=ROOT / "docs" / "e6_monte_carlo.json")
    parser.add_argument("--demonstrate", type=int, default=0, metavar="N",
                        help="run the nonlinear simulation with the rate_profile controller on "
                             "N runs the estimate calls feasible")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "e6_feasibility.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "docs" / "figures" / "e6_feasibility.png")
    args = parser.parse_args()

    mc = json.loads(args.monte_carlo.read_text())
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    nominal = np.array(base.spacecraft["inertia"], dtype=float)
    tau_max = float(base.controller["max_torque"])
    k_q, k_omega = float(base.controller["K_q"]), float(base.controller["K_omega"])
    sigma = float(mc["dispersions"]["inertia"]["relative_sigma"])
    seeds = [int(s) for s in mc["seeds"]]
    passed = [bool(r["passed"]) for r in mc["runs"]]
    assert [r["seed"] for r in mc["runs"]] == seeds

    rows = [analyse_run(s, nominal, tau_max, k_q, k_omega, sigma) for s in seeds]
    for row, run in zip(rows, mc["runs"]):
        row["monte_carlo_passed"] = bool(run["passed"])
        row["monte_carlo_settling_time_s"] = run["settling_time_s"]
        row["monte_carlo_peak_terminal_error_deg"] = run["peak_terminal_error_deg"]

    report: Dict[str, Any] = {
        "label": "E6 feasibility analysis; criteria are project-defined, not mission requirements",
        "source": "docs/e6_monte_carlo.json (unchanged); dispersions rebuilt from the same seeds",
        "assumptions": {"tau_max_n_m": tau_max, "wheel_initial_momentum": 0.0,
                        "gains": {"K_q": k_q, "K_omega": k_omega,
                                  "control_type": base.controller["control_type"]},
                        "stored_result_horizon_s": float(mc["duration_s"])},
        **completeness(len(seeds), seeds, [r["seed"] for r in rows]),
        "principal_axis_linear_analysis": principal_axis_linear_summary(
            nominal, k_q, k_omega, base.controller["control_type"]),
        "summary": summarise(rows, passed),
        "runs": rows,
    }

    if args.demonstrate:
        picks = demonstration_indices(rows, args.demonstrate)
        tasks = [{"index": i, "seed": seeds[i], "attitude_deg": 60.0, "attitude_range_deg": None,
                  "rate_rad_s": 0.1, "dispersions": ["attitude", "rate", "inertia"],
                  "inertia_sigma": sigma, "duration": None,
                  "controller_overrides": DEMONSTRATION_CONTROLLER} for i in picks]
        started = wall_clock.time()
        results = run_tasks(run_one, tasks, args.jobs, label="demonstration run")
        demo_rows = []
        for i, res in zip(picks, results):
            settle = res["settling_time_s"]
            settled = bool(np.isfinite(settle))
            demo_rows.append({
                "seed": seeds[i], "feasibility_estimate_s": rows[i]["feasibility_estimate_s"],
                "stop_then_slew_plan_s": rows[i]["stop_then_slew_plan_s"],
                "settling_time_s": float(settle) if settled else None,
                "settled_within_limit": bool(settled and settle <= SETTLING_LIMIT_S),
                "ratio_to_feasibility_estimate": (float(settle / rows[i]["feasibility_estimate_s"])
                                         if settled else None),
                "passed_all_criteria": bool(res["passed"]),
                "failed_criteria": [n for n, ok in res["checks"].items() if not ok],
                "peak_terminal_error_deg": res["peak_terminal_error_deg"],
                "peak_wheel_momentum_use": res["peak_wheel_momentum_use"],
                "original_gains_passed": rows[i]["monte_carlo_passed"],
            })
        report["demonstration"] = {
            "note": ("rate_profile controller, gyro-driven filter, estimated-state feedback, "
                     "run on runs the feasibility estimate calls feasible. Parameters fixed in "
                     "advance, not tuned. Existing 200-run seeds: a demonstration of "
                     "achievability, not a pass-probability estimate (see B5 for that)."),
            "controller": {**base.controller, **DEMONSTRATION_CONTROLLER},
            **completeness(len(tasks), [t["seed"] for t in tasks], [r["seed"] for r in results]),
            "wall_time_s": wall_clock.time() - started,
            "n_settled_within_limit": int(sum(r["settled_within_limit"] for r in demo_rows)),
            "n_passed_all_criteria": int(sum(r["passed_all_criteria"] for r in demo_rows)),
            "runs": demo_rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figure(rows, passed, args.figure)
    s = report["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "percentiles_5_50_95"}, indent=2))
    print("percentiles (5, 50, 95):", s["percentiles_5_50_95"])
    if args.demonstrate:
        d = report["demonstration"]
        print(f"demonstration: {d['n_settled_within_limit']}/{d['n_requested']} settled within "
              f"{SETTLING_LIMIT_S:.0f} s, {d['n_passed_all_criteria']} passed all criteria")
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
