"""B5: controller comparison for detumble, acquisition and the end-to-end E6 envelope.

Controllers come from a YAML file of named controller overrides (frozen in
``config/b5_controllers.yaml`` before the evaluation run). Scenarios:

``envelope``      the E6 Monte Carlo dispersion: 60 deg about a random axis, 0.1 rad/s in a
                  random direction, inertia sampled from the 5% prior, new sensor noise.
                  Reports GNC-DET-001 (time to |w| <= w_entry) and the end-to-end E6
                  criteria, characterized against the constructive plan time.
``acq_axis60``    GNC-ACQ-001: 60 deg about a random axis, |w0| = w_entry in a random
                  direction, inertia sampled, new sensor noise.
``acq_drifted``   GNC-ACQ-001: the attitude reached when the envelope draw is stopped at
                  full torque (``stop_then_slew_plan``), |w0| = w_entry in the draw's
                  rate direction, same inertia.

Every campaign stops on any failed run and records ``n_requested``, ``n_completed`` and
the seeds. Results do not depend on ``--jobs``. Criteria are project-defined.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time as wall_clock
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "mplconfig"))

import numpy as np
import yaml

from scripts.parallel_runs import completeness, run_tasks
from scripts.run_e6_monte_carlo import (
    build_config,
    clopper_pearson,
    distribution_summary,
    sample_initial_conditions,
)
from scripts.run_experiment import (
    compute_performance_metrics,
    evaluate_e6_project_benchmark,
    load_config,
    run_simulation,
)
from src.control.feasibility import stop_then_slew_plan
from src.dynamics.inertia_sampling import sample_inertia_from_prior

SCENARIOS = ("envelope", "acq_axis60", "acq_drifted")
SEED_STRIDE = 1000
SATURATION_FRACTION = 0.999
WHEEL_TIME_CONSTANT_S = 0.05


def low_pass(signal: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """First-order low-pass along axis 0, started at the first sample."""
    alpha = dt / (tau + dt)
    out = np.empty_like(signal, dtype=float)
    out[0] = signal[0]
    for k in range(1, len(signal)):
        out[k] = out[k - 1] + alpha * (signal[k] - out[k - 1])
    return out


def in_band_effort_fraction(tau_cmd: np.ndarray, tau_rw: np.ndarray, dt: float) -> float:
    """Achieved / requested squared-torque effort after low-passing both at the wheel bandwidth.

    A diagnostic, not a criterion: it removes the command noise above the wheel
    bandwidth that the wheels cannot deliver anyway.
    """
    requested = low_pass(tau_cmd, dt, WHEEL_TIME_CONSTANT_S)
    achieved = low_pass(tau_rw, dt, WHEEL_TIME_CONSTANT_S)
    return float(np.sum(achieved**2) / np.sum(requested**2))


def _inertia_for_seed(seed: int, nominal: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
    return sample_inertia_from_prior(nominal, sigma, rng)[0]


def scenario_initial_state(scenario: str, seed: int, nominal: np.ndarray, sigma: float,
                           entry_rate: float, tau_max: float) -> Dict[str, Any]:
    """Initial attitude and rate for one run; the truth inertia comes from the run's stream 3."""
    inertia = _inertia_for_seed(seed, nominal, sigma)
    if scenario == "envelope":
        c = sample_initial_conditions(seed, 60.0, None, 0.1)
        return {"quaternion": c["quaternion"], "rate": c["rate"], "inertia": inertia}
    if scenario == "acq_axis60":
        c = sample_initial_conditions(seed, 60.0, None, entry_rate)
        return {"quaternion": c["quaternion"], "rate": c["rate"], "inertia": inertia}
    if scenario == "acq_drifted":
        c = sample_initial_conditions(seed, 60.0, None, 0.1)
        stop = stop_then_slew_plan(inertia, c["quaternion"], c["rate"], tau_max)
        return {"quaternion": np.asarray(stop["q_after_stop"]),
                "rate": entry_rate * c["rate_direction"], "inertia": inertia}
    raise ValueError(f"unknown scenario {scenario!r}")


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    base = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    nominal = np.array(base.spacecraft["inertia"], dtype=float)
    tau_max = float(base.controller["max_torque"])
    sigma, entry = task["inertia_sigma"], task["entry_rate"]
    state = scenario_initial_state(task["scenario"], task["seed"], nominal, sigma, entry, tau_max)
    conditions = {"quaternion": state["quaternion"], "rate": state["rate"]}
    config = build_config(base, task["seed"], conditions, ("attitude", "rate", "inertia"), sigma)
    config = replace(config, controller={**config.controller, **task["controller"]})
    if task.get("duration") is not None:
        config = replace(config, duration=float(task["duration"]))
    results = run_simulation(config)
    # The simulation must have drawn the same truth inertia the plan used.
    assert np.array_equal(results["inertia_true_sampled"], state["inertia"])
    metrics = compute_performance_metrics(results)
    benchmark = evaluate_e6_project_benchmark(metrics)

    time = results["time"]
    dt = float(config.dt)
    rate_true = np.linalg.norm(results["omega_true"], axis=1)
    below = np.nonzero(rate_true <= entry)[0]
    det_time = float(time[below[0]]) if below.size else None  # first true crossing
    torque = np.linalg.norm(results["tau_cmd"], axis=1)
    phase = results.get("control_phase")
    if phase is not None:
        changes = np.nonzero(np.diff(np.concatenate([[0], phase.astype(int)])))[0]
        switch_time = float(time[changes[0]]) if changes.size else None
        n_switches = int(changes.size)
        hat_rate = np.linalg.norm(results["omega_hat"], axis=1)
        switch_rate_hat = float(hat_rate[changes[0]]) if changes.size else None
        switch_rate_true = float(rate_true[changes[0]]) if changes.size else None
        signal = results["switch_signal"]
        # Error of the switching signal against the same filter applied to the true
        # rate, over the 10 s before the handover (noise + bias + norm bias).
        tau_f = task["controller"].get("switch_filter_time_constant_s")
        reference = low_pass(rate_true, dt, tau_f) if tau_f else rate_true
        if changes.size:
            window = slice(max(0, changes[0] - int(10.0 / dt)), changes[0])
            error = signal[window] - reference[window]
            switch_error_mean = float(np.mean(error)) if error.size else None
            switch_error_std = float(np.std(error)) if error.size else None
        else:
            switch_error_mean = switch_error_std = None
    else:
        switch_time = n_switches = switch_rate_hat = switch_rate_true = None
        switch_error_mean = switch_error_std = None
    # GNC-DET-001 handover: the mode switch for a mode manager, otherwise the first
    # time the true rate reaches w_entry. Pass if the TRUE rate at the handover is
    # <= w_entry and the handover is within T_det. A later speed-up by the
    # acquisition slew is intended and is not a detumble failure.
    if phase is not None:
        handover_time, handover_true_rate = switch_time, switch_rate_true
    else:
        handover_time = det_time
        handover_true_rate = float(rate_true[below[0]]) if below.size else None

    plan = stop_then_slew_plan(state["inertia"], state["quaternion"], state["rate"], tau_max)
    settle = metrics["settling_time_2pct_s"]
    settled = bool(np.isfinite(settle))
    return {
        "index": task["index"], "seed": task["seed"],
        "passed": bool(benchmark["passed"]),
        "checks": {n: bool(c["passed"]) for n, c in benchmark["checks"].items()},
        "settling_time_s": float(settle) if settled else None,
        "peak_terminal_error_deg": float(np.rad2deg(metrics["pointing_peak_final_20s_rad"])),
        "initial_error_deg": float(np.rad2deg(results["pointing_error"][0])),
        "peak_wheel_momentum_use": float(metrics["peak_wheel_momentum_utilization"]),
        "peak_commanded_torque_nm": float(torque.max()),
        "saturation_duty": float(np.mean(torque >= SATURATION_FRACTION * tau_max)),
        "detumble_time_s": det_time,
        "handover_time_s": handover_time, "handover_true_rate_rad_s": handover_true_rate,
        "switch_signal_error_mean_rad_s": switch_error_mean,
        "switch_signal_error_std_rad_s": switch_error_std,
        "achieved_effort_fraction": float(metrics.get("achieved_effort_fraction", float("nan"))),
        "in_band_effort_fraction": in_band_effort_fraction(
            results["tau_cmd"], results["tau_rw"], dt),
        "mode_switch_time_s": switch_time, "mode_switches": n_switches,
        "switch_rate_hat_rad_s": switch_rate_hat, "switch_rate_true_rad_s": switch_rate_true,
        "plan_s": plan["total_s"], "plan_stop_s": plan["stop_s"],
        # Ratio to the constructive plan, which is NOT a lower bound (ratios below 1 occur).
        "ratio_to_constructive_plan": float(settle / plan["total_s"]) if settled else None,
    }


def seed_range(base: int, n: int) -> Dict[str, int]:
    return {"base": int(base), "stride": SEED_STRIDE, "n": int(n),
            "first": int(base), "last": int(base + SEED_STRIDE * (n - 1))}


def check_disjoint(ranges: Dict[str, Dict[str, int]]) -> None:
    """Raise if any two seed ranges share a seed."""
    sets = {name: {r["base"] + r["stride"] * i for i in range(r["n"])}
            for name, r in ranges.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sets[a] & sets[b]
            if common:
                raise ValueError(f"seed ranges {a!r} and {b!r} overlap, e.g. {min(common)}")


PUBLISHED_RANGE = seed_range(42, 200)


def _ratio(k: int, n: int) -> Dict[str, Any]:
    low, high = clopper_pearson(k, n)
    return {"passes": k, "n": n, "p": k / n, "ci95_clopper_pearson": [low, high]}


def _dist(values: List[Any]) -> Dict[str, Any]:
    finite = [v for v in values if v is not None]
    arr = np.array([np.inf if v is None else v for v in values], dtype=float)
    out = distribution_summary(arr)
    out["n_none"] = len(values) - len(finite)
    return out


def det_passed(run: Dict[str, Any], det_limit_s: float, entry_rate: float) -> bool:
    """GNC-DET-001 at the handover instant, with actuator limits respected."""
    return bool(run["handover_time_s"] is not None and run["handover_time_s"] <= det_limit_s
                and run["handover_true_rate_rad_s"] is not None
                and run["handover_true_rate_rad_s"] <= entry_rate
                and run["checks"]["peak_commanded_torque_nm"] and run["checks"]["peak_wheel_torque_nm"]
                and run["checks"]["peak_wheel_momentum_utilization"])


def aggregate(runs: List[Dict[str, Any]], det_limit_s: float,
              entry_rate: float = 0.01) -> Dict[str, Any]:
    n = len(runs)
    criteria = list(runs[0]["checks"])
    det_ok = [det_passed(r, det_limit_s, entry_rate) for r in runs]
    switches = [r["mode_switches"] for r in runs if r["mode_switches"] is not None]
    return {
        "n_runs": n,
        "e6_all_criteria": _ratio(sum(r["passed"] for r in runs), n),
        "criteria": {c: _ratio(sum(r["checks"][c] for r in runs), n) for c in criteria},
        "detumble_within_limit": _ratio(int(sum(det_ok)), n),
        "settling_time_s": _dist([r["settling_time_s"] for r in runs]),
        "ratio_to_constructive_plan": _dist([r["ratio_to_constructive_plan"] for r in runs]),
        "plan_s": _dist([r["plan_s"] for r in runs]),
        "detumble_time_s": _dist([r["detumble_time_s"] for r in runs]),
        "handover_time_s": _dist([r["handover_time_s"] for r in runs]),
        "handover_true_rate_rad_s": _dist([r["handover_true_rate_rad_s"] for r in runs]),
        "handover_true_rate_above_entry": int(sum(
            r["handover_true_rate_rad_s"] is not None and r["handover_true_rate_rad_s"] > entry_rate
            for r in runs)),
        "switch_signal_error_mean_rad_s": _dist([r["switch_signal_error_mean_rad_s"] for r in runs]),
        "switch_signal_error_std_rad_s": _dist([r["switch_signal_error_std_rad_s"] for r in runs]),
        "achieved_effort_fraction": _dist([r["achieved_effort_fraction"] for r in runs]),
        "in_band_effort_fraction": _dist([r["in_band_effort_fraction"] for r in runs]),
        "in_band_effort_at_or_above_0_85": int(sum(r["in_band_effort_fraction"] >= 0.85 for r in runs)),
        "mode_switch_time_s": _dist([r["mode_switch_time_s"] for r in runs]),
        "mode_switches": ({"max": int(max(switches)), "runs_with_more_than_one": int(
            sum(s > 1 for s in switches)), "runs_with_none": int(sum(s == 0 for s in switches))}
            if switches else None),
        "peak_wheel_momentum_use": _dist([r["peak_wheel_momentum_use"] for r in runs]),
        "peak_commanded_torque_nm_max": float(max(r["peak_commanded_torque_nm"] for r in runs)),
        "saturation_duty": _dist([r["saturation_duty"] for r in runs]),
        "peak_terminal_error_deg": _dist([r["peak_terminal_error_deg"] for r in runs]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers-file", type=Path, default=ROOT / "config" / "b5_controllers.yaml")
    parser.add_argument("--controllers", nargs="*", default=None, help="subset of names")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--stage", choices=("design", "evaluation"), required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--design-base-seed", type=int, default=None,
                        help="evaluation stage: base seed of the design range (recorded, checked)")
    parser.add_argument("--design-runs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None, help="smoke tests only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec_text = args.controllers_file.read_text(encoding="utf-8")
    spec = yaml.safe_load(spec_text)
    entry = float(spec["requirements"]["detumble_entry_rate"])
    det_limit = float(spec["requirements"]["detumble_time_limit_s"])
    sigma = float(spec["dispersion"]["inertia_sigma"])
    names = args.controllers or list(spec["controllers"])
    seeds = [args.base_seed + SEED_STRIDE * i for i in range(args.runs)]
    ranges = {"published_e6_monte_carlo": PUBLISHED_RANGE}
    if args.stage == "design":
        ranges["design"] = seed_range(args.base_seed, args.runs)
    else:
        if args.design_base_seed is None or args.design_runs is None:
            parser.error("the evaluation stage needs --design-base-seed and --design-runs")
        ranges["design"] = seed_range(args.design_base_seed, args.design_runs)
        ranges["evaluation"] = seed_range(args.base_seed, args.runs)
    check_disjoint(ranges)  # raises on any overlap
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        for name in names:
            controller = spec["controllers"][name]
            tasks = [{"index": i, "seed": s, "scenario": scenario, "controller": controller,
                      "entry_rate": entry, "inertia_sigma": sigma, "duration": args.duration}
                     for i, s in enumerate(seeds)]
            started = wall_clock.time()
            runs = run_tasks(run_one, tasks, args.jobs, label=f"{scenario}/{name} run")
            report = {
                "label": "B5 controller comparison; criteria are project-defined, not mission requirements",
                "stage": args.stage, "scenario": scenario, "controller_name": name,
                "controller_overrides": controller,
                "controllers_file": str(args.controllers_file.relative_to(ROOT)
                                        if args.controllers_file.is_relative_to(ROOT)
                                        else args.controllers_file),
                "controllers_file_sha256": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
                "requirements": spec["requirements"], "dispersion": spec["dispersion"],
                "seed_ranges": ranges,
                **completeness(len(tasks), seeds, [r["seed"] for r in runs]),
                "wall_time_s": wall_clock.time() - started,
                "summary": aggregate(runs, det_limit, entry),
                "runs": runs,
            }
            path = args.output_dir / f"{scenario}__{name}.json"
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            s = report["summary"]
            e6, det = s["e6_all_criteria"], s["detumble_within_limit"]
            print(f"{scenario:12s} {name:16s} E6 {e6['passes']:3d}/{e6['n']} "
                  f"[{e6['ci95_clopper_pearson'][0]:.3f},{e6['ci95_clopper_pearson'][1]:.3f}]  "
                  f"DET {det['passes']:3d}/{det['n']}  settle med {s['settling_time_s']['median']}  "
                  f"ratio-to-plan med {s['ratio_to_constructive_plan']['median']}  "
                  f"({report['wall_time_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
