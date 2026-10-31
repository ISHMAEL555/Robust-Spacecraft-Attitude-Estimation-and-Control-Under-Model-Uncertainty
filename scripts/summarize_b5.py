"""Per-phase ratios for the B5 evaluation (post-processing only; no simulation).

For each run of each ``{scenario}__{controller}.json`` in the evaluation directory:

* detumble phase, two ratios: handover time / proven momentum bound to the TRUE rate
  at handover, with the run's own dispersed inertia (``momentum_lower_bound_to_rate_s``;
  must be >= 1), and handover time / the constructive plan's stopping time (the plan
  stops to 1e-4 rad/s, a different end condition, so this ratio can be below 1);
* acquisition phase, ratio to constructive plan = (settling time - handover time) /
  plan slew time, for runs that settled;
* end to end, ratio to constructive plan = settling time / plan time, for runs that
  settled.

The constructive plan is NOT a lower bound (ratios below 1 occur). Evaluation files
written before the rename store the per-run end-to-end ratio under the old key
``optimality_gap_ratio``; this script recomputes every ratio from the stored times.

Runs that did not settle within the 200 s horizon have no acquisition or
end-to-end ratio; they are counted, not dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_b5_comparison import scenario_initial_state  # noqa: E402
from src.control.feasibility import momentum_lower_bound_to_rate_s  # noqa: E402

NOMINAL_INERTIA = np.diag([100.0, 80.0, 60.0])
TAU_MAX = 0.1

ROOT = Path(__file__).resolve().parents[1]


def _pct(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p05": None, "p95": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def phase_gaps(runs: List[Dict[str, Any]], scenario: str = "envelope",
               inertia_sigma: float = 0.05, entry_rate: float = 0.01) -> Dict[str, Any]:
    detumble, detumble_bound, acquisition, end_to_end = [], [], [], []
    for r in runs:
        handover = r["handover_time_s"]
        if handover is not None and r["plan_stop_s"] > 1.0:
            detumble.append(handover / r["plan_stop_s"])
        if scenario == "envelope" and handover is not None:
            state = scenario_initial_state(scenario, r["seed"], NOMINAL_INERTIA, inertia_sigma,
                                           entry_rate, TAU_MAX)
            bound = momentum_lower_bound_to_rate_s(state["inertia"], state["rate"],
                                                   r["handover_true_rate_rad_s"], TAU_MAX)
            detumble_bound.append(handover / bound)
        if r["settling_time_s"] is not None:
            end_to_end.append(r["settling_time_s"] / r["plan_s"])
            slew = r["plan_s"] - r["plan_stop_s"]
            if handover is not None and slew > 0.0:
                acquisition.append((r["settling_time_s"] - handover) / slew)
    return {"n_runs": len(runs),
            "n_not_settled": int(sum(r["settling_time_s"] is None for r in runs)),
            "detumble_ratio_to_momentum_bound": _pct(detumble_bound),
            "detumble_ratio_to_plan_stop": _pct(detumble),
            "detumble_runs_below_bound": int(sum(x < 1.0 for x in detumble_bound)),
            "acquisition_ratio_to_constructive_plan": _pct(acquisition),
            "end_to_end_ratio_to_constructive_plan": _pct(end_to_end),
            "settling_time_settled_runs_s": _pct(
                [r["settling_time_s"] for r in runs if r["settling_time_s"] is not None])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "docs" / "b5_evaluation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out: Dict[str, Any] = {}
    for path in sorted(args.evaluation_dir.glob("*__*.json")):
        scenario, controller = path.stem.split("__")
        data = json.loads(path.read_text())
        assert data["n_completed"] == data["n_requested"] == len(data["runs"])
        out.setdefault(scenario, {})[controller] = phase_gaps(
            data["runs"], scenario, float(data["dispersion"]["inertia_sigma"]),
            float(data["requirements"]["detumble_entry_rate"]))
    output = args.output or args.evaluation_dir / "phase_ratios.json"
    output.write_text(json.dumps(out, indent=2) + "\n")
    fmt = lambda d: "-" if d["median"] is None else f"{d['median']:.2f} [{d['p05']:.2f}, {d['p95']:.2f}] n={d['n']}"
    for scenario, rows in out.items():
        for controller, g in rows.items():
            print(f"{scenario:11s} {controller:13s} not settled {g['n_not_settled']:3d} | "
                  f"detumble/bound {fmt(g['detumble_ratio_to_momentum_bound'])} below {g['detumble_runs_below_bound']} | "
                  f"detumble/plan-stop {fmt(g['detumble_ratio_to_plan_stop'])} | acquisition/plan {fmt(g['acquisition_ratio_to_constructive_plan'])} | "
                  f"end-to-end/plan {fmt(g['end_to_end_ratio_to_constructive_plan'])}")


if __name__ == "__main__":
    main()
