"""Proven lower bounds on E6 end-to-end time (tip-off to acquired attitude at rest), per run.

No simulation: each run's initial attitude, rate and truth inertia are rebuilt from its
seed. Two bounds hold for any controller whose delivered body torque norm is at most
``tau_max`` (wheels start at zero momentum):

* momentum bound to the final-rate criterion: ``(|J w0| - lambda_max 1e-3) / tau_max``;
* angle-reach bound: the time to rotate through the geodesic distance from the initial
  attitude to the 2% settling band, with the rate growth limited by the torque.

Their maximum is a lower bound on end-to-end time. Their sum is NOT: a controller can
turn toward the target while it brakes. The constructive plan is not used: it is not a
lower bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_b5_comparison import scenario_initial_state
from src.control.feasibility import angle_reach_lower_bound_s, momentum_lower_bound_to_rate_s
from src.dynamics.quaternion import error_quat_to_vec, right_error_quat

NOMINAL = np.diag([100.0, 80.0, 60.0])
TAU_MAX = 0.1
FINAL_RATE = 1e-3       # E6 final body-rate criterion (rad/s)
BAND = 0.02             # E6 2% settling band
LIMIT_S = 180.0


def bounds_for_seed(seed: int, inertia_sigma: float = 0.05) -> dict:
    state = scenario_initial_state("envelope", seed, NOMINAL, inertia_sigma, 0.01, TAU_MAX)
    angle = float(np.linalg.norm(error_quat_to_vec(
        right_error_quat(state["quaternion"], np.array([0.0, 0.0, 0.0, 1.0])))))
    angle_bound = angle_reach_lower_bound_s(state["inertia"], state["rate"],
                                            (1.0 - BAND) * angle, TAU_MAX)
    momentum_bound = momentum_lower_bound_to_rate_s(state["inertia"], state["rate"],
                                                    FINAL_RATE, TAU_MAX)
    return {"seed": int(seed), "angle_reach_bound_s": angle_bound,
            "momentum_bound_to_final_rate_s": momentum_bound,
            "end_to_end_lower_bound_s": max(angle_bound, momentum_bound)}


def summarise(rows: list) -> dict:
    out = {"n_runs": len(rows)}
    for key in ("angle_reach_bound_s", "momentum_bound_to_final_rate_s", "end_to_end_lower_bound_s"):
        values = np.array([r[key] for r in rows])
        out[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                    "max": float(values.max())}
    out["runs_with_bound_above_180_s"] = int(sum(r["end_to_end_lower_bound_s"] > LIMIT_S for r in rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "e6_end_to_end_bounds.json")
    args = parser.parse_args()
    sets = {"published_e6_monte_carlo": [42 + 1000 * i for i in range(200)],
            "b5_evaluation_envelope": [20000042 + 1000 * i for i in range(200)]}
    report = {"label": "Proven lower bounds on E6 end-to-end time; no simulation",
              "assumptions": {"tau_max_n_m": TAU_MAX, "wheel_initial_momentum": 0.0,
                              "final_rate_rad_s": FINAL_RATE, "settling_band": BAND},
              "sets": {}}
    for name, seeds in sets.items():
        rows = [bounds_for_seed(s) for s in seeds]
        report["sets"][name] = {"summary": summarise(rows), "runs": rows}
        s = report["sets"][name]["summary"]
        print(name, {k: v for k, v in s.items() if k != "n_runs"})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
