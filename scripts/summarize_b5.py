"""Per-phase optimality gaps for the B5 evaluation (post-processing only; no simulation).

For each run of each ``{scenario}__{controller}.json`` in the evaluation directory:

* detumble-phase gap   = handover time / plan stopping time (the plan's stopping
  phase is within a few percent of the proven momentum bound);
* acquisition-phase gap = (settling time - handover time) / plan slew time, for
  runs that settled;
* end-to-end gap       = settling time / plan time, for runs that settled.

Runs that did not settle within the 200 s horizon have no acquisition or
end-to-end gap; they are counted, not dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _pct(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p05": None, "p95": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def phase_gaps(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    detumble, acquisition, end_to_end = [], [], []
    for r in runs:
        handover = r["handover_time_s"]
        if handover is not None and r["plan_stop_s"] > 1.0:
            detumble.append(handover / r["plan_stop_s"])
        if r["settling_time_s"] is not None:
            end_to_end.append(r["settling_time_s"] / r["plan_s"])
            slew = r["plan_s"] - r["plan_stop_s"]
            if handover is not None and slew > 0.0:
                acquisition.append((r["settling_time_s"] - handover) / slew)
    return {"n_runs": len(runs),
            "n_not_settled": int(sum(r["settling_time_s"] is None for r in runs)),
            "detumble_phase_gap": _pct(detumble),
            "acquisition_phase_gap": _pct(acquisition),
            "end_to_end_gap": _pct(end_to_end),
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
        out.setdefault(scenario, {})[controller] = phase_gaps(data["runs"])
    output = args.output or args.evaluation_dir / "phase_gaps.json"
    output.write_text(json.dumps(out, indent=2) + "\n")
    fmt = lambda d: "-" if d["median"] is None else f"{d['median']:.2f} [{d['p05']:.2f}, {d['p95']:.2f}] n={d['n']}"
    for scenario, rows in out.items():
        for controller, g in rows.items():
            print(f"{scenario:11s} {controller:13s} not settled {g['n_not_settled']:3d} | "
                  f"detumble {fmt(g['detumble_phase_gap'])} | acquisition {fmt(g['acquisition_phase_gap'])} | "
                  f"end-to-end {fmt(g['end_to_end_gap'])}")


if __name__ == "__main__":
    main()
