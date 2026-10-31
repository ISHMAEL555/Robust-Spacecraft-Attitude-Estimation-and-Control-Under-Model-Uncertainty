"""Compare the drifted GNC-ACQ-001 results under the published and the absolute settling band.

Reads ``docs/b5_evaluation/acq_drifted__*.json`` (published, 2% of the initial error)
and ``docs/b5_evaluation_absband/acq_drifted__*.json`` (same seeds and frozen
controllers, re-run to store histories; scored with both bands). Checks that the
re-run reproduces the published per-run results exactly, then tabulates passes with
Clopper-Pearson intervals, settling times, changed verdicts, and exact McNemar tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.b5_paired_tests import exact_mcnemar
from scripts.run_e6_monte_carlo import clopper_pearson

CONTROLLERS = ("pd_zeta08", "rate_profile", "two_phase", "two_phase_ff")
REPRODUCED = ("seed", "passed", "settling_time_s", "peak_terminal_error_deg",
              "peak_wheel_momentum_use", "handover_time_s", "saturation_duty", "plan_s")
PAIRS = (("pd_zeta08", "two_phase"), ("two_phase", "two_phase_ff"),
         ("rate_profile", "two_phase"))


def _pct(values):
    finite = [v for v in values if v is not None]
    if len(finite) < len(values) / 2:
        return {"median": None, "p05": None, "p95": None, "n_none": len(values) - len(finite)}
    a = np.array([np.inf if v is None else v for v in values], dtype=float)
    q = lambda p: float(np.percentile(a, p)) if np.isfinite(np.percentile(a, p)) else None
    return {"median": q(50), "p05": q(5), "p95": q(95), "n_none": len(values) - len(finite)}


def main() -> None:
    old_dir, new_dir = ROOT / "docs" / "b5_evaluation", ROOT / "docs" / "b5_evaluation_absband"
    out = {"controllers": {}, "mcnemar_absolute_band": [], "mcnemar_published_band": []}
    outcomes = {}
    for c in CONTROLLERS:
        old = json.loads((old_dir / f"acq_drifted__{c}.json").read_text())
        new = json.loads((new_dir / f"acq_drifted__{c}.json").read_text())
        assert new["n_completed"] == new["n_requested"] == 200
        for o, n in zip(old["runs"], new["runs"]):
            for key in REPRODUCED:
                if o[key] != n[key]:
                    raise SystemExit(f"{c} seed {o['seed']}: {key} differs, trajectories not identical")
        published = [r["passed"] for r in new["runs"]]
        absolute = [r["absolute_band"]["passed"] for r in new["runs"]]
        outcomes[c] = (published, absolute)
        k_old, k_new = sum(published), sum(absolute)
        changed = [{"seed": r["seed"], "published": p, "absolute": a,
                    "initial_error_deg": r["initial_error_deg"],
                    "settling_published_s": r["settling_time_s"],
                    "settling_absolute_s": r["absolute_band"]["settling_time_s"],
                    "absolute_failed_criteria": [k for k, ok in r["absolute_band"]["checks"].items() if not ok]}
                   for r, p, a in zip(new["runs"], published, absolute) if p != a]
        out["controllers"][c] = {
            "published_band": {"passes": k_old, "ci95": list(clopper_pearson(k_old, 200)),
                               "settling_time_s": _pct([r["settling_time_s"] for r in new["runs"]])},
            "absolute_band": {"passes": k_new, "ci95": list(clopper_pearson(k_new, 200)),
                              "settling_time_s": _pct([r["absolute_band"]["settling_time_s"] for r in new["runs"]])},
            "verdicts_changed": len(changed),
            "pass_to_fail": sum(1 for x in changed if x["published"]),
            "fail_to_pass": sum(1 for x in changed if not x["published"]),
            "changed_runs": changed,
        }
    for a, b in PAIRS:
        out["mcnemar_published_band"].append({"a": a, "b": b, **exact_mcnemar(outcomes[a][0], outcomes[b][0])})
        out["mcnemar_absolute_band"].append({"a": a, "b": b, **exact_mcnemar(outcomes[a][1], outcomes[b][1])})
    (new_dir / "drifted_rescore_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for c, s in out["controllers"].items():
        p, a = s["published_band"], s["absolute_band"]
        print(f"{c:13s} published {p['passes']:3d} [{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] "
              f"settle {p['settling_time_s']}  | absolute {a['passes']:3d} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}] "
              f"settle {a['settling_time_s']}  | changed {s['verdicts_changed']} "
              f"(pass->fail {s['pass_to_fail']}, fail->pass {s['fail_to_pass']})")
    for label in ("mcnemar_published_band", "mcnemar_absolute_band"):
        for t in out[label]:
            print(label, t["a"], "vs", t["b"], t["a_passes"], "/", t["b_passes"],
                  "discordant", t["a_only"], "/", t["b_only"], "p", f"{t['p_value']:.3g}")


if __name__ == "__main__":
    main()
