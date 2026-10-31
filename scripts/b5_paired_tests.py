"""Exact McNemar tests between B5 controllers on the same seeds (post-processing only).

Within a scenario every controller ran the same seeds, so pass/fail outcomes are paired.
For a pair (A, B): b = runs A passed and B failed, c = runs B passed and A failed. The
exact two-sided p-value is min(1, 2 P(X <= min(b, c))), X ~ Binomial(b + c, 1/2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

COMPARISONS = [
    ("envelope", "two_phase", "two_phase_ff"),
    ("envelope", "rate_profile", "two_phase"),
    ("acq_axis60", "pd_zeta08", "two_phase"),
    ("acq_drifted", "pd_zeta08", "two_phase"),
    ("acq_axis60", "two_phase", "two_phase_ff"),
    ("acq_drifted", "two_phase", "two_phase_ff"),
]


def exact_mcnemar(a: List[bool], b: List[bool]) -> Dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("outcome lists must be paired")
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2.0 * float(stats.binom.cdf(min(only_a, only_b), n, 0.5)))
    return {"a_only": int(only_a), "b_only": int(only_b), "discordant": int(n), "p_value": p,
            "a_passes": int(sum(a)), "b_passes": int(sum(b)), "n": len(a)}


def _load(directory: Path, scenario: str, controller: str) -> Dict[str, Any]:
    return json.loads((directory / f"{scenario}__{controller}.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "docs" / "b5_evaluation")
    args = parser.parse_args()
    results = []
    for scenario, first, second in COMPARISONS:
        a, b = _load(args.evaluation_dir, scenario, first), _load(args.evaluation_dir, scenario, second)
        seeds_a, seeds_b = [r["seed"] for r in a["runs"]], [r["seed"] for r in b["runs"]]
        if seeds_a != seeds_b:
            raise ValueError(f"{scenario}: {first} and {second} do not share seeds in order")
        test = exact_mcnemar([r["passed"] for r in a["runs"]], [r["passed"] for r in b["runs"]])
        results.append({"scenario": scenario, "a": first, "b": second, "outcome": "E6 all criteria",
                        "seeds_identical": True, **test})
    env = _load(args.evaluation_dir, "envelope", "rate_profile")
    det_limit = float(env["requirements"]["detumble_time_limit_s"])
    entry = float(env["requirements"]["detumble_entry_rate"])
    from scripts.run_b5_comparison import det_passed  # noqa: E402
    both = sum(r["passed"] and det_passed(r, det_limit, entry) for r in env["runs"])
    overlap = {"rate_profile_envelope_e6_passes": int(sum(r["passed"] for r in env["runs"])),
               "of_which_also_pass_gnc_det_001": int(both)}
    out = {"comparisons": results, "rate_profile_overlap": overlap}
    (args.evaluation_dir / "paired_tests.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in results:
        print(f"{r['scenario']:11s} {r['a']:12s} vs {r['b']:12s} passes {r['a_passes']:3d} / {r['b_passes']:3d}  "
              f"discordant {r['a_only']} / {r['b_only']}  p = {r['p_value']:.3g}")
    print(overlap)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
