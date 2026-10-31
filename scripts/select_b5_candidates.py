"""Apply a B5 design-stage selection rule to design results and record the choice.

Score of a candidate = sum of the listed terms, each a pass count from a design
result file ``{scenario}__{name}.json``:

``<scenario>:e6``   E6 all-criteria passes in that scenario
``<scenario>:det``  GNC-DET-001 passes in that scenario

Highest score wins; ties go to the lower median settling time in ``--tie-scenario``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(design_dir: Path, scenario: str, name: str) -> Dict[str, Any]:
    return json.loads((design_dir / f"{scenario}__{name}.json").read_text())["summary"]


def score(design_dir: Path, name: str, terms: List[str]) -> Dict[str, Any]:
    parts = {}
    for term in terms:
        scenario, kind = term.split(":")
        summary = _load(design_dir, scenario, name)
        key = {"e6": "e6_all_criteria", "det": "detumble_within_limit"}[kind]
        parts[term] = int(summary[key]["passes"])
    return {"name": name, "parts": parts, "score": sum(parts.values())}


def select(design_dir: Path, names: List[str], terms: List[str], tie_scenario: str) -> Dict[str, Any]:
    rows = [score(design_dir, n, terms) for n in names]
    for row in rows:
        median = _load(design_dir, tie_scenario, row["name"])["settling_time_s"]["median"]
        row["tie_break_median_settling_s"] = median if median is not None else float("inf")
    best = max(rows, key=lambda r: (r["score"], -r["tie_break_median_settling_s"]))
    for row in rows:
        if row["tie_break_median_settling_s"] == float("inf"):
            row["tie_break_median_settling_s"] = None
    return {"terms": terms, "tie_scenario": tie_scenario, "candidates": rows,
            "selected": best["name"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--family", nargs="+", action="append", required=True,
                        metavar="NAME", help="one family of candidate names; repeat per family")
    parser.add_argument("--terms", nargs="+", required=True)
    parser.add_argument("--tie-scenario", default="acq_drifted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"families": [select(args.design_dir, family, args.terms, args.tie_scenario)
                           for family in args.family]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for family in result["families"]:
        print("selected:", family["selected"], [(r["name"], r["score"]) for r in family["candidates"]])


if __name__ == "__main__":
    main()
