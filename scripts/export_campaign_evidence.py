#!/usr/bin/env python3
"""Export a completed E0-E7 run and its reproducibility manifest into docs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
SOURCE_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "scripts/run_research_campaigns.py",
    "scripts/run_experiment.py",
    "scripts/plot_campaign_figures.py",
    "scripts/export_campaign_evidence.py",
    "scripts/evaluate_e6_benchmark.py",
    "scripts/verify_model_aided_consistency.py",
    "scripts/verify_robustness_slices.py",
    "scripts/verify_innovation_whiteness.py",
    "scripts/verify_wheel_coupling.py",
    "scripts/plot_readme_figures.py",
    "config/nominal.yaml",
    "config/model_uncertainty.yaml",
    "config/sensor_degradation.yaml",
    "config/closed_loop.yaml",
    "config/monte_carlo.yaml",
    "docs/system_description.md",
    "docs/conventions.md",
    "docs/e0_e7_campaign.md",
    "docs/current_evidence.md",
    "docs/requirements_traceability.md",
    "docs/e6_project_benchmark.json",
    "docs/model_aided_consistency.json",
    "docs/robustness_slices.json",
    "docs/innovation_whiteness.json",
    "docs/wheel_coupling_effect.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _write_csv(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "reports" / "research_campaigns" / "e0_e7_refresh",
        help="Completed campaign directory containing summary.json.",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=ROOT / "docs" / "e0_e7_campaign_summary.csv",
    )
    parser.add_argument(
        "--manifest-output", type=Path,
        default=ROOT / "docs" / "e0_e7_campaign_metadata.json",
    )
    parser.add_argument(
        "--figure-dir", type=Path,
        default=ROOT / "docs" / "figures" / "refreshed_campaign",
    )
    args = parser.parse_args()

    input_path = args.input_dir / "summary.json"
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        parser.error(f"no campaign rows found in {input_path}")
    if any("case_signature" not in row or "seeds" not in row for row in rows):
        parser.error("campaign rows are missing configuration signatures or seed lists")

    rows.sort(key=lambda row: (CAMPAIGNS.index(row["campaign"]), row["case"]))
    _write_csv(rows, args.summary_output)

    campaign_counts = Counter(row["campaign"] for row in rows)
    # Hash every simulation source module, not only the scripts and configs,
    # so the manifest identifies the exact code that produced the evidence.
    source_paths = [ROOT / item for item in SOURCE_PATHS] + sorted(
        (ROOT / "src").rglob("*.py")
    )
    # Hash every published figure, including the README summary figures.
    figure_paths = sorted(
        path for path in (ROOT / "docs" / "figures").rglob("*.png") if path.is_file()
    )
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None

    manifest = {
        "title": "Robust spacecraft attitude estimation and control: refreshed E0-E7 campaign",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head_at_export": git_head,
        "scope": "Simulation characterization; not hardware, mission acceptance, qualification, or flight-readiness evidence.",
        "campaign_design": {
            "cases": len(rows),
            "independent_runs_total": sum(int(row["runs"]) for row in rows),
            "propagation_step_s": 0.01,
            "seed_rule": "seed_i = 42 + 1000*i, i=0..9, independently reset for each run",
            "confidence_intervals": (
                "Two-sided 95% Student-t intervals over independent run-level means. "
                "Time samples within a run are not treated as independent replicates."
            ),
            "nees_statistic": (
                "Each run mean of the full-state NEES time history; confidence intervals "
                "are normalized by state degrees of freedom."
            ),
            "nis_statistic": (
                "Each run mean over available and accepted tracker updates; confidence "
                "intervals are normalized by measurement degrees of freedom. With gating "
                "disabled, every available update is accepted."
            ),
            "campaigns": {
                campaign: {
                    "case_count": campaign_counts[campaign],
                    "duration_s": sorted({
                        float(row["duration_s"]) for row in rows
                        if row["campaign"] == campaign
                    }),
                    "runs_per_case": sorted({
                        int(row["runs"]) for row in rows
                        if row["campaign"] == campaign
                    }),
                }
                for campaign in CAMPAIGNS if campaign in campaign_counts
            },
            "total_propagation_steps": sum(
                int(row["runs"])
                * int(float(row["duration_s"]) / float(row["dt_s"]))
                for row in rows
            ),
        },
        "reproduce": {
            "install": "python -m pip install -r requirements.txt",
            "campaign": (
                "python scripts/run_research_campaigns.py --campaign all --runs 10 "
                "--duration 100 --output-dir "
                "reports/research_campaigns/e0_e7_refresh --resume"
            ),
            "plots": (
                "python scripts/plot_campaign_figures.py --input "
                "reports/research_campaigns/e0_e7_refresh/summary.csv --output-dir "
                "docs/figures/refreshed_campaign"
            ),
            "export": (
                "python scripts/export_campaign_evidence.py --input-dir "
                "reports/research_campaigns/e0_e7_refresh"
            ),
            "duration_note": (
                "With --campaign all, E0 retains nominal.yaml duration 1,000 s; "
                "--duration 100 applies to E1-E7."
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
            "pyyaml": _version("PyYAML"),
        },
        "scenario_inputs": [
            {
                "campaign": row["campaign"],
                "case": row["case"],
                "runs": int(row["runs"]),
                "duration_s": float(row["duration_s"]),
                "dt_s": float(row["dt_s"]),
                "seeds": row["seeds"],
                "case_signature_sha256": row["case_signature"],
            }
            for row in rows
        ],
        "sha256": {
            **{
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in source_paths if path.is_file()
            },
            _display_path(args.summary_output):
                _sha256(args.summary_output),
            **{
                _display_path(path): _sha256(path)
                for path in figure_paths
            },
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest_output.with_suffix(args.manifest_output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest_output)
    print(
        f"Exported {len(rows)} cases / {manifest['campaign_design']['independent_runs_total']} runs; "
        f"manifest: {args.manifest_output}"
    )


if __name__ == "__main__":
    main()
