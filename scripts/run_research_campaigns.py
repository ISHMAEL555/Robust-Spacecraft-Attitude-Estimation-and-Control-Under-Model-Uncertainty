#!/usr/bin/env python3
"""Run reproducible parameter sweeps for research experiments E1-E7."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import (
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
    save_results,
)


ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")


def _case(name: str, campaign: str, config_name: str, **parameters: Any) -> Dict[str, Any]:
    config = load_config(str(ROOT / "config" / config_name))
    overrides = parameters.pop("config_overrides", {})
    if "duration" in overrides:
        config = replace(config, duration=float(overrides.pop("duration")))
    config = replace(
        config,
        filter={**config.filter, **overrides.pop("filter", {})},
        star_tracker={**config.star_tracker, **overrides.pop("star_tracker", {})},
    )
    return {
        "name": name,
        "campaign": campaign,
        "config": config,
        **parameters,
    }


def build_campaign_cases(campaign: str, duration: float | None = None) -> List[Dict[str, Any]]:
    """Build deterministic cases; each sweep changes one named assumption at a time."""
    campaigns = (CAMPAIGNS if campaign == "all" else (campaign,))
    cases: List[Dict[str, Any]] = []
    for selected in campaigns:
        if selected == "E1":
            for variant in ("gyro_driven", "model_aided"):
                for mismatch in (0.0, 0.1, 0.25):
                    cases.append(_case(
                        f"{variant}_inertia_{mismatch:.2f}",
                        selected,
                        "model_uncertainty.yaml",
                        variant=variant,
                        inertia_mismatch=mismatch,
                    ))
        elif selected == "E2":
            for scale in (0.1, 1.0, 10.0):
                cases.append(_case(
                    f"Q_scale_{scale:g}", selected, "model_uncertainty.yaml",
                    config_overrides={"filter": {"Q_scale": scale}},
                    Q_scale=scale,
                ))
        elif selected == "E3":
            for scale in (0.1, 1.0, 10.0):
                cases.append(_case(
                    f"R_scale_{scale:g}", selected, "model_uncertainty.yaml",
                    config_overrides={"filter": {"R_scale": scale}},
                    R_scale=scale,
                ))
        elif selected == "E4":
            for magnitude in (0.1, 0.5, 1.0):
                for gated in (False, True):
                    cases.append(_case(
                        f"outlier_{magnitude:g}_gate_{str(gated).lower()}",
                        selected,
                        "sensor_degradation.yaml",
                        config_overrides={
                            "filter": {"nis_gate_enabled": gated},
                            "star_tracker": {
                                "outlier_prob": 0.05,
                                "outlier_magnitude": magnitude,
                                "outage_prob": 0.0,
                            },
                        },
                        outlier_magnitude_rad=magnitude,
                        nis_gate_enabled=gated,
                        outlier_probability=0.05,
                    ))
        elif selected == "E5":
            for outage_updates in (1, 10, 30):
                cases.append(_case(
                    f"outage_{outage_updates}_updates",
                    selected,
                    "sensor_degradation.yaml",
                    config_overrides={
                        "star_tracker": {
                            "outage_prob": 0.0,
                            "outage_start_time": 5.0,
                            "outage_duration": outage_updates,
                        },
                    },
                    outage_duration_measurements=outage_updates,
                    outage_start_time_s=5.0,
                ))
        elif selected == "E6":
            variations = (
                ("nominal", {}),
                ("Q_scale_10", {"filter": {"Q_scale": 10.0}}),
                ("R_scale_10", {"filter": {"R_scale": 10.0}}),
                ("Q_R_scale_10", {"filter": {"Q_scale": 10.0, "R_scale": 10.0}}),
            )
            for name, overrides in variations:
                cases.append(_case(
                    name, selected, "closed_loop.yaml",
                    config_overrides=overrides,
                    **{key: value for group in overrides.values() for key, value in group.items()},
                ))
            cases.append(_case(
                "inertia_mismatch_25pct", selected, "closed_loop.yaml",
                inertia_mismatch=0.25,
            ))
        elif selected == "E7":
            cases.append(_case("nominal_monte_carlo", selected, "monte_carlo.yaml"))

    if duration is not None:
        for case in cases:
            case["config"] = replace(case["config"], duration=duration)
    return cases


def run_case(case: Dict[str, Any], n_runs: int) -> Dict[str, Any]:
    config = case["config"]
    nominal_inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    mismatch = float(case.get("inertia_mismatch", 0.0))
    true_inertia = nominal_inertia + mismatch * np.diag(np.diag(nominal_inertia))
    filter_inertia = nominal_inertia if case.get("variant") == "model_aided" else None

    if "variant" in case:
        config = replace(config, filter={**config.filter, "variant": case["variant"]})
    results = run_monte_carlo(config, n_runs, true_inertia, filter_inertia)
    results["consistency"] = compute_consistency_stats(
        results,
        config.analysis["nees_confidence"],
        config.analysis["nis_confidence"],
    )
    results["campaign_metadata"] = {
        key: value for key, value in case.items()
        if key not in ("config",)
    }
    return results


def _summary_row(case: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    consistency = results["consistency"]

    def finite_or_none(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    row: Dict[str, Any] = {
        "campaign": case["campaign"],
        "case": case["name"],
        "runs": results["n_runs"],
        "duration_s": case["config"].duration,
        "NEES_dof": consistency["dof_nees"],
        "NIS_dof": consistency["dof_nis"],
        "NEES_mean": finite_or_none(consistency["nees_mean"]),
        "NEES_in_bounds_pct": finite_or_none(100.0 * consistency["nees_in_bounds"]),
        "NIS_mean": finite_or_none(consistency["nis_mean"]),
        "NIS_in_bounds_pct": finite_or_none(100.0 * consistency["nis_in_bounds"]),
        "star_tracker_updates": int(np.sum(results["innovation_count"])),
        "rejected_measurements": int(np.sum(results["measurement_rejection_count"])),
    }
    for key, value in case.items():
        if key not in ("name", "campaign", "config"):
            row[key] = value
    for metric, summary in results["performance_summary"].items():
        for field, value in summary.items():
            row[f"{metric}_{field}"] = finite_or_none(value)
    return row


def _write_campaign_plot(rows: List[Dict[str, Any]], campaign: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["campaign"] == campaign]
    if not selected:
        return
    metrics = (
        ("NEES_mean", "Mean NEES"),
        ("NIS_mean", "Mean NIS"),
        ("pointing_rms_rad_mean", "Mean run pointing RMS (deg)", 180.0 / np.pi),
        ("achieved_control_effort_nm2_s_mean", "Mean applied control effort (N^2 m^2 s)"),
    )
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(9, len(selected) * 0.65), 10))
    for ax, metric in zip(axes, metrics):
        key, label = metric[:2]
        scale = metric[2] if len(metric) == 3 else 1.0
        x = np.arange(len(selected))
        values = np.asarray([row.get(key, np.nan) for row in selected], dtype=float) * scale
        ax.plot(x, values, "o-")
        ax.set_ylabel(label)
        ax.set_xticks(x, [row["case"] for row in selected], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Campaign case")
    fig.suptitle(f"{campaign} Research Campaign")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"{campaign.lower()}_summary.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=(*CAMPAIGNS, "all"), default="all")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--duration", type=float, help="Override each case duration (seconds)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "research_campaigns")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least one")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_campaign_cases(args.campaign, args.duration)
    rows = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case['campaign']} {case['name']}: "
            f"{args.runs} runs x {case['config'].duration:g} s"
        )
        results = run_case(case, args.runs)
        save_results(results, str(args.output_dir / f"{case['campaign'].lower()}_{case['name']}.npz"))
        rows.append(_summary_row(case, results))
        del results

    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
    for campaign in sorted({row["campaign"] for row in rows}):
        _write_campaign_plot(rows, campaign, args.output_dir)
    print(f"Campaign summaries and traces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
