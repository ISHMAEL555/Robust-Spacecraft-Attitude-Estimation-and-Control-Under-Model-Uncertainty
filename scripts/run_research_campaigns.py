#!/usr/bin/env python3
"""Run reproducible parameter sweeps for research experiments E1-E7."""

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.parallel_runs import completeness
from scripts.run_experiment import (
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
    save_results,
)


ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")


def _case(name: str, campaign: str, config_name: str, **parameters: Any) -> Dict[str, Any]:
    config = load_config(str(ROOT / "config" / config_name))
    overrides = parameters.pop("config_overrides", {})
    if "duration" in overrides:
        config = replace(config, duration=float(overrides.pop("duration")))
    config = replace(
        config,
        filter={**config.filter, **overrides.pop("filter", {})},
        star_tracker={**config.star_tracker, **overrides.pop("star_tracker", {})},
        spacecraft={**config.spacecraft, **overrides.pop("spacecraft", {})},
        controller={**config.controller, **overrides.pop("controller", {})},
        reaction_wheels={
            **config.reaction_wheels,
            **overrides.pop("reaction_wheels", {}),
        },
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
        if selected == "E0":
            cases.append(_case(
                "nominal_baseline", selected, "nominal.yaml",
                baseline="nominal_1000s",
            ))
        elif selected == "E1":
            for variant in ("gyro_driven", "model_aided"):
                for mismatch in (0.0, 0.1, 0.25):
                    cases.append(_case(
                        f"{variant}_inertia_{mismatch:.2f}",
                        selected,
                        "model_uncertainty.yaml",
                        config_overrides={
                            "filter": {
                                "inertia_relative_sigma": max(abs(mismatch), 0.05)
                            }
                        },
                        variant=variant,
                        inertia_mismatch=mismatch,
                        inertia_relative_sigma=max(abs(mismatch), 0.05),
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
            cases.extend([
                _case(
                    "zero_initial_rate", selected, "closed_loop.yaml",
                    config_overrides={"spacecraft": {"initial_omega": [0.0, 0.0, 0.0]}},
                    initial_rate_rad_s=0.0,
                ),
                _case(
                    "half_torque_authority", selected, "closed_loop.yaml",
                    config_overrides={
                        "controller": {"max_torque": 0.05},
                        "reaction_wheels": {"max_torque": 0.05},
                    },
                    torque_limit_nm=0.05,
                ),
                _case(
                    "double_torque_authority", selected, "closed_loop.yaml",
                    config_overrides={
                        "controller": {"max_torque": 0.2},
                        "reaction_wheels": {"max_torque": 0.2},
                    },
                    torque_limit_nm=0.2,
                ),
                _case(
                    "low_wheel_momentum_limit", selected, "closed_loop.yaml",
                    config_overrides={"reaction_wheels": {"max_momentum": 1.0}},
                    wheel_momentum_limit_nms=1.0,
                ),
                _case(
                    "near_180deg_attitude", selected, "closed_loop.yaml",
                    config_overrides={
                        "spacecraft": {
                            "initial_attitude": [0.0, 0.0, 0.996194698, 0.087155743]
                        }
                    },
                    initial_attitude_deg=170.0,
                ),
            ])
        elif selected == "E7":
            cases.append(_case("nominal_monte_carlo", selected, "monte_carlo.yaml"))

    if duration is not None:
        for case in cases:
            # The all-campaign refresh keeps E0's long nominal baseline while
            # applying the requested shorter duration to the E1-E7 sweeps.
            if campaign == "all" and case["campaign"] == "E0":
                continue
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


def _execute_case(case: Dict[str, Any], n_runs: int, output_dir: Path) -> Dict[str, Any]:
    """Run one case, save its compressed traces, and return its summary row."""
    results = run_case(case, n_runs)
    save_results(results, str(output_dir / f"{case['campaign'].lower()}_{case['name']}.npz"))
    row = _summary_row(case, results)
    seeds_requested = [case["config"].seed + 1000 * index for index in range(n_runs)]
    row.update(completeness(n_runs, seeds_requested, row["seeds"]))
    return row


def _run_mean_ci95(values: List[float]) -> tuple[float | None, float | None, float | None]:
    """Return mean and Student-t interval over independent run summaries."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return None, None, None
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, None, None
    half_width = float(
        stats.t.ppf(0.975, data.size - 1)
        * np.std(data, ddof=1)
        / np.sqrt(data.size)
    )
    return mean, mean - half_width, mean + half_width


def _code_fingerprint() -> str:
    """Hash the simulation source so rows from older code are never resumed."""
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / "run_experiment.py"]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        # Normalize line endings so the fingerprint is platform independent.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _case_signature(case: Dict[str, Any]) -> str:
    """Hash the scenario inputs and simulation code so resumes cannot reuse stale rows."""
    config = case["config"]
    payload = {
        "code_sha256": _code_fingerprint(),
        "metadata": {key: value for key, value in case.items() if key != "config"},
        "config": {
            field: getattr(config, field)
            for field in (
                "duration", "dt", "seed", "spacecraft", "gyroscope", "star_tracker",
                "filter", "controller", "reaction_wheels", "analysis",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _summary_row(case: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    consistency = results["consistency"]

    def finite_or_none(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    nees_run_means = [
        float(np.mean(run["NEES"])) for run in results["all_runs"]
    ]
    nis_run_means = []
    for run in results["all_runs"]:
        accepted = run["measurement_available"] & run["measurement_accepted"]
        nis = run["NIS"][accepted]
        nis_run_means.append(float(np.mean(nis)) if nis.size else float("nan"))
    nees_run_mean, nees_ci_lower, nees_ci_upper = _run_mean_ci95(nees_run_means)
    nis_run_mean, nis_ci_lower, nis_ci_upper = _run_mean_ci95(nis_run_means)

    row: Dict[str, Any] = {
        "campaign": case["campaign"],
        "case": case["name"],
        "runs": results["n_runs"],
        "base_seed": case["config"].seed,
        "seed_stride": 1000,
        "seeds": [case["config"].seed + 1000 * index for index in range(results["n_runs"])],
        "duration_s": case["config"].duration,
        "dt_s": case["config"].dt,
        "case_signature": _case_signature(case),
        "NEES_dof": consistency["dof_nees"],
        "NIS_dof": consistency["dof_nis"],
        "NEES_mean": finite_or_none(consistency["nees_mean"]),
        "NEES_run_mean": finite_or_none(nees_run_mean),
        "NEES_run_mean_ci95_lower": finite_or_none(nees_ci_lower),
        "NEES_run_mean_ci95_upper": finite_or_none(nees_ci_upper),
        "NEES_in_bounds_pct": finite_or_none(100.0 * consistency["nees_in_bounds"]),
        "NIS_mean": finite_or_none(consistency["nis_mean"]),
        "NIS_accepted_run_mean": finite_or_none(nis_run_mean),
        "NIS_accepted_run_mean_ci95_lower": finite_or_none(nis_ci_lower),
        "NIS_accepted_run_mean_ci95_upper": finite_or_none(nis_ci_upper),
        "NIS_in_bounds_pct": finite_or_none(100.0 * consistency["nis_in_bounds"]),
        "NIS_mean_accepted": finite_or_none(consistency["nis_mean_accepted"]),
        "NIS_accepted_in_bounds_pct": finite_or_none(
            100.0 * consistency["nis_accepted_in_bounds"]
        ),
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


def _write_summaries(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Atomically checkpoint JSON and CSV after each completed case."""
    rows = sorted(rows, key=lambda row: (CAMPAIGNS.index(row["campaign"]), row["case"]))
    fields = sorted({key for row in rows for key in row})
    csv_path = output_dir / "summary.csv"
    csv_tmp = output_dir / "summary.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(csv_path)

    json_path = output_dir / "summary.json"
    json_tmp = output_dir / "summary.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write("\n")
    json_tmp.replace(json_path)


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
    parser.add_argument(
        "--duration", type=float,
        help="Override case duration in seconds; with --campaign all, E0 keeps its 1,000 s nominal duration.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "research_campaigns")
    parser.add_argument(
        "--case", action="append",
        help="Run only this case; use CAMPAIGN:case_name when a name is ambiguous. Repeat as needed.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from matching completed rows in output-dir/summary.json.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of cases to run in parallel worker processes (default 1).",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least one")
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_campaign_cases(args.campaign, args.duration)
    if args.case:
        requested = set()
        for selector in args.case:
            if ":" in selector:
                campaign_name, case_name = selector.split(":", 1)
                requested.add((campaign_name.upper(), case_name))
            else:
                matches = [(case["campaign"], case["name"]) for case in cases if case["name"] == selector]
                if len(matches) != 1:
                    parser.error(
                        f"case {selector!r} is ambiguous or unknown; use CAMPAIGN:case_name"
                    )
                requested.add(matches[0])
        known = {(case["campaign"], case["name"]) for case in cases}
        unknown = requested - known
        if unknown:
            parser.error("unknown case selector(s): " + ", ".join(
                f"{campaign}:{name}" for campaign, name in sorted(unknown)
            ))
        cases = [case for case in cases if (case["campaign"], case["name"]) in requested]

    rows_by_key = {}
    if args.resume and (args.output_dir / "summary.json").exists():
        saved_rows = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        rows_by_key = {(row["campaign"], row["case"]): row for row in saved_rows}

    pending = []
    for index, case in enumerate(cases, start=1):
        key = (case["campaign"], case["name"])
        existing = rows_by_key.get(key)
        signature = _case_signature(case)
        if (
            args.resume
            and existing is not None
            and existing.get("case_signature") == signature
            and int(existing.get("runs", 0)) == args.runs
        ):
            print(f"[{index}/{len(cases)}] {case['campaign']} {case['name']}: already complete; skipping")
            continue
        print(
            f"[{index}/{len(cases)}] {case['campaign']} {case['name']}: "
            f"{args.runs} runs x {case['config'].duration:g} s",
            flush=True,
        )
        pending.append(case)

    # Each case is seeded independently, so results do not depend on the
    # number of worker processes or the order in which cases complete.
    if args.jobs > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_execute_case, case, args.runs, args.output_dir): case
                for case in pending
            }
            failed = []
            for future in as_completed(futures):
                case = futures[future]
                try:
                    rows_by_key[(case["campaign"], case["name"])] = future.result()
                except BaseException as error:  # keep going so finished cases are saved
                    failed.append(f"{case['campaign']} {case['name']}: "
                                  f"{type(error).__name__}: {error}")
                    continue
                _write_summaries(list(rows_by_key.values()), args.output_dir)
                print(f"completed {case['campaign']} {case['name']}", flush=True)
            if failed:
                raise SystemExit(f"{len(failed)} of {len(pending)} campaign cases failed:\n  "
                                 + "\n  ".join(failed))
    else:
        for case in pending:
            rows_by_key[(case["campaign"], case["name"])] = _execute_case(
                case, args.runs, args.output_dir
            )
            _write_summaries(list(rows_by_key.values()), args.output_dir)

    rows = list(rows_by_key.values())
    expected = {(c["campaign"], c["name"]) for c in cases}
    missing = sorted(expected - set(rows_by_key))
    wrong = sorted(k for k in expected & set(rows_by_key) if rows_by_key[k]["runs"] != args.runs)
    if missing or wrong:
        raise SystemExit(f"campaign incomplete: missing cases {missing}, wrong run counts {wrong}")
    _write_summaries(rows, args.output_dir)
    for campaign in sorted({row["campaign"] for row in rows}, key=CAMPAIGNS.index):
        _write_campaign_plot(rows, campaign, args.output_dir)
    print(f"Campaign summaries and traces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
