"""Calibrated ensemble whiteness check for E6 star-tracker innovations.

A single multivariate Ljung-Box p-value is a weak acceptance criterion: at the
E6 sample size (about 200 accepted innovations, 3 axes, 10 lags) the
chi-square approximation is liberal, so the nominal 5% test rejects perfectly
white noise noticeably more often than 5%. This script therefore

1. runs the E6 scenario for an ensemble of independent seeds and records each
   run's Ljung-Box p-value on accepted, covariance-whitened innovations;
2. calibrates the test empirically by applying it to Gaussian white noise of
   the same length and dimension;
3. compares the two p-value distributions with a two-sample
   Kolmogorov-Smirnov test.

Innovations are judged consistent with whiteness when the KS p-value is at
least 0.05, i.e. the ensemble is statistically indistinguishable from
the calibrated white-noise null.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import (  # noqa: E402
    compute_performance_metrics,
    load_config,
    multivariate_ljung_box,
    run_simulation,
)


def _run_seed(arguments: tuple[str, int]) -> dict:
    config_path, seed = arguments
    config = replace(load_config(config_path), seed=seed)
    metrics = compute_performance_metrics(run_simulation(config))
    return {
        "seed": seed,
        "ljung_box_q": metrics["innovation_whiteness_ljung_box_q"],
        "p_value": metrics["innovation_whiteness_ljung_box_pvalue"],
        "lags": int(metrics["innovation_whiteness_lags"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "closed_loop.yaml")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--null-samples", type=int, default=4000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "innovation_whiteness.json"
    )
    args = parser.parse_args()
    if args.runs < 2 or args.null_samples < 100:
        parser.error("--runs must be >= 2 and --null-samples >= 100")

    config = load_config(str(args.config))
    seeds = [config.seed + 1000 * index for index in range(args.runs)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(_run_seed, [(str(args.config), seed) for seed in seeds]))

    # Accepted innovations per run: one per tracker sample over the duration.
    sample_count = int(round(config.duration * config.star_tracker["rate"]))
    null_rng = np.random.default_rng(20261024)
    null_p = np.asarray([
        multivariate_ljung_box(null_rng.standard_normal((sample_count, 3)))[1]
        for _ in range(args.null_samples)
    ])
    run_p = np.asarray([run["p_value"] for run in runs])
    ks = stats.ks_2samp(run_p, null_p)
    null_rate = float(np.mean(null_p < 0.05))
    run_rate = float(np.mean(run_p < 0.05))
    rejections = int(np.sum(run_p < 0.05))

    report = {
        "scope": "E6 closed-loop scenario; simulation evidence only.",
        "config": args.config.resolve().relative_to(ROOT).as_posix(),
        "method": (
            "10-lag multivariate Ljung-Box on Cholesky-whitened accepted innovations; "
            "ensemble p-values compared with an empirical white-noise null of the "
            "same length and dimension (two-sample Kolmogorov-Smirnov)."
        ),
        "runs": runs,
        "ljung_box_degrees_of_freedom": 9 * runs[0]["lags"],
        "innovations_per_run": sample_count,
        "null_samples": args.null_samples,
        "null_rejection_rate_at_nominal_5pct": null_rate,
        "ensemble_rejection_rate_at_nominal_5pct": run_rate,
        "ensemble_rejections": rejections,
        "binomial_p_at_least_observed_rejections_under_null": float(
            stats.binom.sf(rejections - 1, len(runs), null_rate)
        ),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "consistent_with_white_noise": bool(ks.pvalue >= 0.05),
        "interpretation": (
            "A single-run Ljung-Box p-value below 0.05 is expected in a fraction of runs "
            "even for white innovations, and more often than 5% at this sample size. "
            "Whiteness is assessed on the ensemble against the calibrated null."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Ensemble p-values: {np.round(run_p, 3).tolist()}")
    print(f"White-noise null rejection rate at nominal 5%: {null_rate:.3f}")
    print(f"KS p-value vs null: {ks.pvalue:.3f} -> "
          f"{'consistent' if ks.pvalue >= 0.05 else 'NOT consistent'} with white noise")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
