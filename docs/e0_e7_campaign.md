# Refreshed E0–E7 statistical campaign

This report records a reproducible refresh of the repository's E0–E7 simulation campaigns. It contains 33 scenario cases and 330 independent runs. E0 uses the nominal 1,000 s configuration; E1–E7 use 100 s per run. The propagation step is 0.01 s (100 Hz) in every case.

The run seeds are `42 + 1000 × i`, for `i = 0…9`, within each case. The campaign totals 4.2 million propagation steps. Per-case configuration signatures, seeds, run summaries, and environment details are recorded in the [campaign CSV](e0_e7_campaign_summary.csv) and [metadata manifest](e0_e7_campaign_metadata.json).

## Statistical method

Each run is reduced to its own time-averaged NEES, accepted-update NIS, attitude-estimation RMSE, and performance metrics. The 95% intervals below are two-sided Student-t intervals over the ten independent run-level summaries (9 degrees of freedom). Time samples within a run are not treated as independent Monte Carlo replicates. NEES is normalized by state degrees of freedom; NIS is normalized by the three-dimensional tracker measurement degrees of freedom. Accepted-update NIS excludes tracker measurements rejected by the NIS gate.

## Representative results

| Scenario | Mean NEES / DOF (95% run-level interval) | Mean accepted NIS / DOF (95% run-level interval) | Attitude-estimation RMSE / outcome |
|---|---:|---:|---|
| E0 nominal, 1,000 s | 0.999 [0.931, 1.068] | 0.997 [0.979, 1.014] | 0.031°; true pointing RMS 0.639° |
| E1 gyro-driven, matched inertia | 0.994 [0.794, 1.194] | 0.990 [0.942, 1.038] | 0.094° |
| E1 model-aided, matched inertia | 1.003 [0.835, 1.171] | 0.986 [0.920, 1.052] | 0.094° |
| E1 model-aided, 25% inertia mismatch | 17.132 [6.064, 28.199] | 1.070 [1.008, 1.133] | 0.094°; full-state covariance is inconsistent |
| E4 1 rad outliers, gate off | 1.04×10⁶ [7.30×10⁵, 1.36×10⁶] | 6.18×10⁵ [4.24×10⁵, 8.12×10⁵] | 10.80°; 0/1,000 measurements rejected |
| E4 1 rad outliers, gate on | 0.922 [0.790, 1.053] | 0.988 [0.925, 1.050] | 0.094°; 46/1,000 measurements rejected |
| E5 30-update tracker outage | 1.013 [0.688, 1.338] | 0.994 [0.914, 1.074] | 0.119°; 62.78 s recovery in the 100 s case |

The matched nominal gyro-driven and model-aided cases have run-level intervals that contain their state and measurement degrees of freedom. At 25% inertia mismatch, model-aided mean NEES remains far above its 15-DOF reference even though attitude-estimation RMSE stays near 0.094°. This is evidence that attitude accuracy alone does not establish full-state covariance consistency.

The 1 rad outlier experiment shows the effect of measurement gating: without gating, the estimator's attitude error and normalized innovations grow sharply; with gating, the accepted NIS interval contains its 3-DOF reference, while 46 of the 1,000 tracker updates are rejected. The gate result covers the injected scenario only and is not a complete fault-detection design.

## Campaign coverage

- **E0:** nominal 1,000 s baseline.
- **E1:** gyro-driven and model-aided filters at 0%, 10%, and 25% diagonal inertia mismatch.
- **E2–E3:** process-noise and measurement-noise scale factors of 0.1, 1, and 10.
- **E4:** 5% tracker outlier probability at 0.1, 0.5, and 1 rad, with gating on and off.
- **E5:** scheduled tracker outages of 1, 10, and 30 updates.
- **E6:** nominal, Q/R scale changes, 25% inertia mismatch, initial-rate and initial-attitude changes, torque authority changes, and a reduced wheel-momentum limit.
- **E7:** nominal Monte Carlo reference case.

The E6 campaign cases run for 100 s and are characterization slices. The 60° E6 project benchmark is a separate 200 s deterministic run with its own settling, terminal-pointing, wheel-use, and torque-effort criteria. Its results are reported in [the benchmark report](e6_project_benchmark.json).

## Reproduction

Install the declared dependencies, then run the complete campaign. The E0 case retains its 1,000 s nominal duration when `--campaign all` is selected; `--duration 100` applies to E1–E7. `--resume` reuses only completed cases whose configuration signatures and run counts match.

```powershell
python -m pip install -r requirements.txt
python scripts/run_research_campaigns.py --campaign all --runs 10 --duration 100 --output-dir reports/research_campaigns/e0_e7_refresh --resume
python scripts/plot_campaign_figures.py --input reports/research_campaigns/e0_e7_refresh/summary.csv --output-dir docs/figures/refreshed_campaign
python scripts/export_campaign_evidence.py --input-dir reports/research_campaigns/e0_e7_refresh
```

The checked-in [summary CSV](e0_e7_campaign_summary.csv) and [metadata manifest](e0_e7_campaign_metadata.json) preserve the run definitions and summary statistics without checking in the much larger raw simulation traces. The exact simulation sources, configuration files, summary, and figures are identified by SHA-256 hashes in the manifest.

## Interpretation limits

These are simulation results from fixed, one-at-a-time parameter cases. Ten runs per case provide useful repeatability evidence but do not establish a statistically validated operating envelope. The campaign does not randomize full inertia tensors, sensor bias distributions, environmental disturbances, delays, or actuator-fault combinations. The results are not hardware validation, mission acceptance evidence, qualification evidence, or flight-readiness evidence.
