# Current simulation evidence

This page records the regenerated E0–E7 results after aligning the MEKF error-state implementation with its right-multiplicative, body-frame covariance linearization. The campaign is reproducible simulation evidence for this repository's idealized model; it is not hardware or mission validation. Historical outputs in `reports/` predate this correction and remain provenance only.

## Run conditions and statistics

- **E0:** one 1,000 s nominal run with a 0.01 s (100 Hz) propagation step.
- **E1–E7:** 32 deterministic cases, ten runs per case, 100 s per run.
- Run `i` uses `config.seed + i × 1000`, with `i` starting at zero.
- Run-level 95% confidence intervals are Student-t intervals over the ten independent run summaries.
- Campaign mean NEES/NIS values pool available time samples and runs. Fractions inside single-sample chi-square bounds are descriptive time-series statistics because adjacent samples are correlated.
- Regenerated compressed traces and plots are under the ignored `.venv/current_campaigns/` directory. The tracked [campaign CSV](campaign_summary.csv) contains the consolidated 32-case output, including run-level intervals.

## Nominal reference (E0)

| Metric | Result | Reference |
|---|---:|---:|
| Mean NEES | 6.552 | 6 DOF |
| Mean NIS | 2.954 | 3 DOF |
| NEES samples inside single-sample 95% bounds | 96.7% | Descriptive time fraction |
| NIS samples inside single-sample 95% bounds | 95.1% | Descriptive time fraction |
| Attitude-estimation RMSE | 0.036° | 1,000 s |
| True pointing RMS | 0.673° | 1,000 s |

## Inertia mismatch (E1)

Each row is the mean across ten 100 s runs. The true inertia is perturbed on its diagonal while the model-aided filter starts from nominal inertia and estimates an inertia correction. The gyro-driven filter does not propagate with inertia; its pointing still changes as the true spacecraft response changes.

| Filter | Truth mismatch | NEES (DOF) | NIS (DOF) | Attitude RMSE | Attitude component 3σ coverage | Inertia relative RMSE | Pointing RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-driven | 0% | 5.986 (6) | 2.971 (3) | 0.0944° | 99.76% | — | 2.122° |
| Gyro-driven | 10% | 5.986 (6) | 2.971 (3) | 0.0944° | 99.76% | — | 2.350° |
| Gyro-driven | 25% | 5.987 (6) | 2.972 (3) | 0.0944° | 99.76% | — | 2.700° |
| Model-aided | 0% | 18.391 (15) | 2.977 (3) | 0.0940° | 99.25% | 0.94% | 2.097° |
| Model-aided | 10% | 44.850 (15) | 3.022 (3) | 0.0940° | 97.09% | 3.16% | 2.324° |
| Model-aided | 25% | 1,032.234 (15) | 3.299 (3) | 0.0940° | 92.40% | 6.65% | 2.669° |

The augmented model-aided filter is already above its 15-DOF mean-NEES reference at matched inertia, and inconsistency grows sharply with mismatch. At 25%, attitude RMSE remains near 0.094° while full-state NEES exceeds 1,000. This is the clearest limitation in the study: state accuracy does not establish a calibrated covariance, and no acceptable inertia-mismatch envelope has been demonstrated.

## Noise calibration (E2–E3)

| Sweep | Scale | Mean NEES | Mean NIS | Attitude 3σ coverage | Attitude RMSE | Pointing RMS |
|---|---:|---:|---:|---:|---:|---:|
| Process noise `Q` | 0.1× | 45.146 | 11.207 | 80.52% | 0.0946° | 2.122° |
| Process noise `Q` | 1× | 5.986 | 2.971 | 99.76% | 0.0944° | 2.122° |
| Process noise `Q` | 10× | 0.873 | 0.438 | 100.00% | 0.0944° | 2.122° |
| Measurement noise `R` | 0.1× | 8.469 | 4.119 | 97.07% | 0.0944° | 2.122° |
| Measurement noise `R` | 1× | 5.986 | 2.971 | 99.76% | 0.0944° | 2.122° |
| Measurement noise `R` | 10× | 4.540 | 1.147 | 100.00% | 0.0946° | 2.122° |

The sweeps change NEES/NIS substantially with little change in attitude RMSE or pointing. Inflated noise can produce high coverage while NEES/NIS fall far below their expected dimensions, so coverage alone is not evidence of calibration.

## Star-tracker outliers and gating (E4)

Each case uses a 5% outlier probability and ten 100 s runs. Each row summarizes 1,000 tracker updates across the ten runs.

| Outlier magnitude | NIS gate | Attitude RMSE | Mean NEES | Mean NIS, all innovations | Accepted-only NIS | Rejected updates |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.1 rad | Off | 1.085° | 62,457.6 | 55,654.4 | — | 0 / 1,000 |
| 0.1 rad | On | 0.0945° | 5.493 | 29,376.7 | 2.942 | 46 / 1,000 |
| 0.5 rad | Off | 5.402° | 1,560,934.4 | 1,390,736.3 | — | 0 / 1,000 |
| 0.5 rad | On | 0.0945° | 5.493 | 734,212.3 | 2.942 | 46 / 1,000 |
| 1.0 rad | Off | 10.806° | 6,240,093.4 | 5,554,509.4 | — | 0 / 1,000 |
| 1.0 rad | On | 0.0945° | 5.493 | 2,936,772.6 | 2.942 | 46 / 1,000 |

The gate protects the state update from the injected large innovations in these runs. All-innovation NIS remains large because rejected outliers remain included in that diagnostic; accepted-only NIS is close to the three-DOF expectation. This is an outlier-rejection experiment, not complete fault detection, isolation, and recovery.

## Star-tracker outages (E5)

The scheduled outage begins at 5 s. An update is one 1 Hz tracker measurement.

| Outage | Valid tracker updates (of 1,000) | Mean NEES | Mean NIS | Attitude RMSE | Attitude 3σ coverage | Pointing RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 update | 990 | 5.979 | 3.019 | 0.0944° | 99.67% | 2.122° |
| 10 updates | 900 | 6.292 | 2.953 | 0.0962° | 99.58% | 2.125° |
| 30 updates | 700 | 6.063 | 2.977 | 0.1185° | 99.19% | 2.131° |

The 30-update outage modestly degrades the measured estimation and pointing errors in this scenario. It does not establish a maximum outage duration or recovery requirement.

## Closed-loop stress and actuator authority (E6)

Each case is ten 100 s runs. Unless stated, initial conditions are 60° attitude error and 0.1 rad/s body rate. The 0.1 N·m actuator command limit and the wheel momentum limit are distinct constraints.

| Case | Pointing RMS | Final error | Command torque at limit | Peak wheel momentum / limit | Time near wheel limit | Applied / requested effort |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 99.403° | 122.33° | 95.25% | 53.0% | 0.00% | 58.9% |
| `Q × 10` | 99.403° | 122.33° | 95.25% | 53.0% | 0.00% | 58.9% |
| `R × 10` | 99.403° | 122.33° | 95.25% | 53.0% | 0.00% | 58.9% |
| `Q, R × 10` | 99.403° | 122.33° | 95.25% | 53.0% | 0.00% | 58.9% |
| 25% inertia mismatch | 103.181° | 147.56° | 95.04% | 54.2% | 0.00% | 57.1% |
| Zero initial rate | 30.766° | 0.16° | 39.36% | 27.1% | 0.00% | 86.1% |
| Half torque authority | 109.416° | 178.38° | 97.68% | 33.9% | 0.00% | 71.8% |
| Double torque authority | 98.090° | 58.97° | 90.79% | 71.7% | 0.00% | 47.8% |
| 1 N·m·s wheel momentum limit | 114.478° | 132.29° | 88.55% | 99.9% | 12.31% | 17.6% |
| 170° initial attitude | 159.335° | 133.04° | 100.00% | 46.4% | 0.00% | 71.9% |

With the corrected MEKF error convention, E6 nominal mean NEES/NIS are 6.013/2.971 (6/3 DOF). Scaling `Q` by ten moves them to 0.876/0.439 while pointing RMS remains 99.403°. The stress case is dominated by actuator authority: commanded torque is saturated for most of the run even though nominal peak wheel momentum is only 53.0% of its limit. In the low-momentum case, wheel utilization reaches 99.9%, with 12.31% of samples near the limit and only 17.6% of requested effort achieved. The zero-rate case reaches the configured 2%-of-initial-error settling criterion in 76.95 s. These are scenario results, not accepted controller or actuator limits.

## Repeatability (E7)

Ten 100 s nominal runs produce mean NEES 5.986 (6 DOF), mean NIS 2.971 (3 DOF), and 99.76% attitude component 3σ coverage. Mean attitude RMSE is 0.0944° (run-level 95% CI [0.07493°, 0.11389°]); mean pointing RMS is 2.122° (run-level 95% CI [2.11697°, 2.12736°]). The short runs include an initial transient and are not directly comparable with the single 1,000 s E0 run.

## Interpretation limits

- NEES for the model-aided filter includes attitude, rate, gyro bias, and all six inertia parameters (15 DOF); gyro-driven NEES uses six DOF. NIS uses three attitude-measurement DOF.
- Attitude-estimation error, true pointing error, covariance consistency, and actuator usage measure different behaviors and should be read together.
- Time-sample fractions inside single-sample chi-square bounds do not account for temporal correlation.
- The model is a rigid body with simplified wheels, one star tracker, and zero nominal disturbance torque. Flexible dynamics, fuel slosh, detailed hardware effects, orbit/environment models, physical validation, and mission acceptance criteria are absent.
- The data characterize only the documented simulation cases. They establish neither a safe operating envelope nor flight readiness.
