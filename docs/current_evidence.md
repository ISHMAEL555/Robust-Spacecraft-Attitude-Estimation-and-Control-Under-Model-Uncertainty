# Current simulation evidence

The E0–E7 tables below are the original campaign snapshot after aligning the MEKF error-state convention. The [refreshed E0–E7 campaign report](e0_e7_campaign.md) now contains 33 cases and 330 independent runs, with a machine-readable [summary CSV](e0_e7_campaign_summary.csv), [reproduction manifest](e0_e7_campaign_metadata.json), and current campaign figures. The historical E6 tables below use the earlier ideal wheel setup; the updated E6 project benchmark is reported separately. All results remain simulation evidence for this repository's idealized model, not hardware or mission validation.

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

## Post-review verification and updated project evidence

### MEKF math and covariance

`tests/unit/test_mekf_linearization.py` finite-difference checks the gyro-driven and model-aided continuous error dynamics, including every model-aided inertia coordinate. It also checks the star-tracker local residual Jacobian and the exact SO(3) right-error reset. Van Loan Qd is compared with independent Gauss-Legendre quadrature. The reset implementation now uses the full right Jacobian instead of the first-order approximation.

The model-aided rate error was the dominant marginal contribution in the earlier matched-inertia ensemble. Adding the explicit `sigma_omega_process = 1e-7 (rad/s)/sqrt(s)` channel to both G and Q gives the following ten-run matched-inertia result (100 s per run, seeds 42 through 9042 in increments of 1000):

| State group | Mean marginal NEES | DOF | Run-level 95% t interval |
|---|---:|---:|---:|
| Attitude | 3.298 | 3 | [2.660, 3.935] |
| Rate | 2.662 | 3 | [2.315, 3.009] |
| Bias | 2.902 | 3 | [1.838, 3.965] |
| Inertia | 5.830 | 6 | [4.658, 7.001] |

Full-state mean NEES is 15.047 (15 DOF), with a run-mean 95% t interval of [12.530, 17.565]. Mean NIS is 2.958 (3 DOF), with an independent-run 95% t interval of [2.760, 3.156]. Both intervals include their respective DOF references. The across-run mean is inside its pointwise 95% chi-square bounds at 84.96% of epochs; 97% of pointwise NIS means are within their ensemble bounds. These time fractions are descriptive because each run is temporally correlated. The process-noise amplitude is tuned to the simulation and is not hardware-characterized. See [the machine-readable run summary](model_aided_consistency.json) and [NEES plot](figures/model_aided_nees_groups.png).

### E6 project benchmark

This benchmark uses the updated `config/closed_loop.yaml`: 60 deg initial attitude error, 0.1 rad/s initial body rate, 0.1 N m wheel torque, 20 N m s per-wheel momentum, and a 50 ms first-order motor response. Project criteria are settling within 2% of initial error by 180 s, at least 20 s hold, final-20-second peak pointing <=2 deg, final rate <=1e-3 rad/s, peak wheel utilization <=80%, and achieved/requested squared-torque effort >=85%.

Seed 42 settles at 176.58 s and holds for 23.41 s. Final-20-second peak pointing is 1.083 deg; final angular-rate norm is 2.81e-4 rad/s; peak wheel utilization is 49.9%; achieved effort fraction is 87.9%. All torque and momentum checks pass. The full-run pointing RMS is 1.195 rad because the initial spin creates a long acquisition transient. These thresholds are labeled project benchmarks, not mission requirements.

The earlier smooth momentum derating was replaced with predictive hard limiting at the momentum boundary. Approaching the limit no longer reduces torque asymptotically. The optional first-order torque response preserves a shared wheel torque between wheel momentum integration and spacecraft dynamics.

![Updated E6 benchmark pointing and wheel momentum usage](figures/e6_project_benchmark.png)

### Innovation whiteness and sample timing

The updated E6 seed-42 run has covariance-normalized, accepted star-tracker innovations with a 10-lag multivariate Ljung-Box statistic Q = 93.24 (30 DOF), p = 0.387. This is a single-run nominal result. The star-tracker scheduler now retains fractional sample phase; a 3 Hz tracker on a 100 Hz simulation grid produces three updates per simulated second without cumulative drift. Simulation output records each measurement's grid epoch. Delivery latency and out-of-sequence replay remain out of scope.

### Controller and software verification

The controller now forms its attitude correction in current-body coordinates and rotates the desired body rate into that same frame. Regression tests cover non-identity desired attitudes. Wheel tests cover torque/momentum hard limits and motor lag. The automated suite and CI are summarized in the repository README and the risk/traceability document.

### Selected robustness slices

The post-review slices use five seeds per case and report independent-run 95% t intervals. They are exploratory. The 25% model-aided inertia mismatch gives mean full-state NEES/DOF 20.36 while attitude RMSE is 0.092 deg, showing that good attitude accuracy does not make the augmented covariance consistent. The five-run t interval for the raw run-mean NEES crosses zero and is extremely wide, so it is not a useful bound for an acceptance claim.

### Inertia-mismatch and process-noise sensitivity

The local model-aided Jacobian, including all six symmetric-inertia coordinates, agrees with finite differences. In the simulation loop, the model-aided filter and truth plant receive the same achieved body torque. This narrows the current discrepancy to the closed-loop statistical response under model mismatch and process-noise assumptions; it does not prove that the continuous-time model is complete. The test case uses a 25% diagonal inertia offset, a 25% one-sigma inertia prior, zero disturbance torque, and no off-diagonal truth mismatch. Inertia-error coordinates are constant parameters with zero direct process noise; uncertainty enters the rate covariance through the model-aided dynamics Jacobian. `sigma_omega_process` is a continuous rate-state spectral-density amplitude, so its covariance contribution scales with amplitude squared and is discretized by Van Loan's method.

The five-run sensitivity changes only `sigma_omega_process`; seeds, 25% truth mismatch, inertia prior, sensors, and duration remain fixed. Marginal state-group NEES values below use the group's covariance block and do not add to the full-state NEES.

| `sigma_omega_process` [(rad/s)/sqrt(s)] | Full mean NEES/DOF (95% run-mean t interval) | Attitude / rate / bias / inertia marginal NEES per DOF | Mean NIS (3 DOF) |
|---:|---:|---:|---:|
| 0 | 64.905 [-28.214, 158.023] | 2.727 / 11.464 / 0.941 / 26.701 | 3.177 |
| 1e-7 (configured baseline) | 20.358 [-6.078, 46.794] | 2.053 / 4.381 / 0.941 / 21.552 | 3.101 |
| 1e-4 | 0.698 [0.534, 0.862] | 0.808 / 0.358 / 1.013 / 0.692 | 2.508 |
| 1e-3 | 0.884 [0.700, 1.068] | 0.943 / 0.509 / 1.075 / 0.973 | 2.934 |

The baseline mismatch response is driven mainly by inertia and rate marginal NEES, while bias stays near its three-DOF reference. Removing rate-process noise increases the rate and full-state NEES. Larger amplitudes make the covariance more conservative; `1e-4` produces mean NEES below one DOF-normalized and accepted NIS below three, while `1e-3` gives a full-state interval that contains one. This is a sensitivity study, not a noise identification or a validated correction. The five-run t intervals for the low-noise cases are very wide and have negative mathematical lower bounds; the untruncated intervals are retained in [the JSON report](robustness_slices.json).

![Marginal NEES sensitivity to angular-rate process-noise amplitude under 25% inertia mismatch](figures/inertia_mismatch_process_noise.png)

With 5% one-radian star-tracker outliers, disabling the NIS gate yields mean NEES above 6.6 million and accepted NIS above 6.0 million. Enabling the gate rejects 25 of 500 measurements; accepted mean NIS is 3.03 and attitude RMSE is 0.093 deg. A scheduled 30-update outage recovers to the pre-outage 2% pointing band 62.78 s after the outage ends. Tenfold Q and R scaling produce mean NIS 0.439 and 1.131, respectively, illustrating over-conservative covariance in these mismatched settings.

Five-run closed-loop ensembles show 60 deg acquisition from rest settling within 2% at 148.3 s and holding for 51.7 s. A 170 deg initial attitude needs about 235.0 s to reach the 2%-of-initial-error band; by 400 s it has 0.061 deg mean terminal error, 6.57e-5 rad/s mean terminal rate, and 48.4% mean peak wheel utilization. It does not meet the 180 s nominal project benchmark, which was defined for a 60 deg initial error. See [the machine-readable robustness results](robustness_slices.json) and [the scenario comparison plot](figures/robustness_slices.png).
