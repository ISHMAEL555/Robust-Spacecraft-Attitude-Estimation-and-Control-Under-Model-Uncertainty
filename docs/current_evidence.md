# Current simulation evidence

This page records regenerated results after correcting the reaction-wheel sign
and adding inertia uncertainty to the model-aided MEKF. Results are simulation
evidence for this repository's idealized model; they are not hardware or
mission validation. The checked-in artifacts in `reports/` are earlier pilot
runs and should be treated as historical.

## Run conditions

- E0: one 1,000-second nominal run at 100 Hz.
- E1: six cases, each 10 seeded runs of 100 seconds.
- Monte Carlo run seeds are `config.seed + run_index * 1000`.
- Run-level confidence intervals use the ten independent simulation run
  summaries. Consistency-bound fractions across time are descriptive because
  adjacent samples are correlated.
- Regenerated `.npz` files and plots are under the ignored `.venv/current_campaigns/`
  directory during this work. The consolidated 32-case campaign CSV is available here.

## Nominal reference (E0)

| Metric | Result | Reference |
|---|---:|---:|
| Mean NEES | 6.560 | 6 DOF |
| Mean NIS | 2.953 | 3 DOF |
| NEES samples within single-sample 95% bounds | 96.7% | Descriptive time fraction |
| NIS samples within single-sample 95% bounds | 95.1% | Descriptive time fraction |
| Attitude-estimation RMSE | 0.036° | 1,000 s |
| True pointing RMS | 0.673° | 1,000 s |

## Inertia mismatch (E1)

Each row is the mean across ten 100-second runs. Attitude-estimation RMSE is
nearly unchanged, while the model-aided covariance consistency degrades with
mismatch. The gyro-driven filter does not use inertia in its state propagation;
its pointing still changes because the spacecraft truth and control response
change with inertia.

| Filter | Inertia mismatch | NEES (DOF) | NIS (DOF) | Attitude RMSE | Attitude component 3σ coverage | Inertia relative RMSE | Pointing RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-driven | 0% | 6.007 (6) | 2.973 (3) | 0.0944° | 99.76% | — | 2.122° |
| Gyro-driven | 10% | 6.010 (6) | 2.974 (3) | 0.0944° | 99.76% | — | 2.350° |
| Gyro-driven | 25% | 6.013 (6) | 2.974 (3) | 0.0944° | 99.76% | — | 2.700° |
| Model-aided | 0% | 18.409 (15) | 2.979 (3) | 0.0940° | 99.25% | 0.94% | 2.097° |
| Model-aided | 10% | 40.498 (15) | 3.023 (3) | 0.0940° | 97.09% | 3.16% | 2.324° |
| Model-aided | 25% | 953.660 (15) | 3.290 (3) | 0.0940° | 92.68% | 6.69% | 2.669° |

The 25% case is strongly inconsistent in the augmented 15-state estimate even
though attitude RMSE remains small. This demonstrates why accuracy, NEES,
coverage, NIS, and closed-loop pointing need separate reporting. It also means
the present model-aided filter has **not** established a consistency envelope
for large inertia mismatch. The mismatch cases are diagnostic boundaries, not
accepted operating limits.

## Noise calibration (E2–E3)

Ten-run, 100-second noise sweeps reproduce the expected separation between
state performance and covariance consistency. Attitude RMSE and pointing RMS
barely change, while the reported NEES/NIS and coverage move substantially.

| Sweep | Scale | Mean NEES | Mean NIS | Attitude 3σ coverage | Attitude RMSE | Pointing RMS |
|---|---:|---:|---:|---:|---:|---:|
| Process noise Q | 0.1× | 45.323 | 11.203 | 80.57% | 0.0946° | 2.122° |
| Process noise Q | 1× | 6.007 | 2.973 | 99.76% | 0.0944° | 2.122° |
| Process noise Q | 10× | 0.875 | 0.439 | 100.00% | 0.0944° | 2.122° |
| Measurement noise R | 0.1× | 8.493 | 4.123 | 97.13% | 0.0944° | 2.122° |
| Measurement noise R | 1× | 6.007 | 2.973 | 99.76% | 0.0944° | 2.122° |
| Measurement noise R | 10× | 4.558 | 1.147 | 100.00% | 0.0946° | 2.122° |

The optimistic Q case is strongly overconfident; inflated Q/R cases are
conservative in these runs. High coverage alone does not imply a good model:
the 10× noise cases also have far-below-expectation NEES/NIS.

## Outlier rejection (E4)

With a 5% outlier probability, gating rejected 46 of 1,000 tracker updates
(4.6%) for each magnitude. For 0.5 rad outliers, ungated attitude RMSE rose to
5.40° and mean NEES to 1.56 million; gating kept attitude RMSE near 0.094° and
mean NEES at 5.46. The all-innovation NIS remains very large because it also
includes the rejected outlier innovations. Accepted-only NIS is 2.94, close to
its three-degree-of-freedom expectation. Pointing RMS with the gate enabled was
2.124° in this test.

## Tracker outage (E5)

The 30-update outage increased attitude RMSE from 0.0944 degrees to 0.1183
degrees and reduced attitude component 3-sigma coverage from 99.66% (one-update
outage) to 99.40%. Pointing RMS increased from 2.122 degrees to 2.131 degrees.
This is a measured degradation trend for the tested scenario, not a validated
maximum outage duration or recovery requirement.

## Closed-loop stress and actuator authority (E6)

Each case below is ten 100-second runs. The initial condition is a 60-degree
attitude error and 0.1 rad/s body rate unless stated otherwise. Torque saturation
and wheel momentum are separate limits: the nominal case saturates commanded
torque for 95.26% of samples while using 53.0% of available wheel momentum.

| Case | Pointing RMS | Final error | Command torque at limit | Peak wheel momentum / limit | Time near wheel limit | Applied / requested effort |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 99.41 deg | 122.34 deg | 95.26% | 53.0% | 0.00% | 58.9% |
| Q x10 | 99.41 deg | 122.33 deg | 95.26% | 53.0% | 0.00% | 58.9% |
| R x10 | 99.41 deg | 122.34 deg | 95.27% | 53.0% | 0.00% | 58.9% |
| Q and R x10 | 99.41 deg | 122.34 deg | 95.26% | 53.0% | 0.00% | 58.9% |
| 25% inertia mismatch | 103.18 deg | 147.57 deg | 95.05% | 54.2% | 0.00% | 57.1% |
| Zero initial rate | 30.77 deg | 0.17 deg | 39.36% | 27.1% | 0.00% | 86.1% |
| Half torque authority | 109.41 deg | 178.37 deg | 97.77% | 34.0% | 0.00% | 71.9% |
| Double torque authority | 98.09 deg | 59.00 deg | 90.79% | 71.7% | 0.00% | 47.8% |
| 1 N m s wheel limit | 114.48 deg | 132.28 deg | 88.58% | 99.9% | 12.31% | 17.6% |
| 170-degree initial attitude | 159.01 deg | 132.78 deg | 100.00% | 45.4% | 0.00% | 72.7% |

The noise-scaled cases have virtually identical pointing despite large changes in
NEES and NIS, showing that this severe recovery case is dominated by control
and actuator limits. Nominal E6 mean NEES/NIS are 3,273/71.9; multiplying Q by
10 reduces them to 241/5.57, while pointing RMS stays at 99.41 degrees. Thus,
this stress case does not show estimator consistency driving the pointing
failure. The 25% inertia-mismatch E6 case has 103.18-degree pointing RMS versus
99.41 degrees nominal, but both cases are deeply torque-saturated. Zero initial
rate settles within the 100-second window
(76.96 seconds to the configured 2%-of-initial-error criterion). Raising torque
authority helps final error but increases wheel momentum use; the low-momentum
case reaches its limit and realizes only 17.6% of requested effort. The tested
170-degree case remains saturated and does not recover. These are scenario
comparisons, not controller or actuator acceptance limits.

## Repeatability (E7)

Ten 100-second nominal runs produced mean NEES 6.007 (6 DOF), mean NIS 2.973
(3 DOF), and attitude component 3-sigma coverage 99.76%. Mean attitude RMSE
was 0.0944 degrees, with run-level 95% CI [0.0749, 0.1139] degrees. Mean
pointing RMS was 2.122 degrees, with run-level 95% CI [2.117, 2.127] degrees.
This duration includes the initial rate transient; it is not directly
comparable to the 1,000-second E0 RMS.

The consolidated 32-case campaign summary, including run-level intervals, is
available in [campaign_summary.csv](campaign_summary.csv). Regenerated compressed
traces and plots were written to the ignored `.venv/current_campaigns/`
directory for local inspection.

## Interpretation limits

The simulation uses a rigid body, simplified reaction wheels, one star tracker,
and zero nominal environmental disturbance. It omits flexible-body and
high-fidelity hardware effects. There is no mission pointing requirement or
external validation data in the repository. These results answer what the
implemented simulation does under its stated assumptions; they cannot establish
hardware accuracy, mission acceptability, or flight readiness.
