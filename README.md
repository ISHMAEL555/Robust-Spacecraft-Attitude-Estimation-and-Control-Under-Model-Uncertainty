# Robust Spacecraft Attitude Estimation and Control

**A simulation study of multiplicative attitude estimation, covariance consistency, and reaction-wheel pointing under model and sensor uncertainty.**

This repository implements two multiplicative extended Kalman filters (MEKFs), a rigid-body truth model, a quaternion-feedback controller, and a constrained reaction-wheel assembly. It focuses on the engineering evidence behind two separate questions:

1. Does the reported estimator covariance describe the actual state error?
2. Can the closed-loop spacecraft acquire and hold its target with finite wheel torque and momentum?

> **Scope:** Results are simulation evidence for this repository's models. They are not hardware validation, mission acceptance evidence, ECSS compliance evidence, or a flight-readiness claim.

## Current engineering results

- **MEKF local mathematics:** Gyro-driven and augmented model-aided F matrices, measurement residual, attitude reset, and Van Loan covariance discretization are checked against finite differences or independent quadrature.
- **E6 pointing acquisition:** The labeled project benchmark starts at 60 deg error and 0.1 rad/s body rate, with a 0.1 N m wheel torque limit, 20 N m s per-wheel momentum limit, and a 50 ms first-order motor response. The deterministic seed-42 run settles within 2% of its initial error at 176.58 s, holds that bound for 23.41 s, peaks at 1.083 deg during the final 20 s, and ends at 2.81e-4 rad/s. Peak wheel use is 49.9%. It passes the project benchmark in [the machine-readable report](docs/e6_project_benchmark.json).
- **Innovation check:** The E6 run's covariance-normalized star-tracker innovations have a 10-lag multivariate Ljung-Box p-value of 0.387. This is evidence for that run, not a general sensor validation result.
- **Model-aided covariance:** With an explicit angular-rate process-noise term, the 15-state matched-inertia case has mean NEES 15.047 (10-run 95% t interval [12.530, 17.565]) and mean NIS 2.958 (run-level interval [2.760, 3.156]). This is an exploratory matched case; the process-noise setting is not hardware-calibrated and the inertia-mismatch envelope remains open. See the [consistency report](docs/model_aided_consistency.json).
- **Regression and integration tests:** The GitHub Actions workflow runs the suite on Python 3.10 and 3.12.

The E6 targets are explicit **project benchmarks**, not mission requirements. The full checks and assumptions are in [requirements traceability](docs/requirements_traceability.md). The benchmark trajectory and wheel margin are shown below.

![E6 project benchmark: pointing acquisition and wheel momentum use](docs/figures/e6_project_benchmark.png)

## Repository structure

| Path | Purpose |
|---|---|
| `src/dynamics/` | Scalar-last quaternion operations and rigid-body truth dynamics |
| `src/estimation/mekf.py` | Gyro-driven and model-aided MEKF, covariance, and consistency calculations |
| `src/sensors/` | Gyroscope and star-tracker noise, sampling, outlier, and outage models |
| `src/control/` | Quaternion attitude-feedback controller |
| `src/actuators/` | Reaction-wheel allocation, torque response, and hard momentum limits |
| `config/` | Nominal, uncertainty, sensor-degradation, Monte Carlo, and closed-loop scenarios |
| `scripts/run_experiment.py` | Single-run and Monte Carlo simulation |
| `scripts/run_research_campaigns.py` | E1-E7 parameter-sweep campaigns |
| `scripts/evaluate_e6_benchmark.py` | Reproducible E6 benchmark, pass/fail report, and plot |
| `scripts/plot_campaign_figures.py` | Campaign summary plots |
| `tests/` | Unit and integration verification |
| `docs/` | Frame conventions, evidence, traceability, risks, and figures |

## State conventions and models

### Attitude and frames

Quaternions use the Hamilton product and scalar-last ordering `[x, y, z, w]`. The active rotation maps body coordinates to inertial coordinates:

```text
v_I = R(q) v_B
qdot = 0.5 q [omega_B, 0]
q_true = q_hat Exp(delta_theta)
```

The MEKF attitude error, star-tracker residual, correction injection, and reported attitude errors use the same right-multiplicative body-frame convention. The controller forms the right-relative correction `q_error = inverse(q_current) * q_desired`, rotates the desired body rate into current-body coordinates, and combines attitude and rate feedback in the body-torque frame.

### Truth plant and estimator

The truth plant integrates Euler's rigid-body equation with RK4:

```text
J omega_dot + omega x (J omega) = tau_body + tau_disturbance
qdot = 0.5 q [omega_B, 0]
```

The gyro-driven MEKF has error state `[delta_theta, delta_bias]` and propagates attitude from bias-corrected gyro rate. The augmented model-aided MEKF has error state `[delta_theta, delta_omega, delta_bias, delta_inertia]`; it propagates rate from Euler's equation, uses gyro measurements for rate and bias, and estimates six symmetric-inertia coordinates.

The model-aided covariance includes an optional angular-rate process-noise amplitude, `sigma_omega_process`, to represent unmodeled torque and rate dynamics. It is a simulation parameter, not a measured hardware noise value. The matched-inertia study uses `1e-7 (rad/s)/sqrt(s)`. Continuous spectral densities are discretized with Van Loan's method. Measurement updates use Joseph covariance form and the exact SO(3) right-Jacobian reset.

### Sensors and timing

The gyro model uses:

```text
omega_m = omega_true + bias + rate_noise
bias_dot = bias_random_walk
R_gyro = sigma_v^2 / dt
Q_bias = sigma_u^2 * dt
```

The star tracker uses a right-multiplicative small-angle perturbation with per-axis covariance `sigma_theta^2 I`. It supports finite update rates, outliers, and outages. The scheduler retains fractional phase when sensor and simulation rates are not integer multiples; a measurement is timestamped at the current simulation-grid epoch and applied immediately. Transport latency, jitter, and out-of-sequence measurement replay are not modeled.

## Closed-loop project benchmark

The configured E6 scenario uses:

| Quantity | Project value |
|---|---:|
| Initial attitude error | 60 deg |
| Initial body rate | [0.1, 0, 0] rad/s |
| Body command torque limit | 0.1 N m |
| Wheel motor torque limit | 0.1 N m per wheel |
| Wheel momentum limit | 20 N m s per wheel |
| First-order motor time constant | 0.05 s |
| Simulation duration / step | 200 s / 0.01 s |

The benchmark passes when it settles within 2% of initial pointing error by 180 s, holds that bound for at least 20 s, stays below 2 deg throughout the final 20 s, ends below 1e-3 rad/s, uses no more than 80% of wheel momentum, and retains at least 85% of requested squared-torque effort. Commanded and applied torques must respect their 0.1 N m limits.

For seed 42, settling occurs at 176.58 s; the final-20-second peak is 1.083 deg, final rate norm is 2.81e-4 rad/s, peak wheel use is 49.9%, and achieved effort fraction is 87.9%. The 50 ms motor lag explains why applied squared-torque effort is below the commanded value. The **full-run** pointing RMS is 1.195 rad (68.5 deg), dominated by the initial high-rate acquisition transient; use the terminal-window pointing and settling metrics for the steady-state claim.

Re-run the E6 benchmark and model-aided matched-inertia consistency check, then refresh their figures:

```powershell
python scripts/evaluate_e6_benchmark.py
python scripts/verify_model_aided_consistency.py
python scripts/plot_mekf_consistency.py
```

## Estimator statistics and interpretation

NEES and NIS are reported with their state and measurement degrees of freedom. The gyro-driven filter is the statistically consistent baseline in the original nominal campaign. The model-aided filter's matched-inertia result now passes the repository's exploratory 10-run ensemble criterion, with wide confidence intervals; this does not establish consistency under inertia mismatch or randomized prior errors.

Innovation whiteness is evaluated on accepted, covariance-normalized innovations. The multivariate Ljung-Box result is approximate and depends on independent, regularly sampled innovations; outages and rejected outliers change that interpretation. Monte Carlo confidence intervals are computed over independent run summaries, while time-series fractions are descriptive because adjacent epochs are correlated.

Review the [current evidence and limitations](docs/current_evidence.md) alongside the [requirements traceability and risk register](docs/requirements_traceability.md). Earlier E0-E7 campaign results are retained as a campaign snapshot; the E6 project benchmark is reported separately with the updated actuator model.

![Model-aided matched-inertia NEES by state group and full-state ensemble interval](docs/figures/model_aided_nees_groups.png)

## Install, test, and reproduce

Requires Python 3.10 or later.

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Run one nominal simulation and generate analysis figures:

```powershell
python scripts/run_experiment.py --config config/nominal.yaml --output reports/nominal.npz
python scripts/analyze_results.py reports/nominal.npz --all --output-dir reports/nominal_analysis
```

Run an E1-E7 campaign:

```powershell
python scripts/run_research_campaigns.py --campaign all --runs 10 --duration 100 --output-dir reports/research_campaigns
python scripts/plot_campaign_figures.py --input reports/research_campaigns/summary.csv --output-dir docs/figures
```

Campaign outputs include compressed traces, case summaries, and plots. The checked-in [campaign CSV](docs/campaign_summary.csv) is the earlier E0-E7 snapshot and does not replace the current E6 project-benchmark report.

## Selected campaign figures

![E1 inertia uncertainty: NEES and attitude estimation error](docs/figures/e1_inertia_consistency.png)

![E4 star-tracker outlier gating](docs/figures/e4_outlier_gating.png)

![E6 E1-E7 campaign actuator-authority comparison](docs/figures/e6_control_authority.png)

These campaign plots accompany the original campaign snapshot. Read their conditions and confidence intervals in [current evidence](docs/current_evidence.md); use the dedicated E6 benchmark plot above for the updated actuator setup.

## Modeling limitations

- Rigid-body dynamics only; flexible modes, fuel slosh, and orbit/environment models are absent.
- The wheel model includes allocation, torque and momentum limits, and a first-order motor response. It omits wheel-rotor gyroscopic coupling, detailed motor electrical/thermal behavior, friction, and desaturation.
- Nominal disturbance torque is zero. Fault campaigns include sensor outages and outliers, but not a comprehensive fault-detection and recovery design.
- Sensors are sampled on the simulation grid and delivered immediately; latency, jitter, and delayed-update replay are outside scope.
- The model-aided estimator's full-state NEES is an open risk and must be resolved before treating its covariance as statistically calibrated.
- Project benchmark thresholds are not requirements for a real spacecraft and have not been tied to a hardware design.
