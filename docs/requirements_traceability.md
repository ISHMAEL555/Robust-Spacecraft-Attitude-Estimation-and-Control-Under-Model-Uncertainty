# GNC Verification Requirements and Risk Register

This document defines project-level verification targets for the repository's simulation. They are portfolio benchmarks, not mission requirements or hardware acceptance limits.

## Project benchmark requirements

| ID | Requirement and pass criterion | Implementation / verification | Current result |
|---|---|---|---|
| GNC-PB-001 | From 60 deg attitude error and 0.1 rad/s body rate, settle to within 2% of initial error by 180 s; remain there for at least 20 s; final 20 s peak pointing error <= 2 deg; final body-rate norm <= 1e-3 rad/s. | `config/closed_loop.yaml`; `scripts/evaluate_e6_benchmark.py`; E6 regression and report. | Deterministic seed 42: 176.58 s settling, 23.41 s hold, 1.083 deg final-window peak, 2.81e-4 rad/s final rate. **Pass.** |
| GNC-PB-002 | Keep body and wheel torque within 0.1 N m; keep peak wheel momentum below 80% of its 20 N m s limit; preserve at least 85% of requested squared-torque effort with the 50 ms motor response. | Predictive hard momentum limiting and first-order motor model in `src/actuators/reaction_wheel.py`; unit tests. | Peak wheel use 49.9%; effort fraction 87.9%; torque limits respected. **Pass.** |
| EST-MATH-001 | Finite-difference gyro-driven and augmented model-aided continuous error Jacobians agree with the implemented F matrices to the test tolerances. | `tests/unit/test_mekf_linearization.py` | All tested columns pass, including six inertia sensitivities. |
| EST-COV-001 | Van Loan Qd matches independent numerical quadrature; the right-error reset Jacobian matches finite differences at finite correction angles; measurement residual has identity local attitude Jacobian. | `tests/unit/test_mekf_linearization.py` | Pass. The reset uses the full SO(3) right Jacobian. |
| EST-STAT-001 | For the matched-inertia project case, independent-run 95% t intervals for mean NEES and run-mean NIS contain 15 and 3 DOF respectively; marginal state-group NEES intervals contain their DOF references. | `scripts/verify_model_aided_consistency.py`; `docs/model_aided_consistency.json`; `docs/figures/model_aided_nees_groups.png`. | **Pass for this matched case (exploratory n=10).** Mean NEES 15.047; run-mean 95% t interval [12.530, 17.565]. Mean NIS 2.958; run-mean 95% t interval [2.760, 3.156]. Marginal means: attitude 3.298/3, rate 2.662/3, bias 2.902/3, inertia 5.830/6; every marginal run-level interval contains its DOF. 84.96% of epochs fall inside pointwise ensemble NEES bounds and 97% of pointwise NIS means fall inside their bounds; time fractions are descriptive because epochs are correlated. |
| INNOV-001 | Accepted, covariance-normalized star-tracker innovations show no significant multivariate autocorrelation at 10 lags using a 5% Ljung-Box threshold. | `multivariate_ljung_box` in `scripts/run_experiment.py`; `tests/unit/test_experiment_runner.py` | E6 seed 42: Q = 93.24, 30 DOF, p = 0.387. **Pass for this run only.** |
| SENSOR-TS-001 | Non-integer tracker/simulation rate ratios retain phase without accumulating sample drift; updates are associated with the simulation-grid epoch at which the measurement is generated. | `src/sensors/star_tracker.py`; `measurement_time_s` in simulation output; sensor and runner tests. | 3 Hz tracker on a 100 Hz grid yields three samples per simulated second without phase loss. No transport latency or out-of-sequence measurement is modeled. |

## Verification command

```powershell
python -m pytest
python scripts/evaluate_e6_benchmark.py
```

The GitHub Actions workflow runs the regression and integration tests on Python 3.10 and 3.12. Run the E1-E7 campaigns separately when refreshing the broader statistical evidence.

## Risk register

| ID | Risk | Status | Mitigation / next evidence |
|---|---|---|---|
| R-01 | Model-aided consistency outside the matched-inertia case is not established; its rate-process noise parameter is not hardware-calibrated. | Open, medium | Matched case now passes the stated 10-run ensemble check with `sigma_omega_process = 1e-7 (rad/s)/sqrt(s)`. Validate the setting against actuator/torque uncertainty and repeat across inertia mismatch and randomized initial conditions. |
| R-02 | E6's long high-rate acquisition transient drives a large full-run pointing RMS even though the terminal pointing target is met. | Characterized | Report terminal-window peak, settling time, rate, and momentum together; do not present full-run RMS as steady-state accuracy. |
| R-03 | Truth and actuator models omit flexible modes, wheel-rotor gyroscopic coupling, friction, thermal/electrical limits, desaturation logic, and environmental disturbance models. | Open, medium | Add one modeled effect at a time and rerun the benchmark; label parameters as illustrative until tied to hardware. |
| R-04 | Sensor updates are current-epoch samples on the simulation grid; delivery latency, timestamp jitter, and delayed/out-of-sequence updates are absent. | Open, medium | Preserve measurement timestamps and add a delayed-measurement replay case before claiming asynchronous flight-software behavior. |
| R-05 | Current Monte Carlo confidence statements use ten runs and a limited set of fixed scenarios. | Open, medium | Expand randomized initial attitude/rate, inertia tensor, bias, disturbance, outage, and actuator cases; report run-level intervals and pointwise ensemble bounds separately. |
| R-06 | The model-aided angular-rate process-noise amplitude is tuned in simulation and has no hardware characterization. | Open, medium | Treat `sigma_omega_process` as an uncertainty knob; identify it from actuator and torque knowledge before using it in a hardware estimator. |

## Interpretation limits

The E6 pass is deterministic project evidence for the configured rigid-body model. The 10-run matched-inertia consistency check is exploratory. Neither establishes hardware performance, mission acceptance, safe operating limits, ECSS compliance, or flight readiness. The model-aided covariance remains unverified outside the stated matched case; successful pointing performance does not resolve that limit.
