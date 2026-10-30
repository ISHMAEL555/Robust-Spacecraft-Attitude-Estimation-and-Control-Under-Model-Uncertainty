# Verification Requirements, Traceability, and Risk Register

**Release:** 1.0.0 (entries marked *1.1.0, unreleased* were added after it). This document defines the project's verification requirements, traces each to its implementation and evidence, and records open engineering risks. The requirements are **project benchmarks** for a simulation study, not mission requirements or hardware acceptance limits. Results are reported in [current_evidence.md](current_evidence.md); models are specified in [system_description.md](system_description.md).

**Verification methods:** **T** = test (automated, in `tests/`), **A** = analysis (simulation campaign or statistical analysis, in `scripts/`), **I** = inspection.

**Verdicts:** **Pass**, **Fail**, or **Characterized** (no pass/fail criterion; the behavior is measured and reported).

## 1. Requirements and verification matrix

### 1.1 Truth model

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| DYN-001 | With zero external torque, total inertial angular momentum `R(q)(Jω + h_w)` is conserved under arbitrary wheel torque, to relative error ≤ 1e-8. | T, A | `test_rigid_body.py::TestWheelGyroscopicCoupling`; [wheel_coupling_effect.json](wheel_coupling_effect.json) | **Pass**: < 1e-8 (test); 7.1e-15 (E6) and 6.7e-14 (170° slew) |
| DYN-002 | Torque-free rigid-body motion conserves kinetic energy and angular momentum. | T | `test_rigid_body.py` | **Pass** |
| ACT-001 | Body and per-wheel torque never exceed their limits; wheel momentum never exceeds its hard limit; torque is not derated approaching the limit. | T | `test_reaction_wheel.py` | **Pass** |

### 1.2 Estimator mathematics

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| EST-MATH-001 | Gyro-driven and model-aided continuous error Jacobians `F` agree with central finite differences of the exact right-error flow, including all six inertia sensitivities, with and without wheel momentum. | T | `test_mekf_linearization.py` | **Pass** |
| EST-COV-001 | Van Loan `Q_d` matches independent quadrature; the attitude reset uses the exact SO(3) right Jacobian; the star-tracker residual has an identity local Jacobian. | T | `test_mekf_linearization.py` | **Pass** |
| EST-MODEL-001 | The model-aided rate prediction includes the measured wheel momentum; open-loop prediction matches the coupled truth to ≤ 1e-7 rad/s over 20 s. | T | `test_mekf.py::TestModelAidedWheelMomentum` | **Pass** |
| EST-INIT-001 | Initial estimation errors are drawn from the filter prior (mean initial NEES = state DOF over 200 seeds, within ±1). | T | `test_experiment_runner.py` | **Pass** |

### 1.3 Statistical consistency

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| EST-STAT-001 | For matched models, run-level 95% t intervals of NEES/DOF and accepted NIS/DOF contain 1 (E0 1,000 s; E1 matched, both filters; E7). | A | [campaign CSV](e0_e7_campaign_summary.csv) | **Pass**: E0 1.023 / 1.002; E1 gyro-driven 0.965 / 1.037; E1 model-aided 0.885 / 1.011; E7 0.965 / 1.037 |
| EST-STAT-002 | Matched-inertia model-aided filter: full NEES (15 DOF), NIS (3 DOF), and every marginal state-group interval contain their DOF. | A | [model_aided_consistency.json](model_aided_consistency.json) | **Pass**: NEES 13.28 [10.28, 16.28]; NIS 3.03 [2.83, 3.24]; groups 2.48/3, 2.24/3, 2.79/3, 5.60/6 |
| EST-MODEL-002 | Characterize model-aided consistency versus inertia mismatch (0/10/25%) and versus rate-process noise at 25% mismatch. | A | [campaign CSV](e0_e7_campaign_summary.csv); [robustness_slices.json](robustness_slices.json) | **Characterized**: NEES/DOF 0.89, 1.89, 25.98 at 0/10/25%; 44.5, 27.5, 0.68, 0.88 for σ_ω = 0, 1e-7, 1e-4, 1e-3 |
| EST-TUNE-001 | Characterize sensitivity to Q and R mistuning (0.1×, 10×). | A | [campaign CSV](e0_e7_campaign_summary.csv) | **Characterized**: NEES/DOF 7.09 / 0.15 (Q), 1.39 / 0.72 (R) |
| INNOV-001 | Accepted, whitened star-tracker innovations are consistent with white noise: the 12-seed E6 ensemble of 10-lag Ljung–Box p-values is indistinguishable from an empirical white-noise null of matched size (two-sample KS p ≥ 0.05). | A | `scripts/verify_innovation_whiteness.py`; [innovation_whiteness.json](innovation_whiteness.json) | **Pass**: KS p = 0.457. The seed-42 single-run p = 0.006 is within the test's 11.1% white-noise false-alarm rate |

### 1.4 Fault response

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| FAULT-001 | With 5% injected 0.1–1 rad tracker outliers, the NIS gate keeps NEES/DOF and accepted NIS/DOF intervals containing 1 and attitude RMSE within 10% of the fault-free case. | A | [campaign CSV](e0_e7_campaign_summary.csv) (E4) | **Pass**: NEES/DOF 0.974 [0.861, 1.087]; NIS/DOF 1.034 [0.967, 1.102]; RMSE 0.0851° vs 0.0810° (+5%); 58/1,000 rejected |
| FAULT-002 | Through tracker outages of 1–30 samples, NEES/DOF intervals contain 1 and the peak attitude error stays below the peak 3σ. | A | [campaign CSV](e0_e7_campaign_summary.csv) (E5) | **Pass**: 30-sample outage NEES/DOF 0.960 [0.787, 1.132]; peak error 0.153° < 3σ 0.298° |

### 1.5 Closed-loop project benchmark (E6)

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| GNC-PB-001 | From 60° error and 0.1 rad/s: settle within 2% of initial error by 180 s, hold ≥ 20 s, final-20 s peak ≤ 2°, final rate ≤ 1e-3 rad/s. | A | `scripts/evaluate_e6_benchmark.py`; [e6_project_benchmark.json](e6_project_benchmark.json) | **Pass**: 174.51 s; 25.48 s; 1.014°; 3.00e-4 rad/s |
| GNC-PB-002 | Body and wheel torque ≤ 0.1 N·m; peak wheel momentum ≤ 80% of 20 N·m·s; achieved/requested squared-torque effort ≥ 85% with a 50 ms motor lag. | A | same | **Pass**: 0.1000 / 0.1000 N·m; 49.9%; 87.9% |
| GNC-CHAR-001 | Characterize closed-loop stress cases (authority, wheel limit, initial state, estimator tuning) and large-angle acquisition. | A | [campaign CSV](e0_e7_campaign_summary.csv) (E6); [robustness_slices.json](robustness_slices.json); [wheel_coupling_effect.json](wheel_coupling_effect.json) | **Characterized**: 60° from rest settles at 149.2 s; the 170° slew does not settle in 400 s (gyroscopic-torque limited; see R-07) |

### 1.6 Estimation cost (1.1.0, unreleased)

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| EST-DECOMP-001 | The logged pointing, control, and estimation error rotations satisfy `δq_p = δq_c ⊗ δq_e` exactly, and their rotation vectors add to second order (`θ_p − θ_c − θ_e = ½ θ_c × θ_e + O(θ³)`). | T | `test_pointing_decomposition.py` | **Pass** (identity to 1e-14 on random rotations and 1e-12 along closed-loop runs; log-log residual slope 2.0) |
| EST-COST-001 | Characterize the steady-state pointing error caused by estimation: paired estimated-versus-truth feedback, both filters, E0/E5/E6, 10 seeds, 95% t intervals on the per-seed differences. The 0.1° fraction-within-requirement threshold is project-defined. | A | `scripts/verify_estimation_cost.py`; [estimation_cost.json](estimation_cost.json) | **Characterized**: late-window APE increment +0.0425° [0.0361, 0.0490] gyro-driven, +0.0004° [0.0003, 0.0006] model-aided (E6); 0.1° requirement met throughout in steady state in both modes; outage not visible in pointing. See [current_evidence §5.4](current_evidence.md#54-estimation-cost-estimated-versus-true-state-feedback) |
| SW-004 | With `controller.state_feedback` at its default, simulation output is bit-for-bit identical to release 1.0.0 for the same seed, and truth-feedback runs share all random draws with estimated-feedback runs. | T | `test_pointing_decomposition.py` (golden arrays from 1.0.0 in `tests/data/`) | **Pass** (both filters, 10 s closed loop; compared on Python 3.11 / Windows) |

### 1.7 Software and reproducibility

| ID | Requirement | Method | Verification artifact | Result |
|---|---|:---:|---|:---:|
| SW-001 | The regression suite passes on Python 3.10 and 3.12; the package installs and imports outside the checkout. | T | `.github/workflows/tests.yml` | **Pass** locally (156 tests, Python 3.11); CI configured |
| SW-002 | Simulations are deterministic given a seed; sensor random streams are independent and do not use NumPy's global state; campaign results do not depend on the number of parallel workers. | T, A | `test_sensors.py::TestRandomStreams`; campaign rerun with 1 vs 4 workers | **Pass**: E4 summaries with 1 and 4 workers are identical field-for-field; a repeated full campaign reproduced identical NEES values |
| SW-003 | Published evidence identifies the exact code that produced it: campaign rows carry a source fingerprint; the manifest hashes `src/`, scripts, configs, docs, and figures. | I | [e0_e7_campaign_metadata.json](e0_e7_campaign_metadata.json) | **Pass** |
| SENSOR-TS-001 | Non-integer tracker/simulation rate ratios keep phase without drift; measurements are applied at their generation epoch. | T | `test_sensors.py`, `test_experiment_runner.py` | **Pass** |

## 2. Verification commands

```powershell
python -m pytest
python scripts/evaluate_e6_benchmark.py
python scripts/verify_model_aided_consistency.py
python scripts/verify_robustness_slices.py
python scripts/verify_innovation_whiteness.py
python scripts/verify_wheel_coupling.py
```

The full campaign and evidence export are listed in [current_evidence.md §8](current_evidence.md#8-reproduction).

## 3. Risk register

Likelihood and impact are qualitative and assessed for the purpose of this study.

| ID | Risk | Status | Severity | Mitigation / next evidence |
|---|---|---|---|---|
| R-01 | Model-aided covariance becomes inconsistent under inertia mismatch (NEES/DOF 25.98 at 25%), even though attitude accuracy is unchanged. A user who reads RMSE alone would trust an over-confident rate and inertia covariance. | Open | High | Do not use the model-aided rate/inertia covariance downstream without mismatch-aware tuning. Next: full-tensor (off-diagonal) mismatch sweeps, excitation-dependent observability analysis of the inertia states, and a consider-state or Schmidt-Kalman treatment of inertia. |
| R-02 | The E6 benchmark settling margin is thin: 174.51 s against 180 s (3%). Small parameter changes can flip the verdict. Estimation is not the source of the margin: over the 200 s benchmark window the model-aided estimation cost is +0.0001° [−0.0003, 0.0006] and the gyro-driven cost is negative (EST-COST-001); it is a controller-transient matter. | Open | Medium | Report a seed ensemble for the benchmark metrics; consider gain re-design with `compute_control_gains` and gyroscopic feedforward. |
| R-03 | The truth model omits flexible modes, slosh, environmental disturbance torques, wheel friction, motor electrical/thermal limits, and desaturation. | Open | Medium | Add one effect at a time and re-run the benchmark and campaign; label parameters as illustrative until tied to hardware. *Closed sub-item:* wheel gyroscopic coupling was added in 1.0.0. |
| R-04 | Sensor updates are applied at their generation epoch on the simulation grid; latency, jitter, out-of-sequence measurements, gyro scale factor and misalignment are not modeled. | Open | Medium | Add delayed-measurement replay and gyro error terms before claiming flight-software timing behavior. |
| R-05 | Campaigns use 10 runs per case (robustness slices 5) and one-factor-at-a-time variations with diagonal inertia only. | Open | Medium | Randomized dispersions (initial state, full inertia tensor, sensor parameters, disturbances) with larger ensembles; keep run-level intervals separate from pointwise bounds. |
| R-06 | The model-aided rate-process noise `sigma_omega_process` is a simulation tuning parameter. Consistency under mismatch is highly sensitive to it (NEES/DOF 27.5 → 0.88 between 1e-7 and 1e-3). | Open | Medium | Identify the spectral density from torque and model-error budgets; until then treat it as a sensitivity parameter only. |
| R-07 | Large-angle slews that rotate stored wheel momentum through the body frame need gyroscopic torque beyond the 0.1 N·m authority. The 170° case peaks at 0.152 N·m and does not settle in 400 s. | Open | High | Momentum management before large slews (unloading or desaturation), slew planning that accounts for `ω × h_w`, gyroscopic feedforward, or more authority. Define a large-angle acquisition requirement if the mission scope needs one. *The pre-1.0.0 "settles at 235 s" result was a modeling artifact (197% momentum non-conservation) and is withdrawn.* |
| R-08 | The multivariate Ljung–Box test is liberal at the available sample size (11.1% false alarms at nominal 5%), so single-run whiteness verdicts are unreliable. | Mitigated | Low | INNOV-001 uses an ensemble compared with an empirically calibrated null. |
| R-09 | Outliers below the gate threshold (about 5.5 innovation σ) are accepted and not characterized; gating is not a complete FDIR design. | Open | Medium | Sweep outlier magnitude through the gate threshold; evaluate persistent faults and FDIR logic. |
| R-10 | With the gyro-driven filter, gyro noise in the rate feedback becomes real pointing jitter: late-window APE 0.043° against 0.0005° with true-state feedback (1.73 mrad/s rate error through `K_ω = 20`). It stays inside the project-defined 0.1° requirement, with about a factor of two to spare. Not mission-level evidence. | Open | Low | Rate-feedback filtering or the model-aided filter (0.0009°). Test a tighter requirement and other gains. A noise-free-feedback comparison does not isolate which path (rate or attitude) carries the cost. |

## 4. Interpretation limits

The E6 pass is deterministic project evidence for the configured model. Campaign and robustness results are exploratory statistical characterizations over fixed, one-at-a-time cases. None of the results establishes hardware performance, mission acceptance, safe operating limits, standards compliance, or flight readiness.
