# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] — Unreleased

### Added

- **Perfect-knowledge baseline.** `controller.state_feedback` (`"estimated"` default, or `"truth"`) selects whether the controller receives `q̂, ω̂` or the true state. The filter, sensors, and every random stream run identically in both modes. The default reproduces 1.0.0 output bit-for-bit (regression test against golden arrays generated from the 1.0.0 code).
- **Pointing-error decomposition.** `pointing_error_vector`, `control_error_vector`, and `q_desired` are logged. `δq_p = q_d⁻¹⊗q_true`, `δq_c = q_d⁻¹⊗q̂`, `δq_e = q̂⁻¹⊗q_true` satisfy `δq_p = δq_c⊗δq_e` exactly (tested), and the vectors add to second order.
- **Steady-state metrics.** `compute_pointing_decomposition_metrics`: APE and RPE (RMS, 99th percentile) for pointing, control, and estimation error, rate-feedback error, and the fraction of time within a project-defined pointing requirement. The window starts at the 2% settling time or at `analysis.steady_state_start_s`; settings `analysis.rpe_window_s` (10 s) and `analysis.pointing_requirement_deg` (0.1°). `compute_performance_metrics` is unchanged, so existing campaign summaries and hashes are unaffected.
- **`scripts/verify_estimation_cost.py`**, `docs/estimation_cost.json`, `docs/figures/estimation_cost.png`: paired estimated-versus-truth study for E0, E5 and E6, both filters, 10 seeds, with `--jobs` support (output is identical for any job count).
- Requirement IDs EST-DECOMP-001, EST-COST-001, SW-004 and risk R-10; roadmap items 2 and 3 completed, items 1 and 4 partly (see README §8).
- 20 tests (176 total).

### Findings

- With the gyro-driven filter the estimator adds +0.043° [0.036, 0.049] of steady-state APE (400–600 s) over true-state feedback; the model-aided filter adds +0.0004°. The cost comes through rate-feedback noise. The 0.1° project-defined requirement is met throughout in steady state in both modes, and a 30 s tracker outage is not visible in pointing.
- Over the 2%-settling window, gyro-driven estimated feedback gives a slightly *lower* APE than truth feedback (−0.012° over 600 s, −0.054° over the 200 s benchmark window). This is unexplained.
- The E6 terminal-window error (about 1°) is a control-transient effect with no measurable estimation contribution (R-02).

### Notes

- The first run of the estimation-cost study compared each mode over its own settling-based window, which confounded the paired difference. It was corrected to a common per-seed window before any result was published. A fixed late window (400–600 s) was added after seeing that the 2%-settling window still contains the acquisition tail; both are reported.
- Campaign results are not regenerated (the default path is unchanged), so `docs/e0_e7_campaign_metadata.json` still carries the 1.0.0 hashes. Its source and document hashes no longer match the working tree.

## [1.0.0] — 2026-10-24

Engineering review release. It corrects one reporting defect, closes two truth-model and estimator modeling gaps, makes the random-number streams independent, and regenerates all published evidence with the corrected code. **Numbers from before this release are not comparable with those after it**; see [Impact on published results](#impact-on-published-results).

### Fixed

- **Model-aided bias covariance indexing (reporting defect).** Three places selected the gyro-bias covariance block with `6 if P.shape == 9 else 3`, a leftover from the earlier 9-state filter: bias 3σ coverage in `compute_performance_metrics`, the bias 3σ curves in `plot_covariance`, and the final-covariance line of the text report. The model-aided filter has 15 states, so those outputs compared bias errors against the **rate** covariance. The published model-aided bias coverage (about 33–56%) was wrong; correctly computed, it is about 100%. The bias index is now recorded in every result (`bias_index`) and read through one helper, `bias_covariance_index`. A regression test covers it.
- **Global random-state coupling.** `Gyroscope(seed=…)` and `StarTracker(seed=…)` each called `np.random.seed`, so the tracker seed overwrote the gyro seed, and every noise source shared one stream. Each sensor now owns a private `numpy.random.Generator`. The runner derives independent filter-initialization, gyro, and tracker streams from each run seed with `SeedSequence.spawn`. Nothing reads or writes NumPy's global random state any more.
- **Filter initialization inconsistent with its prior.** The initial bias error was drawn with σ = 1e-4 rad/s while the filter prior assumed σ = 1e-3 rad/s, a 100× variance mismatch that biased early NEES low. Initial attitude and bias errors are now drawn from `P0`, so the initial NEES is χ²-distributed with the state dimension (verified over 200 seeds).
- **Stale campaign resume.** The resume signature hashed only scenario inputs, so `--resume` could silently reuse rows produced by older code. It now includes a SHA-256 fingerprint of `src/` and `scripts/run_experiment.py`.
- **Evidence manifest coverage.** `export_campaign_evidence.py` claimed to hash the simulation sources but omitted `src/`. All `src/**/*.py` files are now hashed.
- **E6 report assumptions.** `docs/e6_project_benchmark.json` stated an "ideal reaction-wheel torque response" although the scenario uses a 50 ms motor lag. The assumptions now describe the model actually simulated.
- **`evaluate_e6_benchmark.py`** crashed when its outputs were written outside the repository.
- **Packaging.** `pyproject.toml` installed `dynamics`, `estimation`, and the other subpackages as top-level packages, but the code imports `src.*`, so `pip install .` produced an unusable package. The `src` package is now installed as-is, and CI verifies the install from outside the checkout. The placeholder author e-mail was removed.

### Changed — models and algorithms

- **Wheel gyroscopic coupling in the truth model.** Euler's equation now includes the wheel momentum: `J ω̇ = τ_rw + τ_d − ω × (Jω + h_w)`. The wheel momentum varies linearly over each RK4 step, which is exact for a held wheel torque, so total system angular momentum is conserved to 1e-8 (relative). It can be disabled per scenario (`reaction_wheels.gyroscopic_coupling`) for comparison only.
- **Model-aided MEKF uses the measured wheel momentum.** The rate model, its Jacobian `∂f/∂ω = J⁻¹([Jω + h_w]× − [ω]× J)`, and the inertia sensitivities include `h_w`. The finite-difference Jacobian test now runs with and without wheel momentum.
- **Exact attitude propagation.** Both filters propagate `q̂ ← q̂ ⊗ Exp(ω dt)` instead of a first-order Euler step. The model-aided rate uses a midpoint (RK2) step.
- **Removed an ineffective covariance guard.** The gyro-driven filter added `1e-12·I` when `np.linalg.eigvals` (the non-symmetric solver) reported a negative eigenvalue, which could not correct any meaningful loss of definiteness. Joseph-form updates and explicit symmetrization remain.
- **Controller helpers made consistent.**
  - `quaternion_feedback_torque` used `−K_q q_e`, the opposite sign to the controller's `+K_q q_e` with `q_e = q̂⁻¹ ⊗ q_d`. It now matches, and the controller uses it internally.
  - `compute_control_gains` used `ω_n = 4/T_s`, which ignored the damping ratio, and ignored the factor of two between quaternion feedback and PD stiffness. It now uses `ω_n = 4/(ζ T_s)` and a `control_type` argument, and is verified against a simulated step response.
  - The feedforward option saturated twice, dropping feedforward whenever feedback alone saturated; it now saturates once. It also gained wheel momentum and cancels, rather than adds, the estimated disturbance.

  The campaign scenarios use feedback only, so their control law is unchanged.
- **Input validation** in `RigidBody` raises `ValueError` instead of using `assert`, which is stripped under `python -O`. The controller rejects unknown control types and non-positive torque limits.

### Withdrawn and corrected findings

- **170° acquisition "settles at about 235 s" is withdrawn.** With wheel gyroscopic coupling, the slew needs up to 0.152 N·m of `ω × h_w` torque against 0.1 N·m of authority and does not settle in 400 s. The earlier model reached the target only while violating angular-momentum conservation by 197% (`docs/wheel_coupling_effect.json`). It is tracked as risk R-07.
- **Innovation whiteness criterion (INNOV-001) replaced.** The single-run 10-lag Ljung–Box test rejects Gaussian white noise 11.1% of the time at nominal 5% for the E6 sample size, so a single p-value is not a reliable verdict either way. Whiteness is now assessed as a 12-seed ensemble against an empirically calibrated null (`scripts/verify_innovation_whiteness.py`): KS p = 0.457, consistent with white noise. The previous documentation also quoted 30 χ² degrees of freedom; the correct value is 90 (3² × 10 lags).
- **E5 outage recovery of 62.78 s is withdrawn as outage evidence.** The pointing-based metric measures recovery to 2% of a pre-outage peak that is the controller's start-up transient. It is not reached within 100 s in the regenerated runs, and the estimator-focused outage metrics replace it as evidence.
- **E1 attitude-accuracy figure** used a fixed 0.07–0.12° axis that clipped the run-level intervals; the axis now scales from the data.

### Added

- `scripts/verify_innovation_whiteness.py` and `scripts/verify_wheel_coupling.py`, with their JSON evidence in `docs/`.
- `docs/system_description.md`: a complete design description of every model and algorithm.
- Estimator-focused outage metrics, `outage_peak_attitude_error_rad` and `outage_peak_attitude_3sigma_rad`. The existing pointing-based `outage_recovery_time_s` measures recovery to 2% of the pre-outage peak. In the E5 scenario that peak is the controller's start-up transient, so the pointing metric mostly reflects controller settling, not outage behavior.
- `--jobs N` for `run_research_campaigns.py` (process-parallel cases; results are independent of `N`).
- Regression tests for every item above. The suite grew from 136 to 156 tests.
- `LICENSE` (MIT, as declared in `pyproject.toml`) and this changelog.
- CI steps that verify the package install and smoke-test the simulation entry point.

### Removed

- `docs/campaign_summary.csv`, a pre-refresh snapshot that contained the incorrect bias coverage. It remains in git history.
- Generated simulation outputs under `reports/` are no longer tracked; they were already listed in `.gitignore`. Curated evidence lives in `docs/`.
- Ad-hoc `__main__` self-test blocks inside `src/` modules, which duplicated the pytest suite.

### Impact on published results

All evidence in `docs/` was regenerated with this release. The changes affect results as follows:

| Change | Effect on results |
|---|---|
| Bias-index fix | Model-aided bias 3σ coverage corrected (reporting only; filter behavior unchanged). |
| Independent random streams | Every noise realization differs from 0.x, so individual numbers move within their statistical scatter. |
| Prior-consistent initialization | Early-epoch NEES is no longer biased low; run-mean NEES moves toward the DOF reference. |
| Wheel gyroscopic coupling | Negligible for E6, where spin and slew share an axis (settling 174.51 s with coupling vs 174.52 s without). Decisive for the 170° off-axis slew, which no longer settles (see above). |
| Exact quaternion propagation | Negligible at dt = 0.01 s; removes a first-order modeling inconsistency. |

## [0.1.0]

Initial research release: gyro-driven and model-aided MEKFs, rigid-body truth, quaternion-feedback control, reaction-wheel model, and the E0–E7 campaigns.
