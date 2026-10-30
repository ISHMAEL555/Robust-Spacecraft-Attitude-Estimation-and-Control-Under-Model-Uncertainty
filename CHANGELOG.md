# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [1.2.0] — Unreleased

### Fixed

- **Regression tests are portable.** The 1.0.0 regression test compared against golden arrays with exact equality. The unmodified 1.0.0 code does not reproduce those arrays exactly on other numpy/BLAS builds (numpy 2.5, Python 3.13: about 1e-16 in the states and 1e-10 in NEES), which is rounding, not a regression. The test now uses `assert_allclose` (`rtol=1e-12, atol=1e-14` for states, torques, wheel momentum and pointing error; `rtol=1e-8, atol=1e-10` for NEES, which divides by a small covariance and amplifies rounding). The 1.1.0 wording "bit-for-bit" described the development platform only; the claim is now "matches 1.0.0 to rounding". A separate same-process test requires `np.array_equal` between two identical runs, so hidden global state is still caught.
- **Studies fail loudly instead of losing runs.** `scripts/parallel_runs.py` collects every worker exception, raises `RunFailure` naming the failed runs, and checks that the completed seeds equal the requested seeds. `run_e6_monte_carlo.py` uses it, and `run_research_campaigns.py` now exits with an error if a case fails or a case has the wrong run count. Output JSON records `n_requested`, `n_completed` and the seed list. Tested with a simulated worker failure.

### Added

- CI: Python 3.13 in the test matrix, and a `latest-dependencies` job that installs the newest numpy, scipy and matplotlib (`requirements.txt` stays unpinned). Neither has been run yet: only Python 3.11 and numpy 2.4.4 were available locally.
- `docs/e6_monte_carlo_ablation/*.json` now include per-run records (see Notes).

### Notes

- The four ablation files in `docs/e6_monte_carlo_ablation/` were regenerated only to add per-run records and the completeness fields (`n_requested`, `n_completed`, `runs`). Before replacing them, every published field was compared with the regenerated one: all 100 seeds, `n_runs`, passes per criterion, and the distributions (median, 5th and 95th percentiles of settling time and terminal error) are identical, as are the inertia-sampling and rate-direction blocks. No result changed. The 200-run `docs/e6_monte_carlo.json` was not regenerated and is byte-identical to the 1.1.0 file.
- Test count 211 → 222.

## [1.1.0] — 2026-10-30

### Added

- **Perfect-knowledge baseline.** `controller.state_feedback` (`"estimated"` default, or `"truth"`) selects whether the controller receives `q̂, ω̂` or the true state. The filter, sensors, and every random stream run identically in both modes. The default reproduces 1.0.0 output bit-for-bit (regression test against golden arrays generated from the 1.0.0 code).
- **Pointing-error decomposition.** `pointing_error_vector`, `control_error_vector`, and `q_desired` are logged. `δq_p = q_d⁻¹⊗q_true`, `δq_c = q_d⁻¹⊗q̂`, `δq_e = q̂⁻¹⊗q_true` satisfy `δq_p = δq_c⊗δq_e` exactly (tested), and the vectors add to second order.
- **Steady-state metrics.** `compute_pointing_decomposition_metrics`: APE and RPE (RMS, 99th percentile) for pointing, control, and estimation error, rate-feedback error, and the fraction of time within a project-defined pointing requirement. The window starts at the 2% settling time or at `analysis.steady_state_start_s`; settings `analysis.rpe_window_s` (10 s) and `analysis.pointing_requirement_deg` (0.1°). `compute_performance_metrics` is unchanged, so existing campaign summaries and hashes are unaffected.
- **`scripts/verify_estimation_cost.py`**, `docs/estimation_cost.json`, `docs/figures/estimation_cost.png`: paired estimated-versus-truth study for E0, E5 and E6, both filters, 10 seeds, with `--jobs` support (output is identical for any job count).
- Requirement IDs EST-DECOMP-001, EST-COST-001, SW-004 and risk R-10; roadmap items 2 and 3 completed, items 1 and 4 partly (see README §8).
- 20 tests (176 total).

- **Per-seed inertia sampling.** `spacecraft.inertia_truth_mode` (`"fixed"` default, `"sampled_from_prior"`) draws the six independent inertia elements from `N(J_nom, P0_J)` in the filter's own parameterisation and `inertia_relative_sigma`, from a dedicated fourth `SeedSequence` child. Draws are rejected unless the tensor is SPD with principal moments satisfying the triangle inequality; the rejection rate is reported (`inertia_sampling`). `src/dynamics/inertia_sampling.py`.
- `scripts/verify_inertia_sampling_study.py` (deterministic model-error study and sampled study side by side, 30 seeds, `--jobs`), `docs/model_aided_inertia_sampling.json`, `docs/model_aided_consistency_sampled.json`, `docs/figures/inertia_sampling_consistency.png`; `verify_model_aided_consistency.py --inertia-truth-mode`.
- **E6 Monte Carlo.** `scripts/run_e6_monte_carlo.py` reuses `config/closed_loop.yaml` and `evaluate_e6_project_benchmark` and disperses initial attitude (60° about a random axis; optional magnitude dispersion), initial rate (0.1 rad/s, random direction), truth inertia (per-seed sampling) and sensor noise, each from its own `SeedSequence` stream. Default 200 runs, `--jobs`, Clopper–Pearson 95% intervals per criterion, settling-time, terminal-error and wheel-momentum distributions, most-frequent failing criterion, and rank correlations with the rate direction. `docs/e6_monte_carlo.json`, `docs/e6_monte_carlo_ablation/`, `docs/figures/e6_monte_carlo.png`. 19 more tests (211 total).
- GNC-PB-003; GNC-PB-001/002 results and R-02 updated.
- EST-SAMPLE-001; EST-INIT-001 extended to 15 states; EST-STAT-002, EST-MODEL-002 and R-01 updated. 16 more tests (192 total).

### Findings

- With the gyro-driven filter the estimator adds +0.043° [0.036, 0.049] of steady-state APE (400–600 s) over true-state feedback; the model-aided filter adds +0.0004°. The cost comes through rate-feedback noise. The 0.1° project-defined requirement is met throughout in steady state in both modes, and a 30 s tracker outage is not visible in pointing.
- Over the 2%-settling window, gyro-driven estimated feedback gives a slightly *lower* APE than truth feedback (−0.012° over 600 s, −0.054° over the 200 s benchmark window). This is unexplained.
- **The E6 benchmark does not hold under dispersions.** 1 of 200 randomized runs passes every criterion: P(pass) = 0.005 [0.000, 0.028]. Settling time fails in 199 runs, the median terminal-window error is 16°, and actuator limits are never violated. Sensor noise alone leaves 100/100 passing; a random attitude axis (0/100), a random rate direction (0/100) or inertia dispersion (14/100) each breaks it. The single deterministic pass stands as a statement about one initial condition. R-02 is raised to High. Gains were not changed; a redesign is proposed as a separate follow-up.
- **Fixed truth inertia biased the matched-inertia result.** With 30 seeds the fixed matched case has NEES/DOF 0.795 [0.700, 0.890] (initial NEES 9 of 15). With inertia sampled from the filter prior it is 1.163 [0.943, 1.383], but the inertia group is mildly over-confident (1.44 [1.09, 1.79]). The earlier "all state-group intervals contain their DOF" (10 runs, fixed truth) does not survive either change.
- Model-aided inconsistency begins at a 10% prior (sampled 2.83, fixed offset 1.85) and is severe at 25% (fixed 31.97; sampled median 19.8, one run at 3,700). 55% of 25% draws are non-realisable, so that sampled distribution is heavily truncated; the inconsistency is not a truncation artefact. R-01 stands.
- The E6 terminal-window error (about 1°) is a control-transient effect with no measurable estimation contribution (R-02).

### Notes

- The first analysis of the E6 rate-direction dependence used only pass/fail and settling time, which are degenerate when 199 of 200 runs fail; continuous targets (peak terminal error, final rate) were added for that reason, and all are reported.
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
