# Spacecraft Attitude Estimation Robustness Under Model Uncertainty

**MEKF statistical consistency and its coupling to closed-loop pointing performance, under spacecraft-model and sensor mismatch.**

![Status](https://img.shields.io/badge/status-simulation%20core%20implemented-yellow) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Domain](https://img.shields.io/badge/domain-spacecraft%20GNC-lightgrey)

------------------------------------------------------------------------

## Research Question

> **How wrong can the spacecraft model and sensor assumptions be before the attitude estimator becomes statistically inconsistent, and how much closed-loop pointing performance is lost as a consequence?**

The deliverable is not a statement that "the MEKF works." It is a quantified engineering boundary:

$$\text{Model / Sensor Uncertainty} \;\rightarrow\; \text{Consistency Limit} \;\rightarrow\; \text{Pointing-Performance Limit}$$

supported by NEES/NIS statistics, Monte Carlo distributions, covariance behaviour, and closed-loop metrics.

------------------------------------------------------------------------

## Why This Problem

A filter can have a small RMSE and still be unfit for flight. Downstream consumers (the controller, fault detection, mode-transition logic) act on the **reported covariance** as much as on the state estimate. An optimistic covariance leads to over-trusted estimates, wrongly accepted outliers and mis-sized margins. A pessimistic one wastes performance. This project separates two properties that are often conflated:

| Property | Question | Metric |
|------------------------|------------------------|------------------------|
| **Accuracy** | Is the estimate close to truth? | RMSE, bias, convergence time |
| **Consistency** | Does the covariance describe the actual error? | NEES, NIS, $3\sigma$ containment |

and then asks when degraded accuracy or consistency actually becomes **control-limiting**.

------------------------------------------------------------------------

## Scope

**In scope:** nonlinear rigid-body truth model, gyro + star-tracker sensing, MEKF (attitude + gyro bias), quaternion-feedback control, reaction-wheel torque actuation, controlled model/noise mismatch, star-tracker outage and outlier, Monte Carlo analysis.

**Out of scope (deliberately):** orbital and relative navigation, rendezvous/docking, GNSS, camera/LiDAR navigation, UKF/particle filters, MPC, reinforcement learning, full FDIR, detailed wheel hardware characterisation, full orbital-environment modelling. These are valid topics. They are excluded because they would dilute a single, deep investigation.

------------------------------------------------------------------------

## System Architecture

``` mermaid
flowchart LR
    TS["True Spacecraft<br/>Nonlinear rigid body"] --> GY["Gyroscope<br/>bias + noise"]
    TS --> ST["Star Tracker<br/>absolute attitude"]
    GY --> KF["MEKF<br/>q, b_g, P"]
    ST --> KF
    KF -->|"q̂, ω̂ = ω_m − b̂_g"| CT["Attitude Controller<br/>quaternion feedback"]
    CT -->|"τ_cmd"| RW["Reaction-Wheel Model"]
    RW -->|"τ_rw"| TS
    D["Disturbance torque"] --> TS
```

The architecture is intentionally minimal so that estimator behaviour can be isolated and attributed.

------------------------------------------------------------------------

## Models

### Conventions

All conventions are frozen before implementation and documented in `docs/conventions.md`:

- Quaternion: scalar-last, JPL-style multiplicative error $q_{true} = \delta q \otimes \hat{q}$ (body-frame error), consistent with the error-state notation below
- Frames: inertial $\mathcal{I}$, body $\mathcal{B}$; $q$ maps $\mathcal{I} \rightarrow \mathcal{B}$
- Error definitions: $e_b = b_{g,true} - \hat{b}_g$; attitude error $\delta\theta$ from $\delta q \approx [\tfrac{1}{2}\delta\boldsymbol{\theta},\, 1]^T$
- Units: SI; angles in rad internally, reported in arcsec / deg

### Truth dynamics

$$I\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times(I\boldsymbol{\omega}) = \boldsymbol{\tau}_{rw} + \boldsymbol{\tau}_d, \qquad \dot{q} = \tfrac{1}{2}\Omega(\boldsymbol{\omega})\,q$$

The truth model and the estimator/controller model are **intentionally allowed to differ**, e.g. $I_{model} = I_{true} + \Delta I$.

### Gyroscope

$$\boldsymbol{\omega}_m = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g, \qquad \dot{\mathbf{b}}_g = \mathbf{n}_b$$

with angle random walk ($\sigma_v$) and bias random walk ($\sigma_u$) parameters.

### Star tracker

Absolute attitude measurement $q_{ST} = \delta q_{ST} \otimes q$, with $\delta q_{ST} \approx [\tfrac{1}{2}\delta\boldsymbol{\theta}_{ST},\,1]^T$. Supports finite update rate, measurement noise, isolated outliers and outages.

### Actuator

Reaction-wheel torque is modelled as $\boldsymbol{\tau}_{rw} \approx -\dot{\mathbf{h}}_{rw}$ with torque and momentum limits. The model is kept simple on purpose to keep attention on estimator-to-control coupling.

------------------------------------------------------------------------

## Estimator: Multiplicative Extended Kalman Filter

| Item | Definition |
|------------------------------------|------------------------------------|
| Nominal state | Variant A: $[\hat{q},\, \hat{\mathbf{b}}_g]$; Variant B: $[\hat{q},\,\hat{\boldsymbol{\omega}},\,\hat{\mathbf{b}}_g]$ |
| Error state | Variant A: $[\delta\boldsymbol{\theta},\, \delta\mathbf{b}_g]^T \in \mathbb{R}^6$; Variant B: $[\delta\boldsymbol{\theta},\,\delta\boldsymbol{\omega},\,\delta\mathbf{b}_g]^T \in \mathbb{R}^9$ |
| Error dynamics | $\delta\dot{x} = F\delta x + Gw$ |
| Covariance propagation | $\dot{P} = FP + PF^T + GQG^T$, discretised from the continuous model (van Loan) |
| Measurement | $z_k = H_k\delta x_k + v_k$, innovation $\nu_k = z_k - h(\hat{x}_k^-)$ |
| Innovation covariance | $S_k = H_kP_k^-H_k^T + R_k$ |
| Gain | $K_k = P_k^-H_k^TS_k^{-1}$ |
| Covariance update | Joseph form $P_k^+ = (I-K_kH_k)P_k^-(I-K_kH_k)^T + K_kR_kK_k^T$ |
| Correction | Multiplicative quaternion injection, then **error-state reset** (including the reset Jacobian on $P$) |

### Estimator variants

The role of the inertia matrix depends on how the filter is propagated, so the study makes this explicit:

- **Variant A (gyro-driven, baseline):** the measured rate drives kinematic propagation. The filter does not depend on $I$; inertia error then acts through the controller and feedforward terms only.
- **Variant B (model-aided):** the nine-dimensional error state is $[\delta\boldsymbol{\theta},\delta\boldsymbol{\omega},\delta\mathbf{b}_g]$. Rate is initialized from the gyro, propagated with Euler's equations using $I_{model}$ and applied wheel torque, then corrected by gyro measurements; star-tracker updates correct attitude and correlated states. Inertia mismatch enters the estimator directly through process-model error, and NEES uses nine degrees of freedom.

Comparing A and B isolates *where* inertia uncertainty hurts: the estimator, the controller, or both.

------------------------------------------------------------------------

## Consistency Evaluation

For Variant A, the six-dimensional estimation error is $e = [\delta\boldsymbol{\theta},\; \mathbf{b}_{g,true}-\hat{\mathbf{b}}_g]^T$. Variant B additionally includes angular-rate error, giving nine dimensions:

$$\epsilon_{NEES} = e^TP^{-1}e, \qquad \epsilon_{NIS} = \nu^TS^{-1}\nu$$

Over $N$ independent Monte Carlo runs, the time-indexed average satisfies $N\bar{\epsilon}_{NEES} \sim \chi^2_{N n_x}$, where $n_x=6$ for Variant A and $n_x=9$ for Variant B. NIS uses $n_z = 3$ for the star-tracker attitude measurement; its bounds at each update epoch use the actual number of available measurements across runs. NIS summaries include every available innovation, including measurements later rejected by the gate, so rejected outliers remain visible in the diagnostic. NEES and NIS reporting confidence levels are configured independently.

Additional measures: attitude and bias RMSE, covariance growth, $3\sigma$ containment, convergence time, innovation whiteness (autocorrelation test).

Interpretation:

| Observation | Typical meaning |
|------------------------------------|------------------------------------|
| NEES above upper bound | Optimistic covariance (under-modelled noise, unmodelled dynamics) |
| NEES below lower bound | Conservative covariance (over-tuned $Q$ or $R$) |
| NIS and NEES disagree | Error is hidden from the measurement path: suspect process model or unobservable directions |

------------------------------------------------------------------------

## Experiment Matrix

| ID | Experiment | Independent variable | Primary question | Key outputs |
|---------------|---------------|---------------|---------------|---------------|
| E0 | Nominal | None | Is the implementation correct and consistent? | NEES/NIS inside bounds |
| E1 | Inertia mismatch | $\Delta I$ (magnitude and structure) | How sensitive is consistency to model error? | NEES vs. $\lVert\Delta I\rVert/\lVert I\rVert$, A vs. B |
| E2 | Process-noise mismatch | $Q_{model}/Q_{true}$ | Sensitivity to optimistic/conservative $Q$ | NEES, convergence time |
| E3 | Measurement-noise mismatch | $R_{model}/R_{true}$ | Sensitivity to star-tracker quality assumptions | NIS, $3\sigma$ containment |
| E4 | Outlier | Injected innovation magnitude | Can inconsistent measurements be detected? | Accept vs. reject: state, $P$, pointing |
| E5 | Outage | Outage duration | How fast does uncertainty grow, and what is the recovery cost? | $P(t)$, error, recovery transient |
| E6 | Closed loop | Estimator degradation level | When does estimation become control-limiting? | Pointing error, settling time, effort |
| E7 | Monte Carlo | Initial conditions and parameters | Are conclusions statistically repeatable? | Distributions, confidence intervals |

The experiment configurations define common physical and sensor assumptions. The parameter grids and case labels are defined in `scripts/run_research_campaigns.py`; each Monte Carlo run uses a deterministic seed offset from the case's configured seed. Per-case time histories, a CSV/JSON metric table, and campaign plots are written to the selected output directory.

------------------------------------------------------------------------

## Closed-Loop Study

The controller consumes the estimator output with bias-corrected rate $\hat{\boldsymbol{\omega}} = \boldsymbol{\omega}_m - \hat{\mathbf{b}}_g$ and the quaternion feedback law

$$\boldsymbol{\tau}_c = -K_q\,\boldsymbol{q}_{e,v} - K_\omega\,\hat{\boldsymbol{\omega}}$$

Three feedback configurations separate controller limits from estimator limits:

| Configuration | Feedback state | Purpose |
|------------------------|------------------------|------------------------|
| Truth-state | $q_{true},\,\boldsymbol{\omega}_{true}$ | Ideal controller reference |
| Estimated-state | $\hat{q},\,\hat{\boldsymbol{\omega}}$ | Flight-like baseline |
| Degraded estimate | MEKF under injected mismatch | Quantify estimation-induced loss |

Spacecraft pointing error is measured from the **true attitude** to the desired attitude; attitude-estimation error is reported separately. Metrics include pointing RMS/peak, 2%-of-initial-error settling time, commanded and applied peak torque, commanded and applied control effort ($\int\lVert\tau\rVert^2dt$), estimation error, 3-sigma component coverage, and outage recovery where applicable. No mission pointing limit has been specified, so results compare the metrics across cases but do not declare an acceptable/failed flight-performance boundary.

------------------------------------------------------------------------

## Sensor Degradation Cases

### Star-tracker outage

With no absolute updates, attitude uncertainty is driven by gyro noise and bias. As an analytical cross-check on the simulated covariance, the attitude variance per axis during a long outage should follow approximately

$$\sigma_\theta^2(t) \approx \sigma_{\theta,0}^2 + \sigma_v^2\,t + \sigma_{b,0}^2\,t^2 + \tfrac{1}{3}\sigma_u^2\,t^3$$

Here $\sigma_{\theta,0}^2$ and $\sigma_{b,0}^2$ are the filter's posterior
attitude and bias variances at outage onset, rather than the sensor's initial
bias parameter.

The campaign injects a deterministic one-shot outage at a configured simulation time and maps $\text{outage duration} \rightarrow \text{covariance growth} \rightarrow \text{pointing degradation} \rightarrow \text{recovery transient}$. The implementation also retains probabilistic outages for stochastic sensor-degradation studies.

### Star-tracker outlier

A single anomalous measurement is injected and gated by the NIS statistic against a configurable $\chi^2_3$ threshold. Gating is disabled for the nominal baseline and enabled in `config/sensor_degradation.yaml`; its confidence is configured separately from the NIS reporting confidence. The comparison between accepting and rejecting the outlier covers state, covariance and closed-loop pointing. This is a **measurement-consistency study, not a full FDIR design.**

------------------------------------------------------------------------

## Verification & Validation

Verification proceeds bottom-up so that software defects are separated from genuine GNC behaviour:

``` text
Quaternion algebra → Rigid-body dynamics → Sensor models → MEKF propagation
→ MEKF update → Consistency metrics → Controller → Closed loop → Monte Carlo
```

Planned checks include:

- Quaternion: unit norm, composition/inverse identities, rotation-matrix equivalence
- Dynamics: angular-momentum and energy conservation in the torque-free case
- Jacobians: analytic $F$, $H$ against finite differences
- Covariance: symmetry and positive semi-definiteness after every update; Joseph vs. standard form agreement
- Reset: consistency of the covariance reset with the multiplicative injection
- Filter: NEES/NIS consistent on a **matched** simulation (Q, R, model all correct) before any mismatch is introduced
- Outage: simulated covariance growth against the analytical expression above
- Optional independent cross-validation of selected dynamics/estimation results in MATLAB/Simulink

Unit tests live in `tests/unit/`, system-level tests in `tests/integration/`.

------------------------------------------------------------------------

## Results

### E0 nominal run (single seed)

The first reproducible 1000 s nominal run (`config/nominal.yaml`, seed 42) completed
with 100,000 time steps. The run produced a mean NEES of 6.561 and mean NIS of 2.953;
96.7% of NEES samples and 95.1% of valid NIS samples fell within their raw 95%
chi-squared bounds (6 and 3 degrees of freedom, respectively). True spacecraft
pointing-error RMS was 0.671 deg and peak error was 4.243 deg; the separate
attitude-estimation RMSE was 0.036 deg. The 0.036 deg figure is estimator accuracy,
not closed-loop pointing performance.

These are **single-run, time-series diagnostics**, not Monte Carlo confidence
claims; adjacent samples are correlated. Results and plots are in
[`reports/nominal_analysis/`](reports/nominal_analysis/), with the raw simulation
history in [`reports/nominal.npz`](reports/nominal.npz).

The E1-E7 exploratory grid contains 27 cases with 10 runs of 100 s each. It is
useful for comparing trends, but is not the planned final study of 50-100 runs
at 500-1000 s per case. The corrected run completed all 27 cases; its per-case
histories, summary table, and plots are in
[`reports/research_campaigns/`](reports/research_campaigns/).

Key exploratory findings:

| Finding | Result |
|------------------------------------|------------------------------------|
| Matched gyro-driven filter (E1/E7) | Mean NEES 6.007 and NIS 2.973; 95.9% and 97.0% of samples, respectively, were inside the raw 95% bounds |
| Model-aided inertia sensitivity (E1) | Matched mean NEES/NIS were 10.686/3.033; 10% and 25% inertia mismatch raised mean NEES to $2.26\times10^5$ and $1.17\times10^6$ |
| Process-noise sensitivity (E2) | Scaling $Q$ by 0.1 and 10 produced mean NEES 45.318 and 0.875, respectively, illustrating severe overconfidence and conservatism |
| Outage covariance cross-check (E5) | Observed/predicted attitude variance ratios were 1.073, 1.029, and 1.012 for 1-, 10-, and 30-update outages |
| Outlier gating (E4) | Gating kept mean NEES at 5.460 versus values above $6.2\times10^4$ without gating; NIS summaries include rejected innovations |
| Closed-loop stress case (E6) | Pointing RMS was about 99 deg across the tested estimator-degradation variants; this is a stress-case result, not a flight-performance acceptance decision |

These 10-run results support trend comparisons, not high-confidence mission
claims. E6 remains subject to the lack of a specified mission pointing limit.

| Result | Status |
|------------------------------------|------------------------------------|
| R1 | E0 single-seed baseline and 10-run E7 pilot completed; final Monte Carlo power pending |
| R2 | E1 inertia, E2 process-noise, and E3 measurement-noise exploratory comparisons completed |
| R3 | E5 outage cases and analytical covariance-growth cross-check completed |
| R4 | E4 gated/ungated outlier exploratory comparison completed |
| R5 | E6 closed-loop degradation pilot completed; pointing acceptance threshold is unspecified |
| R6 | Final statistically powered consistency and pointing-performance limits pending |

| Result | Content |
|------------------------------------|------------------------------------|
| R1 | Nominal NEES/NIS with confidence bounds |
| R2 | Consistency-limit curves for $\Delta I$, $Q$ and $R$ mismatch |
| R3 | Outage covariance growth vs. analytical prediction |
| R4 | Outlier accept/reject comparison |
| R5 | Pointing error vs. estimator degradation level |
| R6 | Summary table: mismatch level at which consistency fails and at which pointing requirements fail |

------------------------------------------------------------------------

## Repository Structure

``` text
spacecraft-estimation-control/
├── src/
│   ├── dynamics/        # rigid_body.py, quaternion.py
│   ├── sensors/         # gyro.py, star_tracker.py
│   ├── estimation/      # mekf.py
│   ├── control/         # attitude_controller.py
│   └── actuators/       # reaction_wheel.py
├── experiments/         # nominal, model_uncertainty, noise_mismatch,
│                        # sensor_outlier, sensor_outage, closed_loop
├── analysis/            # consistency, monte_carlo, visualization
├── tests/               # unit, integration
├── config/              # per-experiment configuration and seeds
├── scripts/
├── docs/                # conventions, derivations
├── reports/
└── README.md
```

## Technology

Python · NumPy · SciPy · Matplotlib · pytest · Git/GitHub. MATLAB/Simulink for optional cross-validation.

## Getting Started

Install the package and development dependencies, then run the tests:

``` bash
python -m pip install -e ".[dev]"
pytest -q
```

Run a configured simulation and save its results:

``` bash
python scripts/run_experiment.py --config config/nominal.yaml --output reports/nominal.npz
```

Run the configured Monte Carlo campaign with a smaller run count for a quick check:

``` bash
python scripts/run_experiment.py --config config/nominal.yaml --monte-carlo --runs 5 --output reports/nominal_mc.npz
```

Generate a text report. Add `--all` to save diagnostic plots as well:

``` bash
python scripts/analyze_results.py reports/nominal.npz --output-dir reports/nominal_analysis
python scripts/analyze_results.py reports/nominal.npz --all --output-dir reports/nominal_analysis
python scripts/run_experiment.py --config config/nominal.yaml
```

Run the E1-E7 parameter grids. Omit `--duration` to use each experiment's configured
duration; use a shorter duration for a smoke test, not for final conclusions:

``` bash
python scripts/run_research_campaigns.py --campaign all --runs 10 --duration 100 --output-dir reports/research_campaigns
```

The runner emits one compressed time-history file per case plus `summary.csv`,
`summary.json`, and one comparison plot per experiment. The configured-duration
campaign can be run with `--runs 50` and no duration override; E7's 100-run target
can be run separately with `--campaign E7 --runs 100 --output-dir reports/research_campaigns/E7`.

------------------------------------------------------------------------

## Assumptions & Limitations

- Rigid body; no flexible modes or fuel slosh
- Reaction-wheel model without friction, jitter or detailed hardware dynamics
- Disturbance torque modelled as bounded/stochastic, not from a full environment model
- Single-sensor attitude update (one star tracker); no multi-head blending
- Conclusions apply to the stated sensor class and parameter ranges, and the analysis reports the ranges explicitly

## Roadmap

- [x] Freeze quaternion and frame conventions
- [x] Implement truth dynamics
- [x] Implement sensor models
- [x] Derive and implement gyro-driven and nine-state model-aided MEKF variants
- [x] Verify core estimator behavior with unit and integration tests
- [x] Implement attitude controller and integrate reaction-wheel model
- [x] Run exploratory E1-E7 campaign pilot (10 runs x 100 s per case)
- [ ] Run final statistically powered nominal NEES/NIS and E1-E7 campaigns
- [ ] Establish consistency and control limits at final study duration/run count
- [ ] Define a mission pointing acceptance threshold before declaring pointing pass/fail
- [ ] Write-up and report

## References

1.  Lefferts, Markley, Shuster, "Kalman Filtering for Spacecraft Attitude Estimation," *J. Guidance, Control, and Dynamics*, 1982.
2.  Markley, Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*, Springer, 2014.
3.  Trawny, Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation," UMN Tech. Report, 2005.
4.  Solà, "Quaternion Kinematics for the Error-State Kalman Filter," 2017.
5.  Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 (NEES/NIS).

------------------------------------------------------------------------

## Author

**Kowluri Ishmael** · Spacecraft GNC · Attitude Estimation · Flight Dynamics · Control