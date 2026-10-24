# Robust Spacecraft Attitude Estimation and Control

An exploratory Python simulation study of how spacecraft model and sensor uncertainty affects attitude estimation, covariance consistency, and closed-loop pointing.

> **Research software, not flight software.** Results describe the implemented simulation under its stated assumptions. They do not establish hardware performance, mission acceptance, or flight readiness.

## Research question

When do model or sensor errors make a spacecraft attitude estimator inconsistent, and when does that inconsistency measurably degrade closed-loop pointing?

This repository investigates the question by simulating a rigid-body spacecraft with gyroscope and star-tracker measurements, multiplicative extended Kalman filters (MEKFs), quaternion feedback control, and reaction-wheel actuators. Estimation accuracy, covariance consistency, pointing error, and actuator effort are reported separately: a small attitude error alone does not show that a filter's uncertainty estimate is trustworthy.

## Key findings

The regenerated campaign results below are from ten 100-second runs per case. The nominal E0 reference is one 1,000-second run, so its performance values should not be compared directly with the 100-second results.

| Experiment | Result | What it shows |
|---|---|---|
| Nominal E0 | Mean NEES 6.560 (6 DOF), NIS 2.953 (3 DOF), attitude RMSE 0.036°, pointing RMS 0.673° | Reference behavior for the configured nominal case. |
| Inertia mismatch, model-aided MEKF | NEES increases from 18.409 at 0% mismatch to 40.498 at 10% and 953.660 at 25%; attitude RMSE stays near 0.094° | Attitude accuracy can remain good while the augmented filter becomes strongly inconsistent. |
| Star-tracker outliers, with gating | For 0.5 rad outliers, attitude RMSE remains near 0.094°; 46 of 1,000 updates are rejected | Innovation gating helps reject large outliers in this test. |
| 30-update tracker outage | Attitude RMSE 0.1183°, pointing RMS 2.131° | The tested outage causes a modest degradation; this is not a validated outage limit. |
| Large initial error and rate | Pointing RMS 99.41°; torque command at its limit for 95.26% of samples | This recovery case is dominated by control and actuator authority, not estimator consistency. |

The measurements and conditions are documented in [current simulation evidence](docs/current_evidence.md), and the consolidated 32-case summary is in [campaign_summary.csv](docs/campaign_summary.csv). These are simulation results, not mission acceptance thresholds.

## System at a glance

```text
Rigid-body spacecraft ──> gyro + star-tracker measurements ──> MEKF estimate
         ^                                                       │
         └──────── reaction wheels <── quaternion feedback control┘
```

The project includes two filter variants:

- **Gyro-driven MEKF:** a six-state error model for attitude error and gyroscope bias.
- **Model-aided MEKF:** a 15-state error model for attitude, body rate, gyroscope bias, and six symmetric inertia parameters.

Both filters use a nominal unit quaternion with a local attitude error. Frame, quaternion, and actuator sign conventions are described in [docs/conventions.md](docs/conventions.md).

## Quick start

Requires Python 3.10 or later. From the repository root, install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

Run the nominal configuration and analyze its results:

```bash
python scripts/run_experiment.py --config config/nominal.yaml --output reports/nominal.npz
python scripts/analyze_results.py reports/nominal.npz --all --output-dir reports/nominal_analysis
```

Run the seeded E1–E7 research campaign:

```bash
python scripts/run_research_campaigns.py --campaign all --runs 10 --duration 100 --output-dir reports/research_campaigns
```

The campaign writes one compressed result per case plus CSV/JSON summaries and plots. Use a new output directory to preserve previous outputs. The checked-in `docs/current_evidence.md` and `docs/campaign_summary.csv` capture the regenerated results discussed here; campaign commands write fresh run outputs to the selected directory.

## Experiments

The E0–E7 scenarios examine:

| ID | Study |
|---|---|
| E0 | Long nominal reference run |
| E1 | Inertia mismatch across gyro-driven and model-aided filters |
| E2–E3 | Process-noise and measurement-noise calibration |
| E4 | Star-tracker outliers and innovation gating |
| E5 | Star-tracker outages |
| E6 | Closed-loop initial-condition and actuator stress cases |
| E7 | Repeated nominal runs and run-level uncertainty |

Monte Carlo seeds are reproducible: run `i` uses `config.seed + i * 1000`. Campaign durations, seeds, and stress cases are analysis settings, not operational limits.

## Repository layout

```text
config/                 Simulation configurations
docs/                   Model conventions and current measured evidence
experiments/             Experiment definitions and research notes
reports/                 Historical analysis artifacts
scripts/                 Simulation, analysis, and campaign entry points
src/actuators/           Reaction-wheel model
src/control/             Attitude controller
src/dynamics/            Quaternion and rigid-body dynamics
src/estimation/          MEKF implementations
src/sensors/             Gyroscope and star-tracker models
tests/                   Unit and integration tests
```

Historical files under `reports/` predate the reaction-wheel sign correction and model-aided covariance changes. Use the regenerated evidence in `docs/` for current results; historical reports are retained as project provenance.

## Assumptions and limitations

- Rigid-body dynamics; no flexible modes, fuel slosh, or structural coupling.
- Simplified reaction-wheel allocation and limits; no detailed motor, friction, thermal, or jitter model.
- No orbit propagation or full environmental disturbance model; nominal runs use zero disturbance torque.
- One star tracker, with no multi-head blending, GNSS, or relative navigation.
- Outlier gating is studied, but full fault detection, isolation, and recovery is not implemented.
- No mission pointing requirement or operational acceptance criteria are defined.
- No physical, hardware, or flight validation is included.

In particular, the model-aided MEKF is strongly inconsistent in the tested 25% inertia-mismatch case even though attitude RMSE remains low. This is an open limitation and diagnostic boundary, not an established operating envelope. See [current simulation evidence](docs/current_evidence.md) for detailed metrics and interpretation.

## Dependencies

Runtime dependencies are NumPy, SciPy, Matplotlib, and PyYAML. Development extras include pytest, pytest-cov, Black, Ruff, and mypy. Dependency and tooling configuration is in [pyproject.toml](pyproject.toml).
