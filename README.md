# Robust Spacecraft Attitude Estimation and Control

An exploratory simulation study of when spacecraft model and sensor errors
make an attitude estimator inconsistent, and when that inconsistency affects
closed-loop pointing.

## Problem and approach

The project couples rigid-body attitude dynamics, gyro and star-tracker sensor
models, two multiplicative extended Kalman filters (MEKFs), quaternion feedback
control, and a reaction-wheel actuator model. It studies estimator accuracy,
covariance consistency, true pointing, and actuator effort as separate
quantities.

The gyro-driven MEKF has six error states: attitude and gyro bias. The
model-aided MEKF has 15: attitude, body rate, gyro bias, and six symmetric
inertia parameters. Both filters use a local attitude error with a nominal unit
quaternion. See [the conventions and model definitions](docs/conventions.md).

Experiments E0–E7 cover nominal behavior, inertia mismatch, process- and
measurement-noise mismatch, star-tracker outliers and gating, tracker outages,
closed-loop actuator stress, and repeated nominal runs.

## Current status

The simulator and experiments are implemented. The reaction-wheel torque sign
and model-aided covariance propagation have been corrected and regression
tested. Fresh E0–E7 measurements are summarized in
[current evidence](docs/current_evidence.md), with the full 32-case table in
[campaign_summary.csv](docs/campaign_summary.csv). The checked-in historical
pilot reports predate the wheel-sign correction and are retained for
provenance, not as current evidence.

The model-aided filter estimates inertia uncertainty instead of treating
inertia as exact. In ten 100-second runs, mean NEES rose from 18.4 at matched
inertia to 40.5 at 10% mismatch and 953.7 at 25% mismatch, despite attitude
RMSE remaining near 0.094°. This is a clear consistency boundary and an open
limitation, not a passed consistency result.

## Reproduce

Install the project and its development dependencies, then run the tests:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Run E0 and generate an analysis report:

```bash
python scripts/run_experiment.py --config config/nominal.yaml --output reports/nominal.npz
python scripts/analyze_results.py reports/nominal.npz --all --output-dir reports/nominal_analysis
```

Run the seeded E1–E7 campaign:

```bash
python scripts/run_research_campaigns.py --campaign all --runs 10 --duration 100 --output-dir reports/research_campaigns
```

Each Monte Carlo run uses `config.seed + run_index * 1000`. The campaign writes
one compressed result per case plus CSV/JSON summaries and plots. Choose a new
output directory to preserve earlier runs. The case duration and seed count
are exploratory analysis choices, not mission validation criteria.

## Model boundaries

- Rigid-body dynamics only; no flexible modes, fuel slosh, or structural
  coupling.
- Simplified reaction-wheel allocation and limits; no detailed motor,
  friction, thermal, or jitter model.
- No orbit propagation or full environmental disturbance model; nominal runs
  use zero disturbance torque.
- One star tracker; no multi-head blending, GNSS, or relative navigation.
- Outlier gating is a measurement-consistency study, not complete fault
  detection, isolation, and recovery.
- No mission pointing requirement or operational acceptance criteria are
  defined.

The repository can support software-level verification and internal simulation
comparisons. It cannot establish physical-model validity, hardware performance,
mission acceptance, or flight readiness without independent analysis and
hardware or higher-fidelity validation.

## Repository layout

```text
config/                 Simulation configurations
docs/                   Conventions and measured evidence
reports/                Historical analysis artifacts
scripts/                Simulation, analysis, and campaign entry points
src/actuators/          Reaction-wheel model
src/control/            Attitude controller
src/dynamics/           Quaternion and rigid-body dynamics
src/estimation/         MEKF implementations
src/sensors/            Gyroscope and star tracker
tests/                  Unit and integration checks
```
