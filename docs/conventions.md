# Conventions and model definitions

This document records the frame, sign, and state conventions used in the
simulation. They are part of the model definition and should be kept aligned
with the implementation and tests.

## Frames and quaternions

- `I` is the inertial frame and `B` is the spacecraft body frame.
- Quaternions are scalar-last: `q = [x, y, z, w]`.
- `q_IB` maps inertial-frame vector components into body-frame components:
  `v_B = R(q_IB) v_I`.
- Quaternion composition `q2 ⊗ q1` applies `q1` first and then `q2`.
- The attitude error is a three-component, small-angle multiplicative error.
  The MEKF injects it into the nominal quaternion and resets the local error.
- Angular velocity is expressed in body coordinates.

The implementation in `src/dynamics/quaternion.py` is authoritative for the
algebra and multiplication order.

## Rigid body and wheel signs

The body obeys Euler's rigid-body equation

```text
I ω_dot = τ_body - ω × (I ω)
```

For the wheel array, `A` maps wheel-axis components into body coordinates and
`h_w` is wheel angular momentum along each wheel axis:

```text
H_w = A h_w
τ_sc = -A h_w_dot
```

Thus positive commanded body torque produces wheel momentum change of the
opposite sign. Wheel torque and wheel momentum limits are applied separately.
The reaction-wheel unit tests check conservation of angular momentum and both
limits.

## Sensors and covariance units

The gyro model is

```text
ω_m = ω_true + b_g + n_g
b_dot = n_b
```

`n_g` is white angular-rate noise; `n_b` is bias random walk. Configured gyro
noise values are continuous-time noise spectral densities. For sample interval
`dt`, the corresponding discrete gyro measurement covariance scales as
`σ_g² / dt`, and the bias random-walk covariance increment scales as
`σ_b² dt`. The MEKF uses continuous-time densities in its Van Loan process-noise
discretization.

The star tracker reports a quaternion perturbed by a small-angle measurement
error with covariance `R_ST`. Its update is applied in the tangent space.

## MEKF states

Both filters use a local attitude error, gyro-bias error, and (for the
model-aided filter) body-rate error. The nominal attitude is kept as a unit
quaternion rather than stored as three unconstrained coordinates.

**Gyro-driven MEKF (6 error states):**

```text
[δθ(3), δb_g(3)]
```

The nominal rate is derived from the current gyro measurement minus estimated
bias. This variant does not propagate body rate with the inertia model.

**Model-aided MEKF (15 error states):**

```text
[δθ(3), δω(3), δb_g(3), δI_rel(6)]
```

The six inertia parameters represent the symmetric matrix entries
`[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`. Their error coordinates are dimensionless
relative perturbations, scaled by the corresponding nominal matrix-entry
magnitude. `inertia_relative_sigma` sets the 1σ relative prior used for each
parameter. The matrix estimate is constrained to remain positive definite.

The model-aided propagation uses Euler's equation with the estimated inertia
and applied wheel torque. Gyro measurements update the body-rate and bias
states; tracker updates also correct the coupled state. It begins each run by
initializing rate from the first gyro sample at that epoch, avoiding a
propagation with a measurement that has already been consumed.

## Consistency statistics

For estimation error `e` and error covariance `P`,

```text
NEES = eᵀ P⁻¹ e
```

The six-state filter uses six degrees of freedom. The model-aided filter uses
15, including the estimated inertia parameters. For innovation `ν` and
innovation covariance `S`,

```text
NIS = νᵀ S⁻¹ ν
```

The star-tracker measurement has three degrees of freedom. NIS is reported for
all valid innovations and separately for accepted updates when gating is
enabled. Confidence bounds applied to run ensembles are descriptive statistical
checks; they do not turn temporally correlated simulation samples into
independent trials.

## Feedback controller

The controller uses quaternion attitude error and body-rate error with
proportional and derivative feedback, plus rigid-body feedforward where
configured. Torque clipping and wheel-momentum saturation can make applied
torque differ from requested torque. Pointing error, torque saturation,
momentum utilization, and applied-to-requested effort are therefore reported
separately.

## Experiment labels

- **E0:** nominal implementation and consistency check.
- **E1:** inertia-model mismatch.
- **E2:** process-noise mismatch.
- **E3:** measurement-noise mismatch.
- **E4:** star-tracker outliers and gating.
- **E5:** star-tracker outage.
- **E6:** closed-loop stress and actuator authority.
- **E7:** repeated nominal runs.

All results depend on the stated rigid-body and sensor assumptions. See the
README for study boundaries and current campaign evidence.
