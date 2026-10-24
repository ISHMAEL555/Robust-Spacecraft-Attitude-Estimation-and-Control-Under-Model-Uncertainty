# Frames, signs, and state conventions

These conventions define the simulated plant, estimator, and measurements. They should remain aligned with the implementation and analyses.

## Frames and quaternions

- `I` denotes the inertial frame and `B` the spacecraft body frame.
- Quaternions are scalar-last Hamilton quaternions: `q = [x, y, z, w]`.
- The active rotation matrix `R(q)` maps body-frame vector components to inertial components: `v_I = R(q) v_B`. In this sense, `q` represents body attitude relative to inertial space. Some older variable names use `q_IB`; the matrix and kinematics in `src/dynamics/quaternion.py` define the actual convention.
- Angular velocity `ω` is expressed in body coordinates. Quaternion kinematics are `q_dot = 1/2 q ⊗ [ω, 0]`.
- Quaternion composition `q2 ⊗ q1` applies `q1` first, then `q2`.

### MEKF attitude error

The MEKF uses a right-multiplicative, body-frame local error:

```text
q_true = q_hat ⊗ δq
δθ     = Log(δq)
```

Accordingly, the measurement innovation is `Log(q_hat⁻¹ ⊗ q_meas)` and correction injection is `q_hat ← q_hat ⊗ Exp(δθ_hat)`. The covariance linearization uses this same right-error convention. `right_error_quat`, `right_box_plus`, and `right_box_minus` implement these operations. The older `error_quat`, `box_plus`, and `box_minus` helpers retain their left-multiplicative semantics for compatibility and general quaternion utilities; they are not used for MEKF error-state updates.

## Rigid body and wheel signs

The bus obeys Euler's equation with the wheel momentum in the gyroscopic term:

```text
I ω_dot = τ_sc + τ_disturbance − ω × (I ω + H_w)
```

For the wheel array, `A` maps wheel-axis components into body coordinates and `h_w` is wheel angular momentum along each wheel axis:

```text
H_w  = A h_w
τ_sc = −A h_w_dot
```

Thus wheel momentum rate and spacecraft reaction torque have opposite signs, and total angular momentum `R(q)(I ω + H_w)` is conserved when `τ_disturbance = 0`. Wheel torque and wheel momentum limits are applied separately. The full model is in [system_description.md](system_description.md#2-truth-dynamics).

## Sensors and covariance units

The gyro model is

```text
ω_m = ω_true + b_g + n_g
b_dot = n_b
```

`n_g` is white angular-rate noise and `n_b` is bias random walk. Configured gyro noise values are continuous-time noise spectral densities. For sample interval `dt`, the discrete gyro measurement covariance scales as `σ_g² / dt`, while the bias random-walk covariance increment scales as `σ_b² dt`. The MEKF uses continuous-time densities in its Van Loan process-noise discretization.

The star tracker reports a quaternion perturbed by a small-angle error on the right, `q_meas = q_true ⊗ Exp(δθ_ST)`, with covariance `R_ST`. The update is applied in the same body-frame tangent space.

## MEKF states

The nominal attitude is maintained as a unit quaternion, not as three unconstrained coordinates.

**Gyro-driven MEKF (6 error states):**

```text
[δθ(3), δb_g(3)]
```

The nominal rate is derived from the current gyro measurement minus estimated bias. This variant does not propagate body rate with the inertia model.

**Model-aided MEKF (15 error states):**

```text
[δθ(3), δω(3), δb_g(3), δI_rel(6)]
```

The six inertia parameters represent the symmetric tensor entries `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`. Their error coordinates are dimensionless relative perturbations, scaled by the corresponding nominal matrix-entry magnitude. `inertia_relative_sigma` sets the 1σ prior for each parameter. The estimated inertia matrix is kept positive definite.

The model-aided propagation uses Euler's equation with the estimated inertia, the applied wheel torque held over the step, and the measured (tachometer) wheel momentum. Both variants propagate the nominal attitude with the exact quaternion exponential `q̂ ⊗ Exp(ω dt)`. Gyro measurements update body rate and bias; tracker updates correct attitude and the coupled state. The first gyro sample initializes rate at its measurement epoch; it is not also used to propagate over an interval that has already elapsed.

## Consistency statistics

For estimation error `e` and error covariance `P`,

```text
NEES = eᵀ P⁻¹ e
```

The six-state filter uses six degrees of freedom. The model-aided filter uses 15, including the estimated inertia parameters. For innovation `ν` and innovation covariance `S`,

```text
NIS = νᵀ S⁻¹ ν
```

The star-tracker attitude measurement has three degrees of freedom. NIS is reported for all valid innovations and separately for accepted updates when gating is enabled. Confidence-bound fractions over a time series are descriptive because adjacent samples are correlated; they are not independent trials.

## Feedback controller

The controller forms the shortest-path quaternion error and body-rate error and applies proportional and derivative feedback, with rigid-body feedforward available where configured. Torque command saturation and wheel-momentum limiting can make applied body torque differ from requested torque. Pointing error, command saturation, wheel momentum utilization, and applied-to-requested effort are therefore reported separately.

## Experiment labels

- **E0:** nominal implementation and consistency reference.
- **E1:** inertia-model mismatch.
- **E2:** process-noise mismatch.
- **E3:** measurement-noise mismatch.
- **E4:** star-tracker outliers and gating.
- **E5:** star-tracker outage.
- **E6:** closed-loop stress and actuator authority.
- **E7:** repeated nominal runs.

All results depend on the rigid-body and sensor assumptions stated here and in the configuration files. See the [README](../README.md) for study boundaries and the [current evidence](current_evidence.md) for regenerated results.
