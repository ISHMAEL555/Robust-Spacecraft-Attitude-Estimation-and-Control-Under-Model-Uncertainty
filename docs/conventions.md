# Conventions and model equations

This note records the frame, quaternion, estimator, sensor, and reporting
conventions used by the simulator. Vectors are column vectors unless stated.

## Frames and quaternions

- A quaternion is scalar-last, `q = [q_x, q_y, q_z, q_w]`, and uses the
  Hamilton product.
- `q_IB` maps inertial-frame coordinates into body-frame coordinates:
  `v_B = R(q_IB) v_I`.
- Composition is ordered by application: `q_AC = q_BC ⊗ q_AB`.
- Quaternion kinematics are `q_dot = 0.5 Ω(ω_B) q`, where `ω_B` is body rate.
- The MEKF right-multiplicative error is represented by
  `q_true = δq ⊗ q_hat`; locally `δq ≈ [δθ/2, 1]`.
- A physical rotation is unchanged by `q` versus `-q`. The controller selects
  the nonnegative-scalar representative by default; optional sign hysteresis
  is a separate, disabled-by-default test feature near 180 degrees.

## Truth dynamics and actuators

For body inertia `I`, body rate `ω`, wheel momentum `h_w`, applied wheel
torque `τ_rw`, and external disturbance `τ_d`, the optional momentum-coupled
truth equation is

```text
I ω_dot + ω × (I ω + h_w) = τ_rw + τ_d.
```

The wheel gyroscopic term is disabled by default for compatibility with the
pilot model. The wheel actuator applies torque and momentum limits;
`τ_rw = -h_dot_w` under the reaction-wheel sign convention.

The regulation controller commands bounded body torque from quaternion-vector
and rate feedback. The implementation does not automatically cancel a
disturbance; disturbance compensation is an optional feedforward interface.
Commanded and applied torques and their squared-integral efforts are reported
separately.

## Sensors and sampled noise

The gyro reports

```text
ω_m = ω_true + b_g + n_g,       b_dot_g = n_b.
```

`sigma_v` and `sigma_u` are continuous-time white-noise spectral-density
amplitudes for gyro angle/rate noise and bias random walk, respectively.
Van Loan discretization consumes the continuous spectral densities
`Q_c = diag(sigma_v² I_3, sigma_u² I_3)`; the discrete sample covariances are
not interchangeable with `Q_c`. Optional fixed sensor-axis misalignment is
configurable and disabled by default.

The star tracker returns a multiplicative attitude observation with
three-axis small-angle covariance `R = sigma_theta² I_3`. Sensor rate,
outage schedule/probability, and outlier probability are configuration
parameters.

## Estimator variants and consistency statistics

**Gyro-driven MEKF (6 error states):** attitude and gyro bias are estimated;
gyro measurements drive attitude propagation. Inertia is not part of its
propagation model.

**Model-aided MEKF (9 error states):** attitude, body rate, and gyro bias are
estimated. Rate propagation uses Euler's equation, modeled inertia, and
applied wheel torque. A configured relative inertia-error bound may add
conservative rate process noise; a zero bound reproduces the unbounded
model-aided variant. This bound is a modeling assumption, not an estimate of
the actual inertia error.

For attitude/bias/rate error `e` and reported covariance `P`, NEES is
`eᵀ P⁻¹ e`; its degrees of freedom are 6 for gyro-driven and 9 for
model-aided estimation. For tracker innovation `ν` and innovation covariance
`S`, NIS is `νᵀ S⁻¹ ν` with 3 degrees of freedom. Accepted-only ANIS uses
updates that are available, valid, and accepted by the gate. Pre-gate NIS and
rejected-update counts are reported separately. Ensemble confidence bounds
use the declared run count and chi-squared degrees of freedom; adjacent time
samples within one run are correlated and are not independent trials.

For an inertia perturbation `δI = I_truth - I_model`, the local torque-equivalent
model discrepancy used in the analysis is

```text
δτ = δI ω_dot + ω × (δI ω).
```

It is linearized around the saved truth trajectory. The approximation is
expected to degrade with large attitude/rate errors, saturation, or
time-varying mismatch statistics.

During a tracker outage, a scalar small-error reference for one attitude axis
is

```text
var(δθ(t)) ≈ var(δθ_0) + sigma_v² t
             + var(δb_0) t² + (sigma_u² / 3) t³.
```

It is a local approximation, not a recovery-time guarantee.

## Experiment and evaluation policy

Prediction tolerances, fixed evaluation windows, and experiment definitions
are registered in [`predictions.md`](predictions.md). Unless a case explicitly
requires a longer window, steady-state comparisons use 60–100 seconds and
omit the first ten tracker updates for innovation consistency. Outage,
large-angle, and saturation cases retain their full transient traces and are
not pooled with local steady-state results.
