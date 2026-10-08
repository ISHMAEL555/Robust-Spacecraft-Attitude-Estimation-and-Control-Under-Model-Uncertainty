# Pre-registered engineering predictions

**Status:** Predictions frozen before running any post-`v0.1-pilot` simulation,
test suite, or campaign. The baseline is Git tag [`v0.1-pilot`](../README.md)
at the end of the E1-E7 exploratory pilot. Any later change to a prediction,
model, or tolerance must be versioned and justified; do not silently tune a
prediction after observing its test result.

## Purpose and decision rule

This document states quantitative hypotheses for the next validation
campaign. It is not a report of measured results. Values below are derived
from the existing configuration and first-order models where possible;
where a prediction cannot be calculated before implementation, the formula,
assumptions, and numerical agreement tolerance are stated instead.
The validity boundary for each registered claim is summarized in
[`validity-domain.md`](validity-domain.md).

Use one common steady-state evaluation window of **60-100 s** for all short
comparison cases. At 1 Hz tracker rate, omit the first 10 tracker updates in
that window when computing ANIS. In long E0 runs, use the same 60-100 s
window as the primary comparison, and separately show the 600-1,000 s
late-steady window to expose any residual drift. Seed lists, rejection
counts, and the exact accepted/available sample masks must be retained.

For an ensemble of `N` independent runs, compute average NEES and average
available-measurement NIS at each declared evaluation epoch. Compare them
with the two-sided 95% chi-squared confidence interval for the ensemble:

$$
\left[
\frac{\chi^2_{0.025,N n}}{N},
\frac{\chi^2_{0.975,N n}}{N}
\right],
\qquad n=6\ \text{or}\ 9\ \text{for NEES},\quad n=3\ \text{for NIS}.
$$

Report both the fraction of evaluation epochs within those bounds and a
run-level confidence interval. Time epochs are correlated: the epoch
fraction is descriptive, not a count of independent statistical trials.
Gated measurements are excluded from the accepted-only ANIS statistic and
reported separately as pre-gate NIS and rejection rate.

## Prediction register

| ID | Claim made before the run | Prediction / tolerance | Experiment and evaluation |
|---|---|---|---|
| P0 | The linear gyro-driven MEKF has a well-defined matched-noise steady-state covariance, and the nonlinear implementation converges to it in the small-error regime. | Solve the sampled linear discrete algebraic Riccati equation (DARE) with the configured gyro and star-tracker noise and the actual sample/update schedule. In the steady window, ensemble-mean nonlinear covariance must agree with DARE to **≤10% relative Frobenius error** and each attitude/bias marginal variance to **≤15%**. | E0 matched, small-error truth cases; compare predicted DARE `P_ss` with empirical mean reported `P` in 60-100 s and 600-1,000 s windows. |
| P1 | A matched estimator passes ensemble consistency on both state and measurement residuals. | 95% of the predeclared steady-state ANEES and accepted-only ANIS epoch statistics should lie within the corresponding ensemble 95% χ² bounds, within **±5 percentage points** across independent seed batches; overall run-level mean NEES/NIS confidence intervals must contain their degrees of freedom. | E0, at least 100 independent runs, fixed seed list; evaluate the common 60-100 s window and the 600-1,000 s window separately. |
| P2 | Small inertia mismatch in a model-aided filter behaves like an unmodeled angular-acceleration input. The local torque-equivalent mismatch is `δτ = δI·ω̇ + ω×(δI·ω)`. | For the stated sign convention and first-order linearization, measured mismatch-error covariance must be within **20% relative Frobenius error** of the propagated prediction, and ensemble mean NEES within **20%** of `tr(P_filter⁻¹ P_error)` when the linearization residual remains small. No such agreement is predicted after large-angle, saturation, or strongly time-varying-rate effects dominate. | E1 full factorial over inertia perturbation, assumed Q scale, and tracker rate; log truth/model `I`, `ω`, `ω̇`, torque residual, `P`, state error, saturation, and update masks. |
| P3 | Increasing tracker update rate reduces attitude uncertainty until gyro/process uncertainty dominates; increasing assumed Q reduces the inflation caused by omitted torque only at the cost of conservatism in matched cases. | The analytical covariance model must predict the **direction** of each trend at every tested grid point and measured-vs-predicted NEES inflation must agree within **20%** for unsaturated local-linear cases. Grid points outside that regime are recorded as prediction-domain failures, not retuned into a pass. | E1 factorial: principal-axis inertia errors `δI/I = {-25,-10,0,+10,+25}%`; `Q_scale={0.1,1,10}`; tracker rates `{0.2,1,5} Hz`; matched gyro settings initially. |
| P4 | Adding bounded torque uncertainty to model-aided rate process noise trades model-aided accuracy against conservatism; the bound grows with inertia uncertainty and angular acceleration. | Use the conservative Frobenius inequality `||δτ|| ≤ ε||I||F (||ω̇||+||ω||²)` for relative inertia-error bound `ε`, then map its isotropic bounded-torque variance through `I⁻¹`. At the true configured ε, at least **99%** of rate-error components should lie within the 3σ rate envelope. The matched-model RMSE must not exceed **2x** its unbounded model-aided baseline. | Verify on E1 hold-out seeds after the theory-derived bound is fixed. Retain the exact-model control sweep over gyro ARW/BRW scales `{10⁻⁵,10⁻⁴,3×10⁻⁴,5×10⁻⁴,0.001,0.01,0.025,0.05,0.075,0.1,1,10,100}`; test the operational crossover at **10% inertia mismatch** over `{0.001,0.01,0.1,1,10,100}`, refining the bracketing interval if the two filters change ordering. |
| P5 | Wheel gyroscopic coupling matters only when wheel momentum and body rate make `ω×h_w` non-negligible relative to commanded/disturbance torque. | The full wheel-coupled model must match the reduced model within **5%** when `||ω×h_w||` is below 1% of available control torque; above that threshold, the state/torque difference must exceed numerical integration tolerance and agree in sign and order with the coupled equations. | Paired enabled/disabled runs with fixed seeds; sweep wheel momentum and body rate. Log wheel and body momentum and report normalized coupling ratio. |
| P6 | A single bounded disturbance produces a predictable steady pointing offset/effort increase; long tracker blinding grows attitude covariance according to gyro-noise/bias propagation and delays recovery. | For small-angle unsaturated holds, measured offset must be within **15%** of the linearized PD disturbance equilibrium. During blinding, predicted-vs-measured per-axis attitude variance must agree within **15%** until nonlinear or actuator limits invalidate the approximation. | E0/E5: one constant body disturbance torque sweep; tracker blind intervals `{60,300,900} s`, with post-return pointing and covariance recovery recorded. |
| P7 | The nominal quaternion-feedback controller should choose the shortest physical rotation independently of the quaternion double-cover representation. A near-180° maneuver may nevertheless be sensitive to sign/noise without explicit hysteresis. | For `170°` and `179°` initial errors with `q` and `-q` representations, the physical trajectory/metrics must agree within **1%** away from the 180° ambiguity; near 180°, no sustained sign chatter, uncommanded full rotation, or limit cycle may occur. If hysteresis is enabled, it must remove observed chatter without increasing settling time by more than **10%** in well-conditioned cases. | E6 initial angles `{170°,179°}`, both quaternion signs; torque limits `{0.05,0.1,0.2} N m`; hysteresis off/on; truth-state and estimated-state feedback controls. |
| P8 | Under an unsaturated PD law with bounded estimation error, pointing error remains bounded by a Lyapunov/sublevel-set prediction plus disturbance and actuator residual terms. | In the derived validity region, measured attitude-error envelope must not exceed the predicted bound by more than **10%**; report every sample outside the region separately. A bound is not claimed across torque saturation, the quaternion antipode, arbitrary disturbances, or unbounded estimation error. | Tight-pointing hold with injected bounded attitude/rate estimation error, constant disturbance, and tracker outage. Compare predicted bound and full measured time history. |
| P9 | Near-gate-edge outliers distinguish the detection threshold from gross-outlier rejection; longer tracker blinding exposes recovery dynamics missed by update-count-only short outages. | At the configured **95%** NIS gate, the matched no-outlier case should reject **≤5%** of valid updates; across seeded runs, rejection frequency should trend upward as injected NIS approaches and exceeds the threshold. Long outages are evaluated by covariance and recovery curves, not a single mean metric. | E4 includes a no-outlier baseline and controlled innovations at `{0.9,1.0,1.1}` times the NIS threshold (plus gross-outlier comparator); E5 uses `{60,300,900} s` blind intervals and records time-to-recovery. |
| P10 | A dispersed Monte Carlo produces stable run-level performance summaries once its sample count meets a declared precision target. | Start with **N=100** runs. Increase in fixed batches until the 95% CI half-width for primary run-level means is **≤10% of the mean** and bootstrap percentile-CI width is **≤20% of the percentile estimate** for the 5th, 50th, and 95th percentile, or stop at a preregistered compute cap and label unresolved metrics. | E7 jointly disperses initial attitude/rate, inertia, gyro bias, and sensor misalignment from fixed, documented distributions; use independent deterministic seeds and the common 60-100 s steady window. Report medians, 5th/95th percentiles, bootstrap CIs, and failures. |

## E0 analytical reference

For Variant A around a fixed operating rate, form the sampled error model

$$
x_{k+1} = \Phi x_k + w_k,\qquad z_k = Hx_k+v_k,
\quad x=[\delta\theta,\delta b_g]^T,
$$

using the actual 100 Hz gyro propagation, 1 Hz tracker observation schedule,
continuous gyro ARW/BRW spectral densities, and configured tracker
measurement covariance. The analytical reference is the stabilizing periodic
DARE solution over one sensor-update cycle (not a continuous-time Riccati
solution substituted for the sampled implementation). Compute this reference
independently of the production MEKF implementation.

Check:

1. Positive semidefiniteness and convergence of the independent Riccati
   covariance.
2. Per-window empirical mean covariance against the predicted periodic
   steady-state covariance using P0 tolerances in P0.
3. Ensemble ANEES/ANIS against χ² bounds using the declared independent-run
   count and sample mask in P1.
4. Small-error and nonzero-rate cases separately; any linearization
   discrepancy must be reported, not hidden by pooling.

## E1 analytical reference

Let `I_t = I_m + δI` be the truth inertia and `I_m` the estimator inertia.
For the same applied body torque and disturbance, the first-order torque
equivalent of the model discrepancy is

$$
\delta\tau = \delta I\,\dot{\omega}
             + \omega\times(\delta I\,\omega).
$$

For each saved truth trajectory, compute `δτ` from truth `ω`, `ω̇`, and `δI`,
then map it to model-aided angular acceleration using the declared model
inertia inverse. Propagate its specified bounded/dispersed uncertainty through
the linearized rate-attitude error dynamics and discrete tracker update
schedule to obtain `P_error,predicted`. The predicted mean NEES is

$$
E[\epsilon_{NEES}] \simeq
\mathrm{tr}(P_{filter}^{-1}P_{error,predicted})
+ \mu_e^T P_{filter}^{-1}\mu_e.
$$

The zero-mean covariance-only expression is used only when the residual mean
term is demonstrably negligible. For a fixed unmodeled acceleration trajectory,
include the deterministic mean-error term; do not model a bounded systematic
error as zero-mean white process noise without justification.

The result is an operating-envelope prediction, not a universal scalar
“inertia tolerance”: dependence on `δI`, `Q_scale`, tracker rate, gyro
quality, angular-rate history, and saturation is retained. Predictions stop
claiming validity when the small-error linearization fails, a control or wheel
limit activates, or the model-error statistics assumed by the propagation do
not describe the simulated residual.

## Change log

| Revision | Change |
|---|---|
| v0.1-pilot baseline | Initial prediction register authored before any post-tag tests, analytical calculations, or simulations. |
| P9 erratum | Corrected the nominal gate-rejection target from 1% to 5% to match the explicitly configured 95% NIS gate; added a no-outlier baseline before running E4. |
| P7 erratum | The first E6 run exposed a confound: the reaction-wheel torque cap did not track the controller torque sweep, the inherited initial body rate was 0.1 rad/s, and 10 N·m·s wheel capacity saturated during capture. The controlled E6 matrix now starts from rest, matches wheel and controller torque limits, uses 100 N·m·s capacity, and runs for 600 s. A 170°/0.05 N·m pilot settled in 312.6 s; the full preregistered angle/torque/sign/hysteresis matrix remains the validation set. |
| P4 erratum | The initial gyro-quality sweep found model-aided RMSE lower than gyro-driven RMSE at every tested scale, so it did not bracket the requested crossover. Added five lower-noise scales before collecting the first extension. That extension still found model-aided RMSE lower at all points down to 0.001; added four more lower-noise scales, down to 10⁻⁵. The exact-inertia control still has no crossover at the tested points, as expected for a matched model; added a distinct 10%-mismatch sweep at six gyro-noise scales to test the meaningful operational crossover. Tolerances and comparison metrics are unchanged. |
