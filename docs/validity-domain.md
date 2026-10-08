# Prediction validity map

This map states where the registered claims in
[`predictions.md`](predictions.md) are intended to apply and what invalidates
their assumptions. A numerical pass outside a stated domain is not evidence
that the model is generally valid.

| Prediction | Intended validity domain | Stop condition / boundary |
|---|---|---|
| P0: sampled Riccati covariance | Matched gyro/tracker noise, fixed update schedule, local attitude error, gyro-driven 6-state filter | Nonzero operating-rate coupling, large attitude error, changed update timing, or covariance below the registered tolerance |
| P1: ANEES/ANIS consistency | Independent runs from the declared ensemble, correct truth/noise models, common steady window, accepted/valid/available innovation mask | Model mismatch, dependent seeds, too few accepted innovations, or using correlated time epochs as independent trials |
| P2: inertia-error NEES prediction | First-order `δτ` model and local rate/attitude error dynamics | Large rate/attitude error, torque or wheel saturation, time-varying error statistics, or covariance mismatch above 20% |
| P3: Q/tracker-rate trends | Registered inertia/Q/tracker-rate grid and nominal gyro settings | Extrapolation outside that grid, large nonlinear effects, or a trend reversal at an explicitly tested point |
| P4: bounded model-aided process noise | Actual inertia error no larger than the configured relative bound; torque-to-rate mapping and correlation model are representative | Bound violation, persistent structured error not represented by the assumed process model, or matched-model RMSE above 2x baseline |
| P5: wheel momentum coupling | Rigid body with reported wheel momentum and `ω × h_w` term; coupling ratio computed from telemetry | Flexible dynamics, changing/unreported internal momentum, actuator-limit effects, or the low-coupling 5% comparison failing |
| P6: disturbance equilibrium and outage growth | Constant bounded torque, small angles, local PD dynamics, unsaturated actuators; gyro-noise outage model | Time-varying disturbance, large angle, saturation, bias-model mismatch, or nonlinear covariance departure above tolerance |
| P7: quaternion sign behavior | Stationary 170°/179° capture, equivalent `q`/`-q`, matched controller/wheel torque limits, 100 N·m·s wheel capacity, and a 600 s horizon | Exact 180° ambiguity, nonzero initial-rate capture, actuator saturation outside this envelope, persistent sign chatter, full unwinding, limit cycle, or hysteresis-induced settling penalty above 10% |
| P8: Lyapunov pointing bound | Linearized, unsaturated PD regulation with bounded attitude/rate estimation error and bounded disturbance | Large-angle motion, saturation, unbounded estimation error, or measured pointing above 1.1 times the derived bound |
| P9: gate-edge and long-outage behavior | Configured 95% NIS gate, known injected anomaly, logged gate decisions and scheduled outages | Incorrect innovation covariance, stochastic edge samples interpreted deterministically, or outage recovery not observed within the simulated horizon |
| P10: dispersed Monte Carlo precision | Declared distributions, independent deterministic seeds, one common evaluation window, bootstrap run-level summaries | Changed distributions, underestimated dependence, precision target unmet at the run cap, or extrapolation to unmodeled spacecraft populations |

For each executed case the evidence table records the prediction, measured
quantity, tolerance, pass/fail result, and miss explanation:
[`reports/prediction_validation/predicted_vs_measured.csv`](../reports/prediction_validation/predicted_vs_measured.csv).
An absent case is untested, not a pass.
