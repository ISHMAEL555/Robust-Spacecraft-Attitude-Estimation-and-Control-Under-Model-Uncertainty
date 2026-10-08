"""
Conventions and derivations for the spacecraft attitude estimation project.
"""

# Quaternion Conventions
# ======================
# 
# This document defines the quaternion conventions used throughout the project.
# 
# Representation:
# - Quaternion: q = [x, y, z, w] (scalar-last, JPL convention)
# - Scalar part: w = q[3]
# - Vector part: v = q[:3]
# 
# Rotation:
# - q maps vectors from inertial frame I to body frame B
# - v_B = q * v_I * q^*  (quaternion multiplication)
# - Equivalently: v_B = R(q) @ v_I where R(q) is the rotation matrix
# 
# Composition:
# - q_total = q2 ⊗ q1 means apply q1 first, then q2
# - For frame transformations: q_IB = q_IC ⊗ q_CB
# 
# Error Quaternion:
# - Multiplicative error: q_true = δq ⊗ q_est
# - Body-frame error: δq represents rotation from estimated to true body frame
# - For small errors: δq ≈ [½ δθ, 1]^T where δθ is the 3-vector error
# 
# Frames:
# - I: Inertial frame (ECI, J2000, etc.)
# - B: Body frame (spacecraft body-fixed)
# - q_I_B: Quaternion from I to B
# 
# Angular Velocity:
# - ω: Body angular velocity expressed in body frame
# - Kinematics: q̇ = ½ Ω(ω) q
# - Ω(ω) = [  0  -ωx -ωy -ωz
#             ωx   0  ωz -ωy
#             ωy -ωz   0  ωx
#             ωz  ωy -ωx  0 ]
# 
# Error State:
# - δx = [δθ, δb_g]^T ∈ ℝ⁶
# - δθ: Attitude error vector (rad)
# - δb_g: Gyro bias error (rad/s)
# 
# MEKF Variants:
# ==============
# 
# Variant A (Gyro-driven):
# - Propagation uses measured ω_m - b̂_g
# - Filter does not depend on inertia matrix I
# - Inertia mismatch affects only controller and feedforward
# 
# Variant B (Model-aided):
# - Propagation uses Euler's equations with I_model and τ_cmd
# - ω̂ is part of filter state (propagated)
# - Inertia mismatch directly affects estimator
# 
# Sensor Models:
# ==============
# 
# Gyroscope:
# ω_m = ω_true + b_g + n_g
# ḃ_g = n_b
# 
# n_g ~ N(0, σ_v² I)  - Angle Random Walk (ARW)
# n_b ~ N(0, σ_u² I)  - Bias Random Walk (BRW)
# 
# Discrete-time covariances (sample time Δt):
# R_gyro = σ_v² / Δt · I₃
# Q_bias = σ_u² Δt · I₃
#
# The MEKF Van Loan discretization takes continuous-time spectral densities,
# so its Q_c blocks are σ_v² I₃ and σ_u² I₃. The expressions above are the
# corresponding sampled/discrete covariances and must not be passed as Q_c.
# 
# Star Tracker:
# q_ST = δq_ST ⊗ q_true
# δq_ST ≈ [½ δθ_ST, 1]^T
# δθ_ST ~ N(0, R_ST)
# 
# R_ST = σ_θ² · I₃
# 
# Reaction Wheel:
# τ_rw = -ḣ_rw
# |τ_rw| ≤ τ_max
# |h_rw| ≤ h_max
# 
# Control Law:
# ============
# 
# Quaternion Feedback:
# τ_c = -K_q q_e,v - K_ω ω_e
# 
# q_e = q_des^* ⊗ q_current (error from current to desired)
# q_e,v = vector part of q_e
# ω_e = ω_current - ω_des
# 
# With Feedforward:
# τ_c = -K_q q_e,v - K_ω ω_e + ω × (I ω) + τ_dist
# 
# Consistency Metrics:
# ====================
# 
# NEES (Normalized Estimation Error Squared):
# ε_NEES = e^T P⁻¹ e
# e = [δθ, b_true - b̂]^T
# For consistent filter: N·ε̄_NEES ~ χ²_{N·n_x}, n_x = 6
# 
# NIS (Normalized Innovation Squared):
# ε_NIS = ν^T S⁻¹ ν
# ν = innovation
# S = innovation covariance
# For consistent filter: N·ε̄_NIS ~ χ²_{N·n_z}, n_z = 3
# 
# 95% Confidence Bounds:
# NEES: [χ²_{0.025, 6N}/(6N), χ²_{0.975, 6N}/(6N)]
# NIS:  [χ²_{0.025, 3N}/(3N), χ²_{0.975, 3N}/(3N)]
# 
# Outage Covariance Growth:
# =========================
# 
# During star tracker outage, attitude covariance grows as:
# σ_θ²(t) ≈ σ_θ₀² + σ_v² t + σ_b₀² t² + ⅓ σ_u² t³
# 
# Where:
# - σ_θ₀²: Initial attitude variance
# - σ_v²: Gyro ARW coefficient
# - σ_b₀²: Initial bias variance
# - σ_u²: Gyro BRW coefficient
# 
# Experiment Matrix:
# ==================
# 
# E0: Nominal - Verify implementation correctness
# E1: Inertia mismatch - ΔI magnitude and structure
# E2: Process noise mismatch - Q_model/Q_true
# E3: Measurement noise mismatch - R_model/R_true
# E4: Outlier - Injected innovation magnitude
# E5: Outage - Outage duration
# E6: Closed loop - Estimator degradation level
# E7: Monte Carlo - Statistical repeatability
# 
# File Structure:
# ===============
# 
# src/
#   dynamics/
#     quaternion.py      - Quaternion algebra
#     rigid_body.py      - Rigid body dynamics
#   sensors/
#     gyro.py           - Gyroscope model
#     star_tracker.py   - Star tracker model
#   estimation/
#     mekf.py           - MEKF implementation
#   control/
#     attitude_controller.py - Attitude controller
#   actuators/
#     reaction_wheel.py - Reaction wheel model
# 
# experiments/          - Experiment configurations
# analysis/             - Analysis and visualization
# tests/
#   unit/               - Unit tests
#   integration/        - Integration tests
# config/               - Configuration files
# scripts/              - Run scripts
# docs/                 - Documentation
# reports/              - Generated reports