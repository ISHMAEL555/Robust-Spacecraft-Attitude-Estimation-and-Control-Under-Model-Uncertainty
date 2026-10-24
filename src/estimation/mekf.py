"""
Multiplicative Extended Kalman Filter (MEKF) for spacecraft attitude estimation.

Two variants:
- Variant A (gyro-driven): Uses measured rate for kinematic propagation. Filter does not depend on inertia.
- Variant B (model-aided): Propagates angular rate using Euler's equations with an
  estimated inertia tensor, the applied wheel torque, and the measured wheel momentum.

Variant A error state: [delta_theta (3), delta_bias (3)]
Variant B error state: [delta_theta (3), delta_omega (3), delta_bias (3), delta_inertia (6)]
Nominal state: [q_hat (4), optional omega_hat (3), b_hat (3), optional I_hat (3x3)]

All attitude errors are right-multiplicative body-frame errors,
q_true = q_hat ⊗ Exp(delta_theta); see docs/conventions.md.
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from src.dynamics.quaternion import (
    normalize, propagate_quaternion, right_box_plus, right_error_quat,
    error_quat_to_vec, skew,
)
from src.sensors.gyro import GyroParams, gyro_noise_covariance
from src.sensors.star_tracker import StarTrackerParams, star_tracker_noise_covariance


@dataclass
class MEKFParams:
    """MEKF configuration parameters."""
    # Filter variant: 'gyro_driven' (A) or 'model_aided' (B)
    variant: Literal['gyro_driven', 'model_aided'] = 'gyro_driven'
    # Gyro noise parameters
    gyro_params: GyroParams = None
    # Star tracker noise parameters
    star_tracker_params: StarTrackerParams = None
    # Initial covariance
    P0_attitude: float = 1e-4  # rad^2
    P0_bias: float = 1e-6      # (rad/s)^2
    # Process noise scaling factors (for mismatch studies)
    Q_scale: float = 1.0
    R_scale: float = 1.0
    nis_gate_confidence: float = 0.999999
    nis_gate_enabled: bool = False
    # Inertia matrix (for model-aided variant)
    inertia_model: Optional[np.ndarray] = None
    # Continuous-time unmodeled rate-process spectral density amplitude
    # ((rad/s)/sqrt(s)); represents torque/model uncertainty in model-aided use.
    sigma_omega_process: float = 0.0
    # One-sigma relative prior uncertainty for the six independent elements of
    # the symmetric inertia tensor. Used by the augmented model-aided MEKF.
    inertia_relative_sigma: float = 0.05


class MEKF:
    """
    Multiplicative Extended Kalman Filter for attitude estimation.
    
    Variant A error state: [delta_theta, delta_bias]^T (6x1)
    Variant B error state: [delta_theta, delta_omega, delta_bias, delta_inertia]^T (15x1)
    
    Propagation:
    - Variant A: q_hat propagated with measured omega_m - b_hat
    - Variant B: q_hat and omega_hat propagated with Euler's equations
    
    Update: Star tracker attitude measurement
    """
    
    def __init__(self, params: MEKFParams):
        """
        Initialize MEKF.
        
        Args:
            params: MEKFParams with filter configuration
        """
        self.params = params
        self.variant = params.variant
        if not 0.0 < params.nis_gate_confidence < 1.0:
            raise ValueError("nis_gate_confidence must be between 0 and 1")
        self.nis_threshold = chi2.ppf(params.nis_gate_confidence, 3)
        self.last_measurement_accepted = False
        
        # Default sensor params if not provided
        if params.gyro_params is None:
            params.gyro_params = GyroParams()
        if params.star_tracker_params is None:
            params.star_tracker_params = StarTrackerParams()
        
        self.gyro_params = params.gyro_params
        self.star_tracker_params = params.star_tracker_params
        self.dt_gyro = 1.0 / self.gyro_params.rate
        self.dt_st = 1.0 / self.star_tracker_params.rate
        
        if params.inertia_relative_sigma < 0.0:
            raise ValueError("inertia_relative_sigma must be non-negative")
        if params.sigma_omega_process < 0.0:
            raise ValueError("sigma_omega_process must be non-negative")
        self.state_dim = 15 if self.variant == 'model_aided' else 6
        self.nees_dof = self.state_dim
        self.bias_slice = slice(6, 9) if self.variant == 'model_aided' else slice(3, 6)
        self.inertia_slice = slice(9, 15) if self.variant == 'model_aided' else slice(0, 0)

        # State
        self.q_hat = np.array([0.0, 0.0, 0.0, 1.0])  # Estimated body-to-inertial attitude
        self.b_hat = np.zeros(3)                      # Estimated gyro bias
        self.P = np.eye(self.state_dim)               # Error covariance
        
        # Initialize covariance
        self.P[:3, :3] *= params.P0_attitude
        self.P[self.bias_slice, self.bias_slice] *= params.P0_bias
        
        # Process noise
        self.Q = self._build_process_noise()
        # Measurement noise
        self.R = self._build_measurement_noise()
        
        # For model-aided variant
        if self.variant == 'model_aided':
            if params.inertia_model is None:
                raise ValueError("Model-aided variant requires inertia_model")
            self.I_model = np.asarray(params.inertia_model, dtype=float).copy()
            self.I_model_inv = np.linalg.inv(self.I_model)
            self._inertia_basis = self._build_inertia_basis()
            self._initialize_inertia_scales()
            self._initialize_inertia_covariance()
            self.omega_hat = np.zeros(3)  # Estimated angular velocity
            self._omega_initialized = False
            self._initialize_model_aided_covariance()
        
        # History for analysis
        self.history = {
            'q_hat': [],
            'b_hat': [],
            'P': [],
            'NEES': [],
            'NIS': [],
            'innovation': [],
            'time': []
        }
    
    def _build_process_noise(self) -> np.ndarray:
        """Build continuous-time process noise matrix."""
        Q = np.zeros((self.state_dim, self.state_dim))
        # Van Loan discretization integrates continuous-time noise spectral densities.
        if self.variant == 'gyro_driven':
            Q[:3, :3] = np.eye(3) * self.gyro_params.sigma_v**2 * self.params.Q_scale
        elif self.variant == 'model_aided':
            Q[3:6, 3:6] = (
                np.eye(3) * self.params.sigma_omega_process**2 * self.params.Q_scale
            )
        Q[self.bias_slice, self.bias_slice] = (
            np.eye(3) * self.gyro_params.sigma_u**2 * self.params.Q_scale
        )
        return Q

    def _initialize_model_aided_covariance(self) -> None:
        """Set a gyro-derived prior for rate and its correlation with bias."""
        rate_slice = slice(3, 6)
        bias_covariance = self.P[self.bias_slice, self.bias_slice].copy()
        self.P[rate_slice, rate_slice] = (
            gyro_noise_covariance(self.gyro_params) + bias_covariance
        )
        self.P[rate_slice, self.bias_slice] = -bias_covariance
        self.P[self.bias_slice, rate_slice] = -bias_covariance

    @staticmethod
    def _build_inertia_basis() -> Tuple[np.ndarray, ...]:
        """Return basis tensors for [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]."""
        return (
            np.diag([1.0, 0.0, 0.0]),
            np.diag([0.0, 1.0, 0.0]),
            np.diag([0.0, 0.0, 1.0]),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )

    def _initialize_inertia_covariance(self) -> None:
        """Set an isotropic prior on dimensionless inertia-element errors."""
        relative_sigma = max(self.params.inertia_relative_sigma, 1e-8)
        self.P[self.inertia_slice, self.inertia_slice] = (
            np.eye(6) * relative_sigma**2
        )

    def _initialize_inertia_scales(self) -> None:
        """Scale the six inertia errors by nominal tensor magnitudes."""
        diagonal = np.diag(self.I_model)
        self._inertia_scales = np.array([
            diagonal[0], diagonal[1], diagonal[2],
            np.sqrt(diagonal[0] * diagonal[1]),
            np.sqrt(diagonal[0] * diagonal[2]),
            np.sqrt(diagonal[1] * diagonal[2]),
        ])
    
    def _build_measurement_noise(self) -> np.ndarray:
        """Build measurement noise covariance."""
        return star_tracker_noise_covariance(self.star_tracker_params) * self.params.R_scale
    
    def _van_loan_discretization(self, F: np.ndarray, G: np.ndarray, Qc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Van Loan method for discretizing continuous-time system.
        
        Continuous: dx/dt = F x + G w, w ~ N(0, Qc)
        Discrete: x_k+1 = Phi x_k + w_k, w_k ~ N(0, Qd)
        
        Returns: (Phi, Qd)
        """
        n = F.shape[0]
        m = G.shape[1]
        
        # Build augmented matrix
        M = np.zeros((2*n, 2*n))
        M[:n, :n] = -F
        M[:n, n:] = G @ Qc @ G.T
        M[n:, n:] = F.T
        
        # Matrix exponential
        expM = expm(M * dt)
        
        Phi = expM[n:, n:].T
        Qd = Phi @ expM[:n, n:]
        
        # Ensure symmetry
        Qd = (Qd + Qd.T) / 2
        
        return Phi, Qd
    
    def _compute_F_G(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous-time F and G matrices for error dynamics.
        
        Error state: delta_x = [delta_theta, delta_bias]
        
        For gyro-driven (Variant A):
        delta_theta_dot = -[omega_m - b_hat]_x delta_theta - delta_bias + n_g
        delta_bias_dot = n_b
        
        F = [ -[omega]_x  -I ]
            [    0        0  ]
        G = [ I  0 ]
            [ 0  I ]
        """
        F = np.zeros((6, 6))
        G = np.zeros((6, 6))
        
        F[:3, :3] = -skew(omega)
        F[:3, 3:] = -np.eye(3)
        G[:3, :3] = np.eye(3)
        G[3:, 3:] = np.eye(3)
        
        return F, G
    
    def _compute_F_G_model_aided(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute F and G for model-aided variant (Variant B).
        
        Error state: [delta_theta, delta_omega, delta_bias, delta_inertia].

        Rate dynamics: f(omega, I) = I^-1 (tau - omega x (I omega + h_w)),
        with tau the applied wheel torque and h_w the measured wheel momentum
        held over the propagation interval.
        """
        F = np.zeros((15, 15))
        G = np.zeros((15, 15))

        omega_skew = skew(omega_hat)
        F[:3, :3] = -omega_skew
        F[:3, 3:6] = np.eye(3)

        torque = getattr(self, "_propagation_torque", np.zeros(3))
        wheel_momentum = getattr(self, "_propagation_wheel_momentum", np.zeros(3))
        # df/domega = I^-1 ([I omega + h_w]x - [omega]x I)
        F[3:6, 3:6] = self.I_model_inv @ (
            skew(self.I_model @ omega_hat + wheel_momentum) - omega_skew @ self.I_model
        )
        G[3:6, 3:6] = np.eye(3)
        G[6:9, 6:9] = np.eye(3)

        # Linearize Euler's equation with respect to the six independent
        # inertia elements. If f = I^-1 (tau - omega x I omega), then
        # df/dI_j = -I^-1 E_j f - I^-1 [omega]x E_j omega.
        # The inertia error is a constant parameter state, so its process
        # noise is zero; its uncertainty enters rate covariance through F.
        omega_dot = self._rate_derivative(omega_hat, torque, wheel_momentum)
        for index, basis in enumerate(self._inertia_basis):
            F[3:6, 9 + index] = (
                -self.I_model_inv @ basis @ omega_dot
                -self.I_model_inv @ omega_skew @ basis @ omega_hat
            ) * self._inertia_scales[index]
        
        return F, G
    
    def _rate_derivative(
        self, omega: np.ndarray, torque: np.ndarray, wheel_momentum: np.ndarray
    ) -> np.ndarray:
        """Euler's equation with the current inertia estimate and wheel momentum."""
        gyroscopic = np.cross(omega, self.I_model @ omega + wheel_momentum)
        return self.I_model_inv @ (torque - gyroscopic)

    def propagate(
        self,
        omega_m: np.ndarray,
        tau_cmd: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        wheel_momentum: Optional[np.ndarray] = None,
    ):
        """
        Propagate filter state and covariance.

        Args:
            omega_m: Measured angular velocity (rad/s)
            tau_cmd: Applied body torque (N*m) - required for model-aided variant
            dt: Time step (s), defaults to gyro rate
            wheel_momentum: Measured wheel momentum in body axes (N*m*s),
                representative of the interval; used by the model-aided
                variant only. Zero if None.
        """
        if dt is None:
            dt = self.dt_gyro

        if self.variant == 'gyro_driven':
            self._propagate_gyro_driven(omega_m, dt)
        else:
            if tau_cmd is None:
                raise ValueError("Model-aided variant requires tau_cmd")
            h_w = (
                np.zeros(3) if wheel_momentum is None
                else np.asarray(wheel_momentum, dtype=float)
            )
            self._propagate_model_aided(omega_m, np.asarray(tau_cmd, dtype=float), dt, h_w)

    def _propagate_gyro_driven(self, omega_m: np.ndarray, dt: float):
        """Variant A: Gyro-driven propagation."""
        # Exact zeroth-order-hold attitude propagation about the same held
        # rate used to linearize the covariance.
        omega_est = omega_m - self.b_hat
        self.q_hat = propagate_quaternion(self.q_hat, omega_est, dt)

        F, G = self._compute_F_G(omega_est)
        Phi, Qd = self._van_loan_discretization(F, G, self.Q, dt)
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = (self.P + self.P.T) / 2

    def _propagate_model_aided(
        self,
        omega_m: np.ndarray,
        tau_cmd: np.ndarray,
        dt: float,
        wheel_momentum: np.ndarray,
    ):
        """Variant B: Model-aided propagation."""
        if not self._omega_initialized:
            self.omega_hat = omega_m - self.b_hat
            self._omega_initialized = True

        # Linearize at the start-of-interval rate, then advance the nominal
        # state with a midpoint (RK2) step of Euler's equation and an exact
        # quaternion exponential at the midpoint rate.
        self._propagation_torque = tau_cmd
        self._propagation_wheel_momentum = wheel_momentum
        F, G = self._compute_F_G_model_aided(self.omega_hat)

        omega_start = self.omega_hat.copy()
        omega_mid = omega_start + 0.5 * dt * self._rate_derivative(
            omega_start, tau_cmd, wheel_momentum
        )
        self.omega_hat = omega_start + dt * self._rate_derivative(
            omega_mid, tau_cmd, wheel_momentum
        )
        self.q_hat = propagate_quaternion(self.q_hat, omega_mid, dt)
        Phi, Qd = self._van_loan_discretization(F, G, self.Q, dt)
        
        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = (self.P + self.P.T) / 2
    
    def update(self, q_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Measurement update with star tracker attitude.
        
        Args:
            q_meas: Measured body-to-inertial quaternion from the star tracker
            
        Returns:
            (innovation, innovation_covariance, NIS)
        """
        # Right innovation for the covariance model's body-frame error:
        # q_meas = q_hat ⊗ Exp(delta_theta).
        delta_q = right_error_quat(q_meas, self.q_hat)
        innovation = error_quat_to_vec(delta_q)  # 3-vector
        
        # Star tracker observes attitude error only.
        H = np.zeros((3, self.state_dim))
        H[:3, :3] = np.eye(3)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # NIS gate protects the state and covariance from inconsistent measurements.
        NIS = innovation @ np.linalg.solve(S, innovation)
        if self.params.nis_gate_enabled and NIS > self.nis_threshold:
            self.last_measurement_accepted = False
            return innovation, S, NIS
        self.last_measurement_accepted = True

        # Kalman gain
        K = np.linalg.solve(S, H @ self.P).T
        
        # State update (error state)
        delta_x = K @ innovation
        delta_theta = delta_x[:3]
        delta_bias = delta_x[self.bias_slice]
        
        # Multiplicative quaternion update
        self.q_hat = right_box_plus(self.q_hat, delta_theta)
        if self.variant == 'model_aided':
            self.omega_hat += delta_x[3:6]
            self._inject_inertia_correction(delta_x[9:15])
        self.b_hat += delta_bias
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        self._reset_attitude_error_covariance(delta_theta)
        
        return innovation, S, NIS

    def update_gyro(self, omega_m: np.ndarray) -> None:
        """Update model-aided rate and bias from the gyro measurement."""
        H = np.zeros((3, self.state_dim))
        H[:, 3:6] = np.eye(3)
        H[:, 6:9] = np.eye(3)
        innovation = omega_m - (self.omega_hat + self.b_hat)
        R_gyro = gyro_noise_covariance(self.gyro_params)
        S = H @ self.P @ H.T + R_gyro
        K = np.linalg.solve(S, H @ self.P).T
        delta_x = K @ innovation

        self.q_hat = right_box_plus(self.q_hat, delta_x[:3])
        self.omega_hat += delta_x[3:6]
        self.b_hat += delta_x[6:9]
        self._inject_inertia_correction(delta_x[9:15])

        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gyro @ K.T
        self.P = (self.P + self.P.T) / 2
        self._reset_attitude_error_covariance(delta_x[:3])

    def _inject_inertia_correction(self, delta_parameters: np.ndarray) -> None:
        """Apply a dimensionless EKF correction while keeping inertia positive definite."""
        delta_inertia = sum(
            delta_parameters[index] * self._inertia_scales[index] * basis
            for index, basis in enumerate(self._inertia_basis)
        )
        candidate_inertia = self.I_model + delta_inertia
        if np.min(np.linalg.eigvalsh(candidate_inertia)) > 0.0:
            self.I_model = candidate_inertia
            self.I_model_inv = np.linalg.inv(self.I_model)

    def _reset_attitude_error_covariance(self, delta_theta: np.ndarray) -> None:
        reset = np.eye(self.state_dim)
        theta_sq = float(delta_theta @ delta_theta)
        theta = np.sqrt(theta_sq)
        theta_skew = skew(delta_theta)
        if theta < 1e-5:
            theta_fourth = theta_sq * theta_sq
            first = 0.5 - theta_sq / 24.0 + theta_fourth / 720.0
            second = 1.0 / 6.0 - theta_sq / 120.0 + theta_fourth / 5040.0
        else:
            first = (1.0 - np.cos(theta)) / theta_sq
            second = (theta - np.sin(theta)) / (theta_sq * theta)
        # Exact SO(3) right Jacobian for the reset map
        # Log(Exp(-delta_theta) Exp(delta_theta + error)).
        reset[:3, :3] = (
            np.eye(3) - first * theta_skew + second * (theta_skew @ theta_skew)
        )
        self.P = reset @ self.P @ reset.T
        self.P = (self.P + self.P.T) / 2
    
    def step(self, omega_m: np.ndarray, q_meas: Optional[np.ndarray], 
             tau_cmd: Optional[np.ndarray] = None, dt: Optional[float] = None,
             time: float = 0.0,
             wheel_momentum: Optional[np.ndarray] = None) -> dict:
        """
        Complete filter step: propagate and optionally update.
        
        Args:
            omega_m: Gyro measurement
            q_meas: Star tracker measurement (None if not available)
            tau_cmd: Applied body torque (for model-aided)
            dt: Time step
            time: Current simulation time
            wheel_momentum: Measured wheel momentum in body axes (model-aided)

        Returns:
            Dictionary with filter outputs
        """
        if self.variant == 'model_aided' and not self._omega_initialized:
            # The first gyro sample initializes the rate at the current epoch.
            # Do not also propagate it forward and then reuse that same sample
            # as a measurement; later calls propagate the prior epoch to this
            # epoch before applying the current gyro update.
            self.omega_hat = omega_m - self.b_hat
            self._omega_initialized = True
        else:
            self.propagate(omega_m, tau_cmd, dt, wheel_momentum)
            if self.variant == 'model_aided':
                self.update_gyro(omega_m)
        
        # Update if measurement available
        innovation = None
        S = None
        NIS = None
        NEES = None
        self.last_measurement_accepted = False
        
        if q_meas is not None:
            innovation, S, NIS = self.update(q_meas)
        
        # Compute NEES if true state available (for analysis)
        # This would be computed externally with true state
        
        # Store history
        self.history['q_hat'].append(self.q_hat.copy())
        self.history['b_hat'].append(self.b_hat.copy())
        self.history['P'].append(self.P.copy())
        self.history['time'].append(time)
        if innovation is not None:
            self.history['innovation'].append(innovation)
            self.history['NIS'].append(NIS)
        
        return {
            'q_hat': self.q_hat.copy(),
            'b_hat': self.b_hat.copy(),
            'omega_hat': self.get_estimated_omega(omega_m),
            'P': self.P.copy(),
            'innovation': innovation,
            'S': S,
            'NIS': NIS,
            'NEES': NEES,
            'measurement_accepted': self.last_measurement_accepted
        }
    
    def compute_nees(
        self,
        q_true: np.ndarray,
        b_true: np.ndarray,
        omega_true: Optional[np.ndarray] = None,
        inertia_true: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Normalized Estimation Error Squared (NEES).
        
        Args:
            q_true: True quaternion
            b_true: True gyro bias
            omega_true: True angular velocity, required for the model-aided variant
            inertia_true: True inertia tensor, optional for the model-aided variant
            
        Returns:
            NEES value
        """
        error = self._compute_error_state(
            q_true, b_true, omega_true, inertia_true
        )
        # NEES = e^T P^{-1} e
        NEES = error @ np.linalg.solve(self.P, error)
        self.history['NEES'].append(NEES)
        return NEES

    def compute_nees_components(
        self,
        q_true: np.ndarray,
        b_true: np.ndarray,
        omega_true: Optional[np.ndarray] = None,
        inertia_true: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Return marginal NEES by state group; these do not sum to full NEES."""
        error = self._compute_error_state(
            q_true, b_true, omega_true, inertia_true
        )
        groups = {'attitude': slice(0, 3)}
        if self.variant == 'model_aided':
            groups.update({
                'rate': slice(3, 6),
                'bias': slice(6, 9),
                'inertia': slice(9, 15),
            })
        else:
            groups['bias'] = slice(3, 6)
        return {
            name: float(error[group] @ np.linalg.solve(
                self.P[group, group], error[group]
            ))
            for name, group in groups.items()
        }

    def _compute_error_state(
        self,
        q_true: np.ndarray,
        b_true: np.ndarray,
        omega_true: Optional[np.ndarray],
        inertia_true: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build the true-minus-estimated error in the filter's tangent frame."""
        delta_q = right_error_quat(q_true, self.q_hat)
        delta_theta = error_quat_to_vec(delta_q)
        delta_bias = b_true - self.b_hat
        if self.variant == 'model_aided':
            if omega_true is None:
                raise ValueError("Model-aided NEES requires true angular velocity")
            delta_omega = omega_true - self.omega_hat
            if inertia_true is None:
                inertia_true = self.I_model
            delta_inertia = np.asarray(inertia_true) - self.I_model
            inertia_error = np.array([
                delta_inertia[0, 0], delta_inertia[1, 1], delta_inertia[2, 2],
                delta_inertia[0, 1], delta_inertia[0, 2], delta_inertia[1, 2],
            ]) / self._inertia_scales
            error = np.concatenate([
                delta_theta, delta_omega, delta_bias, inertia_error
            ])
        else:
            error = np.concatenate([delta_theta, delta_bias])
        return error
    
    def get_estimated_omega(self, omega_m: np.ndarray) -> np.ndarray:
        """Get bias-corrected angular velocity estimate."""
        if self.variant == 'gyro_driven':
            return omega_m - self.b_hat
        else:
            return self.omega_hat.copy()
    
    def reset(self, q_hat: np.ndarray, b_hat: np.ndarray, P: Optional[np.ndarray] = None):
        """Reset filter state."""
        self.q_hat = normalize(q_hat)
        self.b_hat = np.array(b_hat)
        if P is not None:
            covariance = np.asarray(P, dtype=float)
            if covariance.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"Covariance must have shape {(self.state_dim, self.state_dim)}"
                )
            self.P = covariance.copy()
        else:
            self.P = np.eye(self.state_dim)
            self.P[:3, :3] *= self.params.P0_attitude
            self.P[self.bias_slice, self.bias_slice] *= self.params.P0_bias
            if self.variant == 'model_aided':
                self.I_model = np.asarray(self.params.inertia_model, dtype=float).copy()
                self.I_model_inv = np.linalg.inv(self.I_model)
                self._initialize_inertia_scales()
                self._initialize_model_aided_covariance()
                self._initialize_inertia_covariance()
        if self.variant == 'model_aided':
            self.omega_hat = np.zeros(3)
            self._omega_initialized = False
        self.history = {k: [] for k in self.history}


def create_mekf_nominal(variant: str = 'gyro_driven', 
                         gyro_sigma_v: float = 1e-4,
                         gyro_sigma_u: float = 1e-6,
                         st_sigma_theta: float = 10.0 * np.pi / 180.0 / 3600.0,
                         inertia: Optional[np.ndarray] = None,
                         inertia_relative_sigma: float = 0.05,
                         sigma_omega_process: float = 0.0) -> MEKF:
    """Create MEKF with nominal parameters."""
    gyro_params = GyroParams(sigma_v=gyro_sigma_v, sigma_u=gyro_sigma_u)
    st_params = StarTrackerParams(sigma_theta=st_sigma_theta)
    
    params = MEKFParams(
        variant=variant,
        gyro_params=gyro_params,
        star_tracker_params=st_params,
        inertia_model=inertia,
        inertia_relative_sigma=inertia_relative_sigma,
        sigma_omega_process=sigma_omega_process,
    )
    return MEKF(params)
