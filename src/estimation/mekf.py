"""
Multiplicative Extended Kalman Filter (MEKF) for spacecraft attitude estimation.

Two variants:
- Variant A (gyro-driven): Uses measured rate for kinematic propagation. Filter does not depend on inertia.
- Variant B (model-aided): Propagates angular rate using Euler's equations with model inertia.

Variant A error state: [delta_theta (3), delta_bias (3)]
Variant B error state: [delta_theta (3), delta_omega (3), delta_bias (3)]
Nominal state: [q_hat (4), optional omega_hat (3), b_hat (3)]
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from src.dynamics.quaternion import (
    normalize, quat_derivative, box_plus, error_quat, error_quat_to_vec
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


class MEKF:
    """
    Multiplicative Extended Kalman Filter for attitude estimation.
    
    Variant A error state: [delta_theta, delta_bias]^T (6x1)
    Variant B error state: [delta_theta, delta_omega, delta_bias]^T (9x1)
    
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
        
        self.state_dim = 9 if self.variant == 'model_aided' else 6
        self.nees_dof = self.state_dim
        self.bias_slice = slice(6, 9) if self.variant == 'model_aided' else slice(3, 6)

        # State
        self.q_hat = np.array([0.0, 0.0, 0.0, 1.0])  # Estimated quaternion (I->B)
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
            self.I_model = params.inertia_model
            self.I_model_inv = np.linalg.inv(self.I_model)
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
        
        # Skew-symmetric matrix
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        F[:3, :3] = -omega_skew
        F[:3, 3:] = -np.eye(3)
        G[:3, :3] = np.eye(3)
        G[3:, 3:] = np.eye(3)
        
        return F, G
    
    def _compute_F_G_model_aided(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute F and G for model-aided variant (Variant B).
        
        Error state: [delta_theta, delta_omega, delta_bias].
        """
        F = np.zeros((9, 9))
        G = np.zeros((9, 9))
        
        # Attitude error dynamics
        omega_skew = np.array([
            [0, -omega_hat[2], omega_hat[1]],
            [omega_hat[2], 0, -omega_hat[0]],
            [-omega_hat[1], omega_hat[0], 0]
        ])
        
        F[:3, :3] = -omega_skew
        F[:3, 3:6] = np.eye(3)

        inertia_rate = self.I_model @ omega_hat
        inertia_rate_skew = np.array([
            [0.0, -inertia_rate[2], inertia_rate[1]],
            [inertia_rate[2], 0.0, -inertia_rate[0]],
            [-inertia_rate[1], inertia_rate[0], 0.0],
        ])
        F[3:6, 3:6] = self.I_model_inv @ (
            inertia_rate_skew - omega_skew @ self.I_model
        )
        G[6:9, 6:9] = np.eye(3)
        
        return F, G
    
    def propagate(self, omega_m: np.ndarray, tau_cmd: Optional[np.ndarray] = None, dt: Optional[float] = None):
        """
        Propagate filter state and covariance.
        
        Args:
            omega_m: Measured angular velocity (rad/s)
            tau_cmd: Applied body torque (N*m) - required for model-aided variant
            dt: Time step (s), defaults to gyro rate
        """
        if dt is None:
            dt = self.dt_gyro
        
        if self.variant == 'gyro_driven':
            self._propagate_gyro_driven(omega_m, dt)
        else:
            if tau_cmd is None:
                raise ValueError("Model-aided variant requires tau_cmd")
            self._propagate_model_aided(omega_m, tau_cmd, dt)
    
    def _propagate_gyro_driven(self, omega_m: np.ndarray, dt: float):
        """Variant A: Gyro-driven propagation."""
        # Nominal quaternion propagation
        omega_est = omega_m - self.b_hat
        q_dot = quat_derivative(self.q_hat, omega_est)
        self.q_hat = normalize(self.q_hat + q_dot * dt)
        
        # Error state transition
        F, G = self._compute_F_G(omega_est)
        Phi, Qd = self._van_loan_discretization(F, G, self.Q, dt)
        
        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Qd
        
        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        eigvals = np.linalg.eigvals(self.P)
        if np.any(eigvals < 0):
            self.P += np.eye(6) * 1e-12
    
    def _propagate_model_aided(self, omega_m: np.ndarray, tau_cmd: np.ndarray, dt: float):
        """Variant B: Model-aided propagation."""
        if not self._omega_initialized:
            self.omega_hat = omega_m - self.b_hat
            self._omega_initialized = True

        # Propagate estimated angular velocity using Euler's equations
        # I * omega_dot + omega x (I * omega) = applied body torque.
        omega_cross_Iomega = np.cross(self.omega_hat, self.I_model @ self.omega_hat)
        omega_dot = self.I_model_inv @ (tau_cmd - omega_cross_Iomega)
        self.omega_hat += omega_dot * dt
        
        # Nominal quaternion propagation with estimated omega
        q_dot = quat_derivative(self.q_hat, self.omega_hat)
        self.q_hat = normalize(self.q_hat + q_dot * dt)
        
        # Error state transition (using estimated omega)
        F, G = self._compute_F_G_model_aided(self.omega_hat)
        Phi, Qd = self._van_loan_discretization(F, G, self.Q, dt)
        
        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = (self.P + self.P.T) / 2
    
    def update(self, q_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Measurement update with star tracker attitude.
        
        Args:
            q_meas: Measured quaternion from star tracker (I->B)
            
        Returns:
            (innovation, innovation_covariance, NIS)
        """
        # Innovation: delta_q = q_meas ⊗ q_hat^*
        delta_q = error_quat(q_meas, self.q_hat)
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
        self.q_hat = box_plus(self.q_hat, delta_theta)
        if self.variant == 'model_aided':
            self.omega_hat += delta_x[3:6]
        self.b_hat += delta_bias
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2
        self._reset_attitude_error_covariance(delta_theta)
        
        return innovation, S, NIS

    def update_gyro(self, omega_m: np.ndarray) -> None:
        """Update model-aided rate and bias from the gyro measurement."""
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)
        H[:, 6:9] = np.eye(3)
        innovation = omega_m - (self.omega_hat + self.b_hat)
        R_gyro = gyro_noise_covariance(self.gyro_params)
        S = H @ self.P @ H.T + R_gyro
        K = np.linalg.solve(S, H @ self.P).T
        delta_x = K @ innovation

        self.q_hat = box_plus(self.q_hat, delta_x[:3])
        self.omega_hat += delta_x[3:6]
        self.b_hat += delta_x[6:9]

        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gyro @ K.T
        self.P = (self.P + self.P.T) / 2
        self._reset_attitude_error_covariance(delta_x[:3])

    def _reset_attitude_error_covariance(self, delta_theta: np.ndarray) -> None:
        reset = np.eye(self.state_dim)
        reset[:3, :3] -= 0.5 * np.array([
            [0.0, -delta_theta[2], delta_theta[1]],
            [delta_theta[2], 0.0, -delta_theta[0]],
            [-delta_theta[1], delta_theta[0], 0.0],
        ])
        self.P = reset @ self.P @ reset.T
        self.P = (self.P + self.P.T) / 2
    
    def step(self, omega_m: np.ndarray, q_meas: Optional[np.ndarray], 
             tau_cmd: Optional[np.ndarray] = None, dt: Optional[float] = None,
             time: float = 0.0) -> dict:
        """
        Complete filter step: propagate and optionally update.
        
        Args:
            omega_m: Gyro measurement
            q_meas: Star tracker measurement (None if not available)
            tau_cmd: Applied body torque (for model-aided)
            dt: Time step
            time: Current simulation time
            
        Returns:
            Dictionary with filter outputs
        """
        was_omega_initialized = (
            self._omega_initialized if self.variant == 'model_aided' else False
        )

        # Propagate
        self.propagate(omega_m, tau_cmd, dt)

        if self.variant == 'model_aided' and was_omega_initialized:
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
    ) -> float:
        """
        Compute Normalized Estimation Error Squared (NEES).
        
        Args:
            q_true: True quaternion
            b_true: True gyro bias
            omega_true: True angular velocity, required for the model-aided variant
            
        Returns:
            NEES value
        """
        # Error state
        delta_q = error_quat(q_true, self.q_hat)
        delta_theta = error_quat_to_vec(delta_q)
        delta_bias = b_true - self.b_hat
        if self.variant == 'model_aided':
            if omega_true is None:
                raise ValueError("Model-aided NEES requires true angular velocity")
            delta_omega = omega_true - self.omega_hat
            error = np.concatenate([delta_theta, delta_omega, delta_bias])
        else:
            error = np.concatenate([delta_theta, delta_bias])
        
        # NEES = e^T P^{-1} e
        NEES = error @ np.linalg.inv(self.P) @ error
        self.history['NEES'].append(NEES)
        return NEES
    
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
                self._initialize_model_aided_covariance()
        if self.variant == 'model_aided':
            self.omega_hat = np.zeros(3)
            self._omega_initialized = False
        self.history = {k: [] for k in self.history}


def create_mekf_nominal(variant: str = 'gyro_driven', 
                         gyro_sigma_v: float = 1e-4,
                         gyro_sigma_u: float = 1e-6,
                         st_sigma_theta: float = 10.0 * np.pi / 180.0 / 3600.0,
                         inertia: Optional[np.ndarray] = None) -> MEKF:
    """Create MEKF with nominal parameters."""
    gyro_params = GyroParams(sigma_v=gyro_sigma_v, sigma_u=gyro_sigma_u)
    st_params = StarTrackerParams(sigma_theta=st_sigma_theta)
    
    params = MEKFParams(
        variant=variant,
        gyro_params=gyro_params,
        star_tracker_params=st_params,
        inertia_model=inertia
    )
    return MEKF(params)


# Test functions
def test_mekf():
    """Run basic MEKF tests."""
    # Create nominal filter
    mekf = create_mekf_nominal('gyro_driven')
    
    # Initial true state
    q_true = np.array([0.1, 0.2, 0.3, 0.9])
    q_true = normalize(q_true)
    b_true = np.array([0.001, -0.002, 0.0005])
    omega_true = np.array([0.01, 0.005, 0.002])
    
    # Simulate for a few steps
    dt = 0.01
    for i in range(100):
        # Gyro measurement
        omega_m = omega_true + b_true + np.random.randn(3) * 1e-4
        
        # Star tracker measurement every 10 steps
        q_meas = None
        if i % 10 == 0:
            noise = np.random.randn(3) * 10.0 * np.pi / 180.0 / 3600.0
            from src.dynamics.quaternion import vec_to_error_quat, multiply
            delta_q = vec_to_error_quat(noise)
            q_meas = multiply(delta_q, q_true)
            q_meas = normalize(q_meas)
        
        # Filter step
        mekf.step(omega_m, q_meas, dt=dt, time=i*dt)
    
    # Check results
    print(f"Final q_hat: {mekf.q_hat}")
    print(f"Final b_hat: {mekf.b_hat}")
    print(f"Final P diag: {np.diag(mekf.P)}")
    
    # Compute NEES
    nees = mekf.compute_nees(q_true, b_true)
    print(f"NEES: {nees:.3f}")
    
    print("MEKF tests passed!")


if __name__ == "__main__":
    test_mekf()