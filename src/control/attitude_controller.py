"""Spacecraft attitude controller with body-frame quaternion feedback.

The correction quaternion is q_e = q_current^{-1} ⊗ q_desired, so its vector
part points along the body-frame rotation that carries the current attitude
to the desired one. The rate error and commanded torque are both expressed in
the current body frame:

    quaternion_feedback:  tau = +K_q q_e,v - K_omega (omega - omega_d)
    pd:                   tau = 2 K_q q_e,v - K_omega (omega - omega_d)

For small angles q_e,v ≈ theta_e / 2, so the effective attitude stiffness is
K_q / 2 for 'quaternion_feedback' and K_q for 'pd'. The command is limited
in Euclidean norm, which preserves the commanded torque direction.
"""

import numpy as np
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from src.dynamics.quaternion import normalize, multiply, conjugate, rotate_vector


@dataclass
class ControllerParams:
    """Attitude controller parameters."""
    # Quaternion feedback gain (N*m per unit quaternion vector error)
    K_q: float = 1.0
    # Rate feedback gain (N*m*s/rad)
    K_omega: float = 10.0
    # Maximum torque command norm (N*m)
    max_torque: float = 0.1
    # Control law type
    control_type: Literal['quaternion_feedback', 'pd'] = 'quaternion_feedback'


class AttitudeController:
    """
    Quaternion feedback controller.

    The right-relative attitude correction and body-rate error are expressed in
    current-body coordinates so their feedback terms can be combined directly.
    """

    def __init__(self, params: ControllerParams):
        """
        Initialize attitude controller.

        Args:
            params: ControllerParams
        """
        if params.control_type not in ('quaternion_feedback', 'pd'):
            raise ValueError(f"Unknown control_type: {params.control_type!r}")
        if params.max_torque <= 0.0:
            raise ValueError("max_torque must be positive")
        self.params = params
        self.K_q = params.K_q
        self.K_omega = params.K_omega
        self.max_torque = params.max_torque

        # Desired state
        self.q_desired = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega_desired = np.zeros(3)

    def set_desired(self, q_desired: np.ndarray, omega_desired: Optional[np.ndarray] = None):
        """Set desired attitude and desired-body-frame angular velocity."""
        self.q_desired = normalize(q_desired)
        if omega_desired is not None:
            self.omega_desired = np.array(omega_desired, dtype=float)
        else:
            self.omega_desired = np.zeros(3)

    def compute_control(self, q_current: np.ndarray, omega_current: np.ndarray) -> np.ndarray:
        """
        Compute the saturated feedback torque command.

        Args:
            q_current: Current body-to-inertial quaternion estimate
            omega_current: Current angular velocity estimate (rad/s)

        Returns:
            Commanded torque (N*m) in body frame
        """
        return self._saturate(self._feedback(q_current, omega_current))

    def compute_control_with_feedforward(
        self,
        q_current: np.ndarray,
        omega_current: np.ndarray,
        inertia: np.ndarray,
        tau_dist: Optional[np.ndarray] = None,
        wheel_momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute feedback plus gyroscopic and disturbance feedforward.

        tau = tau_fb + omega x (J omega + h_w) - tau_dist_hat

        The feedforward cancels the gyroscopic torque of the bus and wheels and
        the estimated environmental disturbance. The sum is saturated once, so
        the feedforward is not lost when the feedback alone is at the limit.

        Args:
            q_current: Current quaternion
            omega_current: Current angular velocity
            inertia: Bus inertia matrix used for feedforward
            tau_dist: Estimated disturbance torque acting on the bus (N*m)
            wheel_momentum: Wheel momentum in body axes (N*m*s)

        Returns:
            Commanded torque
        """
        h_w = np.zeros(3) if wheel_momentum is None else np.asarray(wheel_momentum, dtype=float)
        tau_ff = np.cross(omega_current, inertia @ omega_current + h_w)
        if tau_dist is not None:
            tau_ff = tau_ff - np.asarray(tau_dist, dtype=float)
        return self._saturate(self._feedback(q_current, omega_current) + tau_ff)

    def attitude_error(self, q_current: np.ndarray) -> np.ndarray:
        """Return the shortest-path body-frame correction quaternion q_c^-1 ⊗ q_d."""
        q_error = normalize(multiply(conjugate(normalize(q_current)), self.q_desired))
        return -q_error if q_error[3] < 0 else q_error

    def _feedback(self, q_current: np.ndarray, omega_current: np.ndarray) -> np.ndarray:
        """Unsaturated feedback torque."""
        q_error = self.attitude_error(q_current)
        # Desired rate is given in the desired body frame; express it in the
        # current body frame before forming the rate error.
        omega_error = omega_current - rotate_vector(q_error, self.omega_desired)
        stiffness = self.K_q if self.params.control_type == 'quaternion_feedback' else 2.0 * self.K_q
        return quaternion_feedback_torque(q_error, omega_error, stiffness, self.K_omega)

    def _saturate(self, tau_cmd: np.ndarray) -> np.ndarray:
        """Limit the command norm while preserving direction."""
        tau_norm = np.linalg.norm(tau_cmd)
        if tau_norm > self.max_torque:
            return tau_cmd * self.max_torque / tau_norm
        return tau_cmd


def compute_control_gains(
    settling_time: float,
    damping_ratio: float,
    inertia: np.ndarray,
    control_type: Literal['quaternion_feedback', 'pd'] = 'pd',
) -> Tuple[float, float]:
    """
    Compute linear-design gains for a desired 2% settling time and damping ratio.

    Small-angle closed loop per axis: J theta_ddot + K_omega theta_dot + k theta = 0
    with effective stiffness k = K_q for 'pd' and k = K_q / 2 for
    'quaternion_feedback' (because q_e,v ≈ theta / 2).

    omega_n = 4 / (damping_ratio * settling_time)   (2% criterion)
    k       = J omega_n^2
    K_omega = 2 damping_ratio J omega_n

    The mean principal inertia is used for J; the result is a starting point,
    not a substitute for checking the actuator saturation limits.

    Args:
        settling_time: Desired settling time (s)
        damping_ratio: Desired damping ratio
        inertia: Inertia matrix (diagonal elements used)
        control_type: Control law the gains are intended for

    Returns:
        (K_q, K_omega) gains
    """
    if settling_time <= 0.0 or damping_ratio <= 0.0:
        raise ValueError("settling_time and damping_ratio must be positive")
    omega_n = 4.0 / (damping_ratio * settling_time)
    I_avg = float(np.mean(np.diag(inertia)))
    stiffness = I_avg * omega_n**2
    K_q = 2.0 * stiffness if control_type == 'quaternion_feedback' else stiffness
    K_omega = 2.0 * damping_ratio * I_avg * omega_n
    return K_q, K_omega


def quaternion_feedback_torque(q_error: np.ndarray, omega_error: np.ndarray,
                               K_q: float, K_omega: float) -> np.ndarray:
    """
    Compute quaternion feedback torque tau = K_q q_e,v - K_omega omega_e.

    Args:
        q_error: Correction quaternion q_current^-1 ⊗ q_desired (vector part used)
        omega_error: Angular velocity error omega - omega_desired (rad/s)
        K_q: Quaternion gain
        K_omega: Rate gain

    Returns:
        Torque command (unsaturated)
    """
    return K_q * q_error[:3] - K_omega * omega_error
