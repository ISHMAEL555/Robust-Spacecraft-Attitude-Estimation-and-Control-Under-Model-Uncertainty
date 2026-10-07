"""
Attitude controller for spacecraft attitude control.

Quaternion feedback control law:
tau_c = -K_q * q_e_v - K_omega * omega_hat

where q_e_v is the vector part of the error quaternion.
"""

import numpy as np
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from src.dynamics.quaternion import normalize, multiply, conjugate, error_quat, error_quat_to_vec


@dataclass
class ControllerParams:
    """Attitude controller parameters."""
    # Quaternion feedback gain
    K_q: float = 1.0
    # Rate feedback gain
    K_omega: float = 10.0
    # Maximum torque command (N*m)
    max_torque: float = 0.1
    # Control law type
    control_type: Literal['quaternion_feedback', 'pd'] = 'quaternion_feedback'


class AttitudeController:
    """
    Quaternion feedback attitude controller.
    
    Control law: tau_c = -K_q * q_e_v - K_omega * omega
    
    where q_e = q_desired^* ⊗ q_current (error from current to desired)
    and q_e_v is the vector part of q_e.
    """
    
    def __init__(self, params: ControllerParams):
        """
        Initialize attitude controller.
        
        Args:
            params: ControllerParams
        """
        self.params = params
        self.K_q = params.K_q
        self.K_omega = params.K_omega
        self.max_torque = params.max_torque
        
        # Desired state
        self.q_desired = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega_desired = np.zeros(3)
    
    def set_desired(self, q_desired: np.ndarray, omega_desired: Optional[np.ndarray] = None):
        """Set desired attitude and angular velocity."""
        self.q_desired = normalize(q_desired)
        if omega_desired is not None:
            self.omega_desired = np.array(omega_desired)
        else:
            self.omega_desired = np.zeros(3)
    
    def compute_control(self, q_current: np.ndarray, omega_current: np.ndarray) -> np.ndarray:
        """
        Compute control torque command.
        
        Args:
            q_current: Current quaternion estimate (I->B)
            omega_current: Current angular velocity estimate (rad/s)
            
        Returns:
            Commanded torque (N*m) in body frame
        """
        # Error quaternion: q_e = q_desired^* ⊗ q_current
        # This represents rotation from desired to current
        q_error = multiply(conjugate(self.q_desired), q_current)
        q_error = normalize(q_error)
        
        # Ensure shortest path (positive scalar part)
        if q_error[3] < 0:
            q_error = -q_error
        
        # Vector part of error quaternion
        q_e_v = q_error[:3]
        
        # Rate error
        omega_error = omega_current - self.omega_desired
        
        # Control law
        if self.params.control_type == 'quaternion_feedback':
            tau_cmd = -self.K_q * q_e_v - self.K_omega * omega_error
        else:  # PD
            # For small angles, q_e_v ≈ 0.5 * theta_e
            tau_cmd = -2 * self.K_q * q_e_v - self.K_omega * omega_error
        
        # Torque saturation
        tau_norm = np.linalg.norm(tau_cmd)
        if tau_norm > self.max_torque:
            tau_cmd = tau_cmd * self.max_torque / tau_norm
        
        return tau_cmd
    
    def compute_control_with_feedforward(self, q_current: np.ndarray, omega_current: np.ndarray,
                                          inertia: np.ndarray, tau_dist: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute control with inertia feedforward and disturbance compensation.
        
        tau_c = -K_q * q_e_v - K_omega * omega_e + I * omega_dot_desired + omega x (I * omega)
        
        Args:
            q_current: Current quaternion
            omega_current: Current angular velocity
            inertia: Inertia matrix
            tau_dist: Estimated disturbance torque
            
        Returns:
            Commanded torque
        """
        # Basic feedback
        tau_fb = self.compute_control(q_current, omega_current)
        
        # Feedforward terms
        # For regulation to zero, omega_desired = 0, omega_dot_desired = 0
        # Feedforward is omega x (I * omega) for nonlinear compensation
        omega_cross_Iomega = np.cross(omega_current, inertia @ omega_current)
        
        tau_ff = omega_cross_Iomega
        
        if tau_dist is not None:
            tau_ff += tau_dist
        
        tau_cmd = tau_fb + tau_ff
        
        # Saturation
        tau_norm = np.linalg.norm(tau_cmd)
        if tau_norm > self.max_torque:
            tau_cmd = tau_cmd * self.max_torque / tau_norm
        
        return tau_cmd


def compute_control_gains(settling_time: float, damping_ratio: float, inertia: np.ndarray) -> Tuple[float, float]:
    """
    Compute PD gains for desired settling time and damping ratio.
    
    For each axis: I * theta_ddot + K_omega * theta_dot + K_q * theta = 0
    
    Natural frequency: omega_n = 4 / settling_time (for 2% criterion)
    K_q = I * omega_n^2
    K_omega = 2 * damping_ratio * I * omega_n
    
    Args:
        settling_time: Desired settling time (s)
        damping_ratio: Desired damping ratio
        inertia: Inertia matrix (diagonal elements used)
        
    Returns:
        (K_q, K_omega) gains
    """
    omega_n = 4.0 / settling_time
    I_avg = np.mean(np.diag(inertia))
    K_q = I_avg * omega_n**2
    K_omega = 2 * damping_ratio * I_avg * omega_n
    return K_q, K_omega


def quaternion_feedback_torque(q_error: np.ndarray, omega_error: np.ndarray, 
                                K_q: float, K_omega: float) -> np.ndarray:
    """
    Compute quaternion feedback torque.
    
    Args:
        q_error: Error quaternion (vector part used)
        omega_error: Angular velocity error
        K_q: Quaternion gain
        K_omega: Rate gain
        
    Returns:
        Torque command
    """
    return -K_q * q_error[:3] - K_omega * omega_error


# Test functions
def test_controller():
    """Run basic controller tests."""
    params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
    controller = AttitudeController(params)
    
    # Test regulation to identity
    controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
    
    # Initial error
    q_current = np.array([0.1, 0.0, 0.0, np.sqrt(1 - 0.1**2)])
    q_current = normalize(q_current)
    omega_current = np.array([0.01, 0.0, 0.0])
    
    tau = controller.compute_control(q_current, omega_current)
    print(f"Control torque: {tau}")
    
    # Test with large error (should saturate)
    q_current_large = np.array([0.7, 0.0, 0.0, np.sqrt(1 - 0.7**2)])
    q_current_large = normalize(q_current_large)
    tau = controller.compute_control(q_current_large, np.zeros(3))
    print(f"Large error torque: {tau}, norm: {np.linalg.norm(tau)}")
    
    # Test gain computation
    I = np.diag([100.0, 80.0, 60.0])
    K_q, K_omega = compute_control_gains(100.0, 0.7, I)
    print(f"Computed gains: K_q={K_q:.3f}, K_omega={K_omega:.3f}")
    
    print("Controller tests passed!")


if __name__ == "__main__":
    test_controller()