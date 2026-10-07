"""
Rigid body dynamics for spacecraft attitude simulation.

Truth model: nonlinear rigid body dynamics with reaction wheel torque and disturbances.
I * omega_dot + omega x (I * omega) = tau_rw + tau_d
q_dot = 0.5 * Omega(omega) * q
"""

import numpy as np
from typing import Tuple, Optional
from src.dynamics.quaternion import (
    normalize, quat_derivative, quat_to_rotmat, rotate_vector
)


class RigidBody:
    """
    Nonlinear rigid body dynamics for spacecraft attitude.
    
    State: [q (4), omega (3)] - quaternion (I->B) and body angular velocity
    """
    
    def __init__(
        self,
        inertia: np.ndarray,
        initial_q: np.ndarray,
        initial_omega: np.ndarray,
        disturbance_torque: Optional[np.ndarray] = None
    ):
        """
        Initialize rigid body.
        
        Args:
            inertia: 3x3 inertia matrix (kg*m^2) in body frame
            initial_q: Initial quaternion [x, y, z, w] (I->B)
            initial_omega: Initial body angular velocity (rad/s)
            disturbance_torque: Constant disturbance torque (N*m) in body frame
        """
        self.I = np.array(inertia, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.q = normalize(np.array(initial_q, dtype=float))
        self.omega = np.array(initial_omega, dtype=float)
        self.tau_d = np.zeros(3) if disturbance_torque is None else np.array(disturbance_torque, dtype=float)
        
        # Validate inertia matrix
        assert self.I.shape == (3, 3), "Inertia must be 3x3"
        assert np.allclose(self.I, self.I.T), "Inertia must be symmetric"
        eigvals = np.linalg.eigvals(self.I)
        assert np.all(eigvals > 0), "Inertia must be positive definite"
    
    def dynamics(self, tau_rw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state derivatives.
        
        Args:
            tau_rw: Reaction wheel torque in body frame (N*m)
            
        Returns:
            q_dot: Quaternion derivative
            omega_dot: Angular acceleration
        """
        # Euler's equation: I * omega_dot + omega x (I * omega) = tau_rw + tau_d
        omega_cross_Iomega = np.cross(self.omega, self.I @ self.omega)
        omega_dot = self.I_inv @ (tau_rw + self.tau_d - omega_cross_Iomega)
        
        # Quaternion kinematics
        q_dot = quat_derivative(self.q, self.omega)
        
        return q_dot, omega_dot
    
    def step(self, tau_rw: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate dynamics using RK4.
        
        Args:
            tau_rw: Reaction wheel torque (N*m)
            dt: Time step (s)
            
        Returns:
            Updated quaternion and angular velocity
        """
        # RK4 integration
        def f(state, tau):
            q, omega = state[:4], state[4:]
            q_dot, omega_dot = self._derivatives(q, omega, tau)
            return np.concatenate([q_dot, omega_dot])
        
        state = np.concatenate([self.q, self.omega])
        
        k1 = f(state, tau_rw)
        k2 = f(state + 0.5 * dt * k1, tau_rw)
        k3 = f(state + 0.5 * dt * k2, tau_rw)
        k4 = f(state + dt * k3, tau_rw)
        
        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.q = normalize(state_new[:4])
        self.omega = state_new[4:]
        
        return self.q.copy(), self.omega.copy()
    
    def _derivatives(self, q: np.ndarray, omega: np.ndarray, tau_rw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute derivatives for given state."""
        omega_cross_Iomega = np.cross(omega, self.I @ omega)
        omega_dot = self.I_inv @ (tau_rw + self.tau_d - omega_cross_Iomega)
        q_dot = quat_derivative(q, omega)
        return q_dot, omega_dot
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state."""
        return self.q.copy(), self.omega.copy()
    
    def set_state(self, q: np.ndarray, omega: np.ndarray):
        """Set state."""
        self.q = normalize(q)
        self.omega = np.array(omega, dtype=float)
    
    def kinetic_energy(self) -> float:
        """Compute rotational kinetic energy."""
        return 0.5 * self.omega @ self.I @ self.omega
    
    def angular_momentum(self) -> np.ndarray:
        """Compute angular momentum in body frame."""
        return self.I @ self.omega
    
    def angular_momentum_inertial(self) -> np.ndarray:
        """Compute angular momentum in inertial frame."""
        R = quat_to_rotmat(self.q)
        return R.T @ (self.I @ self.omega)


def inertia_matrix_from_principal(Ixx: float, Iyy: float, Izz: float, 
                                   Ixy: float = 0.0, Ixz: float = 0.0, Iyz: float = 0.0) -> np.ndarray:
    """Create inertia matrix from principal moments and products of inertia."""
    return np.array([
        [Ixx, -Ixy, -Ixz],
        [-Ixy, Iyy, -Iyz],
        [-Ixz, -Iyz, Izz]
    ])


def add_inertia_uncertainty(I_nominal: np.ndarray, delta_I: np.ndarray) -> np.ndarray:
    """Add uncertainty to inertia matrix."""
    return I_nominal + delta_I


def random_inertia_uncertainty(I_nominal: np.ndarray, max_rel_error: float = 0.1) -> np.ndarray:
    """
    Generate random inertia uncertainty.
    
    Args:
        I_nominal: Nominal inertia matrix
        max_rel_error: Maximum relative error (e.g., 0.1 for 10%)
        
    Returns:
        Uncertainty matrix delta_I
    """
    # Generate symmetric uncertainty matrix
    delta = np.random.randn(3, 3) * max_rel_error
    delta = (delta + delta.T) / 2  # Make symmetric
    # Scale by nominal inertia magnitudes
    scale = np.sqrt(np.diag(I_nominal))
    delta = delta * scale[:, None] * scale[None, :]
    return delta


# Test functions
def test_rigid_body():
    """Run basic rigid body dynamics tests."""
    # Test torque-free motion (conservation of energy and momentum)
    I = np.diag([100.0, 80.0, 60.0])  # kg*m^2
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    omega0 = np.array([0.1, 0.05, 0.02])  # rad/s
    
    body = RigidBody(I, q0, omega0)
    
    # Simulate torque-free motion
    dt = 0.1
    steps = 1000
    energy_initial = body.kinetic_energy()
    momentum_initial = body.angular_momentum_inertial()
    
    for _ in range(steps):
        body.step(np.zeros(3), dt)
    
    energy_final = body.kinetic_energy()
    momentum_final = body.angular_momentum_inertial()
    
    # Energy should be conserved (within numerical error)
    energy_error = abs(energy_final - energy_initial) / energy_initial
    momentum_error = np.linalg.norm(momentum_final - momentum_initial) / np.linalg.norm(momentum_initial)
    
    print(f"Energy conservation error: {energy_error:.2e}")
    print(f"Momentum conservation error: {momentum_error:.2e}")
    
    assert energy_error < 1e-6, f"Energy not conserved: {energy_error}"
    assert momentum_error < 1e-6, f"Momentum not conserved: {momentum_error}"
    
    # Test with constant torque
    body2 = RigidBody(I, q0, np.zeros(3))
    tau = np.array([0.1, 0.0, 0.0])
    body2.step(tau, 1.0)
    q, omega = body2.get_state()
    
    # Should have accelerated about x-axis
    assert omega[0] > 0, "Should accelerate with applied torque"
    
    print("All rigid body tests passed!")


if __name__ == "__main__":
    test_rigid_body()