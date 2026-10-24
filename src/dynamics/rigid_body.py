"""
Rigid body dynamics for spacecraft attitude simulation.

Truth model: nonlinear rigid-body dynamics of a spacecraft bus carrying
reaction wheels, with the wheel angular momentum included in the gyroscopic
term:

    J omega_dot + omega x (J omega + h_w) = tau_rw + tau_d
    q_dot = 0.5 q ⊗ [omega_body, 0]

J is the bus inertia (including the wheels' transverse inertia), h_w is the
total wheel angular momentum relative to the body expressed in body axes, and
tau_rw = -dh_w/dt is the reaction torque applied to the bus by the wheel motors.
"""

import numpy as np
from typing import Tuple, Optional
from src.dynamics.quaternion import (
    normalize, quat_derivative, quat_to_rotmat
)


class RigidBody:
    """
    Nonlinear rigid body dynamics for spacecraft attitude.

    State: [q (4), omega (3)] - body-to-inertial quaternion and body angular velocity
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
            initial_q: Initial body-to-inertial quaternion [x, y, z, w]
            initial_omega: Initial body angular velocity (rad/s)
            disturbance_torque: Constant disturbance torque (N*m) in body frame
        """
        self.I = np.array(inertia, dtype=float)
        self.q = normalize(np.array(initial_q, dtype=float))
        self.omega = np.array(initial_omega, dtype=float)
        self.tau_d = np.zeros(3) if disturbance_torque is None else np.array(disturbance_torque, dtype=float)

        if self.I.shape != (3, 3):
            raise ValueError("Inertia must be 3x3")
        if not np.allclose(self.I, self.I.T):
            raise ValueError("Inertia must be symmetric")
        if not np.all(np.linalg.eigvalsh(self.I) > 0):
            raise ValueError("Inertia must be positive definite")
        self.I_inv = np.linalg.inv(self.I)

    def dynamics(
        self,
        tau_rw: np.ndarray,
        wheel_momentum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state derivatives at the current state.

        Args:
            tau_rw: Reaction wheel torque on the bus, body frame (N*m)
            wheel_momentum: Total wheel momentum in body axes (N*m*s); zero if None

        Returns:
            q_dot: Quaternion derivative
            omega_dot: Angular acceleration
        """
        h_w = np.zeros(3) if wheel_momentum is None else wheel_momentum
        return self._derivatives(self.q, self.omega, tau_rw, h_w)

    def step(
        self,
        tau_rw: np.ndarray,
        dt: float,
        wheel_momentum: Optional[np.ndarray] = None,
        wheel_momentum_rate: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate dynamics over one step using RK4.

        The bus torque is held constant over the step (zeroth-order hold). The
        wheel momentum evolves linearly over the step as
        h_w(s) = wheel_momentum + wheel_momentum_rate * s, which is exact for a
        held wheel torque.

        Args:
            tau_rw: Reaction wheel torque on the bus, body frame (N*m)
            dt: Time step (s)
            wheel_momentum: Wheel momentum in body axes at the start of the step
                (N*m*s). None neglects wheel gyroscopic coupling.
            wheel_momentum_rate: Body-axis wheel momentum rate over the step
                (N*m). Defaults to -tau_rw when wheel_momentum is given.

        Returns:
            Updated quaternion and angular velocity
        """
        tau_rw = np.asarray(tau_rw, dtype=float)
        if wheel_momentum is None:
            h0 = np.zeros(3)
            h_dot = np.zeros(3)
        else:
            h0 = np.asarray(wheel_momentum, dtype=float)
            h_dot = (
                -tau_rw if wheel_momentum_rate is None
                else np.asarray(wheel_momentum_rate, dtype=float)
            )

        def f(state, s):
            q_dot, omega_dot = self._derivatives(
                state[:4], state[4:], tau_rw, h0 + h_dot * s
            )
            return np.concatenate([q_dot, omega_dot])

        state = np.concatenate([self.q, self.omega])

        k1 = f(state, 0.0)
        k2 = f(state + 0.5 * dt * k1, 0.5 * dt)
        k3 = f(state + 0.5 * dt * k2, 0.5 * dt)
        k4 = f(state + dt * k3, dt)

        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.q = normalize(state_new[:4])
        self.omega = state_new[4:]

        return self.q.copy(), self.omega.copy()

    def _derivatives(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        tau_rw: np.ndarray,
        wheel_momentum: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute derivatives for a given state and wheel momentum."""
        gyroscopic = np.cross(omega, self.I @ omega + wheel_momentum)
        omega_dot = self.I_inv @ (tau_rw + self.tau_d - gyroscopic)
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
    
    def angular_momentum(self, wheel_momentum: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute system angular momentum (bus plus wheels) in the body frame."""
        h_w = np.zeros(3) if wheel_momentum is None else np.asarray(wheel_momentum)
        return self.I @ self.omega + h_w

    def angular_momentum_inertial(self, wheel_momentum: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute system angular momentum in the inertial frame."""
        R = quat_to_rotmat(self.q)
        return R @ self.angular_momentum(wheel_momentum)


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


def random_inertia_uncertainty(
    I_nominal: np.ndarray,
    max_rel_error: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate random inertia uncertainty.
    
    Args:
        I_nominal: Nominal inertia matrix
        max_rel_error: Maximum relative error (e.g., 0.1 for 10%)
        rng: Random generator; a fresh unseeded generator is used if None
        
    Returns:
        Uncertainty matrix delta_I
    """
    if max_rel_error < 0:
        raise ValueError("max_rel_error must be non-negative")

    # Generate a symmetric uncertainty matrix bounded by max_rel_error.
    rng = np.random.default_rng() if rng is None else rng
    delta = rng.uniform(-max_rel_error, max_rel_error, size=(3, 3))
    delta = (delta + delta.T) / 2  # Make symmetric
    # Scale by nominal inertia magnitudes
    scale = np.sqrt(np.diag(I_nominal))
    delta = delta * scale[:, None] * scale[None, :]
    return delta
