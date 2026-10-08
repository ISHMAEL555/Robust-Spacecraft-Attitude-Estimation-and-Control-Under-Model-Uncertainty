"""
Reaction wheel actuator model for spacecraft attitude control.

Simple reaction-wheel allocation model with torque and momentum limits.\n\nSign convention: wheel motor torque increases wheel momentum, while the\nspacecraft reaction torque is equal to the negative of the wheel momentum\nrate projected into the body frame.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReactionWheelParams:
    """Reaction wheel parameters."""
    # Maximum torque per wheel (N*m)
    max_torque: float = 0.1
    # Maximum momentum per wheel (N*m*s)
    max_momentum: float = 10.0
    # Wheel inertia (kg*m^2)
    wheel_inertia: float = 0.01
    # Number of wheels (typically 3 or 4)
    num_wheels: int = 3
    # Wheel axis directions (body frame) - 3xN matrix
    # Default: orthogonal axes aligned with body frame
    axes: Optional[np.ndarray] = None


class ReactionWheelAssembly:
    """
    Reaction wheel assembly model.
    
    Simple model converting commanded torque to wheel torque with saturation.
    """
    
    def __init__(self, params: ReactionWheelParams):
        """
        Initialize reaction wheel assembly.
        
        Args:
            params: ReactionWheelParams
        """
        self.params = params
        
        # Default to orthogonal axes
        if params.axes is not None:
            self.axes = params.axes
        else:
            if params.num_wheels == 3:
                self.axes = np.eye(3)
            elif params.num_wheels == 4:
                # Pyramid configuration
                beta = np.arccos(1.0 / np.sqrt(3))  # ~54.7 deg
                self.axes = np.array([
                    [np.sin(beta), 0, -np.sin(beta), 0],
                    [0, np.sin(beta), 0, -np.sin(beta)],
                    [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]
                ])
            else:
                raise ValueError("Only 3 or 4 wheels supported by default")
        
        # Wheel states
        self.momentum = np.zeros(params.num_wheels)  # N*m*s per wheel
        self.torque = np.zeros(params.num_wheels)    # N*m per wheel
    
    def compute_torque(self, tau_cmd: np.ndarray) -> np.ndarray:
        """
        Compute achievable reaction wheel torque from commanded torque.
        
        Args:
            tau_cmd: Commanded body torque (N*m) - 3-vector
            
        Returns:
            Achieved body torque (N*m) - 3-vector
        """
        # Wheel motor torque increases wheel momentum. The spacecraft reaction
        # torque is equal and opposite: tau_body = -A @ tau_wheel.
        A = self.axes
        A_pinv = np.linalg.pinv(A)
        tau_wheel_cmd = -A_pinv @ tau_cmd
        
        # Apply torque saturation
        tau_wheel_sat = np.clip(tau_wheel_cmd, -self.params.max_torque, self.params.max_torque)
        
        # Check momentum saturation (simplified: limit torque if near momentum limit)
        momentum_margin = 1.0 - np.abs(self.momentum) / self.params.max_momentum
        momentum_margin = np.clip(momentum_margin, 0.0, 1.0)
        
        # Reduce torque if momentum is saturated
        tau_wheel_final = tau_wheel_sat * momentum_margin
        
        # Project wheel momentum rate into body frame with equal-and-opposite sign.
        tau_body = -A @ tau_wheel_final
        
        # Update wheel states (simplified)
        self.torque = tau_wheel_final
        # Momentum update would be done in integration step
        
        return tau_body
    
    def update_momentum(self, dt: float):
        """Update wheel momentum based on current torque."""
        self.momentum += self.torque * dt
        
        # Enforce momentum limits
        self.momentum = np.clip(self.momentum, -self.params.max_momentum, self.params.max_momentum)
    
    def get_momentum(self) -> np.ndarray:
        """Get current wheel momentum."""
        return self.momentum.copy()
    
    def get_torque(self) -> np.ndarray:
        """Get current wheel torque."""
        return self.torque.copy()
    
    def get_body_momentum(self) -> np.ndarray:
        """Get total angular momentum in body frame."""
        return self.axes @ self.momentum
    
    def reset(self):
        """Reset wheel states."""
        self.momentum = np.zeros(self.params.num_wheels)
        self.torque = np.zeros(self.params.num_wheels)


def allocate_torque_3wheel(tau_cmd: np.ndarray, max_torque: float) -> np.ndarray:
    """
    Simple 3-wheel torque allocation (orthogonal axes).
    
    Args:
        tau_cmd: Commanded body torque (3-vector)
        max_torque: Maximum torque per wheel
        
    Returns:
        Wheel torques (3-vector)
    """
    return np.clip(tau_cmd, -max_torque, max_torque)


def allocate_torque_4wheel_pyramid(tau_cmd: np.ndarray, max_torque: float, beta: float = None) -> np.ndarray:
    """
    4-wheel pyramid configuration torque allocation.
    
    Args:
        tau_cmd: Commanded body torque (3-vector)
        max_torque: Maximum torque per wheel
        beta: Pyramid angle (default: arccos(1/sqrt(3)))
        
    Returns:
        Wheel torques (4-vector)
    """
    if beta is None:
        beta = np.arccos(1.0 / np.sqrt(3))
    
    # Allocation matrix for pyramid configuration
    A = np.array([
        [np.sin(beta), 0, -np.sin(beta), 0],
        [0, np.sin(beta), 0, -np.sin(beta)],
        [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]
    ])
    
    A_pinv = np.linalg.pinv(A)
    tau_wheel = A_pinv @ tau_cmd
    
    return np.clip(tau_wheel, -max_torque, max_torque)


# Test functions
def test_reaction_wheel():
    """Run basic reaction wheel tests."""
    params = ReactionWheelParams(
        max_torque=0.1,
        max_momentum=10.0,
        num_wheels=3
    )
    
    rwa = ReactionWheelAssembly(params)
    
    # Test torque allocation
    tau_cmd = np.array([0.05, 0.03, 0.02])
    tau_achieved = rwa.compute_torque(tau_cmd)
    print(f"Commanded torque: {tau_cmd}")
    print(f"Achieved torque: {tau_achieved}")
    
    # Test saturation
    tau_cmd_large = np.array([0.5, 0.5, 0.5])
    tau_achieved = rwa.compute_torque(tau_cmd_large)
    print(f"Large commanded torque: {tau_cmd_large}")
    print(f"Saturated achieved torque: {tau_achieved}")
    
    # Test momentum update
    rwa.update_momentum(1.0)
    print(f"Momentum after 1s: {rwa.get_momentum()}")
    print(f"Body momentum: {rwa.get_body_momentum()}")
    
    # Test 4-wheel pyramid
    params4 = ReactionWheelParams(num_wheels=4)
    rwa4 = ReactionWheelAssembly(params4)
    tau_achieved4 = rwa4.compute_torque(tau_cmd)
    print(f"4-wheel achieved torque: {tau_achieved4}")
    
    print("Reaction wheel tests passed!")


if __name__ == "__main__":
    test_reaction_wheel()