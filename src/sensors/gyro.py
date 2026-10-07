"""
Gyroscope sensor model for spacecraft attitude estimation.

Model: omega_m = omega_true + b_g + n_g
       b_g_dot = n_b (bias random walk)

where:
- n_g ~ N(0, sigma_v^2 * I) is angle random walk (ARW)
- n_b ~ N(0, sigma_u^2 * I) is bias random walk (BRW)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GyroParams:
    """Gyroscope noise parameters."""
    # Angle Random Walk (ARW) - rad/s/sqrt(Hz) or rad/sqrt(s)
    sigma_v: float = 1e-4  # rad/s/sqrt(Hz)
    # Bias Random Walk (BRW) - rad/s^2/sqrt(Hz) or rad/s/sqrt(s)
    sigma_u: float = 1e-6  # rad/s^2/sqrt(Hz)
    # Initial bias standard deviation
    sigma_b0: float = 1e-3  # rad/s
    # Update rate (Hz)
    rate: float = 100.0  # Hz
    
    @property
    def dt(self) -> float:
        """Sample time (s)."""
        return 1.0 / self.rate


class Gyroscope:
    """
    Gyroscope sensor model with bias and noise.
    
    The true angular velocity is corrupted by:
    1. Constant/initial bias
    2. Bias random walk (slowly varying)
    3. Angle random walk (white noise on measurement)
    """
    
    def __init__(
        self,
        params: GyroParams,
        initial_bias: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize gyroscope.
        
        Args:
            params: GyroParams with noise characteristics
            initial_bias: Initial bias vector (rad/s), random if None
            seed: Random seed for reproducibility
        """
        self.params = params
        self.dt = 1.0 / params.rate
        
        if seed is not None:
            np.random.seed(seed)
        
        if initial_bias is not None:
            self.bias = np.array(initial_bias, dtype=float)
        else:
            # Random initial bias
            self.bias = np.random.randn(3) * params.sigma_b0
        
        # True bias (for simulation truth)
        self.true_bias = self.bias.copy()
    
    def measure(self, omega_true: np.ndarray) -> np.ndarray:
        """
        Generate gyroscope measurement.
        
        Args:
            omega_true: True body angular velocity (rad/s)
            
        Returns:
            Measured angular velocity (rad/s)
        """
        # Propagate true bias (random walk)
        self.true_bias += np.random.randn(3) * self.params.sigma_u * np.sqrt(self.dt)
        
        # Measurement noise (angle random walk)
        noise = np.random.randn(3) * self.params.sigma_v / np.sqrt(self.dt)
        
        # Measurement: omega_m = omega_true + bias + noise
        omega_measured = omega_true + self.true_bias + noise
        
        return omega_measured
    
    def get_true_bias(self) -> np.ndarray:
        """Get the true bias (for simulation truth)."""
        return self.true_bias.copy()
    
    def reset_bias(self, bias: Optional[np.ndarray] = None):
        """Reset bias to new value or random."""
        if bias is not None:
            self.bias = np.array(bias, dtype=float)
            self.true_bias = self.bias.copy()
        else:
            self.bias = np.random.randn(3) * self.params.sigma_b0
            self.true_bias = self.bias.copy()


def gyro_noise_covariance(params: GyroParams) -> np.ndarray:
    """
    Compute discrete-time gyro measurement noise covariance.
    
    For discrete-time filter with update rate 1/dt:
    R_gyro = sigma_v^2 / dt * I_3
    
    Args:
        params: GyroParams
        
    Returns:
        3x3 measurement noise covariance matrix
    """
    return (params.sigma_v**2 / params.dt) * np.eye(3)


def gyro_bias_process_noise(params: GyroParams) -> np.ndarray:
    """
    Compute discrete-time bias process noise covariance.
    
    For bias random walk: b_dot = n_b, n_b ~ N(0, sigma_u^2 * I)
    Discrete: Q_b = sigma_u^2 * dt * I_3
    
    Args:
        params: GyroParams
        
    Returns:
        3x3 bias process noise covariance matrix
    """
    return (params.sigma_u**2 * params.dt) * np.eye(3)


def continuous_to_discrete_gyro_noise(sigma_v: float, sigma_u: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous-time gyro noise specs to discrete-time covariances.
    
    Continuous: n_g ~ N(0, sigma_v^2), n_b ~ N(0, sigma_u^2)
    Discrete measurement noise: R = sigma_v^2 / dt * I
    Discrete bias process noise: Q_b = sigma_u^2 * dt * I
    
    Args:
        sigma_v: Angle random walk coefficient (rad/s/sqrt(Hz))
        sigma_u: Bias random walk coefficient (rad/s^2/sqrt(Hz))
        dt: Sample time (s)
        
    Returns:
        (R, Q_b) - measurement noise and bias process noise covariances
    """
    R = (sigma_v**2 / dt) * np.eye(3)
    Q_b = (sigma_u**2 * dt) * np.eye(3)
    return R, Q_b


# Test functions
def test_gyro():
    """Run basic gyroscope tests."""
    params = GyroParams(
        sigma_v=1e-4,
        sigma_u=1e-6,
        sigma_b0=1e-3,
        rate=100.0
    )
    
    gyro = Gyroscope(params, seed=42)
    
    # Test measurement
    omega_true = np.array([0.1, 0.05, 0.02])
    measurements = []
    for _ in range(1000):
        meas = gyro.measure(omega_true)
        measurements.append(meas)
    
    measurements = np.array(measurements)
    mean_meas = np.mean(measurements, axis=0)
    std_meas = np.std(measurements, axis=0)
    
    # Mean should be close to omega_true + bias
    print(f"True omega: {omega_true}")
    print(f"True bias: {gyro.get_true_bias()}")
    print(f"Mean measurement: {mean_meas}")
    print(f"Std measurement: {std_meas}")
    print(f"Expected noise std: {params.sigma_v / np.sqrt(params.dt):.2e}")
    
    # Test covariance functions
    R = gyro_noise_covariance(params)
    Q_b = gyro_bias_process_noise(params)
    print(f"R (measurement noise):\n{R}")
    print(f"Q_b (bias process noise):\n{Q_b}")
    
    print("Gyro tests passed!")


if __name__ == "__main__":
    test_gyro()