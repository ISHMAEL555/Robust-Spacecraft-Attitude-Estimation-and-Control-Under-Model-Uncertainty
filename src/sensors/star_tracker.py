"""
Star tracker sensor model for spacecraft attitude estimation.

Model: q_ST = q_true ⊗ delta_q_ST
       delta_q_ST ≈ [0.5 * delta_theta_ST, 1]^T
       delta_theta_ST ~ N(0, R_ST)

Supports:
- Finite update rate
- Measurement noise
- Isolated outliers
- Outages
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from src.dynamics.quaternion import (
    normalize, multiply, right_error_quat, vec_to_error_quat, error_quat_to_vec
)


@dataclass
class StarTrackerParams:
    """Star tracker parameters."""
    # Measurement noise standard deviation (rad) per axis
    sigma_theta: float = 10.0 * np.pi / 180.0 / 3600.0  # 10 arcsec -> rad
    # Update rate (Hz)
    rate: float = 1.0  # Hz
    # Outlier probability per measurement
    outlier_prob: float = 0.0
    # Outlier magnitude (rad) - typically large
    outlier_magnitude: float = 1.0  # rad
    # Outage probability per measurement
    outage_prob: float = 0.0
    # Outage duration (number of measurements)
    outage_duration: int = 10
    # Optional one-shot outage start time (seconds from simulation start)
    outage_start_time: Optional[float] = None
    
    @property
    def dt(self) -> float:
        """Sample time (s)."""
        return 1.0 / self.rate


class StarTracker:
    """
    Star tracker sensor model.
    
    Provides absolute attitude measurements with configurable noise,
    outliers, and outages.
    """
    
    def __init__(
        self,
        params: StarTrackerParams,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize star tracker.
        
        Args:
            params: StarTrackerParams with noise and degradation characteristics
            seed: Seed for this sensor's private random generator
            rng: Explicit random generator; takes precedence over seed
        """
        self.params = params
        self.dt = 1.0 / params.rate
        self.time_since_update = 0.0
        self.elapsed_time = 0.0
        self.in_outage = False
        self.outage_counter = 0
        self.scheduled_outage_started = False
        self.rng = rng if rng is not None else np.random.default_rng(seed)
    
    def measure(self, q_true: np.ndarray, dt: float) -> Tuple[Optional[np.ndarray], bool]:
        """
        Generate star tracker measurement.
        
        Args:
            q_true: True body-to-inertial quaternion
            dt: Time since last call (s)
            
        Returns:
            (q_measured, valid) where q_measured is None if no measurement (outage)
            and valid indicates if measurement is valid (not outlier/outage)
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.elapsed_time += dt
        self.time_since_update += dt
        
        # Check if it's time for an update
        if self.time_since_update + 1e-12 < self.dt:
            return None, False
        
        # Preserve fractional phase so non-integer ratios between sensor and
        # simulation rates do not accumulate scheduling drift.
        self.time_since_update -= self.dt
        if self.time_since_update < 1e-12:
            self.time_since_update = 0.0
        
        # Check for outage
        if self.in_outage:
            if self.outage_counter > 0:
                self.outage_counter -= 1
                return None, False
            else:
                self.in_outage = False

        if (
            self.params.outage_start_time is not None
            and not self.scheduled_outage_started
            and self.elapsed_time >= self.params.outage_start_time
        ):
            if self.params.outage_duration < 1:
                raise ValueError("outage_duration must be at least one measurement")
            self.scheduled_outage_started = True
            self.in_outage = True
            self.outage_counter = self.params.outage_duration - 1
            return None, False
        
        # Check for new outage
        if self.params.outage_prob > 0 and self.rng.random() < self.params.outage_prob:
            self.in_outage = True
            self.outage_counter = max(self.params.outage_duration - 1, 0)
            return None, False
        
        # Generate measurement noise
        noise = self.rng.standard_normal(3) * self.params.sigma_theta
        delta_q_noise = vec_to_error_quat(noise)
        q_meas = multiply(q_true, delta_q_noise)
        q_meas = normalize(q_meas)
        
        # Check for outlier
        if self.params.outlier_prob > 0 and self.rng.random() < self.params.outlier_prob:
            # Add large outlier
            outlier_dir = self.rng.standard_normal(3)
            outlier_dir = outlier_dir / np.linalg.norm(outlier_dir)
            outlier_noise = outlier_dir * self.params.outlier_magnitude
            delta_q_outlier = vec_to_error_quat(outlier_noise)
            q_meas = multiply(q_meas, delta_q_outlier)
            q_meas = normalize(q_meas)
            return q_meas, False  # Invalid measurement (outlier)
        
        return q_meas, True
    
    def is_available(self, dt: float) -> bool:
        """Check if measurement would be available at next call."""
        # Don't modify internal state, just check
        next_time = self.time_since_update + dt
        available = next_time >= self.dt
        return available and not self.in_outage


def star_tracker_noise_covariance(params: StarTrackerParams) -> np.ndarray:
    """
    Compute star tracker measurement noise covariance.
    
    For small angle approximation: R = sigma_theta^2 * I_3
    
    Args:
        params: StarTrackerParams
        
    Returns:
        3x3 measurement noise covariance matrix
    """
    return (params.sigma_theta**2) * np.eye(3)


def generate_outlier_measurement(
    q_true: np.ndarray,
    magnitude: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an outlier measurement with specified magnitude."""
    rng = np.random.default_rng() if rng is None else rng
    outlier_dir = rng.standard_normal(3)
    outlier_dir = outlier_dir / np.linalg.norm(outlier_dir)
    outlier_noise = outlier_dir * magnitude
    delta_q_outlier = vec_to_error_quat(outlier_noise)
    q_meas = multiply(q_true, delta_q_outlier)
    return normalize(q_meas)


def nis_threshold(dof: int = 3, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Get NIS chi-squared thresholds for consistency checking.
    
    Args:
        dof: Degrees of freedom (3 for star tracker attitude)
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        (lower_bound, upper_bound)
    """
    from scipy import stats
    alpha = 1.0 - confidence
    lower = stats.chi2.ppf(alpha / 2, dof)
    upper = stats.chi2.ppf(1 - alpha / 2, dof)
    return lower, upper
