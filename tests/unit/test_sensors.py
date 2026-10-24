"""
Unit tests for sensor models.
"""

import numpy as np
import pytest
from src.sensors.gyro import (
    Gyroscope, GyroParams, gyro_noise_covariance, 
    gyro_bias_process_noise, continuous_to_discrete_gyro_noise
)
from src.sensors.star_tracker import (
    StarTracker, StarTrackerParams, star_tracker_noise_covariance,
    generate_outlier_measurement, nis_threshold
)
from src.dynamics.quaternion import normalize


class TestGyroscope:
    """Test gyroscope sensor model."""
    
    def test_initialization(self):
        params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(params, seed=42)
        
        assert gyro.params == params
        assert gyro.dt == 0.01
        assert gyro.bias is not None
        assert gyro.true_bias is not None
    
    def test_initialization_with_bias(self):
        params = GyroParams()
        initial_bias = np.array([0.001, -0.002, 0.0005])
        gyro = Gyroscope(params, initial_bias=initial_bias, seed=42)
        
        assert np.allclose(gyro.bias, initial_bias)
        assert np.allclose(gyro.true_bias, initial_bias)
    
    def test_measure(self):
        params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(params, seed=42)
        
        omega_true = np.array([0.1, 0.05, 0.02])
        omega_meas = gyro.measure(omega_true)
        
        # Measurement should be omega_true + bias + noise
        # Can't check exact value due to noise, but should be close
        bias = gyro.get_true_bias()
        diff = omega_meas - omega_true - bias
        # Noise std should be sigma_v / sqrt(dt)
        expected_std = params.sigma_v / np.sqrt(params.dt)
        assert np.all(np.abs(diff) < 5 * expected_std)  # 5-sigma bound
    
    def test_bias_random_walk(self):
        params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(params, seed=42)
        
        omega_true = np.zeros(3)
        biases = []
        for _ in range(1000):
            gyro.measure(omega_true)
            biases.append(gyro.get_true_bias())
        
        biases = np.array(biases)
        # Bias should drift
        bias_diff = biases[-1] - biases[0]
        # Expected std after 1000 steps: sigma_u * sqrt(1000 * dt)
        expected_std = params.sigma_u * np.sqrt(1000 * params.dt)
        assert np.all(np.abs(bias_diff) < 5 * expected_std)
    
    def test_reset_bias(self):
        params = GyroParams()
        gyro = Gyroscope(params, seed=42)
        
        new_bias = np.array([0.01, 0.02, 0.03])
        gyro.reset_bias(new_bias)
        
        assert np.allclose(gyro.bias, new_bias)
        assert np.allclose(gyro.true_bias, new_bias)
    
    def test_gyro_noise_covariance(self):
        params = GyroParams(sigma_v=1e-4, rate=100.0)
        R = gyro_noise_covariance(params)
        
        expected = (params.sigma_v**2 / params.dt) * np.eye(3)
        assert np.allclose(R, expected)
    
    def test_gyro_bias_process_noise(self):
        params = GyroParams(sigma_u=1e-6, rate=100.0)
        Q_b = gyro_bias_process_noise(params)
        
        expected = (params.sigma_u**2 * params.dt) * np.eye(3)
        assert np.allclose(Q_b, expected)
    
    def test_continuous_to_discrete(self):
        sigma_v = 1e-4
        sigma_u = 1e-6
        dt = 0.01
        
        R, Q_b = continuous_to_discrete_gyro_noise(sigma_v, sigma_u, dt)
        
        expected_R = (sigma_v**2 / dt) * np.eye(3)
        expected_Q = (sigma_u**2 * dt) * np.eye(3)
        
        assert np.allclose(R, expected_R)
        assert np.allclose(Q_b, expected_Q)


class TestStarTracker:
    """Test star tracker sensor model."""
    
    def test_initialization(self):
        params = StarTrackerParams(sigma_theta=1e-3, rate=1.0)
        tracker = StarTracker(params, seed=42)
        
        assert tracker.params == params
        assert tracker.dt == 1.0
        assert not tracker.in_outage
    
    def test_measure_nominal(self):
        params = StarTrackerParams(sigma_theta=1e-3, rate=1.0)
        tracker = StarTracker(params, seed=42)
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        
        q_meas, valid = tracker.measure(q_true, 1.0)
        
        assert q_meas is not None
        assert valid
        assert np.abs(np.linalg.norm(q_meas) - 1.0) < 1e-10
    
    def test_measure_rate(self):
        params = StarTrackerParams(sigma_theta=1e-3, rate=1.0)
        tracker = StarTracker(params, seed=42)
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        
        # First call at t=0 should return measurement
        q_meas1, valid1 = tracker.measure(q_true, 1.0)
        assert q_meas1 is not None
        assert valid1
        
        # Second call at t=0.5 should not return measurement (rate=1Hz)
        q_meas2, valid2 = tracker.measure(q_true, 0.5)
        assert q_meas2 is None
        assert not valid2
        
        # Third call at t=1.0 should return measurement
        q_meas3, valid3 = tracker.measure(q_true, 0.5)
        assert q_meas3 is not None
        assert valid3

    def test_noninteger_sample_period_does_not_drift(self):
        tracker = StarTracker(StarTrackerParams(sigma_theta=1e-3, rate=3.0), seed=7)
        q_true = np.array([0.0, 0.0, 0.0, 1.0])
        availability = [
            tracker.measure(q_true, 0.01)[0] is not None
            for _ in range(100)
        ]

        # Three 3 Hz samples fit in one second; resetting the fractional
        # scheduler phase would yield only two.
        assert sum(availability) == 3
        assert 0.0 <= tracker.time_since_update < tracker.dt
    
    def test_outlier(self):
        params = StarTrackerParams(sigma_theta=1e-3, rate=1.0, outlier_prob=1.0, outlier_magnitude=1.0)
        tracker = StarTracker(params, seed=42)
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        
        q_meas, valid = tracker.measure(q_true, 1.0)
        
        assert q_meas is not None
        assert not valid  # Should be marked as invalid (outlier)
        
        # Error should be large
        from src.dynamics.quaternion import error_quat, error_quat_to_vec
        delta_q = error_quat(q_true, q_meas)
        delta_theta = error_quat_to_vec(delta_q)
        assert np.linalg.norm(delta_theta) > 0.5  # Large outlier
    
    def test_outage(self):
        params = StarTrackerParams(sigma_theta=1e-3, rate=1.0, outage_prob=1.0, outage_duration=5)
        tracker = StarTracker(params, seed=42)
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        
        # First measurement should trigger outage
        q_meas1, valid1 = tracker.measure(q_true, 1.0)
        assert q_meas1 is None
        assert not valid1
        tracker.params.outage_prob = 0.0
        
        # Next 4 measurements should also be None (total 5 outage measurements)
        for _ in range(4):
            q_meas, valid = tracker.measure(q_true, 1.0)
            assert q_meas is None
            assert not valid
        
        # 6th measurement should be available again
        q_meas6, valid6 = tracker.measure(q_true, 1.0)
        assert q_meas6 is not None
        assert valid6
    
    def test_star_tracker_noise_covariance(self):
        params = StarTrackerParams(sigma_theta=1e-3)
        R = star_tracker_noise_covariance(params)
        
        expected = (params.sigma_theta**2) * np.eye(3)
        assert np.allclose(R, expected)
    
    def test_generate_outlier_measurement(self):
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        
        q_outlier = generate_outlier_measurement(q_true, 1.0)
        
        assert np.abs(np.linalg.norm(q_outlier) - 1.0) < 1e-10
        
        from src.dynamics.quaternion import error_quat, error_quat_to_vec
        delta_q = error_quat(q_true, q_outlier)
        delta_theta = error_quat_to_vec(delta_q)
        assert np.linalg.norm(delta_theta) > 0.5
    
    def test_nis_threshold(self):
        lower, upper = nis_threshold(3, 0.95)
        
        # For dof=3, 95% bounds
        assert lower > 0
        assert upper > lower
        # Approximate values: lower ~ 0.216, upper ~ 9.348
        assert 0.2 < lower < 0.3
        assert 9.0 < upper < 9.5


class TestRandomStreams:
    """Sensors own private generators and never touch NumPy's global RNG."""

    def test_sensor_construction_does_not_reseed_global_rng(self):
        np.random.seed(123)
        expected = np.random.random_sample(3)
        np.random.seed(123)
        Gyroscope(GyroParams(), seed=1)
        StarTracker(StarTrackerParams(), seed=2)
        assert np.array_equal(np.random.random_sample(3), expected)

    def test_tracker_stream_is_independent_of_gyro_stream(self):
        params = StarTrackerParams(rate=1.0)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        reference = StarTracker(params, seed=5).measure(q, 1.0)[0]

        tracker = StarTracker(params, seed=5)
        gyro = Gyroscope(GyroParams(), seed=99)
        for _ in range(10):
            gyro.measure(np.zeros(3))
        assert np.array_equal(tracker.measure(q, 1.0)[0], reference)

    def test_explicit_generator_is_used(self):
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        gyro_a = Gyroscope(GyroParams(), initial_bias=np.zeros(3), rng=rng_a)
        gyro_b = Gyroscope(GyroParams(), initial_bias=np.zeros(3), rng=rng_b)
        assert np.array_equal(gyro_a.measure(np.zeros(3)), gyro_b.measure(np.zeros(3)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
