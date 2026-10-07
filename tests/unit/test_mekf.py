"""
Unit tests for MEKF.
"""

import numpy as np
import pytest
from src.estimation.mekf import MEKF, MEKFParams, create_mekf_nominal
from src.sensors.gyro import GyroParams
from src.sensors.star_tracker import StarTrackerParams
from src.dynamics.quaternion import normalize, error_quat, error_quat_to_vec, box_plus


class TestMEKF:
    """Test MEKF filter."""
    
    def test_initialization_gyro_driven(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        assert mekf.variant == 'gyro_driven'
        assert mekf.q_hat is not None
        assert mekf.b_hat is not None
        assert mekf.P.shape == (6, 6)
        assert np.allclose(mekf.q_hat, [0.0, 0.0, 0.0, 1.0])
        assert np.allclose(mekf.b_hat, 0.0)
    
    def test_initialization_model_aided(self):
        I = np.diag([100.0, 80.0, 60.0])
        mekf = create_mekf_nominal('model_aided', inertia=I)
        
        assert mekf.variant == 'model_aided'
        assert mekf.I_model is not None
        assert mekf.omega_hat is not None
        assert np.allclose(mekf.omega_hat, 0.0)
    
    def test_model_aided_requires_inertia(self):
        params = MEKFParams(variant='model_aided', inertia_model=None)
        with pytest.raises(ValueError):
            MEKF(params)
    
    def test_propagate_gyro_driven(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        omega_m = np.array([0.01, 0.005, 0.002])
        mekf.propagate(omega_m, dt=0.01)
        
        # Quaternion should have rotated
        assert not np.allclose(mekf.q_hat, [0.0, 0.0, 0.0, 1.0])
        # Bias should not change during propagation
        assert np.allclose(mekf.b_hat, 0.0)
        # Covariance should grow
        assert np.trace(mekf.P) > 6 * 1e-4  # Initial trace
    
    def test_propagate_model_aided(self):
        I = np.diag([100.0, 80.0, 60.0])
        mekf = create_mekf_nominal('model_aided', inertia=I)
        
        omega_m = np.array([0.01, 0.005, 0.002])
        tau_cmd = np.array([0.01, 0.0, 0.0])
        mekf.propagate(omega_m, tau_cmd, dt=0.01)
        
        # Both quaternion and omega_hat should change
        assert not np.allclose(mekf.q_hat, [0.0, 0.0, 0.0, 1.0])
        assert not np.allclose(mekf.omega_hat, 0.0)
    
    def test_update(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        # Set initial state close to true
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        b_true = np.array([0.001, -0.002, 0.0005])
        
        mekf.reset(q_true, b_true)
        
        # Perfect measurement
        q_meas = q_true
        innovation, S, NIS = mekf.update(q_meas)
        
        # Innovation should be near zero
        assert np.linalg.norm(innovation) < 1e-10
        # NIS should be near zero
        assert NIS < 1e-10
    
    def test_update_with_noise(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        b_true = np.array([0.001, -0.002, 0.0005])
        
        mekf.reset(q_true, b_true)
        
        # Measurement with noise
        noise = np.array([0.001, -0.002, 0.0005])
        from src.dynamics.quaternion import vec_to_error_quat, multiply
        delta_q = vec_to_error_quat(noise)
        q_meas = multiply(delta_q, q_true)
        q_meas = normalize(q_meas)
        
        innovation, S, NIS = mekf.update(q_meas)
        
        # Innovation should match noise (approximately)
        assert np.allclose(innovation, noise, atol=1e-3)
        # NIS should be reasonable
        assert 0 < NIS < 100
    
    def test_step(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        b_true = np.array([0.001, -0.002, 0.0005])
        omega_true = np.array([0.01, 0.005, 0.002])
        
        mekf.reset(q_true, b_true)
        
        # Gyro measurement
        omega_m = omega_true + b_true
        
        # Star tracker measurement
        q_meas = q_true
        
        result = mekf.step(omega_m, q_meas, dt=0.01, time=0.0)
        
        assert 'q_hat' in result
        assert 'b_hat' in result
        assert 'P' in result
        assert 'NIS' in result
    
    def test_compute_nees(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        b_true = np.array([0.001, -0.002, 0.0005])
        
        mekf.reset(q_true, b_true)
        
        # Perfect estimate
        nees = mekf.compute_nees(q_true, b_true)
        assert nees < 1e-10
        
        # With error
        delta = np.array([0.01, -0.01, 0.005])
        q_est = box_plus(q_true, -delta)
        mekf.reset(q_est, b_true + np.array([0.0001, -0.0001, 0.00005]))
        
        nees = mekf.compute_nees(q_true, b_true)
        assert nees > 0
    
    def test_get_estimated_omega(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        b_hat = np.array([0.001, -0.002, 0.0005])
        mekf.b_hat = b_hat
        
        omega_m = np.array([0.01, 0.005, 0.002])
        omega_est = mekf.get_estimated_omega(omega_m)
        
        expected = omega_m - b_hat
        assert np.allclose(omega_est, expected)
    
    def test_reset(self):
        mekf = create_mekf_nominal('gyro_driven')
        
        # Run a few steps
        for _ in range(10):
            mekf.step(np.array([0.01, 0.0, 0.0]), None, dt=0.01)
        
        # Reset
        q_new = np.array([0.1, 0.2, 0.3, 0.9])
        q_new = normalize(q_new)
        b_new = np.array([0.01, 0.02, 0.03])
        mekf.reset(q_new, b_new)
        
        assert np.allclose(mekf.q_hat, q_new)
        assert np.allclose(mekf.b_hat, b_new)
        assert len(mekf.history['q_hat']) == 0
    
    def test_covariance_symmetry(self):
        """Test that covariance remains symmetric and positive definite."""
        mekf = create_mekf_nominal('gyro_driven')
        
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        b_true = np.array([0.001, -0.002, 0.0005])
        mekf.reset(q_true, b_true)
        
        for i in range(100):
            omega_m = np.array([0.01, 0.005, 0.002]) + b_true
            q_meas = q_true if i % 10 == 0 else None
            mekf.step(omega_m, q_meas, dt=0.01)
            
            # Check symmetry
            assert np.allclose(mekf.P, mekf.P.T, atol=1e-10)
            # Check positive definite
            eigvals = np.linalg.eigvals(mekf.P)
            assert np.all(eigvals > 0)


class TestMEKFVariants:
    """Test both MEKF variants."""
    
    def test_variant_a_gyro_driven(self):
        """Variant A: Gyro-driven propagation."""
        mekf = create_mekf_nominal('gyro_driven')
        
        # Should not have omega_hat
        assert not hasattr(mekf, 'omega_hat') or mekf.omega_hat is None
        
        # Propagation should only use gyro measurement
        omega_m = np.array([0.01, 0.005, 0.002])
        mekf.propagate(omega_m, dt=0.01)
        
        # Quaternion updated with omega_m - b_hat
        assert not np.allclose(mekf.q_hat, [0.0, 0.0, 0.0, 1.0])
    
    def test_variant_b_model_aided(self):
        """Variant B: Model-aided propagation."""
        I = np.diag([100.0, 80.0, 60.0])
        mekf = create_mekf_nominal('model_aided', inertia=I)
        
        # Should have omega_hat
        assert hasattr(mekf, 'omega_hat')
        assert mekf.omega_hat is not None
        
        # Propagation should use tau_cmd
        omega_m = np.array([0.01, 0.005, 0.002])
        tau_cmd = np.array([0.01, 0.0, 0.0])
        mekf.propagate(omega_m, tau_cmd, dt=0.01)
        
        # Both quaternion and omega_hat updated
        assert not np.allclose(mekf.q_hat, [0.0, 0.0, 0.0, 1.0])
        assert not np.allclose(mekf.omega_hat, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])