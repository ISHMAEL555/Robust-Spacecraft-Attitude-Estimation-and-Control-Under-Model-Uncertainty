"""
Unit tests for quaternion algebra.
"""

import numpy as np
import pytest
from src.dynamics.quaternion import (
    normalize, is_unit, conjugate, inverse, multiply,
    rotate_vector, quat_to_rotmat, rotmat_to_quat,
    error_quat, error_quat_to_vec, vec_to_error_quat,
    quat_derivative, box_plus, box_minus, slerp
)


class TestQuaternionBasics:
    """Test basic quaternion operations."""
    
    def test_normalize(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = normalize(q)
        assert is_unit(q_norm)
        assert np.allclose(q_norm, q / np.linalg.norm(q))
    
    def test_normalize_zero_raises(self):
        q = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            normalize(q)
    
    def test_is_unit(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        assert is_unit(q)
        
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert is_unit(q)
        
        q = np.array([1.0, 0.0, 0.0, 0.0])
        assert not is_unit(q)
    
    def test_conjugate(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        q_conj = conjugate(q)
        expected = np.array([-0.1, -0.2, -0.3, 0.9])
        expected = normalize(expected)
        assert np.allclose(q_conj, expected)
    
    def test_inverse(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        q_inv = inverse(q)
        # For unit quaternions, inverse = conjugate
        assert np.allclose(q_inv, conjugate(q))
        
        # q * q^-1 = identity
        q_mult = multiply(q, q_inv)
        assert np.allclose(q_mult, [0.0, 0.0, 0.0, 1.0], atol=1e-10)
    
    def test_multiply_identity(self):
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        
        assert np.allclose(multiply(q_id, q), q)
        assert np.allclose(multiply(q, q_id), q)
    
    def test_multiply_inverse(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        q_inv = inverse(q)
        result = multiply(q, q_inv)
        assert np.allclose(result, [0.0, 0.0, 0.0, 1.0], atol=1e-10)


class TestQuaternionRotation:
    """Test quaternion rotation operations."""
    
    def test_rotate_vector_90_deg_z(self):
        # 90 degree rotation about z-axis
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        v = np.array([1.0, 0.0, 0.0])
        v_rot = rotate_vector(q, v)
        assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-10)
    
    def test_rotate_vector_180_deg_x(self):
        # 180 degree rotation about x-axis
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        v_rot = rotate_vector(q, v)
        assert np.allclose(v_rot, [0.0, -1.0, 0.0], atol=1e-10)
    
    def test_quat_to_rotmat_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        R = quat_to_rotmat(q)
        assert np.allclose(R, np.eye(3))
    
    def test_quat_to_rotmat_90_deg_z(self):
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        R = quat_to_rotmat(q)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        assert np.allclose(R, expected, atol=1e-10)
    
    def test_rotmat_to_quat_roundtrip(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        R = quat_to_rotmat(q)
        q_back = rotmat_to_quat(R)
        # Quaternions q and -q represent same rotation
        assert np.allclose(np.abs(q_back), np.abs(q), atol=1e-10)


class TestErrorQuaternion:
    """Test error quaternion operations."""
    
    def test_error_quat_identity(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        delta_q = error_quat(q, q)
        assert np.allclose(delta_q, [0.0, 0.0, 0.0, 1.0], atol=1e-10)
    
    def test_error_quat_small(self):
        q_true = np.array([0.1, 0.2, 0.3, 0.9])
        q_true = normalize(q_true)
        delta_theta = np.array([0.01, -0.02, 0.005])
        q_est = box_plus(q_true, -delta_theta)  # q_est = q_true * exp(-delta_theta)
        
        delta_q = error_quat(q_true, q_est)
        delta_theta_back = error_quat_to_vec(delta_q)
        assert np.allclose(delta_theta, delta_theta_back, atol=1e-10)
    
    def test_vec_to_error_quat_small(self):
        delta_theta = np.array([0.01, -0.02, 0.005])
        delta_q = vec_to_error_quat(delta_theta)
        delta_theta_back = error_quat_to_vec(delta_q)
        assert np.allclose(delta_theta, delta_theta_back, atol=1e-10)
    
    def test_box_plus_minus(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        delta = np.array([0.01, -0.02, 0.005])
        
        q_new = box_plus(q, delta)
        delta_back = box_minus(q_new, q)
        assert np.allclose(delta, delta_back, atol=1e-10)


class TestQuaternionDerivative:
    """Test quaternion kinematics."""
    
    def test_quat_derivative_zero_omega(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = normalize(q)
        omega = np.zeros(3)
        q_dot = quat_derivative(q, omega)
        assert np.allclose(q_dot, 0.0)
    
    def test_quat_derivative_constant(self):
        # Constant rotation about z-axis
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s about z
        q_dot = quat_derivative(q, omega)
        # q_dot = 0.5 * [0, 0, 1, 0]^T for this case
        expected = np.array([0.0, 0.0, 0.5, 0.0])
        assert np.allclose(q_dot, expected)


class TestSlerp:
    """Test spherical linear interpolation."""
    
    def test_slerp_endpoints(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        
        assert np.allclose(slerp(q1, q2, 0.0), q1)
        assert np.allclose(np.abs(slerp(q1, q2, 1.0)), np.abs(q2))
    
    def test_slerp_midpoint(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about z
        q_mid = slerp(q1, q2, 0.5)
        # Should be 90 deg about z
        expected = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        assert np.allclose(np.abs(q_mid), np.abs(expected), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])