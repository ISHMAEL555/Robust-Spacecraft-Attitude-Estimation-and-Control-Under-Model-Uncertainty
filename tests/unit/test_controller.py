"""
Unit tests for attitude controller.
"""

import numpy as np
import pytest
from src.control.attitude_controller import (
    AttitudeController, ControllerParams, 
    compute_control_gains, quaternion_feedback_torque
)
from src.dynamics.quaternion import normalize


class TestAttitudeController:
    """Test attitude controller."""
    
    def test_initialization(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        
        assert controller.K_q == 1.0
        assert controller.K_omega == 10.0
        assert controller.max_torque == 0.1
        assert np.allclose(controller.q_desired, [0.0, 0.0, 0.0, 1.0])
        assert np.allclose(controller.omega_desired, 0.0)
    
    def test_set_desired(self):
        params = ControllerParams()
        controller = AttitudeController(params)
        
        q_des = np.array([0.1, 0.2, 0.3, 0.9])
        q_des = normalize(q_des)
        omega_des = np.array([0.01, 0.0, 0.0])
        
        controller.set_desired(q_des, omega_des)
        
        assert np.allclose(controller.q_desired, q_des)
        assert np.allclose(controller.omega_desired, omega_des)
    
    def test_compute_control_regulation(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        # Small attitude error
        q_current = np.array([0.01, 0.0, 0.0, np.sqrt(1 - 0.01**2)])
        q_current = normalize(q_current)
        omega_current = np.array([0.001, 0.0, 0.0])
        
        tau = controller.compute_control(q_current, omega_current)
        
        # Should produce negative torque to correct error
        assert tau[0] < 0
        assert np.abs(tau[0]) < 0.1  # Within saturation limit
    
    def test_compute_control_large_error(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        # Large attitude error (90 deg about x)
        q_current = np.array([np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)])
        q_current = normalize(q_current)
        omega_current = np.zeros(3)
        
        tau = controller.compute_control(q_current, omega_current)
        
        # Should saturate
        assert np.linalg.norm(tau) <= 0.1 + 1e-10
    
    def test_compute_control_with_rate_error(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        q_current = np.array([0.0, 0.0, 0.0, 1.0])
        omega_current = np.array([0.1, 0.0, 0.0])
        
        tau = controller.compute_control(q_current, omega_current)
        
        # Should produce damping torque
        assert tau[0] < 0
    
    def test_compute_control_saturation(self):
        params = ControllerParams(K_q=100.0, K_omega=1000.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        q_current = np.array([0.1, 0.0, 0.0, np.sqrt(1 - 0.1**2)])
        q_current = normalize(q_current)
        omega_current = np.array([1.0, 0.0, 0.0])
        
        tau = controller.compute_control(q_current, omega_current)
        
        # Should be saturated
        assert np.linalg.norm(tau) <= 0.1 + 1e-10
    
    def test_compute_control_with_feedforward(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        I = np.diag([100.0, 80.0, 60.0])
        q_current = np.array([0.0, 0.0, 0.0, 1.0])
        omega_current = np.array([0.1, 0.05, 0.02])
        
        tau = controller.compute_control_with_feedforward(q_current, omega_current, I)
        
        # Feedforward should add omega x I*omega term
        omega_cross_Iomega = np.cross(omega_current, I @ omega_current)
        tau_fb = controller.compute_control(q_current, omega_current)
        expected = tau_fb + omega_cross_Iomega
        expected_norm = np.linalg.norm(expected)
        if expected_norm > params.max_torque:
            expected *= params.max_torque / expected_norm
        
        assert np.allclose(tau, expected, atol=1e-10)
    
    def test_compute_control_with_disturbance(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        I = np.diag([100.0, 80.0, 60.0])
        tau_dist = np.array([0.001, 0.002, 0.003])
        
        q_current = np.array([0.0, 0.0, 0.0, 1.0])
        omega_current = np.zeros(3)
        
        tau = controller.compute_control_with_feedforward(q_current, omega_current, I, tau_dist)
        
        tau_fb = controller.compute_control(q_current, omega_current)
        expected = tau_fb + tau_dist
        
        assert np.allclose(tau, expected, atol=1e-10)


class TestControlGains:
    """Test control gain computation."""
    
    def test_compute_control_gains(self):
        I = np.diag([100.0, 80.0, 60.0])
        settling_time = 100.0
        damping_ratio = 0.7
        
        K_q, K_omega = compute_control_gains(settling_time, damping_ratio, I)
        
        # Check positive gains
        assert K_q > 0
        assert K_omega > 0
        
        # Check approximate values
        omega_n = 4.0 / settling_time
        I_avg = np.mean(np.diag(I))
        expected_K_q = I_avg * omega_n**2
        expected_K_omega = 2 * damping_ratio * I_avg * omega_n
        
        assert np.allclose(K_q, expected_K_q, rtol=0.01)
        assert np.allclose(K_omega, expected_K_omega, rtol=0.01)
    
    def test_quaternion_feedback_torque(self):
        q_error = np.array([0.01, -0.02, 0.005, 0.999])
        q_error = normalize(q_error)
        omega_error = np.array([0.001, -0.002, 0.0005])
        K_q = 1.0
        K_omega = 10.0
        
        tau = quaternion_feedback_torque(q_error, omega_error, K_q, K_omega)
        
        expected = -K_q * q_error[:3] - K_omega * omega_error
        assert np.allclose(tau, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])