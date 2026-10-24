"""
Unit tests for attitude controller.
"""

import numpy as np
import pytest
from src.control.attitude_controller import (
    AttitudeController, ControllerParams, 
    compute_control_gains, quaternion_feedback_torque
)
from src.dynamics.quaternion import normalize, multiply, vec_to_error_quat, rotate_vector


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

    def test_nonidentity_target_error_is_expressed_in_current_body_frame(self):
        params = ControllerParams(K_q=0.8, K_omega=0.0, max_torque=10.0)
        controller = AttitudeController(params)
        q_desired = vec_to_error_quat(np.array([0.7, -0.4, 0.2]))
        attitude_offset_body = np.array([0.2, -0.1, 0.05])
        q_current = multiply(q_desired, vec_to_error_quat(attitude_offset_body))
        controller.set_desired(q_desired)

        torque = controller.compute_control(q_current, np.zeros(3))
        expected = -params.K_q * vec_to_error_quat(attitude_offset_body)[:3]

        assert np.allclose(torque, expected, atol=1e-12)

    def test_desired_rate_is_rotated_into_current_body_frame(self):
        params = ControllerParams(K_q=0.0, K_omega=2.0, max_torque=10.0)
        controller = AttitudeController(params)
        q_desired = vec_to_error_quat(np.array([0.6, 0.2, -0.3]))
        q_current = multiply(q_desired, vec_to_error_quat(np.array([0.3, -0.2, 0.1])))
        omega_desired = np.array([0.01, -0.02, 0.03])
        omega_current = np.array([-0.02, 0.04, 0.01])
        controller.set_desired(q_desired, omega_desired)
        q_error = multiply(np.array([-q_current[0], -q_current[1], -q_current[2], q_current[3]]), q_desired)
        q_error = normalize(q_error)

        torque = controller.compute_control(q_current, omega_current)
        expected = -params.K_omega * (
            omega_current - rotate_vector(q_error, omega_desired)
        )

        assert np.allclose(torque, expected, atol=1e-12)
    
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
    
    def test_compute_control_with_feedforward_saturates_once(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))

        I = np.diag([100.0, 80.0, 60.0])
        q_current = np.array([0.0, 0.0, 0.0, 1.0])
        omega_current = np.array([0.1, 0.05, 0.02])

        tau = controller.compute_control_with_feedforward(q_current, omega_current, I)

        # The unsaturated feedback and gyroscopic feedforward are summed and
        # then limited once; the feedforward is not lost when feedback alone
        # would already saturate.
        tau_fb = -params.K_omega * omega_current
        expected = tau_fb + np.cross(omega_current, I @ omega_current)
        expected *= params.max_torque / np.linalg.norm(expected)
        assert np.allclose(tau, expected, atol=1e-12)
        assert np.linalg.norm(tau) <= params.max_torque + 1e-12

    def test_feedforward_includes_wheel_momentum(self):
        params = ControllerParams(K_q=1.0, K_omega=0.0, max_torque=10.0)
        controller = AttitudeController(params)
        I = np.diag([100.0, 80.0, 60.0])
        omega = np.array([0.0, 0.0, 0.01])
        h_w = np.array([2.0, 0.0, 0.0])

        tau = controller.compute_control_with_feedforward(
            np.array([0.0, 0.0, 0.0, 1.0]), omega, I, wheel_momentum=h_w
        )

        assert np.allclose(tau, np.cross(omega, I @ omega + h_w), atol=1e-14)

    def test_feedforward_cancels_estimated_disturbance(self):
        params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))

        I = np.diag([100.0, 80.0, 60.0])
        tau_dist = np.array([0.001, 0.002, 0.003])

        tau = controller.compute_control_with_feedforward(
            np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), I, tau_dist
        )

        # A disturbance acting on the bus is cancelled by commanding its negative.
        assert np.allclose(tau, -tau_dist, atol=1e-14)

    def test_rejects_unknown_control_type(self):
        with pytest.raises(ValueError):
            AttitudeController(ControllerParams(control_type="lqr"))


class TestControlGains:
    """Test control gain computation."""

    def test_compute_control_gains_formula(self):
        I = np.diag([100.0, 80.0, 60.0])
        settling_time = 100.0
        damping_ratio = 0.7

        K_q, K_omega = compute_control_gains(settling_time, damping_ratio, I)

        omega_n = 4.0 / (damping_ratio * settling_time)
        I_avg = np.mean(np.diag(I))
        assert np.isclose(K_q, I_avg * omega_n**2)
        assert np.isclose(K_omega, 2 * damping_ratio * I_avg * omega_n)

        K_q_qf, K_omega_qf = compute_control_gains(
            settling_time, damping_ratio, I, control_type="quaternion_feedback"
        )
        # q_e,v ≈ theta / 2 halves the stiffness, so the gain is doubled.
        assert np.isclose(K_q_qf, 2.0 * K_q)
        assert np.isclose(K_omega_qf, K_omega)

    @pytest.mark.parametrize("control_type", ["pd", "quaternion_feedback"])
    def test_designed_gains_meet_settling_time(self, control_type):
        """Small-angle single-axis response settles near the design time."""
        inertia = 80.0
        settling_time, damping_ratio = 60.0, 0.8
        K_q, K_omega = compute_control_gains(
            settling_time, damping_ratio, np.eye(3) * inertia, control_type
        )
        controller = AttitudeController(ControllerParams(
            K_q=K_q, K_omega=K_omega, max_torque=1e3, control_type=control_type
        ))
        theta0, theta, rate, dt = 0.01, 0.01, 0.0, 0.01
        times, errors = [], []
        for step in range(int(2 * settling_time / dt)):
            q = np.array([np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2)])
            torque = controller.compute_control(q, np.array([rate, 0.0, 0.0]))[0]
            rate += torque / inertia * dt
            theta += rate * dt
            times.append(step * dt)
            errors.append(abs(theta))
        errors = np.asarray(errors)
        outside = np.flatnonzero(errors > 0.02 * theta0)
        measured_settling = times[outside[-1] + 1]
        assert abs(measured_settling - settling_time) / settling_time < 0.25

    def test_quaternion_feedback_torque_sign(self):
        q_error = normalize(np.array([0.01, -0.02, 0.005, 0.999]))
        omega_error = np.array([0.001, -0.002, 0.0005])

        tau = quaternion_feedback_torque(q_error, omega_error, 1.0, 10.0)

        # q_error = q_current^-1 ⊗ q_desired points toward the target, so the
        # attitude term is positive; the rate term damps the rate error.
        assert np.allclose(tau, q_error[:3] - 10.0 * omega_error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
