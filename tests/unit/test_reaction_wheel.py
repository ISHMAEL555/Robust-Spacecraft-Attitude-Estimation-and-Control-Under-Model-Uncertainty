"""
Unit tests for reaction wheel actuator.
"""

import numpy as np
import pytest
from src.actuators.reaction_wheel import (
    ReactionWheelAssembly, ReactionWheelParams,
    allocate_torque_3wheel, allocate_torque_4wheel_pyramid
)


class TestReactionWheelAssembly:
    """Test reaction wheel assembly."""
    
    def test_initialization_3wheel(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        assert rwa.params.num_wheels == 3
        assert rwa.axes.shape == (3, 3)
        assert np.allclose(rwa.axes, np.eye(3))
        assert np.allclose(rwa.momentum, 0.0)
        assert np.allclose(rwa.torque, 0.0)
    
    def test_initialization_4wheel(self):
        params = ReactionWheelParams(num_wheels=4, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        assert rwa.params.num_wheels == 4
        assert rwa.axes.shape == (3, 4)
        # Check pyramid configuration
        beta = np.arccos(1.0 / np.sqrt(3))
        expected_axes = np.array([
            [np.sin(beta), 0, -np.sin(beta), 0],
            [0, np.sin(beta), 0, -np.sin(beta)],
            [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]
        ])
        assert np.allclose(rwa.axes, expected_axes)
    
    def test_compute_torque_3wheel(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.05, 0.03, 0.02])
        tau_achieved = rwa.compute_torque(tau_cmd)
        
        # With orthogonal axes and no saturation, should match
        assert np.allclose(tau_achieved, tau_cmd)
    
    def test_compute_torque_saturation(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.5, 0.5, 0.5])
        tau_achieved = rwa.compute_torque(tau_cmd)
        
        # Should be saturated at 0.1 per axis
        assert np.allclose(tau_achieved, [0.1, 0.1, 0.1])
    
    def test_compute_torque_4wheel(self):
        params = ReactionWheelParams(num_wheels=4, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.05, 0.03, 0.02])
        tau_achieved = rwa.compute_torque(tau_cmd)
        
        # Should be close to commanded (pseudo-inverse allocation)
        assert np.allclose(tau_achieved, tau_cmd, atol=1e-10)
    
    def test_update_momentum(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.05, 0.03, 0.02])
        rwa.compute_torque(tau_cmd, 1.0)
        rwa.update_momentum(1.0)
        
        # Momentum should equal torque * dt
        expected_momentum = np.array([-0.05, -0.03, -0.02])
        assert np.allclose(rwa.get_momentum(), expected_momentum)
    
    def test_momentum_saturation_does_not_block_unloading(self):
        """Torque that reduces momentum magnitude remains available at the limit."""
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=1.0)
        rwa = ReactionWheelAssembly(params)
        rwa.momentum[0] = -1.0

        tau_cmd = np.array([-0.1, 0.0, 0.0])
        tau_achieved = rwa.compute_torque(tau_cmd)

        # A negative body torque commands positive wheel torque, which unloads
        # a negatively saturated wheel under tau_body = -A @ tau_wheel.
        assert np.allclose(tau_achieved, tau_cmd)

    def test_momentum_saturation(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=1.0)
        rwa = ReactionWheelAssembly(params)
        
        # Apply torque for long time to saturate momentum
        tau_cmd = np.array([0.1, 0.0, 0.0])
        for _ in range(20):
            rwa.compute_torque(tau_cmd, 1.0)
            rwa.update_momentum(1.0)
        
        momentum = rwa.get_momentum()
        # Should be saturated at max_momentum
        assert momentum[0] <= 1.0 + 1e-10
        assert np.isclose(momentum[0], -1.0)
        assert np.isclose(rwa.get_torque()[0], 0.0)

    def test_momentum_limit_is_hard_and_predictive(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=1.0)
        rwa = ReactionWheelAssembly(params)
        rwa.momentum[0] = -0.999

        # A 0.1 s command would cross the lower limit; clip only the final
        # increment needed to reach the hard bound, preserving exact balance.
        tau_body = rwa.compute_torque(np.array([0.1, 0.0, 0.0]), dt=0.1)
        rwa.update_momentum(0.1)

        assert np.isclose(tau_body[0], 0.01)
        assert np.isclose(rwa.get_momentum()[0], -1.0)

    def test_first_order_motor_torque_response(self):
        params = ReactionWheelParams(
            num_wheels=3,
            max_torque=0.1,
            max_momentum=1.0,
            motor_time_constant_s=0.05,
        )
        rwa = ReactionWheelAssembly(params)
        dt = 0.01

        tau_body = rwa.compute_torque(np.array([0.1, 0.0, 0.0]), dt=dt)
        expected_motor_torque = -0.1 * (1.0 - np.exp(-dt / 0.05))
        rwa.update_momentum(dt)

        assert np.isclose(tau_body[0], -expected_motor_torque)
        assert np.isclose(rwa.get_torque()[0], expected_motor_torque)
        assert np.isclose(rwa.get_momentum()[0], expected_motor_torque * dt)
    
    def test_get_body_momentum(self):
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.05, 0.03, 0.02])
        rwa.compute_torque(tau_cmd)
        rwa.update_momentum(1.0)
        
        body_momentum = rwa.get_body_momentum()
        # With orthogonal axes, body-frame wheel momentum equals the wheel-state vector
        assert np.allclose(body_momentum, rwa.get_momentum())

    def test_total_angular_momentum_conservation(self):
        """Wheel/body exchange must conserve total angular momentum."""
        params = ReactionWheelParams(num_wheels=3, max_torque=0.1, max_momentum=10.0)
        rwa = ReactionWheelAssembly(params)
        tau_cmd = np.array([0.05, -0.03, 0.02])
        body_momentum_change = np.zeros(3)
        for _ in range(10):
            tau_body = rwa.compute_torque(tau_cmd)
            body_momentum_change += tau_body * 0.1
            rwa.update_momentum(0.1)
        assert np.allclose(body_momentum_change + rwa.get_body_momentum(), 0.0, atol=1e-12)

    def test_reset(self):
        params = ReactionWheelParams(num_wheels=3)
        rwa = ReactionWheelAssembly(params)
        
        tau_cmd = np.array([0.05, 0.03, 0.02])
        rwa.compute_torque(tau_cmd)
        rwa.update_momentum(1.0)
        
        rwa.reset()
        
        assert np.allclose(rwa.momentum, 0.0)
        assert np.allclose(rwa.torque, 0.0)


class TestAllocationFunctions:
    """Test torque allocation functions."""
    
    def test_allocate_torque_3wheel(self):
        tau_cmd = np.array([0.05, 0.03, 0.02])
        max_torque = 0.1
        
        tau_wheel = allocate_torque_3wheel(tau_cmd, max_torque)
        
        assert np.allclose(tau_wheel, -tau_cmd)
    
    def test_allocate_torque_3wheel_saturation(self):
        tau_cmd = np.array([0.5, 0.5, 0.5])
        max_torque = 0.1
        
        tau_wheel = allocate_torque_3wheel(tau_cmd, max_torque)
        
        assert np.allclose(tau_wheel, [-0.1, -0.1, -0.1])
    
    def test_allocate_torque_4wheel_pyramid(self):
        tau_cmd = np.array([0.05, 0.03, 0.02])
        max_torque = 0.1
        
        tau_wheel = allocate_torque_4wheel_pyramid(tau_cmd, max_torque)
        
        assert tau_wheel.shape == (4,)
        # Check that allocation produces correct body torque
        beta = np.arccos(1.0 / np.sqrt(3))
        A = np.array([
            [np.sin(beta), 0, -np.sin(beta), 0],
            [0, np.sin(beta), 0, -np.sin(beta)],
            [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]
        ])
        tau_body = -A @ tau_wheel
        assert np.allclose(tau_body, tau_cmd, atol=1e-10)
    
    def test_allocate_torque_4wheel_pyramid_saturation(self):
        tau_cmd = np.array([0.5, 0.5, 0.5])
        max_torque = 0.1
        
        tau_wheel = allocate_torque_4wheel_pyramid(tau_cmd, max_torque)
        
        assert np.all(np.abs(tau_wheel) <= max_torque + 1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
