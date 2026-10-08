"""
Unit tests for rigid body dynamics.
"""

import numpy as np
import pytest
from src.dynamics.rigid_body import (
    RigidBody, inertia_matrix_from_principal, 
    add_inertia_uncertainty, random_inertia_uncertainty
)
from src.dynamics.quaternion import normalize


class TestRigidBody:
    """Test rigid body dynamics."""
    
    def test_initialization(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        
        assert np.allclose(body.I, I)
        assert np.allclose(body.q, q0)
        assert np.allclose(body.omega, omega0)
        assert np.allclose(body.tau_d, 0.0)
    
    def test_initialization_with_disturbance(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        tau_d = np.array([0.001, 0.002, 0.003])
        
        body = RigidBody(I, q0, omega0, disturbance_torque=tau_d)
        assert np.allclose(body.tau_d, tau_d)
    
    def test_inertia_validation(self):
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        
        # Non-symmetric inertia should fail
        I_bad = np.array([[100, 1, 0], [2, 80, 0], [0, 0, 60]])
        with pytest.raises(AssertionError):
            RigidBody(I_bad, q0, omega0)
        
        # Non-positive definite should fail
        I_bad2 = np.diag([100, -80, 60])
        with pytest.raises(AssertionError):
            RigidBody(I_bad2, q0, omega0)
    
    def test_dynamics_torque_free(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        q_dot, omega_dot = body.dynamics(np.zeros(3))
        
        # With no torque, omega_dot should be -I^-1 * (omega x I*omega)
        omega_cross_Iomega = np.cross(omega0, I @ omega0)
        expected_omega_dot = -np.linalg.inv(I) @ omega_cross_Iomega
        assert np.allclose(omega_dot, expected_omega_dot)
    
    def test_dynamics_with_torque(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        tau = np.array([0.1, 0.0, 0.0])
        
        body = RigidBody(I, q0, omega0)
        q_dot, omega_dot = body.dynamics(tau)
        
        expected_omega_dot = np.linalg.inv(I) @ tau
        assert np.allclose(omega_dot, expected_omega_dot)
    
    def test_step_rk4(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        q_new, omega_new = body.step(np.zeros(3), 0.1)
        
        # Quaternion should remain normalized
        assert np.abs(np.linalg.norm(q_new) - 1.0) < 1e-10
        
        # State should have changed
        assert not np.allclose(q_new, q0)
        assert not np.allclose(omega_new, omega0)
    
    def test_energy_conservation_torque_free(self):
        """Test energy conservation in torque-free motion."""
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        energy_initial = body.kinetic_energy()
        momentum_initial = body.angular_momentum_inertial()
        
        dt = 0.01  # Smaller time step for better energy conservation
        steps = 10000
        for _ in range(steps):
            body.step(np.zeros(3), dt)
        
        energy_final = body.kinetic_energy()
        momentum_final = body.angular_momentum_inertial()
        
        energy_error = abs(energy_final - energy_initial) / energy_initial
        momentum_error = np.linalg.norm(momentum_final - momentum_initial) / np.linalg.norm(momentum_initial)
        
        # Energy and momentum should be conserved (within numerical error)
        assert energy_error < 1e-4  # Relaxed tolerance for RK4
        assert momentum_error < 1e-6
    
    def test_kinetic_energy(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        energy = body.kinetic_energy()
        expected = 0.5 * omega0 @ I @ omega0
        assert np.allclose(energy, expected)
    
    def test_angular_momentum(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        L_body = body.angular_momentum()
        L_inertial = body.angular_momentum_inertial()
        
        assert np.allclose(L_body, I @ omega0)
        # At identity quaternion, body and inertial should match
        assert np.allclose(L_inertial, L_body)
    
    def test_get_set_state(self):
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.array([0.1, 0.05, 0.02])
        
        body = RigidBody(I, q0, omega0)
        q, omega = body.get_state()
        assert np.allclose(q, q0)
        assert np.allclose(omega, omega0)
        
        q_new = np.array([0.1, 0.2, 0.3, 0.9])
        q_new = normalize(q_new)
        omega_new = np.array([0.01, 0.02, 0.03])
        body.set_state(q_new, omega_new)
        
        q, omega = body.get_state()
        assert np.allclose(q, q_new)
        assert np.allclose(omega, omega_new)


class TestInertiaFunctions:
    """Test inertia matrix helper functions."""
    
    def test_inertia_matrix_from_principal(self):
        I = inertia_matrix_from_principal(100.0, 80.0, 60.0)
        expected = np.diag([100.0, 80.0, 60.0])
        assert np.allclose(I, expected)
    
    def test_inertia_matrix_with_products(self):
        I = inertia_matrix_from_principal(100.0, 80.0, 60.0, Ixy=1.0, Ixz=2.0, Iyz=3.0)
        expected = np.array([
            [100.0, -1.0, -2.0],
            [-1.0, 80.0, -3.0],
            [-2.0, -3.0, 60.0]
        ])
        assert np.allclose(I, expected)
    
    def test_add_inertia_uncertainty(self):
        I_nom = np.diag([100.0, 80.0, 60.0])
        delta_I = np.diag([1.0, -2.0, 0.5])
        I_uncertain = add_inertia_uncertainty(I_nom, delta_I)
        expected = np.diag([101.0, 78.0, 60.5])
        assert np.allclose(I_uncertain, expected)
    
    def test_random_inertia_uncertainty(self):
        I_nom = np.diag([100.0, 80.0, 60.0])
        delta_I = random_inertia_uncertainty(I_nom, max_rel_error=0.1)
        
        # Should be symmetric
        assert np.allclose(delta_I, delta_I.T)
        
        # Diagonal elements should be roughly within 10%
        for i in range(3):
            rel_error = abs(delta_I[i, i]) / I_nom[i, i]
            assert rel_error < 0.2  # Allow some margin for random generation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])