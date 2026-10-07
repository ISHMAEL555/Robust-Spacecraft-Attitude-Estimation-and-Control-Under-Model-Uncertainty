"""
Integration tests for the complete spacecraft simulation.
"""

import numpy as np
import pytest
from src.dynamics.rigid_body import RigidBody
from src.sensors.gyro import Gyroscope, GyroParams
from src.sensors.star_tracker import StarTracker, StarTrackerParams
from src.estimation.mekf import MEKF, create_mekf_nominal
from src.control.attitude_controller import AttitudeController, ControllerParams
from src.actuators.reaction_wheel import ReactionWheelAssembly, ReactionWheelParams
from src.dynamics.quaternion import normalize, error_quat, error_quat_to_vec


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_nominal_simulation_short(self):
        """Run a short nominal simulation to verify integration."""
        # Configuration
        dt = 0.01
        duration = 10.0
        n_steps = int(duration / dt)
        
        # Spacecraft
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.1, 0.2, 0.3, 0.9])
        q0 = normalize(q0)
        omega0 = np.array([0.01, 0.005, 0.002])
        
        spacecraft = RigidBody(I, q0, omega0)
        
        # Gyroscope
        gyro_params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(gyro_params, initial_bias=np.array([0.001, -0.002, 0.0005]), seed=42)
        
        # Star tracker
        st_params = StarTrackerParams(sigma_theta=10.0 * np.pi / 180.0 / 3600.0, rate=1.0)
        star_tracker = StarTracker(st_params, seed=43)
        
        # Filter
        mekf = create_mekf_nominal('gyro_driven')
        mekf.reset(q0, gyro.get_true_bias())
        
        # Controller
        ctrl_params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(ctrl_params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        # Reaction wheels
        rw_params = ReactionWheelParams(max_torque=0.1, max_momentum=10.0, num_wheels=3)
        rwa = ReactionWheelAssembly(rw_params)
        
        # Run simulation
        pointing_errors = []
        nees_values = []
        
        for i in range(n_steps):
            # True state
            q_true, omega_true = spacecraft.get_state()
            
            # Gyro measurement
            omega_m = gyro.measure(omega_true)
            
            # Star tracker measurement (1 Hz)
            q_meas = None
            if i % 100 == 0:
                q_meas, valid = star_tracker.measure(q_true, dt)
            
            # Filter step
            tau_cmd_prev = np.zeros(3) if i == 0 else tau_cmd
            result = mekf.step(omega_m, q_meas, tau_cmd_prev, dt, i*dt)
            
            # Controller
            tau_cmd = controller.compute_control(result['q_hat'], result['omega_hat'])
            
            # Reaction wheels
            tau_rw = rwa.compute_torque(tau_cmd)
            rwa.update_momentum(dt)
            
            # Spacecraft dynamics
            spacecraft.step(tau_rw, dt)
            
            # Metrics
            delta_q = error_quat(q_true, result['q_hat'])
            delta_theta = error_quat_to_vec(delta_q)
            pointing_errors.append(np.linalg.norm(delta_theta))
            
            b_true = gyro.get_true_bias()
            nees = mekf.compute_nees(q_true, b_true)
            nees_values.append(nees)
        
        # Check that simulation ran without errors
        assert len(pointing_errors) == n_steps
        assert len(nees_values) == n_steps
        
        # Pointing error should be reasonable (less than 1 degree)
        max_pointing = np.max(pointing_errors)
        assert max_pointing < np.radians(1.0)
        
        # NEES should be reasonable (not extremely large)
        mean_nees = np.mean(nees_values)
        assert mean_nees < 100  # Very loose bound
    
    def test_filter_convergence(self):
        """Test that filter converges to true state."""
        dt = 0.01
        duration = 100.0
        n_steps = int(duration / dt)
        
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        
        spacecraft = RigidBody(I, q0, omega0)
        
        gyro_params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(gyro_params, initial_bias=np.array([0.01, -0.01, 0.005]), seed=42)
        
        st_params = StarTrackerParams(sigma_theta=10.0 * np.pi / 180.0 / 3600.0, rate=1.0)
        star_tracker = StarTracker(st_params, seed=43)
        
        mekf = create_mekf_nominal('gyro_driven')
        # Initialize with large error
        q_init = np.array([0.1, 0.1, 0.1, np.sqrt(1 - 0.03)])
        q_init = normalize(q_init)
        b_init = np.array([0.0, 0.0, 0.0])
        mekf.reset(q_init, b_init)
        
        ctrl_params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(ctrl_params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        rw_params = ReactionWheelParams(max_torque=0.1, max_momentum=10.0, num_wheels=3)
        rwa = ReactionWheelAssembly(rw_params)
        
        attitude_errors = []
        bias_errors = []
        
        for i in range(n_steps):
            q_true, omega_true = spacecraft.get_state()
            omega_m = gyro.measure(omega_true)
            
            q_meas = None
            if i % 100 == 0:
                q_meas, _ = star_tracker.measure(q_true, dt)
            
            tau_cmd_prev = np.zeros(3) if i == 0 else tau_cmd
            result = mekf.step(omega_m, q_meas, tau_cmd_prev, dt, i*dt)
            
            tau_cmd = controller.compute_control(result['q_hat'], result['omega_hat'])
            tau_rw = rwa.compute_torque(tau_cmd)
            rwa.update_momentum(dt)
            spacecraft.step(tau_rw, dt)
            
            # Track errors
            delta_q = error_quat(q_true, result['q_hat'])
            delta_theta = error_quat_to_vec(delta_q)
            attitude_errors.append(np.linalg.norm(delta_theta))
            
            b_true = gyro.get_true_bias()
            bias_errors.append(np.linalg.norm(b_true - result['b_hat']))
        
        # Errors should decrease over time
        # Check that final errors are smaller than initial
        assert attitude_errors[-1] < attitude_errors[0]
        assert bias_errors[-1] < bias_errors[0]
        
        # Final errors should be small
        assert attitude_errors[-1] < np.radians(0.1)  # < 0.1 deg
        assert bias_errors[-1] < 1e-3  # < 1 mrad/s
    
    def test_closed_loop_stability(self):
        """Test closed-loop stability with controller."""
        dt = 0.01
        duration = 50.0
        n_steps = int(duration / dt)
        
        I = np.diag([100.0, 80.0, 60.0])
        # Start with large initial attitude error
        q0 = np.array([0.5, 0.0, 0.0, np.sqrt(1 - 0.25)])
        q0 = normalize(q0)
        omega0 = np.array([0.1, 0.0, 0.0])
        
        spacecraft = RigidBody(I, q0, omega0)
        
        gyro_params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(gyro_params, seed=42)
        
        st_params = StarTrackerParams(sigma_theta=10.0 * np.pi / 180.0 / 3600.0, rate=1.0)
        star_tracker = StarTracker(st_params, seed=43)
        
        mekf = create_mekf_nominal('gyro_driven')
        mekf.reset(q0, gyro.get_true_bias())
        
        ctrl_params = ControllerParams(K_q=2.0, K_omega=20.0, max_torque=0.1)
        controller = AttitudeController(ctrl_params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        rw_params = ReactionWheelParams(max_torque=0.1, max_momentum=10.0, num_wheels=3)
        rwa = ReactionWheelAssembly(rw_params)
        
        attitude_errors = []
        
        for i in range(n_steps):
            q_true, omega_true = spacecraft.get_state()
            omega_m = gyro.measure(omega_true)
            
            q_meas = None
            if i % 100 == 0:
                q_meas, _ = star_tracker.measure(q_true, dt)
            
            tau_cmd_prev = np.zeros(3) if i == 0 else tau_cmd
            result = mekf.step(omega_m, q_meas, tau_cmd_prev, dt, i*dt)
            
            tau_cmd = controller.compute_control(result['q_hat'], result['omega_hat'])
            tau_rw = rwa.compute_torque(tau_cmd)
            rwa.update_momentum(dt)
            spacecraft.step(tau_rw, dt)
            
            delta_q = error_quat(q_true, result['q_hat'])
            delta_theta = error_quat_to_vec(delta_q)
            attitude_errors.append(np.linalg.norm(delta_theta))
        
        # Attitude should converge to near zero
        final_error = attitude_errors[-1]
        assert final_error < np.radians(0.5)  # < 0.5 deg
        
        # Error should be decreasing overall
        # Check last 1000 steps average < first 1000 steps average
        early_avg = np.mean(attitude_errors[:1000])
        late_avg = np.mean(attitude_errors[-1000:])
        assert late_avg < early_avg


class TestModelMismatch:
    """Test filter behavior with model mismatch."""
    
    def test_inertia_mismatch_gyro_driven(self):
        """Variant A should be unaffected by inertia mismatch in propagation."""
        dt = 0.01
        duration = 50.0
        n_steps = int(duration / dt)
        
        # True inertia
        I_true = np.diag([100.0, 80.0, 60.0])
        # Filter uses different inertia (but Variant A doesn't use it)
        I_filter = np.diag([110.0, 88.0, 66.0])  # 10% error
        
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        
        spacecraft = RigidBody(I_true, q0, omega0)
        
        gyro_params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(gyro_params, seed=42)
        
        st_params = StarTrackerParams(sigma_theta=10.0 * np.pi / 180.0 / 3600.0, rate=1.0)
        star_tracker = StarTracker(st_params, seed=43)
        
        # Variant A - gyro driven
        mekf = create_mekf_nominal('gyro_driven')
        mekf.reset(q0, gyro.get_true_bias())
        
        ctrl_params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(ctrl_params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        rw_params = ReactionWheelParams(max_torque=0.1, max_momentum=10.0, num_wheels=3)
        rwa = ReactionWheelAssembly(rw_params)
        
        nees_values = []
        
        for i in range(n_steps):
            q_true, omega_true = spacecraft.get_state()
            omega_m = gyro.measure(omega_true)
            
            q_meas = None
            if i % 100 == 0:
                q_meas, _ = star_tracker.measure(q_true, dt)
            
            tau_cmd_prev = np.zeros(3) if i == 0 else tau_cmd
            result = mekf.step(omega_m, q_meas, tau_cmd_prev, dt, i*dt)
            
            tau_cmd = controller.compute_control(result['q_hat'], result['omega_hat'])
            tau_rw = rwa.compute_torque(tau_cmd)
            rwa.update_momentum(dt)
            spacecraft.step(tau_rw, dt)
            
            b_true = gyro.get_true_bias()
            nees = mekf.compute_nees(q_true, b_true)
            nees_values.append(nees)
        
        # NEES should be consistent (around 1.0 for 6 DOF)
        mean_nees = np.mean(nees_values[1000:])  # Skip initial transient
        assert 0.5 < mean_nees < 2.0  # Loose bounds for short simulation
    
    def test_process_noise_mismatch(self):
        """Test filter with optimistic/conservative process noise."""
        dt = 0.01
        duration = 50.0
        n_steps = int(duration / dt)
        
        I = np.diag([100.0, 80.0, 60.0])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        
        spacecraft = RigidBody(I, q0, omega0)
        
        gyro_params = GyroParams(sigma_v=1e-4, sigma_u=1e-6, rate=100.0)
        gyro = Gyroscope(gyro_params, seed=42)
        
        st_params = StarTrackerParams(sigma_theta=10.0 * np.pi / 180.0 / 3600.0, rate=1.0)
        star_tracker = StarTracker(st_params, seed=43)
        
        # Optimistic Q (too small)
        mekf = create_mekf_nominal('gyro_driven')
        mekf.params.Q_scale = 0.1  # 10x too small
        mekf.Q = mekf._build_process_noise()
        mekf.reset(q0, gyro.get_true_bias())
        
        ctrl_params = ControllerParams(K_q=1.0, K_omega=10.0, max_torque=0.1)
        controller = AttitudeController(ctrl_params)
        controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))
        
        rw_params = ReactionWheelParams(max_torque=0.1, max_momentum=10.0, num_wheels=3)
        rwa = ReactionWheelAssembly(rw_params)
        
        nees_values = []
        
        for i in range(n_steps):
            q_true, omega_true = spacecraft.get_state()
            omega_m = gyro.measure(omega_true)
            
            q_meas = None
            if i % 100 == 0:
                q_meas, _ = star_tracker.measure(q_true, dt)
            
            tau_cmd_prev = np.zeros(3) if i == 0 else tau_cmd
            result = mekf.step(omega_m, q_meas, tau_cmd_prev, dt, i*dt)
            
            tau_cmd = controller.compute_control(result['q_hat'], result['omega_hat'])
            tau_rw = rwa.compute_torque(tau_cmd)
            rwa.update_momentum(dt)
            spacecraft.step(tau_rw, dt)
            
            b_true = gyro.get_true_bias()
            nees = mekf.compute_nees(q_true, b_true)
            nees_values.append(nees)
        
        # With optimistic Q, NEES should be large (overconfident)
        mean_nees = np.mean(nees_values[1000:])
        assert mean_nees > 5.0  # Should be inconsistent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])