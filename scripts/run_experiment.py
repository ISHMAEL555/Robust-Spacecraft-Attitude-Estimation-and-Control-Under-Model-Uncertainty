#!/usr/bin/env python3
"""
Main simulation script for spacecraft attitude estimation and control.

Runs experiments defined in config files.
"""

import numpy as np
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add the repository root so the src package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dynamics.quaternion import right_box_plus, right_error_quat, error_quat_to_vec
from src.dynamics.rigid_body import RigidBody
from src.sensors.gyro import Gyroscope, GyroParams
from src.sensors.star_tracker import StarTracker, StarTrackerParams
from src.estimation.mekf import MEKF, MEKFParams
from src.control.attitude_controller import AttitudeController, ControllerParams
from src.actuators.reaction_wheel import ReactionWheelAssembly, ReactionWheelParams


@dataclass
class SimulationConfig:
    """Simulation configuration loaded from YAML."""
    duration: float
    dt: float
    seed: int
    spacecraft: Dict[str, Any]
    gyroscope: Dict[str, Any]
    star_tracker: Dict[str, Any]
    filter: Dict[str, Any]
    controller: Dict[str, Any]
    reaction_wheels: Dict[str, Any]
    analysis: Dict[str, Any]


def load_config(config_path: str) -> SimulationConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    simulation = config_dict['simulation']
    return SimulationConfig(
        duration=simulation['duration'],
        dt=simulation['dt'],
        seed=simulation['seed'],
        spacecraft=config_dict['spacecraft'],
        gyroscope=config_dict['gyroscope'],
        star_tracker=config_dict['star_tracker'],
        filter=config_dict['filter'],
        controller=config_dict['controller'],
        reaction_wheels=config_dict['reaction_wheels'],
        analysis=config_dict['analysis'],
    )


def run_simulation(config: SimulationConfig, 
                   true_inertia: Optional[np.ndarray] = None,
                   filter_inertia: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Run a single simulation.
    
    Args:
        config: Simulation configuration
        true_inertia: True inertia matrix (for mismatch studies)
        filter_inertia: Inertia matrix used by filter (for model-aided variant)
        
    Returns:
        Dictionary with simulation results
    """
    np.random.seed(config.seed)
    if config.dt <= 0 or config.duration <= 0:
        raise ValueError("Simulation duration and time step must be positive")

    # Time array
    n_steps = int(config.duration / config.dt)
    if n_steps < 1:
        raise ValueError("Simulation duration must be at least one time step")
    time = np.arange(n_steps) * config.dt
    
    # Spacecraft
    if true_inertia is None:
        true_inertia = np.array(config.spacecraft['inertia'])
    spacecraft = RigidBody(
        inertia=true_inertia,
        initial_q=np.array(config.spacecraft['initial_attitude']),
        initial_omega=np.array(config.spacecraft['initial_omega']),
        disturbance_torque=np.array(config.spacecraft['disturbance_torque'])
    )
    
    # Gyroscope
    gyro_params = GyroParams(
        sigma_v=config.gyroscope['sigma_v'],
        sigma_u=config.gyroscope['sigma_u'],
        sigma_b0=config.gyroscope['sigma_b0'],
        rate=config.gyroscope['rate']
    )
    gyro = Gyroscope(
        gyro_params,
        initial_bias=np.array(config.gyroscope['initial_bias']),
        seed=config.seed + 1
    )
    
    # Star tracker
    st_params = StarTrackerParams(
        sigma_theta=config.star_tracker['sigma_theta'],
        rate=config.star_tracker['rate'],
        outlier_prob=config.star_tracker['outlier_prob'],
        outlier_magnitude=config.star_tracker.get('outlier_magnitude', 1.0),
        outage_prob=config.star_tracker['outage_prob'],
        outage_duration=config.star_tracker.get('outage_duration', 10),
        outage_start_time=config.star_tracker.get('outage_start_time'),
    )
    star_tracker = StarTracker(st_params, seed=config.seed + 2)
    
    # Filter
    filter_params = MEKFParams(
        variant=config.filter['variant'],
        P0_attitude=config.filter['P0_attitude'],
        P0_bias=config.filter['P0_bias'],
        Q_scale=config.filter['Q_scale'],
        R_scale=config.filter['R_scale'],
        nis_gate_confidence=config.filter.get('nis_gate_confidence', 0.999999),
        nis_gate_enabled=config.filter.get('nis_gate_enabled', False),
        inertia_model=(
            filter_inertia
            if filter_inertia is not None
            else np.array(config.spacecraft['inertia'])
            if config.filter['variant'] == 'model_aided'
            else None
        ),
        inertia_relative_sigma=config.filter.get('inertia_relative_sigma', 0.05),
        sigma_omega_process=config.filter.get('sigma_omega_process', 0.0),
    )
    filter_params.gyro_params = gyro_params
    filter_params.star_tracker_params = st_params
    mekf = MEKF(filter_params)
    
    # Initialize filter with true state + small error
    q_init = spacecraft.q.copy()
    # Add small initial error
    delta_init = np.random.randn(3) * 0.01
    q_init = right_box_plus(q_init, delta_init)
    b_init = gyro.get_true_bias() + np.random.randn(3) * 1e-4
    mekf.reset(q_init, b_init)
    
    # Controller
    ctrl_params = ControllerParams(
        K_q=config.controller['K_q'],
        K_omega=config.controller['K_omega'],
        max_torque=config.controller['max_torque'],
        control_type=config.controller['control_type']
    )
    controller = AttitudeController(ctrl_params)
    controller.set_desired(np.array([0.0, 0.0, 0.0, 1.0]))  # Regulate to identity
    
    # Reaction wheels
    rw_params = ReactionWheelParams(
        max_torque=config.reaction_wheels['max_torque'],
        max_momentum=config.reaction_wheels['max_momentum'],
        motor_time_constant_s=config.reaction_wheels.get('motor_time_constant_s', 0.0),
        num_wheels=config.reaction_wheels['num_wheels']
    )
    rwa = ReactionWheelAssembly(rw_params)
    
    # Storage
    results = {
        'time': time,
        'q_true': np.zeros((n_steps, 4)),
        'omega_true': np.zeros((n_steps, 3)),
        'q_hat': np.zeros((n_steps, 4)),
        'b_hat': np.zeros((n_steps, 3)),
        'b_true': np.zeros((n_steps, 3)),
        'omega_hat': np.zeros((n_steps, 3)),
        'P': np.zeros((n_steps, mekf.state_dim, mekf.state_dim)),
        'nees_dof': mekf.nees_dof,
        'tau_cmd': np.zeros((n_steps, 3)),
        'tau_rw': np.zeros((n_steps, 3)),
        'wheel_torque': np.zeros((n_steps, config.reaction_wheels['num_wheels'])),
        'wheel_momentum': np.zeros((n_steps, config.reaction_wheels['num_wheels'])),
        'NEES': np.zeros(n_steps),
        'NEES_components': np.full((n_steps, 4), np.nan),
        'NIS': np.zeros(n_steps),
        'innovation': np.zeros((n_steps, 3)),
        'innovation_covariance': np.zeros((n_steps, 3, 3)),
        'innovation_valid': np.zeros(n_steps, dtype=bool),
        'measurement_available': np.zeros(n_steps, dtype=bool),
        'measurement_time_s': np.full(n_steps, np.nan),
        'measurement_accepted': np.zeros(n_steps, dtype=bool),
        'star_tracker_outage': np.zeros(n_steps, dtype=bool),
        'pointing_error': np.zeros(n_steps),
        'pointing_error_deg': np.zeros(n_steps),
        'attitude_estimation_error': np.zeros(n_steps),
        'attitude_estimation_error_vector': np.zeros((n_steps, 3)),
        'bias_error': np.zeros((n_steps, 3)),
        'max_torque': float(config.controller['max_torque']),
        'max_wheel_momentum': float(config.reaction_wheels['max_momentum']),
        'max_wheel_torque': float(config.reaction_wheels['max_torque']),
    }
    if config.filter['variant'] == 'model_aided':
        results['inertia_true'] = true_inertia.copy()
        results['inertia_hat'] = np.zeros((n_steps, 3, 3))
    
    # Star tracker update tracking
    for i in range(n_steps):
        t = time[i]
        
        # Get true state
        q_true, omega_true = spacecraft.get_state()
        results['q_true'][i] = q_true
        results['omega_true'][i] = omega_true
        
        # Gyro measurement
        omega_m = gyro.measure(omega_true)
        
        # Star tracker measurement (at lower rate)
        q_meas = None
        q_meas, sensor_valid = star_tracker.measure(q_true, config.dt)
        results['innovation_valid'][i] = sensor_valid
        results['measurement_available'][i] = q_meas is not None
        if q_meas is not None:
            results['measurement_time_s'][i] = t
        results['star_tracker_outage'][i] = star_tracker.in_outage
        
        # The filter state is at epoch t here. Apply measurements from this
        # epoch before recording metrics or computing the control command.
        # Propagate to t + dt only after the current control torque is known.
        if config.filter['variant'] == 'model_aided':
            if i == 0:
                # Initialize rate from the first gyro sample at the initial
                # epoch. Subsequent samples update the propagated rate state.
                mekf.omega_hat = omega_m - mekf.b_hat
                mekf._omega_initialized = True
            else:
                mekf.update_gyro(omega_m)

        innovation = None
        innovation_covariance = None
        nis = None
        mekf.last_measurement_accepted = False
        if q_meas is not None:
            innovation, innovation_covariance, nis = mekf.update(q_meas)
        results['measurement_accepted'][i] = mekf.last_measurement_accepted

        q_hat = mekf.q_hat.copy()
        b_hat = mekf.b_hat.copy()
        covariance = mekf.P.copy()
        results['q_hat'][i] = q_hat
        results['b_hat'][i] = b_hat
        results['P'][i] = covariance
        results['omega_hat'][i] = mekf.get_estimated_omega(omega_m)
        if 'inertia_hat' in results:
            results['inertia_hat'][i] = mekf.I_model
        
        if innovation is not None:
            results['innovation'][i] = innovation
            results['innovation_covariance'][i] = innovation_covariance
            results['NIS'][i] = nis
        
        # Compute NEES
        b_true = gyro.get_true_bias()
        results['b_true'][i] = b_true
        results['bias_error'][i] = b_true - b_hat
        nees = mekf.compute_nees(q_true, b_true, omega_true, true_inertia)
        results['NEES'][i] = nees
        components = mekf.compute_nees_components(
            q_true, b_true, omega_true, true_inertia
        )
        for column, name in enumerate(('attitude', 'rate', 'bias', 'inertia')):
            if name in components:
                results['NEES_components'][i, column] = components[name]
        
        # Controller
        tau_cmd = controller.compute_control(q_hat, results['omega_hat'][i])
        results['tau_cmd'][i] = tau_cmd
        
        # Reaction wheels
        tau_rw = rwa.compute_torque(tau_cmd, config.dt)
        results['tau_rw'][i] = tau_rw
        results['wheel_torque'][i] = rwa.get_torque()
        rwa.update_momentum(config.dt)
        results['wheel_momentum'][i] = rwa.get_momentum()
        
        # Spacecraft dynamics
        spacecraft.step(tau_rw, config.dt)

        # Advance the estimator over [t, t + dt] with the gyro sample and
        # body torque held over the same interval. The next loop then applies
        # measurements at the matching t + dt epoch.
        mekf.propagate(omega_m, tau_rw, config.dt)
        
        # Separate true spacecraft pointing from estimator attitude error.
        pointing_delta = right_error_quat(q_true, controller.q_desired)
        pointing_err = np.linalg.norm(error_quat_to_vec(pointing_delta))
        attitude_delta = right_error_quat(q_true, q_hat)
        attitude_error_vector = error_quat_to_vec(attitude_delta)
        results['pointing_error'][i] = pointing_err
        results['pointing_error_deg'][i] = np.degrees(pointing_err)
        results['attitude_estimation_error'][i] = np.linalg.norm(attitude_error_vector)
        results['attitude_estimation_error_vector'][i] = attitude_error_vector
    
    return results


def compute_performance_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize pointing, estimation, actuator, and covariance performance."""
    time = results['time']
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.0
    pointing = np.asarray(results['pointing_error'])
    initial_error = float(pointing[0])
    settling_time = float('nan')
    if initial_error > 0.0:
        within_tolerance = pointing <= 0.02 * initial_error
        remains_within = np.logical_and.accumulate(within_tolerance[::-1])[::-1]
        settling_indices = np.flatnonzero(remains_within)
        if settling_indices.size:
            settling_time = float(time[settling_indices[0]])
    within_2deg = pointing <= np.deg2rad(2.0)
    remains_within_2deg = np.logical_and.accumulate(within_2deg[::-1])[::-1]
    settling_2deg_indices = np.flatnonzero(remains_within_2deg)
    final_window_start = max(float(time[0]), float(time[-1]) - 20.0)
    final_window = pointing[time >= final_window_start]

    torque = np.linalg.norm(results['tau_cmd'], axis=1)
    achieved_torque = np.linalg.norm(results['tau_rw'], axis=1)
    metrics = {
        'pointing_rms_rad': float(np.sqrt(np.mean(pointing**2))),
        'pointing_peak_rad': float(np.max(pointing)),
        'pointing_final_rad': float(pointing[-1]),
        'settling_time_2pct_s': settling_time,
        'settling_time_2deg_s': (
            float(time[settling_2deg_indices[0]])
            if settling_2deg_indices.size else float('nan')
        ),
        'settling_hold_duration_s': (
            float(time[-1] - settling_time) if np.isfinite(settling_time) else 0.0
        ),
        'pointing_rms_final_20s_rad': float(np.sqrt(np.mean(final_window**2))),
        'pointing_peak_final_20s_rad': float(np.max(final_window)),
        'control_effort_nm2_s': float(np.sum(results['tau_cmd']**2) * dt),
        'achieved_control_effort_nm2_s': float(np.sum(results['tau_rw']**2) * dt),
        'peak_commanded_torque_nm': float(np.max(torque)),
        'peak_achieved_torque_nm': float(np.max(achieved_torque)),
        'max_torque_nm': float(results['max_torque']),
        'command_torque_saturation_fraction': float(
            np.mean(torque >= results['max_torque'] * (1.0 - 1e-9))
        ),
    }
    if 'wheel_momentum' in results:
        wheel_utilization = (
            np.abs(results['wheel_momentum']) / results['max_wheel_momentum']
        )
        metrics['peak_wheel_momentum_utilization'] = float(np.max(wheel_utilization))
        metrics['wheel_momentum_near_limit_fraction'] = float(
            np.mean(wheel_utilization >= 0.95)
        )
    if 'wheel_torque' in results:
        wheel_torque = np.abs(results['wheel_torque'])
        metrics['peak_wheel_torque_nm'] = float(np.max(wheel_torque))
        metrics['max_wheel_torque_nm'] = float(results['max_wheel_torque'])
        metrics['wheel_torque_saturation_fraction'] = float(np.mean(
            wheel_torque >= results['max_wheel_torque'] * (1.0 - 1e-9)
        ))
    if 'omega_true' in results:
        rate_norm = np.linalg.norm(results['omega_true'], axis=1)
        metrics['final_angular_rate_norm_rad_s'] = float(rate_norm[-1])
        metrics['peak_angular_rate_norm_rad_s'] = float(np.max(rate_norm))
    if metrics['control_effort_nm2_s'] > 0.0:
        metrics['achieved_effort_fraction'] = float(
            metrics['achieved_control_effort_nm2_s'] / metrics['control_effort_nm2_s']
        )
    if 'attitude_estimation_error' in results:
        metrics['attitude_estimation_rmse_rad'] = float(
            np.sqrt(np.mean(results['attitude_estimation_error']**2))
        )
    if 'omega_hat' in results and 'omega_true' in results:
        metrics['angular_rate_estimation_rmse_rad_s'] = float(
            np.sqrt(np.mean((results['omega_hat'] - results['omega_true'])**2))
        )
    if 'inertia_hat' in results and 'inertia_true' in results:
        inertia_error = results['inertia_hat'] - results['inertia_true']
        inertia_scale = np.sqrt(
            np.outer(np.diag(results['inertia_true']), np.diag(results['inertia_true']))
        )
        metrics['inertia_relative_rmse'] = float(
            np.sqrt(np.mean((inertia_error / inertia_scale) ** 2))
        )
    if 'bias_error' in results:
        metrics['bias_estimation_rmse_rad_s'] = float(
            np.sqrt(np.mean(results['bias_error']**2))
        )
    if 'attitude_estimation_error_vector' in results and 'P' in results:
        sigma_attitude = 3.0 * np.sqrt(
            np.maximum(np.diagonal(results['P'][:, :3, :3], axis1=1, axis2=2), 0.0)
        )
        metrics['attitude_3sigma_component_coverage'] = float(np.mean(
            np.abs(results['attitude_estimation_error_vector']) <= sigma_attitude
        ))
    if 'bias_error' in results and 'P' in results:
        bias_start = 6 if results['P'].shape[1] == 9 else 3
        bias_covariance = results['P'][
            :, bias_start:bias_start + 3, bias_start:bias_start + 3
        ]
        sigma_bias = 3.0 * np.sqrt(
            np.maximum(np.diagonal(bias_covariance, axis1=1, axis2=2), 0.0)
        )
        metrics['bias_3sigma_component_coverage'] = float(np.mean(
            np.abs(results['bias_error']) <= sigma_bias
        ))
    if (
        'innovation' in results
        and 'innovation_covariance' in results
        and 'measurement_available' in results
    ):
        accepted = results.get(
            'measurement_accepted', np.ones_like(results['measurement_available'])
        )
        sensor_valid = results.get(
            'innovation_valid', np.ones_like(results['measurement_available'])
        )
        valid_indices = np.flatnonzero(
            results['measurement_available'] & accepted & sensor_valid
        )
        whitened = []
        for index in valid_indices:
            try:
                lower = np.linalg.cholesky(results['innovation_covariance'][index])
                whitened.append(np.linalg.solve(lower, results['innovation'][index]))
            except np.linalg.LinAlgError:
                continue
        if len(whitened) >= 20:
            whitened = np.asarray(whitened)
            centered = whitened - np.mean(whitened, axis=0)
            denominator = np.sum(centered**2, axis=0)
            lag1_denominator = np.sqrt(denominator**2)
            lag1_numerator = np.sum(centered[:-1] * centered[1:], axis=0)
            lag1 = np.divide(
                lag1_numerator,
                lag1_denominator,
                out=np.zeros(3),
                where=lag1_denominator > 0.0,
            )
            metrics['innovation_whitened_lag1_autocorrelation_rms'] = float(
                np.sqrt(np.mean(lag1**2))
            )
            statistic, p_value, n_lags = multivariate_ljung_box(whitened)
            metrics['innovation_whiteness_ljung_box_q'] = statistic
            metrics['innovation_whiteness_ljung_box_pvalue'] = p_value
            metrics['innovation_whiteness_lags'] = n_lags
            metrics['innovation_whiteness_pass_95'] = float(p_value >= 0.05)
    if 'star_tracker_outage' in results and np.any(results['star_tracker_outage']):
        outage_indices = np.flatnonzero(results['star_tracker_outage'])
        outage_end_index = int(outage_indices[-1])
        pre_outage_error = pointing[:outage_indices[0]]
        if pre_outage_error.size and float(np.max(pre_outage_error)) > 0.0:
            threshold = 0.02 * float(np.max(pre_outage_error))
            after_outage = pointing[outage_end_index + 1:]
            remains_within = np.logical_and.accumulate((after_outage <= threshold)[::-1])[::-1]
            recovery_indices = np.flatnonzero(remains_within)
            metrics['outage_recovery_time_s'] = (
                float(time[outage_end_index + 1 + recovery_indices[0]] - time[outage_end_index])
                if recovery_indices.size
                else float('nan')
            )
    return metrics


def multivariate_ljung_box(
    innovations: np.ndarray, max_lags: int = 10
) -> tuple[float, float, int]:
    """Approximate multivariate Ljung-Box whiteness test for vector innovations."""
    from scipy.stats import chi2

    samples = np.asarray(innovations, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 20 or samples.shape[1] < 1:
        return float('nan'), float('nan'), 0
    count, dimensions = samples.shape
    n_lags = min(int(max_lags), count // 5)
    if n_lags < 1:
        return float('nan'), float('nan'), 0
    centered = samples - np.mean(samples, axis=0)
    covariance = centered.T @ centered / count
    inverse_covariance = np.linalg.pinv(covariance)
    statistic = 0.0
    for lag in range(1, n_lags + 1):
        lag_covariance = centered[lag:].T @ centered[:-lag] / (count - lag)
        quadratic = np.trace(
            inverse_covariance
            @ lag_covariance
            @ inverse_covariance
            @ lag_covariance.T
        )
        statistic += count * (count + 2.0) * quadratic / (count - lag)
    degrees_of_freedom = dimensions**2 * n_lags
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return float(statistic), p_value, n_lags


def evaluate_e6_project_benchmark(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the explicitly project-level E6 closed-loop benchmark."""
    limits = {
        'settling_time_2pct_s': ('max', 180.0),
        'settling_hold_duration_s': ('min', 20.0),
        'pointing_peak_final_20s_rad': ('max', float(np.deg2rad(2.0))),
        'final_angular_rate_norm_rad_s': ('max', 1e-3),
        'peak_wheel_momentum_utilization': ('max', 0.8),
        'achieved_effort_fraction': ('min', 0.85),
        'peak_commanded_torque_nm': ('max', metrics.get('max_torque_nm', 0.1)),
        'peak_wheel_torque_nm': ('max', metrics.get('max_wheel_torque_nm', 0.1)),
    }
    checks = {}
    for name, (direction, limit) in limits.items():
        value = float(metrics.get(name, float('nan')))
        passed = np.isfinite(value) and (
            value <= limit + 1e-12 if direction == 'max' else value >= limit - 1e-12
        )
        checks[name] = {
            'measured': value if np.isfinite(value) else None,
            'operator': '<=' if direction == 'max' else '>=',
            'limit': limit,
            'passed': bool(passed),
        }
    return {'label': 'project benchmark; not a mission requirement',
            'passed': all(item['passed'] for item in checks.values()),
            'checks': checks}


def summarize_performance(
    performance_by_run: list[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute run-level means, standard deviations, and 95% t intervals."""
    from scipy import stats

    summary: Dict[str, Dict[str, Any]] = {}
    metric_names = performance_by_run[0].keys() if performance_by_run else ()
    for name in metric_names:
        values = np.asarray([run.get(name, np.nan) for run in performance_by_run])
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        half_width = (
            float(stats.t.ppf(0.975, values.size - 1) * std / np.sqrt(values.size))
            if values.size > 1
            else float('nan')
        )
        summary[name] = {
            'mean': mean,
            'std': std,
            'ci95_lower': mean - half_width if values.size > 1 else None,
            'ci95_upper': mean + half_width if values.size > 1 else None,
            'n': int(values.size),
        }
    return summary


def run_monte_carlo(config: SimulationConfig, n_runs: int, 
                    true_inertia: Optional[np.ndarray] = None,
                    filter_inertia: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation.
    
    Args:
        config: Base configuration
        n_runs: Number of Monte Carlo runs
        true_inertia: True inertia (can vary per run)
        filter_inertia: Filter inertia (can vary per run)
        
    Returns:
        Aggregated results
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    compact_runs = []
    performance_by_run = []
    accumulators: Dict[str, np.ndarray] = {}
    squared_accumulators: Dict[str, np.ndarray] = {}
    n_steps = int(config.duration / config.dt)
    time = np.arange(n_steps) * config.dt

    for run in range(n_runs):
        # Modify seed for each run
        run_config = SimulationConfig(
            duration=config.duration,
            dt=config.dt,
            seed=config.seed + run * 1000,
            spacecraft=config.spacecraft,
            gyroscope=config.gyroscope,
            star_tracker=config.star_tracker,
            filter=config.filter,
            controller=config.controller,
            reaction_wheels=config.reaction_wheels,
            analysis=config.analysis
        )
        
        result = run_simulation(run_config, true_inertia, filter_inertia)
        if run == 0:
            time = result['time']
        compact_runs.append({
            key: result[key]
            for key in (
                'NEES', 'NEES_components', 'NIS',
                'measurement_available', 'measurement_accepted'
            )
        })
        performance_by_run.append(compute_performance_metrics(result))

        mean_fields = {
            'b_hat_mean': 'b_hat',
            'b_true_mean': 'b_true',
            'P_mean': 'P',
            'NEES_mean': 'NEES',
            'pointing_error_mean': 'pointing_error',
            'attitude_estimation_error_mean': 'attitude_estimation_error',
        }
        for aggregate_key, run_key in mean_fields.items():
            values = result[run_key]
            accumulators[aggregate_key] = accumulators.get(
                aggregate_key, np.zeros_like(values, dtype=float)
            ) + values
            squared_accumulators[aggregate_key] = squared_accumulators.get(
                aggregate_key, np.zeros_like(values, dtype=float)
            ) + values**2
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
    
    # Aggregate time histories while retaining only consistency samples per run.
    means = {key: value / n_runs for key, value in accumulators.items()}
    stds = {
        key: np.sqrt(np.maximum(
            squared_accumulators[key] / n_runs - means[key]**2, 0.0
        ))
        for key in accumulators
    }
    aggregated = {
        'time': time,
        'nees_dof': 15 if config.filter['variant'] == 'model_aided' else 6,
        **means,
        'NEES_std': stds['NEES_mean'],
        'pointing_error_std': stds['pointing_error_mean'],
        'attitude_estimation_error_std': stds['attitude_estimation_error_mean'],
        'NIS_mean': np.zeros(n_steps),
        'NIS_std': np.zeros(n_steps),
        'innovation_count': np.zeros(n_steps, dtype=int),
        'measurement_rejection_count': np.zeros(n_steps, dtype=int),
        'max_torque': float(config.controller['max_torque']),
        'is_monte_carlo': True,
        'n_runs': n_runs,
        'all_runs': compact_runs,
        'performance_by_run': performance_by_run,
        'performance_summary': summarize_performance(performance_by_run),
    }
    nis_values = np.stack([r['NIS'] for r in compact_runs])
    nis_valid = np.stack([r['measurement_available'] for r in compact_runs])
    nis_accepted = np.stack([r['measurement_accepted'] for r in compact_runs])
    aggregated['innovation_count'] = np.sum(nis_valid, axis=0)
    aggregated['measurement_rejection_count'] = np.sum(nis_valid & ~nis_accepted, axis=0)
    nis_sum = np.sum(np.where(nis_valid, nis_values, 0.0), axis=0)
    aggregated['NIS_mean'] = np.divide(
        nis_sum,
        aggregated['innovation_count'],
        out=np.zeros(n_steps),
        where=aggregated['innovation_count'] > 0,
    )
    nis_squared_sum = np.sum(np.where(nis_valid, nis_values**2, 0.0), axis=0)
    nis_variance = np.divide(
        nis_squared_sum,
        aggregated['innovation_count'],
        out=np.zeros(n_steps),
        where=aggregated['innovation_count'] > 0,
    ) - aggregated['NIS_mean']**2
    aggregated['NIS_std'] = np.sqrt(np.maximum(nis_variance, 0.0))
    component_values = np.stack([r['NEES_components'] for r in compact_runs])
    component_counts = np.sum(np.isfinite(component_values), axis=0)
    aggregated['NEES_components_mean'] = np.divide(
        np.nansum(component_values, axis=0),
        component_counts,
        out=np.full(component_values.shape[1:], np.nan),
        where=component_counts > 0,
    )
    
    return aggregated


def compute_consistency_stats(
    results: Dict[str, Any],
    confidence: float = 0.95,
    nis_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute consistency statistics (NEES/NIS bounds).
    
    Args:
        results: Simulation results (single run or Monte Carlo)
        confidence: Confidence level
        
    Returns:
        Dictionary with consistency statistics
    """
    from scipy import stats

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if nis_confidence is None:
        nis_confidence = confidence
    if not 0.0 < nis_confidence < 1.0:
        raise ValueError("nis_confidence must be between 0 and 1")
    
    is_monte_carlo = 'all_runs' in results
    nees_component_summary = {}
    all_nees: Optional[np.ndarray] = None
    all_nis: Optional[np.ndarray] = None
    all_nis_valid: Optional[np.ndarray] = None
    nis_counts = np.zeros_like(results.get('NEES_mean', results.get('NEES')), dtype=int)
    nis_valid = np.zeros_like(nis_counts, dtype=bool)
    if is_monte_carlo:
        all_nees = np.stack([run['NEES'] for run in results['all_runs']])
        all_nis = np.stack([run['NIS'] for run in results['all_runs']])
        all_nis_valid = np.stack([
            run['measurement_available'] for run in results['all_runs']
        ])
        all_nis_accepted = np.stack([
            run['measurement_available'] & run['measurement_accepted']
            for run in results['all_runs']
        ])
        if results['all_runs'] and 'NEES_components' in results['all_runs'][0]:
            names = ('attitude', 'rate', 'bias', 'inertia')
            component_values = np.stack([
                run['NEES_components'] for run in results['all_runs']
            ])
            for column, name in enumerate(names):
                values = component_values[:, :, column]
                valid_values = values[np.isfinite(values)]
                run_means = np.asarray([
                    np.nanmean(row) if np.any(np.isfinite(row)) else np.nan
                    for row in values
                ])
                run_means = run_means[np.isfinite(run_means)]
                if valid_values.size:
                    component_mean = float(np.mean(valid_values))
                    if run_means.size > 1:
                        half = float(
                            stats.t.ppf(0.975, run_means.size - 1)
                            * np.std(run_means, ddof=1) / np.sqrt(run_means.size)
                        )
                    else:
                        half = float('nan')
                    nees_component_summary[name] = {
                        'mean': component_mean,
                        'dof': 6 if name == 'inertia' else 3,
                        'run_mean_ci95_lower': component_mean - half,
                        'run_mean_ci95_upper': component_mean + half,
                        'n_runs': int(run_means.size),
                    }
        NEES = results['NEES_mean']
        nis_counts = np.sum(all_nis_valid, axis=0)
        NIS = np.divide(
            np.sum(np.where(all_nis_valid, all_nis, 0.0), axis=0),
            nis_counts,
            out=np.zeros_like(NEES),
            where=nis_counts > 0,
        )
    else:
        NEES = results['NEES']
        NIS = results['NIS']
        nis_valid = results.get(
            'measurement_available',
            results.get('innovation_valid', NIS > 0),
        )
        nis_accepted = nis_valid & results.get(
            'measurement_accepted', np.ones_like(nis_valid, dtype=bool)
        )
        if 'NEES_components' in results:
            for column, name in enumerate(('attitude', 'rate', 'bias', 'inertia')):
                values = results['NEES_components'][:, column]
                values = values[np.isfinite(values)]
                if values.size:
                    nees_component_summary[name] = {
                        'mean': float(np.mean(values)),
                        'dof': 6 if name == 'inertia' else 3,
                    }
    
    # Degrees of freedom
    dof_nees = int(results.get('nees_dof', 6))
    dof_nis = 3   # 3 attitude measurement
    
    # Chi-squared bounds
    alpha_nees = 1.0 - confidence
    alpha_nis = 1.0 - nis_confidence
    nees_lower = stats.chi2.ppf(alpha_nees / 2, dof_nees)
    nees_upper = stats.chi2.ppf(1 - alpha_nees / 2, dof_nees)
    nis_lower = stats.chi2.ppf(alpha_nis / 2, dof_nis)
    nis_upper = stats.chi2.ppf(1 - alpha_nis / 2, dof_nis)
    
    # For Monte Carlo, average NEES over runs
    if is_monte_carlo:
        assert (
            all_nees is not None
            and all_nis is not None
            and all_nis_valid is not None
            and all_nis_accepted is not None
        )
        n_runs = len(results['all_runs'])
        nees_avg = np.mean(all_nees, axis=0)
        nis_avg = NIS
        accepted_values = all_nis[all_nis_accepted]
        nis_mean_accepted = float(np.mean(accepted_values)) if accepted_values.size else float('nan')
        accepted_epoch_counts = np.sum(all_nis_accepted, axis=0)
        accepted_sum = np.sum(np.where(all_nis_accepted, all_nis, 0.0), axis=0)
        nis_avg_accepted = np.divide(
            accepted_sum, accepted_epoch_counts,
            out=np.zeros_like(NIS), where=accepted_epoch_counts > 0
        )
        
        # Bounds for average over N runs
        nees_lower_mc = stats.chi2.ppf(
            alpha_nees / 2, n_runs * dof_nees
        ) / n_runs
        nees_upper_mc = stats.chi2.ppf(
            1 - alpha_nees / 2, n_runs * dof_nees
        ) / n_runs
        nis_lower_mc = np.full(nis_counts.shape, np.nan, dtype=float)
        nis_upper_mc = np.full(nis_counts.shape, np.nan, dtype=float)
        valid_nis_epochs = nis_counts > 0
        nis_lower_mc[valid_nis_epochs] = (
            stats.chi2.ppf(
                alpha_nis / 2, nis_counts[valid_nis_epochs] * dof_nis
            ) / nis_counts[valid_nis_epochs]
        )
        nis_upper_mc[valid_nis_epochs] = (
            stats.chi2.ppf(
                1 - alpha_nis / 2, nis_counts[valid_nis_epochs] * dof_nis
            ) / nis_counts[valid_nis_epochs]
        )
        
        nees_in_bounds = np.mean((nees_avg >= nees_lower_mc) & (nees_avg <= nees_upper_mc))
        nis_in_bounds = (
            np.mean(
                (nis_avg[valid_nis_epochs] >= nis_lower_mc[valid_nis_epochs])
                & (nis_avg[valid_nis_epochs] <= nis_upper_mc[valid_nis_epochs])
            )
            if np.any(valid_nis_epochs)
            else float('nan')
        )
        accepted_valid_epochs = accepted_epoch_counts > 0
        nis_accepted_in_bounds = (
            np.mean(
                (nis_avg_accepted[accepted_valid_epochs] >=
                 stats.chi2.ppf(alpha_nis / 2, accepted_epoch_counts[accepted_valid_epochs] * dof_nis) /
                 accepted_epoch_counts[accepted_valid_epochs])
                & (nis_avg_accepted[accepted_valid_epochs] <=
                   stats.chi2.ppf(1 - alpha_nis / 2, accepted_epoch_counts[accepted_valid_epochs] * dof_nis) /
                   accepted_epoch_counts[accepted_valid_epochs])
            )
            if np.any(accepted_valid_epochs)
            else float('nan')
        )
    else:
        nees_lower_mc = nees_lower
        nees_upper_mc = nees_upper
        nis_lower_mc = nis_lower
        nis_upper_mc = nis_upper
        nees_in_bounds = np.mean((NEES >= nees_lower) & (NEES <= nees_upper))
        nis_in_bounds = (
            np.mean((NIS[nis_valid] >= nis_lower) & (NIS[nis_valid] <= nis_upper))
            if np.any(nis_valid) else float('nan')
        )
        accepted_values = NIS[nis_accepted]
        nis_mean_accepted = float(np.mean(accepted_values)) if accepted_values.size else float('nan')
        nis_accepted_in_bounds = (
            np.mean((accepted_values >= nis_lower) & (accepted_values <= nis_upper))
            if accepted_values.size else float('nan')
        )
    
    return {
        'nees_bounds': (nees_lower, nees_upper),
        'nis_bounds': (nis_lower, nis_upper),
        'nees_bounds_mc': (nees_lower_mc, nees_upper_mc),
        'nis_bounds_mc': (nis_lower_mc, nis_upper_mc),
        'nees_in_bounds': nees_in_bounds,
        'nis_in_bounds': nis_in_bounds,
        'nees_mean': np.mean(all_nees) if all_nees is not None else np.mean(NEES),
        'nis_mean': (
            np.mean(all_nis[all_nis_valid])
            if all_nis is not None and all_nis_valid is not None
            else np.mean(NIS[nis_valid]) if np.any(nis_valid) else float('nan')
        ),
        'nis_mean_accepted': nis_mean_accepted,
        'nis_accepted_in_bounds': nis_accepted_in_bounds,
        'dof_nees': dof_nees,
        'dof_nis': dof_nis,
        'nees_component_summary': nees_component_summary,
    }


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to NPZ file."""
    # Convert arrays to saveable format
    save_dict = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            save_dict[key] = np.array(value)
        elif key not in ('all_runs', 'performance_by_run'):
            save_dict[key] = value
    
    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run spacecraft attitude estimation simulation')
    parser.add_argument('--config', type=str, default='config/nominal.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='results/nominal.npz',
                        help='Output file path')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo simulation')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of Monte Carlo runs')
    parser.add_argument('--inertia-mismatch', type=float, default=0.0,
                        help='Relative inertia mismatch (e.g., 0.1 for 10%)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply inertia mismatch if specified
    true_inertia = None
    filter_inertia = None
    if args.inertia_mismatch > 0:
        I_nominal = np.array(config.spacecraft['inertia'])
        # Add diagonal mismatch
        mismatch = np.diag(np.diag(I_nominal)) * args.inertia_mismatch
        true_inertia = I_nominal + mismatch
        filter_inertia = I_nominal  # Filter uses nominal
        print(f"Inertia mismatch: {args.inertia_mismatch*100:.1f}%")
        print(f"True inertia diag: {np.diag(true_inertia)}")
        print(f"Filter inertia diag: {np.diag(filter_inertia)}")
    
    # Run simulation
    if args.monte_carlo:
        print(f"Running Monte Carlo with {args.runs} runs...")
        results = run_monte_carlo(config, args.runs, true_inertia, filter_inertia)
    else:
        print("Running single simulation...")
        results = run_simulation(config, true_inertia, filter_inertia)
    
    # Compute consistency stats
    stats = compute_consistency_stats(
        results,
        config.analysis['nees_confidence'],
        config.analysis['nis_confidence'],
    )
    results['consistency'] = stats
    if not args.monte_carlo:
        results['performance_summary'] = summarize_performance([
            compute_performance_metrics(results)
        ])
    
    print(f"\nConsistency Statistics:")
    print(f"  NEES mean: {stats['nees_mean']:.3f}")
    print(f"  NEES in bounds: {stats['nees_in_bounds']*100:.1f}%")
    print(f"  NIS mean: {stats['nis_mean']:.3f}")
    print(f"  NIS in bounds: {stats['nis_in_bounds']*100:.1f}%")
    print(f"  NEES bounds (single): [{stats['nees_bounds'][0]:.3f}, {stats['nees_bounds'][1]:.3f}]")
    print(f"  NIS bounds (single): [{stats['nis_bounds'][0]:.3f}, {stats['nis_bounds'][1]:.3f}]")
    
    # Pointing error stats
    performance = results['performance_summary']['pointing_rms_rad']
    peak_pointing = results['performance_summary']['pointing_peak_rad']
    print("\nSpacecraft Pointing Performance:")
    print(f"  Run-mean RMS: {np.degrees(performance['mean']):.3f} deg")
    print(f"  Run-mean peak: {np.degrees(peak_pointing['mean']):.3f} deg")
    if 'attitude_estimation_rmse_rad' in results['performance_summary']:
        estimate_rmse = results['performance_summary']['attitude_estimation_rmse_rad']
        print(f"  Attitude-estimation RMSE: {np.degrees(estimate_rmse['mean']):.3f} deg")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
