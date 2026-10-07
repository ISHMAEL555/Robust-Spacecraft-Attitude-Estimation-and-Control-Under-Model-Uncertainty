#!/usr/bin/env python3
"""
Main simulation script for spacecraft attitude estimation and control.

Runs experiments defined in config files.
"""

import numpy as np
import yaml
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.dynamics.quaternion import normalize, multiply, conjugate, error_quat, error_quat_to_vec, quat_to_rotmat
from src.dynamics.rigid_body import RigidBody, inertia_matrix_from_principal
from src.sensors.gyro import Gyroscope, GyroParams
from src.sensors.star_tracker import StarTracker, StarTrackerParams
from src.estimation.mekf import MEKF, MEKFParams, create_mekf_nominal
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
    return SimulationConfig(**config_dict)


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
    
    # Time array
    n_steps = int(config.duration / config.dt)
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
        outage_prob=config.star_tracker['outage_prob']
    )
    star_tracker = StarTracker(st_params, seed=config.seed + 2)
    
    # Filter
    filter_params = MEKFParams(
        variant=config.filter['variant'],
        P0_attitude=config.filter['P0_attitude'],
        P0_bias=config.filter['P0_bias'],
        Q_scale=config.filter['Q_scale'],
        R_scale=config.filter['R_scale'],
        inertia_model=filter_inertia
    )
    filter_params.gyro_params = gyro_params
    filter_params.star_tracker_params = st_params
    mekf = MEKF(filter_params)
    
    # Initialize filter with true state + small error
    q_init = spacecraft.q.copy()
    # Add small initial error
    delta_init = np.random.randn(3) * 0.01
    from src.dynamics.quaternion import box_plus
    q_init = box_plus(q_init, delta_init)
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
        'omega_hat': np.zeros((n_steps, 3)),
        'P': np.zeros((n_steps, 6, 6)),
        'tau_cmd': np.zeros((n_steps, 3)),
        'tau_rw': np.zeros((n_steps, 3)),
        'NEES': np.zeros(n_steps),
        'NIS': np.zeros(n_steps),
        'innovation': np.zeros((n_steps, 3)),
        'innovation_valid': np.zeros(n_steps, dtype=bool),
        'pointing_error': np.zeros(n_steps),
        'pointing_error_deg': np.zeros(n_steps),
    }
    
    # Star tracker update tracking
    st_counter = 0
    st_interval = int(1.0 / (st_params.rate * config.dt))
    
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
        if i % st_interval == 0:
            q_meas, valid = star_tracker.measure(q_true, config.dt)
            results['innovation_valid'][i] = valid
            if q_meas is not None:
                # Store for filter update
                pass
        
        # Filter step
        tau_cmd_prev = results['tau_cmd'][i-1] if i > 0 else np.zeros(3)
        filter_result = mekf.step(
            omega_m, q_meas, tau_cmd_prev, config.dt, t
        )
        
        results['q_hat'][i] = filter_result['q_hat']
        results['b_hat'][i] = filter_result['b_hat']
        results['P'][i] = filter_result['P']
        results['omega_hat'][i] = mekf.get_estimated_omega(omega_m)
        
        if filter_result['innovation'] is not None:
            results['innovation'][i] = filter_result['innovation']
            results['NIS'][i] = filter_result['NIS']
        
        # Compute NEES
        b_true = gyro.get_true_bias()
        nees = mekf.compute_nees(q_true, b_true)
        results['NEES'][i] = nees
        
        # Controller
        tau_cmd = controller.compute_control(filter_result['q_hat'], results['omega_hat'][i])
        results['tau_cmd'][i] = tau_cmd
        
        # Reaction wheels
        tau_rw = rwa.compute_torque(tau_cmd)
        results['tau_rw'][i] = tau_rw
        rwa.update_momentum(config.dt)
        
        # Spacecraft dynamics
        spacecraft.step(tau_rw, config.dt)
        
        # Pointing error
        q_error = error_quat(q_true, filter_result['q_hat'])
        delta_theta = error_quat_to_vec(q_error)
        pointing_err = np.linalg.norm(delta_theta)
        results['pointing_error'][i] = pointing_err
        results['pointing_error_deg'][i] = np.degrees(pointing_err)
    
    # Convert history to arrays
    for key in ['q_hat', 'b_hat', 'P', 'NEES', 'NIS', 'innovation']:
        if key in mekf.history and len(mekf.history[key]) > 0:
            results[f'filter_{key}'] = np.array(mekf.history[key])
    
    return results


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
    all_results = []
    
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
        all_results.append(result)
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
    
    # Aggregate statistics
    n_steps = int(config.duration / config.dt)
    aggregated = {
        'time': all_results[0]['time'],
        'q_true_mean': np.mean([r['q_true'] for r in all_results], axis=0),
        'q_true_std': np.std([r['q_true'] for r in all_results], axis=0),
        'q_hat_mean': np.mean([r['q_hat'] for r in all_results], axis=0),
        'q_hat_std': np.std([r['q_hat'] for r in all_results], axis=0),
        'b_hat_mean': np.mean([r['b_hat'] for r in all_results], axis=0),
        'b_hat_std': np.std([r['b_hat'] for r in all_results], axis=0),
        'NEES_mean': np.mean([r['NEES'] for r in all_results], axis=0),
        'NEES_std': np.std([r['NEES'] for r in all_results], axis=0),
        'NIS_mean': np.mean([r['NIS'] for r in all_results], axis=0),
        'NIS_std': np.std([r['NIS'] for r in all_results], axis=0),
        'pointing_error_mean': np.mean([r['pointing_error'] for r in all_results], axis=0),
        'pointing_error_std': np.std([r['pointing_error'] for r in all_results], axis=0),
        'all_runs': all_results
    }
    
    return aggregated


def compute_consistency_stats(results: Dict[str, Any], confidence: float = 0.95) -> Dict[str, Any]:
    """
    Compute consistency statistics (NEES/NIS bounds).
    
    Args:
        results: Simulation results (single run or Monte Carlo)
        confidence: Confidence level
        
    Returns:
        Dictionary with consistency statistics
    """
    from scipy import stats
    
    NEES = results['NEES']
    NIS = results['NIS']
    
    # Degrees of freedom
    dof_nees = 6  # 3 attitude + 3 bias
    dof_nis = 3   # 3 attitude measurement
    
    # Chi-squared bounds
    alpha = 1.0 - confidence
    nees_lower = stats.chi2.ppf(alpha / 2, dof_nees) / dof_nees
    nees_upper = stats.chi2.ppf(1 - alpha / 2, dof_nees) / dof_nees
    nis_lower = stats.chi2.ppf(alpha / 2, dof_nis) / dof_nis
    nis_upper = stats.chi2.ppf(1 - alpha / 2, dof_nis) / dof_nis
    
    # For Monte Carlo, average NEES over runs
    if 'all_runs' in results:
        n_runs = len(results['all_runs'])
        nees_avg = results['NEES_mean']
        nis_avg = results['NIS_mean']
        
        # Bounds for average over N runs
        nees_lower_mc = stats.chi2.ppf(alpha / 2, n_runs * dof_nees) / (n_runs * dof_nees)
        nees_upper_mc = stats.chi2.ppf(1 - alpha / 2, n_runs * dof_nees) / (n_runs * dof_nees)
        nis_lower_mc = stats.chi2.ppf(alpha / 2, n_runs * dof_nis) / (n_runs * dof_nis)
        nis_upper_mc = stats.chi2.ppf(1 - alpha / 2, n_runs * dof_nis) / (n_runs * dof_nis)
        
        nees_in_bounds = np.mean((nees_avg >= nees_lower_mc) & (nees_avg <= nees_upper_mc))
        nis_in_bounds = np.mean((nis_avg >= nis_lower_mc) & (nis_avg <= nis_upper_mc))
    else:
        nees_lower_mc = nees_lower
        nees_upper_mc = nees_upper
        nis_lower_mc = nis_lower
        nis_upper_mc = nis_upper
        nees_in_bounds = np.mean((NEES >= nees_lower) & (NEES <= nees_upper))
        nis_in_bounds = np.mean((NIS >= nis_lower) & (NIS <= nis_upper))
    
    return {
        'nees_bounds': (nees_lower, nees_upper),
        'nis_bounds': (nis_lower, nis_upper),
        'nees_bounds_mc': (nees_lower_mc, nees_upper_mc),
        'nis_bounds_mc': (nis_lower_mc, nis_upper_mc),
        'nees_in_bounds': nees_in_bounds,
        'nis_in_bounds': nis_in_bounds,
        'nees_mean': np.mean(NEES),
        'nis_mean': np.mean(NIS[NIS > 0]),
        'dof_nees': dof_nees,
        'dof_nis': dof_nis
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
        elif key != 'all_runs':  # Skip large nested structures
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
    stats = compute_consistency_stats(results, config.analysis['nees_confidence'])
    results['consistency'] = stats
    
    print(f"\nConsistency Statistics:")
    print(f"  NEES mean: {stats['nees_mean']:.3f}")
    print(f"  NEES in bounds: {stats['nees_in_bounds']*100:.1f}%")
    print(f"  NIS mean: {stats['nis_mean']:.3f}")
    print(f"  NIS in bounds: {stats['nis_in_bounds']*100:.1f}%")
    print(f"  NEES bounds (single): [{stats['nees_bounds'][0]:.3f}, {stats['nees_bounds'][1]:.3f}]")
    print(f"  NIS bounds (single): [{stats['nis_bounds'][0]:.3f}, {stats['nis_bounds'][1]:.3f}]")
    
    # Pointing error stats
    if 'pointing_error_mean' in results:
        print(f"\nPointing Error:")
        print(f"  Mean: {np.mean(results['pointing_error_mean'])*180/np.pi:.3f} deg")
        print(f"  RMS: {np.sqrt(np.mean(results['pointing_error_mean']**2))*180/np.pi:.3f} deg")
        print(f"  Peak: {np.max(results['pointing_error_mean'])*180/np.pi:.3f} deg")
    else:
        print(f"\nPointing Error:")
        print(f"  Mean: {np.mean(results['pointing_error'])*180/np.pi:.3f} deg")
        print(f"  RMS: {np.sqrt(np.mean(results['pointing_error']**2))*180/np.pi:.3f} deg")
        print(f"  Peak: {np.max(results['pointing_error'])*180/np.pi:.3f} deg")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()