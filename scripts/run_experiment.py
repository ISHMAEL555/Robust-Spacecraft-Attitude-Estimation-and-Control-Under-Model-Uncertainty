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

from src.dynamics.quaternion import error_quat, error_quat_to_vec
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
        nis_gate_confidence=config.filter.get('nis_gate_confidence', 0.999999),
        nis_gate_enabled=config.filter.get('nis_gate_enabled', False),
        inertia_model=(
            filter_inertia
            if filter_inertia is not None
            else np.array(config.spacecraft['inertia'])
            if config.filter['variant'] == 'model_aided'
            else None
        )
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
        'b_true': np.zeros((n_steps, 3)),
        'omega_hat': np.zeros((n_steps, 3)),
        'P': np.zeros((n_steps, 6, 6)),
        'tau_cmd': np.zeros((n_steps, 3)),
        'tau_rw': np.zeros((n_steps, 3)),
        'NEES': np.zeros(n_steps),
        'NIS': np.zeros(n_steps),
        'innovation': np.zeros((n_steps, 3)),
        'innovation_valid': np.zeros(n_steps, dtype=bool),
        'measurement_available': np.zeros(n_steps, dtype=bool),
        'measurement_accepted': np.zeros(n_steps, dtype=bool),
        'pointing_error': np.zeros(n_steps),
        'pointing_error_deg': np.zeros(n_steps),
    }
    
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
        
        # Filter step
        tau_cmd_prev = results['tau_cmd'][i-1] if i > 0 else np.zeros(3)
        filter_result = mekf.step(
            omega_m, q_meas, tau_cmd_prev, config.dt, t
        )
        results['measurement_accepted'][i] = filter_result['measurement_accepted']
        
        results['q_hat'][i] = filter_result['q_hat']
        results['b_hat'][i] = filter_result['b_hat']
        results['P'][i] = filter_result['P']
        results['omega_hat'][i] = mekf.get_estimated_omega(omega_m)
        
        if filter_result['innovation'] is not None:
            results['innovation'][i] = filter_result['innovation']
            results['NIS'][i] = filter_result['NIS']
        
        # Compute NEES
        b_true = gyro.get_true_bias()
        results['b_true'][i] = b_true
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
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

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
        'b_true_mean': np.mean([r['b_true'] for r in all_results], axis=0),
        'P_mean': np.mean([r['P'] for r in all_results], axis=0),
        'NEES_mean': np.mean([r['NEES'] for r in all_results], axis=0),
        'NEES_std': np.std([r['NEES'] for r in all_results], axis=0),
        'NIS_mean': np.zeros(n_steps),
        'NIS_std': np.zeros(n_steps),
        'innovation_count': np.zeros(n_steps, dtype=int),
        'measurement_rejection_count': np.zeros(n_steps, dtype=int),
        'pointing_error_mean': np.mean([r['pointing_error'] for r in all_results], axis=0),
        'pointing_error_std': np.std([r['pointing_error'] for r in all_results], axis=0),
        'is_monte_carlo': True,
        'n_runs': n_runs,
        'all_runs': all_results
    }
    nis_values = np.stack([r['NIS'] for r in all_results])
    nis_valid = np.stack([r['measurement_available'] for r in all_results])
    nis_accepted = np.stack([r['measurement_accepted'] for r in all_results])
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
    
    is_monte_carlo = 'all_runs' in results
    if is_monte_carlo:
        all_nees = np.stack([run['NEES'] for run in results['all_runs']])
        all_nis = np.stack([run['NIS'] for run in results['all_runs']])
        all_        nis_valid = np.stack([run['measurement_available'] for run in results['all_runs']])
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
    
    # Degrees of freedom
    dof_nees = 6  # 3 attitude + 3 bias
    dof_nis = 3   # 3 attitude measurement
    
    # Chi-squared bounds
    alpha = 1.0 - confidence
    nees_lower = stats.chi2.ppf(alpha / 2, dof_nees)
    nees_upper = stats.chi2.ppf(1 - alpha / 2, dof_nees)
    nis_lower = stats.chi2.ppf(alpha / 2, dof_nis)
    nis_upper = stats.chi2.ppf(1 - alpha / 2, dof_nis)
    
    # For Monte Carlo, average NEES over runs
    if is_monte_carlo:
        n_runs = len(results['all_runs'])
        nees_avg = np.mean(all_nees, axis=0)
        nis_avg = NIS
        
        # Bounds for average over N runs
        nees_lower_mc = stats.chi2.ppf(alpha / 2, n_runs * dof_nees) / n_runs
        nees_upper_mc = stats.chi2.ppf(1 - alpha / 2, n_runs * dof_nees) / n_runs
        nis_lower_mc = stats.chi2.ppf(alpha / 2, n_runs * dof_nis) / n_runs
        nis_upper_mc = stats.chi2.ppf(1 - alpha / 2, n_runs * dof_nis) / n_runs
        
        nees_in_bounds = np.mean((nees_avg >= nees_lower_mc) & (nees_avg <= nees_upper_mc))
        valid_nis_epochs = nis_counts > 0
        nis_in_bounds = (
            np.mean(
                (nis_avg[valid_nis_epochs] >= nis_lower_mc)
                & (nis_avg[valid_nis_epochs] <= nis_upper_mc)
            )
            if np.any(valid_nis_epochs)
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
            if np.any(nis_valid)
            else float('nan')
        )
    
    return {
        'nees_bounds': (nees_lower, nees_upper),
        'nis_bounds': (nis_lower, nis_upper),
        'nees_bounds_mc': (nees_lower_mc, nees_upper_mc),
        'nis_bounds_mc': (nis_lower_mc, nis_upper_mc),
        'nees_in_bounds': nees_in_bounds,
        'nis_in_bounds': nis_in_bounds,
        'nees_mean': np.mean(all_nees) if is_monte_carlo else np.mean(NEES),
        'nis_mean': (
            np.mean(all_nis[all_nis_valid])
            if is_monte_carlo
            else np.mean(NIS[nis_valid]) if np.any(nis_valid) else float('nan')
        ),
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()