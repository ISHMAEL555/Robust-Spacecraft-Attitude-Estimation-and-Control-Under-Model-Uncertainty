#!/usr/bin/env python3
"""
Analysis and visualization tools for spacecraft attitude estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dynamics.quaternion import error_quat, error_quat_to_vec


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from NPZ file."""
    data = np.load(filepath, allow_pickle=True)
    results = {}
    for key in data.files:
        value = data[key]
        if value.shape == () and value.dtype == object:
            value = value.item()
        results[key] = value
    return results


def _save_or_show(fig, save_path: Optional[Union[str, Path]]) -> None:
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_attitude_error(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot attitude estimation error."""
    if 'attitude_estimation_error_mean' in results and 'q_hat' not in results:
        time = results['time']
        error_mean = np.degrees(results['attitude_estimation_error_mean'])
        error_std = np.degrees(results['attitude_estimation_error_std'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(time, error_mean, label='Mean attitude estimation error')
        ax.fill_between(
            time,
            np.maximum(error_mean - error_std, 0.0),
            error_mean + error_std,
            alpha=0.2,
            label='±1σ across runs',
        )
        ax.set_ylabel('Attitude Estimation Error (deg)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle('Monte Carlo Attitude Estimation Error')
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return

    if 'q_true' not in results or 'q_hat' not in results:
        time = results['time']
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(time, np.degrees(results['pointing_error_mean']), label='Mean')
        ax.set_ylabel('Attitude Error (deg)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle('Mean Attitude Estimation Error')
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return

    time = results['time']
    q_true = results['q_true']
    q_hat = results['q_hat']
    
    # Compute error quaternions
    errors = []
    for i in range(len(time)):
        delta_q = error_quat(q_true[i], q_hat[i])
        delta_theta = error_quat_to_vec(delta_q)
        errors.append(delta_theta)
    errors = np.array(errors)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for j in range(3):
        ax = axes[j]
        ax.plot(time, np.degrees(errors[:, j]), color=colors[j], label=labels[j], linewidth=0.5)
        ax.set_ylabel('Error (deg)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Attitude Estimation Error')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_bias_estimation(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot gyro bias estimation."""
    time = results['time']
    b_hat = results.get('b_hat', results.get('b_hat_mean'))
    
    # True bias (if available)
    b_true = results.get('b_true', results.get('b_true_mean'))
    has_true = b_true is not None
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['X', 'Y', 'Z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for j in range(3):
        ax = axes[j]
        ax.plot(time, b_hat[:, j] * 1000, color=colors[j], label=f'Estimated {labels[j]}', linewidth=0.5)
        if has_true:
            ax.plot(time, b_true[:, j] * 1000, '--', color=colors[j], label=f'True {labels[j]}', linewidth=0.5)
        ax.set_ylabel('Bias (mrad/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Gyro Bias Estimation')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_covariance(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot covariance diagonals (3-sigma bounds)."""
    time = results['time']
    P = results.get('P', results.get('P_mean'))
    
    # 3-sigma bounds
    sigma_att = 3 * np.sqrt(P[:, :3, :3].diagonal(axis1=1, axis2=2))
    bias_start = 6 if P.shape[1] == 9 else 3
    sigma_bias = 3 * np.sqrt(
        P[:, bias_start:bias_start + 3, bias_start:bias_start + 3].diagonal(
            axis1=1, axis2=2
        )
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Attitude covariance
    ax = axes[0]
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for j in range(3):
        ax.plot(time, np.degrees(sigma_att[:, j]), color=colors[j], label=labels[j], linewidth=0.5)
    ax.set_ylabel('3σ Attitude (deg)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Bias covariance
    ax = axes[1]
    for j in range(3):
        ax.plot(time, sigma_bias[:, j] * 1000, color=colors[j], label=labels[j], linewidth=0.5)
    ax.set_ylabel('3σ Bias (mrad/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Covariance 3-Sigma Bounds')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_nees_nis(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot NEES and NIS with consistency bounds."""
    time = results['time']
    NEES = results.get('NEES', results.get('NEES_mean'))
    NIS = results.get('NIS', results.get('NIS_mean'))
    
    # Get bounds from consistency stats if available
    if 'consistency' in results:
        stats = results['consistency']
        nees_lower, nees_upper = stats['nees_bounds']
        nis_lower, nis_upper = stats['nis_bounds']
    else:
        # Raw NEES and NIS follow chi-squared distributions for their state dimensions.
        from scipy import stats as sp_stats
        nees_dof = int(results.get('nees_dof', 6))
        nees_lower = sp_stats.chi2.ppf(0.025, nees_dof)
        nees_upper = sp_stats.chi2.ppf(0.975, nees_dof)
        nis_lower = sp_stats.chi2.ppf(0.025, 3)
        nis_upper = sp_stats.chi2.ppf(0.975, 3)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # NEES
    ax = axes[0]
    ax.plot(time, NEES, 'b-', linewidth=0.5, label='NEES')
    ax.axhline(nees_lower, color='r', linestyle='--', label=f'Lower bound ({nees_lower:.2f})')
    ax.axhline(nees_upper, color='r', linestyle='--', label=f'Upper bound ({nees_upper:.2f})')
    nees_dof = int(results.get('nees_dof', 6))
    ax.axhline(nees_dof, color='k', linestyle=':', label=f'Expected ({nees_dof})')
    ax.set_ylabel('NEES')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # NIS
    ax = axes[1]
    if 'measurement_available' in results:
        valid_nis = results['measurement_available']
    elif 'innovation_valid' in results:
        valid_nis = results['innovation_valid']
    elif 'innovation_count' in results:
        valid_nis = results['innovation_count'] > 0
    else:
        valid_nis = NIS > 0
    if np.any(valid_nis):
        ax.plot(time[valid_nis], NIS[valid_nis], 'g-', linewidth=0.5, label='NIS')
    if np.ndim(nis_lower) == 0:
        ax.axhline(nis_lower, color='r', linestyle='--', label=f'Lower bound ({nis_lower:.2f})')
        ax.axhline(nis_upper, color='r', linestyle='--', label=f'Upper bound ({nis_upper:.2f})')
    else:
        ax.plot(time, nis_lower, 'r--', label='Lower bound')
        ax.plot(time, nis_upper, 'r--', label='Upper bound')
    ax.axhline(3.0, color='k', linestyle=':', label='Expected (3)')
    ax.set_ylabel('NIS')
    ax.set_xlabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Consistency Metrics (NEES/NIS)')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_pointing_error(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot closed-loop pointing error."""
    time = results['time']
    pointing_error = results.get('pointing_error', results.get('pointing_error_mean'))
    pointing_error_deg = results.get('pointing_error_deg', np.degrees(pointing_error))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Pointing error in degrees
    ax = axes[0]
    ax.plot(time, pointing_error_deg, 'b-', linewidth=0.5)
    ax.set_ylabel('Pointing Error (deg)')
    ax.grid(True, alpha=0.3)
    
    # Pointing error in arcsec
    ax = axes[1]
    ax.plot(time, pointing_error * 180/np.pi * 3600, 'r-', linewidth=0.5)
    ax.set_ylabel('Pointing Error (arcsec)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Closed-Loop Pointing Error')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_control_torque(results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None):
    """Plot control torque commands."""
    time = results['time']
    tau_cmd = results['tau_cmd']
    tau_rw = results['tau_rw']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['X', 'Y', 'Z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for j in range(3):
        ax = axes[j]
        ax.plot(time, tau_cmd[:, j], color=colors[j], label=f'Cmd {labels[j]}', linewidth=0.5)
        ax.plot(time, tau_rw[:, j], '--', color=colors[j], label=f'RW {labels[j]}', linewidth=0.5)
        ax.set_ylabel('Torque (N·m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Control Torque')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_monte_carlo_consistency(
    results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
):
    """Plot Monte Carlo consistency statistics."""
    if 'NEES_mean' not in results:
        print("No Monte Carlo data available")
        return
    
    time = results['time']
    NEES_mean = results['NEES_mean']
    NEES_std = results['NEES_std']
    NIS_mean = results['NIS_mean']
    NIS_std = results['NIS_std']
    
    # Get bounds
    if 'consistency' in results:
        stats = results['consistency']
        nees_lower, nees_upper = stats['nees_bounds_mc']
        nis_lower, nis_upper = stats['nis_bounds_mc']
    else:
        from scipy import stats as sp_stats
        n_runs = int(results.get('n_runs', 1))
        nees_dof = int(results.get('nees_dof', 6))
        nees_lower = sp_stats.chi2.ppf(0.025, n_runs * nees_dof) / n_runs
        nees_upper = sp_stats.chi2.ppf(0.975, n_runs * nees_dof) / n_runs
        nis_lower = sp_stats.chi2.ppf(0.025, n_runs * 3) / n_runs
        nis_upper = sp_stats.chi2.ppf(0.975, n_runs * 3) / n_runs
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # NEES
    ax = axes[0]
    ax.plot(time, NEES_mean, 'b-', label='Mean NEES', linewidth=1)
    ax.fill_between(time, NEES_mean - NEES_std, NEES_mean + NEES_std, alpha=0.2, color='blue')
    if np.ndim(nees_lower) == 0:
        ax.axhline(nees_lower, color='r', linestyle='--', label='Lower bound')
        ax.axhline(nees_upper, color='r', linestyle='--', label='Upper bound')
    else:
        ax.plot(time, nees_lower, 'r--', label='Lower bound')
        ax.plot(time, nees_upper, 'r--', label='Upper bound')
    nees_dof = int(results.get('nees_dof', 6))
    ax.axhline(nees_dof, color='k', linestyle=':', label=f'Expected ({nees_dof})')
    ax.set_ylabel('NEES (avg over runs)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # NIS
    ax = axes[1]
    ax.plot(time, NIS_mean, 'g-', label='Mean NIS', linewidth=1)
    ax.fill_between(time, NIS_mean - NIS_std, NIS_mean + NIS_std, alpha=0.2, color='green')
    if np.ndim(nis_lower) == 0:
        ax.axhline(nis_lower, color='r', linestyle='--', label='Lower bound')
        ax.axhline(nis_upper, color='r', linestyle='--', label='Upper bound')
    else:
        ax.plot(time, nis_lower, 'r--', label='Lower bound')
        ax.plot(time, nis_upper, 'r--', label='Upper bound')
    ax.axhline(3.0, color='k', linestyle=':', label='Expected (3)')
    ax.set_ylabel('NIS (avg over runs)')
    ax.set_xlabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Monte Carlo Consistency Metrics')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_monte_carlo_pointing(
    results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None
):
    """Plot Monte Carlo pointing error statistics."""
    if 'pointing_error_mean' not in results:
        print("No Monte Carlo data available")
        return
    
    time = results['time']
    pointing_mean = results['pointing_error_mean']
    pointing_std = results['pointing_error_std']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(time, np.degrees(pointing_mean), 'b-', label='Mean', linewidth=1)
    ax.fill_between(time, 
                    np.degrees(pointing_mean - pointing_std),
                    np.degrees(pointing_mean + pointing_std),
                    alpha=0.2, color='blue', label='±1σ')
    ax.fill_between(time,
                    np.degrees(pointing_mean - 2*pointing_std),
                    np.degrees(pointing_mean + 2*pointing_std),
                    alpha=0.1, color='blue', label='±2σ')
    
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Monte Carlo Pointing Error')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def generate_report(results: Dict[str, Any], output_path: Union[str, Path]):
    """Generate a text report from results."""
    with open(output_path, 'w') as f:
        f.write("Spacecraft Attitude Estimation Simulation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Simulation parameters
        time = results['time']
        dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.0
        f.write(f"Duration: {time[-1] + dt:.1f} s\n")
        f.write(f"Time step: {dt:.3f} s\n")
        f.write(f"Steps: {len(results['time'])}\n\n")
        
        # Consistency
        if 'consistency' in results:
            stats = results['consistency']
            f.write("Consistency Statistics:\n")
            f.write(
                f"  NEES mean (dof={stats['dof_nees']}): "
                f"{stats['nees_mean']:.3f}\n"
            )
            f.write(f"  NIS mean (dof={stats['dof_nis']}): {stats['nis_mean']:.3f}\n")
            f.write(f"  NEES in bounds: {stats['nees_in_bounds']*100:.1f}%\n")
            f.write(f"  NIS in bounds: {stats['nis_in_bounds']*100:.1f}%\n\n")

        if 'measurement_available' in results:
            attempts = int(np.sum(results['measurement_available']))
            rejected = int(np.sum(
                results['measurement_available'] & ~results['measurement_accepted']
            ))
            f.write(f"Star-tracker updates: {attempts} attempted, {rejected} rejected\n\n")
        elif 'innovation_count' in results:
            attempts = int(np.sum(results['innovation_count']))
            rejected = int(np.sum(results['measurement_rejection_count']))
            f.write(f"Star-tracker updates: {attempts} attempted, {rejected} rejected\n\n")
        
        # Pointing error
        if 'pointing_error_mean' in results:
            pe = results['pointing_error_mean']
        else:
            pe = results['pointing_error']
        
        f.write("Pointing Error:\n")
        f.write(f"  Mean: {np.mean(pe)*180/np.pi:.3f} deg\n")
        f.write(f"  RMS: {np.sqrt(np.mean(pe**2))*180/np.pi:.3f} deg\n")
        f.write(f"  Peak: {np.max(pe)*180/np.pi:.3f} deg\n")
        f.write(f"  Mean (arcsec): {np.mean(pe)*180/np.pi*3600:.1f}\n")
        f.write(f"  RMS (arcsec): {np.sqrt(np.mean(pe**2))*180/np.pi*3600:.1f}\n\n")

        if 'performance_summary' in results:
            is_monte_carlo = bool(results.get('is_monte_carlo', False))
            heading = (
                "Monte Carlo Performance (mean +/- 1 SD across runs):\n"
                if is_monte_carlo
                else "Performance Metrics:\n"
            )
            f.write(heading)
            units = {
                'pointing_rms_rad': ('Pointing RMS', 180 / np.pi, 'deg'),
                'pointing_peak_rad': ('Peak pointing error', 180 / np.pi, 'deg'),
                'pointing_final_rad': ('Final pointing error', 180 / np.pi, 'deg'),
                'settling_time_2pct_s': ('2% settling time', 1.0, 's'),
                'control_effort_nm2_s': ('Commanded control effort', 1.0, 'N^2 m^2 s'),
                'achieved_control_effort_nm2_s': ('Applied control effort', 1.0, 'N^2 m^2 s'),
                'peak_commanded_torque_nm': ('Peak commanded torque', 1.0, 'N m'),
                'peak_achieved_torque_nm': ('Peak applied torque', 1.0, 'N m'),
                'command_torque_saturation_fraction': ('Command saturation', 100.0, '%'),
                'attitude_estimation_rmse_rad': ('Attitude estimation RMSE', 180 / np.pi, 'deg'),
                'angular_rate_estimation_rmse_rad_s': ('Angular-rate estimation RMSE', 1.0, 'rad/s'),
                'bias_estimation_rmse_rad_s': ('Bias estimation RMSE', 1.0, 'rad/s'),
                'attitude_3sigma_component_coverage': ('Attitude 3-sigma component coverage', 100.0, '%'),
                'bias_3sigma_component_coverage': ('Bias 3-sigma component coverage', 100.0, '%'),
                'outage_recovery_time_s': ('Outage recovery time', 1.0, 's'),
            }
            for key, (label, scale, unit) in units.items():
                metric = results['performance_summary'].get(key)
                if metric is not None:
                    if is_monte_carlo:
                        line = (
                            f"  {label}: {metric['mean'] * scale:.4g} +/- "
                            f"{metric['std'] * scale:.3g} {unit}"
                        )
                    else:
                        line = f"  {label}: {metric['mean'] * scale:.4g} {unit}"
                    if is_monte_carlo and metric['ci95_lower'] is not None:
                        line += (
                            f" (95% t CI [{metric['ci95_lower'] * scale:.4g}, "
                            f"{metric['ci95_upper'] * scale:.4g}], n={metric['n']})"
                        )
                    elif is_monte_carlo:
                        line += f" (95% CI unavailable, n={metric['n']})"
                    f.write(line + "\n")
            f.write("\n")

        if 'attitude_estimation_error' in results:
            estimate_rmse = np.sqrt(np.mean(results['attitude_estimation_error']**2))
            f.write(f"Attitude estimation RMSE: {np.degrees(estimate_rmse):.4g} deg\n\n")
        
        # Final covariance
        covariance = results.get('P', results.get('P_mean'))
        if covariance is not None:
            P_final = covariance[-1]
            f.write("Final Covariance (3-sigma):\n")
            attitude_bound = 3 * np.sqrt(np.diag(P_final[:3, :3])) * 180 / np.pi * 3600
            bias_start = 6 if P_final.shape[0] == 9 else 3
            bias_bound = 3 * np.sqrt(
                np.diag(P_final[bias_start:bias_start + 3, bias_start:bias_start + 3])
            ) * 1000
            f.write(f"  Attitude: {np.array2string(attitude_bound, precision=1)} arcsec\n")
            f.write(f"  Bias: {np.array2string(bias_bound, precision=3)} mrad/s\n")
    
    print(f"Report saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze simulation results')
    parser.add_argument('input', type=str, help='Input NPZ file')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    args = parser.parse_args()
    
    results = load_results(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.input).stem
    
    if args.all:
        plot_attitude_error(results, output_dir / f"{base_name}_attitude_error.png")
        plot_bias_estimation(results, output_dir / f"{base_name}_bias.png")
        plot_covariance(results, output_dir / f"{base_name}_covariance.png")
        plot_nees_nis(results, output_dir / f"{base_name}_nees_nis.png")
        plot_pointing_error(results, output_dir / f"{base_name}_pointing.png")
        if 'tau_cmd' in results and 'tau_rw' in results:
            plot_control_torque(results, output_dir / f"{base_name}_torque.png")

        if 'NEES_mean' in results:
            plot_monte_carlo_consistency(results, output_dir / f"{base_name}_mc_consistency.png")
            plot_monte_carlo_pointing(results, output_dir / f"{base_name}_mc_pointing.png")
        
        generate_report(results, output_dir / f"{base_name}_report.txt")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()