#!/usr/bin/env python3
"""
Analysis and visualization tools for spacecraft attitude estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
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


def _save_or_show(fig, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_attitude_error(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot attitude estimation error."""
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


def plot_bias_estimation(results: Dict[str, Any], save_path: Optional[str] = None):
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


def plot_covariance(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot covariance diagonals (3-sigma bounds)."""
    time = results['time']
    P = results.get('P', results.get('P_mean'))
    
    # 3-sigma bounds
    sigma_att = 3 * np.sqrt(P[:, :3, :3].diagonal(axis1=1, axis2=2))
    sigma_bias = 3 * np.sqrt(P[:, 3:, 3:].diagonal(axis1=1, axis2=2))
    
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


def plot_nees_nis(results: Dict[str, Any], save_path: Optional[str] = None):
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
        # Raw NEES and NIS follow chi-squared distributions with 6 and 3 DOF.
        from scipy import stats as sp_stats
        nees_lower = sp_stats.chi2.ppf(0.025, 6)
        nees_upper = sp_stats.chi2.ppf(0.975, 6)
        nis_lower = sp_stats.chi2.ppf(0.025, 3)
        nis_upper = sp_stats.chi2.ppf(0.975, 3)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # NEES
    ax = axes[0]
    ax.plot(time, NEES, 'b-', linewidth=0.5, label='NEES')
    ax.axhline(nees_lower, color='r', linestyle='--', label=f'Lower bound ({nees_lower:.2f})')
    ax.axhline(nees_upper, color='r', linestyle='--', label=f'Upper bound ({nees_upper:.2f})')
    ax.axhline(6.0, color='k', linestyle=':', label='Expected (6)')
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
    ax.axhline(nis_lower, color='r', linestyle='--', label=f'Lower bound ({nis_lower:.2f})')
    ax.axhline(nis_upper, color='r', linestyle='--', label=f'Upper bound ({nis_upper:.2f})')
    ax.axhline(3.0, color='k', linestyle=':', label='Expected (3)')
    ax.set_ylabel('NIS')
    ax.set_xlabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Consistency Metrics (NEES/NIS)')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_pointing_error(results: Dict[str, Any], save_path: Optional[str] = None):
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


def plot_control_torque(results: Dict[str, Any], save_path: Optional[str] = None):
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


def plot_monte_carlo_consistency(results: Dict[str, Any], save_path: Optional[str] = None):
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
        nees_lower = sp_stats.chi2.ppf(0.025, n_runs * 6) / n_runs
        nees_upper = sp_stats.chi2.ppf(0.975, n_runs * 6) / n_runs
        nis_lower = sp_stats.chi2.ppf(0.025, n_runs * 3) / n_runs
        nis_upper = sp_stats.chi2.ppf(0.975, n_runs * 3) / n_runs
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # NEES
    ax = axes[0]
    ax.plot(time, NEES_mean, 'b-', label='Mean NEES', linewidth=1)
    ax.fill_between(time, NEES_mean - NEES_std, NEES_mean + NEES_std, alpha=0.2, color='blue')
    ax.axhline(nees_lower, color='r', linestyle='--', label=f'Lower bound')
    ax.axhline(nees_upper, color='r', linestyle='--', label=f'Upper bound')
    ax.axhline(6.0, color='k', linestyle=':', label='Expected (6)')
    ax.set_ylabel('NEES (avg over runs)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # NIS
    ax = axes[1]
    ax.plot(time, NIS_mean, 'g-', label='Mean NIS', linewidth=1)
    ax.fill_between(time, NIS_mean - NIS_std, NIS_mean + NIS_std, alpha=0.2, color='green')
    ax.axhline(nis_lower, color='r', linestyle='--', label=f'Lower bound')
    ax.axhline(nis_upper, color='r', linestyle='--', label=f'Upper bound')
    ax.axhline(3.0, color='k', linestyle=':', label='Expected (3)')
    ax.set_ylabel('NIS (avg over runs)')
    ax.set_xlabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.suptitle('Monte Carlo Consistency Metrics')
    plt.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_monte_carlo_pointing(results: Dict[str, Any], save_path: Optional[str] = None):
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


def generate_report(results: Dict[str, Any], output_path: str):
    """Generate a text report from results."""
    with open(output_path, 'w') as f:
        f.write("Spacecraft Attitude Estimation Simulation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Simulation parameters
        f.write(f"Duration: {results['time'][-1]:.1f} s\n")
        f.write(f"Time step: {results['time'][1] - results['time'][0]:.3f} s\n")
        f.write(f"Steps: {len(results['time'])}\n\n")
        
        # Consistency
        if 'consistency' in results:
            stats = results['consistency']
            f.write("Consistency Statistics:\n")
            f.write(f"  NEES mean: {stats['nees_mean']:.3f}\n")
            f.write(f"  NEES in bounds: {stats['nees_in_bounds']*100:.1f}%\n")
            f.write(f"  NIS mean: {stats['nis_mean']:.3f}\n")
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
        
        # Final covariance
        covariance = results.get('P', results.get('P_mean'))
        if covariance is not None:
            P_final = covariance[-1]
            f.write("Final Covariance (3-sigma):\n")
            attitude_bound = 3 * np.sqrt(np.diag(P_final[:3, :3])) * 180 / np.pi * 3600
            bias_bound = 3 * np.sqrt(np.diag(P_final[3:, 3:])) * 1000
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