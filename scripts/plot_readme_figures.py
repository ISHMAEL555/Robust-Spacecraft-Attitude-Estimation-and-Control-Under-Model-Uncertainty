"""Build the summary figures used in README.md from the published evidence.

Inputs are the checked-in evidence files in docs/ (campaign summary CSV), so
the consistency, noise-mistuning, and outage figures need no new simulation.
The wheel-coupling figure re-runs the two deterministic seed-42 cases of
scripts/verify_wheel_coupling.py because that report stores no time series.

Colors follow the project figures (Okabe-Ito blue, vermillion, green); every
series also carries a distinct marker or line style and a direct label, so
identity never depends on color alone.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_config, run_simulation  # noqa: E402
from src.dynamics.quaternion import quat_to_rotmat  # noqa: E402

BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#202124", "#5F6368", "#D9DEE5"
ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0


def _style(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(True, color=GRID, linewidth=0.7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#9AA0A6")
    axis.tick_params(colors=MUTED, labelsize=9)


def _rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return {(row["campaign"], row["case"]): row for row in csv.DictReader(stream)}


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def consistency_overview(rows: dict, output: Path) -> None:
    """NEES/DOF and accepted NIS/DOF with run-level 95% intervals, key cases."""
    cases = [
        ("E0", "nominal_baseline", "E0  nominal, 1,000 s"),
        ("E7", "nominal_monte_carlo", "E7  nominal, 100 s"),
        ("E1", "gyro_driven_inertia_0.25", "E1  gyro-driven, 25% inertia error"),
        ("E1", "model_aided_inertia_0.00", "E1  model-aided, matched inertia"),
        ("E1", "model_aided_inertia_0.10", "E1  model-aided, 10% inertia error"),
        ("E1", "model_aided_inertia_0.25", "E1  model-aided, 25% inertia error"),
        ("E2", "Q_scale_0.1", "E2  Q × 0.1"),
        ("E2", "Q_scale_10", "E2  Q × 10"),
        ("E3", "R_scale_0.1", "E3  R × 0.1"),
        ("E3", "R_scale_10", "E3  R × 10"),
        ("E4", "outlier_1_gate_false", "E4  1 rad outliers, gate off"),
        ("E4", "outlier_1_gate_true", "E4  1 rad outliers, gate on"),
        ("E5", "outage_30_updates", "E5  30 s tracker outage"),
        ("E6", "nominal", "E6  closed-loop acquisition"),
    ]
    labels = [label for *_, label in cases]
    y = np.arange(len(cases))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), sharey=True,
                                constrained_layout=True)
    panels = (
        ("NEES", "NEES_dof", "Full-state NEES / DOF"),
        ("NIS_accepted", None, "Accepted NIS / DOF"),
    )
    for axis, (prefix, dof_key, title) in zip(axes, panels):
        _style(axis)
        for yi, (campaign, case, _) in zip(y, cases):
            row = rows[(campaign, case)]
            dof = _f(row, dof_key) if dof_key else 3.0
            mean = _f(row, f"{prefix}_run_mean") / dof
            low = _f(row, f"{prefix}_run_mean_ci95_lower") / dof
            high = _f(row, f"{prefix}_run_mean_ci95_upper") / dof
            consistent = low <= 1.0 <= high
            axis.plot([max(low, 1e-2), high], [yi, yi], color=BLUE, linewidth=2.0,
                      solid_capstyle="round")
            axis.plot(mean, yi, marker="o", markersize=7,
                      markerfacecolor=BLUE if consistent else "white",
                      markeredgecolor=BLUE, markeredgewidth=1.6, linestyle="none")
        axis.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlim(5e-2, 1e7)
        axis.set_title(title, fontsize=11, color=INK)
        axis.set_xlabel("Normalized mean (1 = consistent)", fontsize=9, color=MUTED)
    axes[0].set_yticks(y, labels, fontsize=9)
    axes[0].plot([], [], marker="o", color=BLUE, linestyle="none",
                 label="interval contains 1 (consistent)")
    axes[0].plot([], [], marker="o", markerfacecolor="white", markeredgecolor=BLUE,
                 linestyle="none", label="interval excludes 1")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Estimator consistency across the E0–E7 campaign "
                    "(10 runs per case; 95% t intervals over run means)",
                    fontsize=12, fontweight="bold", color=INK)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def noise_mistuning(rows: dict, output: Path) -> None:
    """E2/E3: statistics move with Q/R scaling; attitude accuracy does not."""
    scales = np.array([0.1, 1.0, 10.0])
    sweeps = (
        ("E2", "Q_scale", "Process noise Q", BLUE, "o", "-"),
        ("E3", "R_scale", "Measurement noise R", VERMILLION, "s", "--"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    panels = (
        ("NEES / DOF", lambda r: _f(r, "NEES_run_mean") / _f(r, "NEES_dof"), True),
        ("Accepted NIS / DOF", lambda r: _f(r, "NIS_accepted_run_mean") / 3.0, True),
        ("Attitude RMSE (arcsec)",
         lambda r: _f(r, "attitude_estimation_rmse_rad_mean") * ARCSEC_PER_RAD, False),
    )
    for axis, (title, value, reference) in zip(axes, panels):
        _style(axis)
        peak = 0.0
        for index, (campaign, prefix, label, color, marker, style) in enumerate(sweeps):
            values = [value(rows[(campaign, f"{prefix}_{scale:g}")]) for scale in scales]
            peak = max(peak, max(values))
            axis.plot(scales, values, color=color, marker=marker, linestyle=style,
                      linewidth=2.0, markersize=7, label=label)
            # On the flat RMSE panel the two end points coincide; stack the labels.
            offset = (7, 0) if reference else (7, 9 - 18 * index)
            axis.annotate(label.split()[-1], (scales[-1], values[-1]),
                          textcoords="offset points", xytext=offset, va="center",
                          fontsize=9, color=INK)
        if reference:
            axis.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.0)
            axis.set_yscale("log")
        else:
            axis.set_ylim(0.0, 1.35 * peak)
        axis.set_xscale("log")
        axis.set_xticks(scales, ["0.1×", "1×", "10×"])
        axis.set_xlim(0.07, 20)
        axis.set_xlabel("Filter noise scale factor", fontsize=9, color=MUTED)
        axis.set_title(title, fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    figure.suptitle("Noise-model mistuning (E2/E3): consistency statistics detect it; "
                    "attitude RMSE does not", fontsize=12, fontweight="bold", color=INK)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def outage_response(rows: dict, output: Path) -> None:
    """E5: peak estimation error versus filter 3-sigma for each outage length."""
    lengths = [1, 10, 30]
    keys = [("E5", f"outage_{n}_updates") for n in lengths]
    error = [_f(rows[k], "outage_peak_attitude_error_rad_mean") * ARCSEC_PER_RAD for k in keys]
    sigma = [_f(rows[k], "outage_peak_attitude_3sigma_rad_mean") * ARCSEC_PER_RAD for k in keys]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    _style(axis)
    axis.plot(lengths, sigma, color=BLUE, linestyle="--", marker="s", markersize=7,
              linewidth=2.0)
    axis.plot(lengths, error, color=VERMILLION, marker="o", markersize=7, linewidth=2.0)
    axis.annotate("filter peak 3σ", (lengths[-1], sigma[-1]), textcoords="offset points",
                  xytext=(-8, 8), ha="right", fontsize=9, color=INK)
    midpoint = 20.0
    axis.annotate("actual peak error", (midpoint, float(np.interp(midpoint, lengths, error))),
                  textcoords="offset points", xytext=(6, -24), ha="left",
                  fontsize=9, color=INK)
    axis.set_xticks(lengths, [f"{n} s" for n in lengths])
    axis.set_xlim(-1.0, 33.0)
    axis.set_xlabel("Star-tracker outage length (1 Hz samples missed)", fontsize=9, color=MUTED)
    axis.set_ylabel("Attitude angle (arcsec)", fontsize=9, color=MUTED)
    axis.set_ylim(0.0, 1.15 * max(sigma))
    axis.set_title("E5 outages: the covariance bounds the real error\n"
                   "(10-run means, gyro-driven MEKF)",
                   fontsize=11, color=INK)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def _coupling_run(coupled: bool) -> dict:
    config = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    config = replace(
        config,
        duration=400.0,
        spacecraft={**config.spacecraft,
                    "initial_attitude": [0.0, 0.0, 0.996194698, 0.087155743]},
        reaction_wheels={**config.reaction_wheels, "gyroscopic_coupling": coupled},
    )
    results = run_simulation(config)
    inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    wheel = np.vstack([np.zeros(3), results["wheel_momentum"][:-1]])
    total = np.array([
        quat_to_rotmat(q) @ (inertia @ omega + h)
        for q, omega, h in zip(results["q_true"], results["omega_true"], wheel)
    ])
    step = 50  # 0.5 s decimation for plotting
    return {
        "coupled": coupled,
        "time": results["time"][::step],
        "pointing_deg": np.degrees(results["pointing_error"][::step]),
        "momentum_drift": (np.linalg.norm(total - total[0], axis=1)
                           / np.linalg.norm(total[0]))[::step],
        "gyroscopic_nm": np.linalg.norm(np.cross(results["omega_true"], wheel), axis=1)[::step],
    }


def wheel_coupling(output: Path) -> None:
    """170 deg slew with stored momentum: correct physics versus the old model."""
    with ProcessPoolExecutor(max_workers=2) as pool:
        coupled, uncoupled = pool.map(_coupling_run, (True, False))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for axis in axes:
        _style(axis)
        axis.set_xlabel("Time (s)", fontsize=9, color=MUTED)
    series = ((coupled, "with ω × h_w (physical)", BLUE, "-"),
              (uncoupled, "without ω × h_w (pre-1.0)", VERMILLION, "--"))
    for run, label, color, style in series:
        axes[0].plot(run["time"], run["pointing_deg"], color=color, linestyle=style,
                     linewidth=1.8, label=label)
        axes[2].plot(run["time"], 100.0 * run["momentum_drift"], color=color,
                     linestyle=style, linewidth=1.8, label=label)
    axes[1].plot(coupled["time"], coupled["gyroscopic_nm"], color=BLUE, linewidth=1.8)
    axes[1].axhline(0.1, color=MUTED, linestyle=":", linewidth=1.2)
    axes[1].annotate("0.1 N·m control authority", (400, 0.1), textcoords="offset points",
                     xytext=(-4, 5), ha="right", fontsize=8, color=INK)
    axes[0].set_ylabel("True pointing error (deg)", fontsize=9, color=MUTED)
    axes[0].set_title("Acquisition", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_ylabel("‖ω × h_w‖ (N·m)", fontsize=9, color=MUTED)
    axes[1].set_title("Gyroscopic torque to overcome", fontsize=11, color=INK)
    axes[2].set_ylabel("Total momentum drift (%)", fontsize=9, color=MUTED)
    axes[2].set_title("Angular-momentum conservation", fontsize=11, color=INK)
    figure.suptitle("170° slew with 10 N·m·s stored wheel momentum (seed 42)",
                    fontsize=12, fontweight="bold", color=INK)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "e0_e7_campaign_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures" / "summary")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Only build the figures that need no new simulation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.summary)
    consistency_overview(rows, args.output_dir / "consistency_overview.png")
    noise_mistuning(rows, args.output_dir / "noise_mistuning.png")
    outage_response(rows, args.output_dir / "outage_response.png")
    if not args.skip_simulation:
        wheel_coupling(args.output_dir / "wheel_coupling.png")
    print(f"Saved README figures to {args.output_dir}")


if __name__ == "__main__":
    main()
