"""Run and plot the labeled E6 project-level pointing benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_experiment import (
    compute_performance_metrics,
    evaluate_e6_project_benchmark,
    load_config,
    run_simulation,
)


def _display_path(path: Path) -> str:
    """Repository-relative POSIX path when possible, otherwise absolute."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "closed_loop.yaml")
    parser.add_argument(
        "--figure", type=Path,
        default=ROOT / "docs" / "figures" / "e6_project_benchmark.png",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "docs" / "e6_project_benchmark.json",
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    results = run_simulation(config)
    metrics = compute_performance_metrics(results)
    benchmark = evaluate_e6_project_benchmark(metrics)

    time = results["time"]
    pointing_deg = np.rad2deg(results["pointing_error"])
    wheel_utilization = np.max(
        np.abs(results["wheel_momentum"]) / results["max_wheel_momentum"], axis=1
    )
    final_window_start = max(float(time[0]), float(time[-1]) - 20.0)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    figure.patch.set_facecolor("white")
    colors = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73"}
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(True, color="#D9DEE5", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].plot(time, pointing_deg, color=colors["blue"], linewidth=1.5)
    axes[0].axhline(2.0, color=colors["orange"], linestyle="--", linewidth=1.3,
                    label="2° steady pointing bound")
    axes[0].axvspan(final_window_start, time[-1], color=colors["green"], alpha=0.10,
                    label="final 20 s hold window")
    axes[0].set_ylabel("True pointing error (deg)")
    axes[0].set_title("Pointing acquisition")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(time, 100.0 * wheel_utilization, color=colors["blue"], linewidth=1.5)
    axes[1].axhline(80.0, color=colors["orange"], linestyle="--", linewidth=1.3,
                    label="80% project momentum-use limit")
    axes[1].axhline(100.0, color="#A61C3C", linestyle=":", linewidth=1.2,
                    label="hard wheel momentum limit")
    axes[1].set_ylabel("Peak wheel utilization (%)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_title("Reaction-wheel momentum margin")
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle("E6 · Labeled project benchmark", fontsize=15, fontweight="bold")
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    summary = {
        "benchmark": benchmark,
        "scenario": {
            "duration_s": config.duration,
            "dt_s": config.dt,
            "initial_attitude_deg": 60.0,
            "initial_rate_rad_s": config.spacecraft["initial_omega"],
            "controller_gains": {
                "K_q": config.controller["K_q"],
                "K_omega": config.controller["K_omega"],
            },
            "body_torque_limit_nm": config.controller["max_torque"],
            "wheel_torque_limit_nm": config.reaction_wheels["max_torque"],
            "wheel_momentum_limit_nms": config.reaction_wheels["max_momentum"],
            "wheel_motor_time_constant_s": config.reaction_wheels.get(
                "motor_time_constant_s", 0.0
            ),
            "star_tracker_rate_hz": config.star_tracker["rate"],
            "gyro_rate_hz": config.gyroscope["rate"],
            "wheel_gyroscopic_coupling": bool(results["wheel_gyroscopic_coupling"]),
            "assumptions": [
                "rigid bus with wheel gyroscopic coupling omega x h_w",
                "first-order wheel motor torque response with hard torque and momentum limits",
                "zero environmental disturbance torque; no sensor or actuator transport latency",
                "sensor measurements are applied at their simulation-grid epochs",
                "project benchmark values are not mission requirements",
            ],
        },
        "metrics": metrics,
        "figure": _display_path(args.figure),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Project benchmark: {'PASS' if benchmark['passed'] else 'FAIL'}")
    for name, check in benchmark["checks"].items():
        print(
            f"  {'PASS' if check['passed'] else 'FAIL'} {name}: "
            f"{check['measured']} {check['operator']} {check['limit']}"
        )
    print(f"Saved figure: {args.figure}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
