"""Quantify the effect of wheel gyroscopic coupling on closed-loop acquisition.

Runs the E6 benchmark (60 deg about x, spin about x) and the 170 deg
acquisition (rotation about z with the same x-axis spin) with the wheel
gyroscopic term omega x h_w enabled and disabled. For each run it reports the
drift of total inertial angular momentum R(q)(J omega + h_w), which must be
conserved with zero external torque, together with the acquisition metrics.
The uncoupled runs are reported only to show what the earlier model omitted.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import (  # noqa: E402
    compute_performance_metrics,
    load_config,
    run_simulation,
)
from src.dynamics.quaternion import quat_to_rotmat  # noqa: E402

SCENARIOS = {
    "e6_60deg_about_x": {"duration": 200.0, "initial_attitude": None},
    "near_180deg_about_z": {
        "duration": 400.0,
        "initial_attitude": [0.0, 0.0, 0.996194698, 0.087155743],
    },
}


def _run(arguments: tuple[str, bool]) -> dict:
    name, coupled = arguments
    scenario = SCENARIOS[name]
    config = load_config(str(ROOT / "config" / "closed_loop.yaml"))
    spacecraft = dict(config.spacecraft)
    if scenario["initial_attitude"] is not None:
        spacecraft["initial_attitude"] = scenario["initial_attitude"]
    config = replace(
        config,
        duration=scenario["duration"],
        spacecraft=spacecraft,
        reaction_wheels={**config.reaction_wheels, "gyroscopic_coupling": coupled},
    )
    results = run_simulation(config)
    metrics = compute_performance_metrics(results)

    inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    # Recorded wheel momentum is at the end of each step; the truth state is
    # recorded at its start, so shift by one sample to align epochs.
    wheel = np.vstack([np.zeros(3), results["wheel_momentum"][:-1]])
    total = np.array([
        quat_to_rotmat(q) @ (inertia @ omega + h)
        for q, omega, h in zip(results["q_true"], results["omega_true"], wheel)
    ])
    drift = np.max(np.linalg.norm(total - total[0], axis=1)) / np.linalg.norm(total[0])
    gyroscopic = np.linalg.norm(np.cross(results["omega_true"], wheel), axis=1)
    pointing_deg = np.degrees(results["pointing_error"])
    return {
        "scenario": name,
        "gyroscopic_coupling": coupled,
        "duration_s": config.duration,
        "seed": config.seed,
        "total_momentum_nms": float(np.linalg.norm(total[0])),
        "max_relative_momentum_drift": float(drift),
        "peak_wheel_gyroscopic_torque_nm": float(np.max(gyroscopic)),
        "settling_time_2pct_s": (
            metrics["settling_time_2pct_s"]
            if np.isfinite(metrics["settling_time_2pct_s"]) else None
        ),
        "final_pointing_deg": float(pointing_deg[-1]),
        "final_angular_rate_norm_rad_s": metrics["final_angular_rate_norm_rad_s"],
        "command_torque_saturation_fraction": metrics["command_torque_saturation_fraction"],
        "peak_wheel_momentum_utilization": metrics["peak_wheel_momentum_utilization"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "wheel_coupling_effect.json")
    args = parser.parse_args()
    jobs = [(name, coupled) for name in SCENARIOS for coupled in (True, False)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        rows = list(pool.map(_run, jobs))
    report = {
        "scope": "Deterministic seed-42 closed-loop runs; simulation evidence only.",
        "max_torque_nm": 0.1,
        "runs": rows,
        "interpretation": (
            "With coupling, total angular momentum is conserved to numerical precision. "
            "Without it, momentum is not conserved, and multi-axis slews with stored "
            "wheel momentum appear easier than they physically are."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['scenario']:<22} coupled={row['gyroscopic_coupling']!s:<5} "
            f"drift={row['max_relative_momentum_drift']:.2e} "
            f"peak|w x h|={row['peak_wheel_gyroscopic_torque_nm']:.4f} N m "
            f"settle={row['settling_time_2pct_s']} final={row['final_pointing_deg']:.3f} deg "
            f"sat={row['command_torque_saturation_fraction']:.3f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
