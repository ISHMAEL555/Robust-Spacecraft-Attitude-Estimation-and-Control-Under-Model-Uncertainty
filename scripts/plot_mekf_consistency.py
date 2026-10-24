"""Plot marginal and full-state NEES for the matched-inertia model-aided check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "docs" / "model_aided_consistency.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs" / "figures" / "model_aided_nees_groups.png",
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8-sig"))
    groups = report["marginal_nees_by_state_group"]
    names = ["attitude", "rate", "bias", "inertia"]
    dof = np.asarray([groups[name]["dof"] for name in names], dtype=float)
    means = np.asarray([groups[name]["mean"] for name in names]) / dof
    lower = np.asarray([
        groups[name]["run_mean_ci95_lower"] for name in names
    ]) / dof
    upper = np.asarray([
        groups[name]["run_mean_ci95_upper"] for name in names
    ]) / dof

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ink, blue, orange, grid = "#202124", "#0072B2", "#D55E00", "#D9DEE5"
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(True, color=grid, linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors="#5F6368", labelsize=9)
    x = np.arange(len(names))
    axes[0].errorbar(
        x, means,
        yerr=np.vstack((means - lower, upper - means)),
        fmt="o", capsize=4, linewidth=1.6, markersize=6, color=blue,
    )
    axes[0].axhline(1.0, color=orange, linestyle="--", linewidth=1.3,
                    label="expected NEES / DOF = 1")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Marginal mean NEES / state-group DOF")
    axes[0].set_title("State-group consistency")
    axes[0].legend(frameon=False, fontsize=8)

    total_mean = report["mean_nees"] / report["nees_dof"]
    total_ci = np.asarray(report["run_mean_nees_ci95"]) / report["nees_dof"]
    axes[1].errorbar(
        [0], [total_mean],
        yerr=np.array([[total_mean - total_ci[0]], [total_ci[1] - total_mean]]),
        fmt="o", capsize=4, linewidth=1.6, markersize=7, color=blue,
    )
    axes[1].axhline(1.0, color=orange, linestyle="--", linewidth=1.3)
    axes[1].set_xlim(-0.7, 0.7)
    axes[1].set_xticks([0], ["full state"])
    axes[1].set_ylabel("Mean NEES / 15 DOF")
    axes[1].set_title("Independent-run 95% interval")

    figure.suptitle(
        f"Matched inertia · model-aided MEKF · {report['n_runs']} runs",
        fontsize=14, fontweight="bold", color=ink,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
