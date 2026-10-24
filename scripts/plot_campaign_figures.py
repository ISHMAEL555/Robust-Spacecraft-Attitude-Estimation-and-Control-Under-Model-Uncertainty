"""Create README-ready figures from the refreshed E0-E7 campaign summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
INK = "#202124"
MUTED = "#5F6368"
GRID = "#D9DEE5"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metric(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"Missing {name!r} for campaign case {row.get('case')!r}")
    return float(value)


def case_rows(
    rows: list[dict[str, str]], campaign: str, cases: list[str]
) -> list[dict[str, str]]:
    selected = {
        row["case"]: row for row in rows if row.get("campaign") == campaign
    }
    missing = [case for case in cases if case not in selected]
    if missing:
        raise ValueError(f"Missing {campaign} cases in summary CSV: {', '.join(missing)}")
    return [selected[case] for case in cases]


def style_axes(axes: list[plt.Axes] | np.ndarray) -> None:
    for axis in np.ravel(axes):
        axis.set_axisbelow(True)
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9)
        axis.xaxis.label.set_color(INK)
        axis.yaxis.label.set_color(INK)
        axis.title.set_color(INK)


def save_estimator_consistency(rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in rows if row.get("campaign") == "E1"]
    variants = [
        ("gyro_driven", "Gyro-driven MEKF", BLUE),
        ("model_aided", "Model-aided MEKF", VERMILLION),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    style_axes(axes)

    for variant, label, color in variants:
        series = sorted(
            (row for row in selected if row.get("variant") == variant),
            key=lambda row: metric(row, "inertia_mismatch"),
        )
        offset = -0.006 if variant == "gyro_driven" else 0.006
        x = np.array([metric(row, "inertia_mismatch") for row in series]) + offset
        dof = np.array([metric(row, "NEES_dof") for row in series])
        nees_per_dof = np.array(
            [metric(row, "NEES_run_mean") / value for row, value in zip(series, dof)]
        )
        nees_low = np.array([
            metric(row, "NEES_run_mean_ci95_lower") / value
            for row, value in zip(series, dof)
        ])
        nees_high = np.array([
            metric(row, "NEES_run_mean_ci95_upper") / value
            for row, value in zip(series, dof)
        ])
        axes[0].errorbar(
            x,
            nees_per_dof,
            yerr=np.vstack((nees_per_dof - nees_low, nees_high - nees_per_dof)),
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )

        rmse = np.array(
            [metric(row, "attitude_estimation_rmse_rad_mean") * 180.0 / np.pi for row in series]
        )
        ci_low = np.array(
            [
                metric(row, "attitude_estimation_rmse_rad_ci95_lower") * 180.0 / np.pi
                for row in series
            ]
        )
        ci_high = np.array(
            [
                metric(row, "attitude_estimation_rmse_rad_ci95_upper") * 180.0 / np.pi
                for row in series
            ]
        )
        axes[1].errorbar(
            x,
            rmse,
            yerr=np.vstack((rmse - ci_low, ci_high - rmse)),
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )

    axes[0].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, label="NEES / DOF = 1")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.5, 100)
    axes[0].set_ylabel("Run-mean NEES / DOF")
    axes[0].set_title("Covariance consistency (95% run-level t CI)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].set_ylabel("Attitude-estimation RMSE (deg)")
    # Scale from the data: RMSE is non-negative, and a fixed window clips the
    # run-level intervals whenever the results move.
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("Attitude accuracy")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    for axis in axes:
        axis.set_xticks([0.0, 0.1, 0.25], ["0%", "10%", "25%"])
        axis.set_xlabel("Truth inertia mismatch")

    fig.suptitle("E1 · Inertia mismatch", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.5,
        -0.025,
        "10 independent runs per case · intervals are over run-level means",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_outlier_gating(rows: list[dict[str, str]], output: Path) -> None:
    magnitudes = ["0.1", "0.5", "1.0"]
    selected = [row for row in rows if row.get("campaign") == "E4"]
    ungated = {
        row["outlier_magnitude_rad"]: row
        for row in selected
        if row.get("nis_gate_enabled", "").lower() == "false"
    }
    gated = {
        row["outlier_magnitude_rad"]: row
        for row in selected
        if row.get("nis_gate_enabled", "").lower() == "true"
    }
    if not ungated or not gated:
        raise ValueError("E4 summary must contain gated and ungated outlier cases")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    style_axes(axes)
    x = np.array([float(value) for value in magnitudes])

    for dataset, label, color in [
        (ungated, "Gate off", BLUE),
        (gated, "Gate on", VERMILLION),
    ]:
        series = [dataset[value] for value in magnitudes]
        rmse = np.array(
            [metric(row, "attitude_estimation_rmse_rad_mean") * 180.0 / np.pi for row in series]
        )
        ci_low = np.array(
            [
                metric(row, "attitude_estimation_rmse_rad_ci95_lower") * 180.0 / np.pi
                for row in series
            ]
        )
        ci_high = np.array(
            [
                metric(row, "attitude_estimation_rmse_rad_ci95_upper") * 180.0 / np.pi
                for row in series
            ]
        )
        axes[0].errorbar(
            x,
            rmse,
            yerr=np.vstack((rmse - ci_low, ci_high - rmse)),
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )

    axes[0].set_xlabel("Injected outlier magnitude (rad)")
    axes[0].set_ylabel("Attitude-estimation RMSE (deg)")
    axes[0].set_title("State-estimation error")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xticks(x)

    for dataset, label, color in [
        (ungated, "All updates, gate off", BLUE),
        (gated, "Accepted updates, gate on", VERMILLION),
    ]:
        series = [dataset[value] for value in magnitudes]
        mean = np.array([metric(row, "NIS_accepted_run_mean") / 3.0 for row in series])
        low = np.array([
            metric(row, "NIS_accepted_run_mean_ci95_lower") / 3.0 for row in series
        ])
        high = np.array([
            metric(row, "NIS_accepted_run_mean_ci95_upper") / 3.0 for row in series
        ])
        axes[1].errorbar(
            x,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=3,
            color=color,
            label=label,
        )
    axes[1].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, label="NIS / 3 DOF = 1")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1.0e-1, 3.0e6)
    axes[1].set_xlabel("Injected outlier magnitude (rad)")
    axes[1].set_ylabel("Run-mean accepted NIS / 3 DOF")
    axes[1].set_title("Innovation consistency")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xticks(x)

    rejected = metric(gated[magnitudes[0]], "rejected_measurements")
    updates = metric(gated[magnitudes[0]], "star_tracker_updates")
    fig.suptitle("E4 · Star-tracker outlier rejection", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.5,
        -0.025,
        f"5% outlier probability · 10 runs × 100 s per case · NIS gate rejected {rejected:.0f}/{updates:.0f} updates",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_nominal_baseline(rows: list[dict[str, str]], output: Path) -> None:
    row = case_rows(rows, "E0", ["nominal_baseline"])[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    style_axes(axes)

    consistency = [
        ("NEES_run_mean", "NEES_run_mean_ci95_lower", "NEES_run_mean_ci95_upper", 6.0, "NEES / 6 DOF"),
        ("NIS_accepted_run_mean", "NIS_accepted_run_mean_ci95_lower", "NIS_accepted_run_mean_ci95_upper", 3.0, "Accepted NIS / 3 DOF"),
    ]
    for axis, (mean_key, low_key, high_key, dof, title) in zip(axes[:2], consistency):
        mean = metric(row, mean_key) / dof
        low = metric(row, low_key) / dof
        high = metric(row, high_key) / dof
        axis.errorbar(
            [0], [mean], yerr=[[mean - low], [high - mean]],
            fmt="o", markersize=7, capsize=4, color=BLUE, linewidth=1.8,
        )
        axis.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
        axis.set_xlim(-0.6, 0.6)
        axis.set_xticks([0], ["Nominal"])
        axis.set_ylim(0.5, 1.5)
        axis.set_ylabel("Normalized mean")
        axis.set_title(title)

    angle_metrics = [
        ("attitude_estimation_rmse_rad", "Attitude estimate RMSE", BLUE),
        ("pointing_rms_rad", "True pointing RMS", VERMILLION),
    ]
    for index, (prefix, label, color) in enumerate(angle_metrics):
        mean = metric(row, f"{prefix}_mean") * 180.0 / np.pi
        low = metric(row, f"{prefix}_ci95_lower") * 180.0 / np.pi
        high = metric(row, f"{prefix}_ci95_upper") * 180.0 / np.pi
        axes[2].errorbar(
            [index], [mean], yerr=[[mean - low], [high - mean]],
            fmt="o", markersize=6, capsize=3, color=color, label=label,
        )
    axes[2].set_xticks([0, 1], ["Estimator", "Spacecraft"])
    axes[2].set_ylabel("Angle RMS (deg)")
    axes[2].set_title("Accuracy and true pointing")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("E0 · Nominal 1,000 s baseline", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.5,
        -0.025,
        "10 independent runs · bars are 95% Student-t intervals over run means",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_actuator_authority(rows: list[dict[str, str]], output: Path) -> None:
    cases = [
        ("nominal", "Nominal stress"),
        ("Q_scale_10", "Process noise ×10"),
        ("R_scale_10", "Measurement noise ×10"),
        ("Q_R_scale_10", "Q and R ×10"),
        ("inertia_mismatch_25pct", "25% inertia mismatch"),
        ("zero_initial_rate", "Zero initial rate"),
        ("half_torque_authority", "Half torque limit"),
        ("double_torque_authority", "Double torque limit"),
        ("low_wheel_momentum_limit", "1 N·m·s wheel limit"),
        ("near_180deg_attitude", "170° initial attitude"),
    ]
    series = case_rows(rows, "E6", [case for case, _ in cases])
    labels = [label for _, label in cases]
    y = np.arange(len(cases))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 7.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.0], "wspace": 0.28},
    )
    style_axes(axes)

    metrics = [
        ("pointing_rms_rad", "Pointing RMS (deg)", 180.0 / np.pi, 180.0),
        ("command_torque_saturation_fraction", "Command torque saturated (%)", 100.0, 105.0),
        ("peak_wheel_momentum_utilization", "Peak wheel momentum (%)", 100.0, 105.0),
    ]
    for axis, (name, title, scale, xmax) in zip(axes, metrics):
        values = np.array([metric(row, f"{name}_mean") * scale for row in series])
        lower = np.array([metric(row, f"{name}_ci95_lower") * scale for row in series])
        upper = np.array([metric(row, f"{name}_ci95_upper") * scale for row in series])
        colors = [VERMILLION if case == "nominal" else BLUE for case, _ in cases]
        for yi, value, low, high, color in zip(y, values, lower, upper, colors):
            axis.errorbar(
                value,
                yi,
                xerr=[[value - low], [high - value]],
                fmt="o",
                markersize=5.5,
                capsize=2.5,
                color=color,
                ecolor=color,
                linewidth=1.4,
            )
        axis.set_title(title, fontsize=10, pad=10)
        axis.set_xlim(0, xmax)
        axis.set_ylim(-0.6, len(cases) - 0.4)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        if scale == 100.0:
            axis.axvline(100, color=MUTED, linestyle="--", linewidth=1.0)

    fig.suptitle(
        "E6 · Closed-loop stress and control authority",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Points show 10-run means; bars show run-level 95% Student-t intervals. Each run is 100 s.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs" / "e0_e7_campaign_summary.csv",
        help="Refreshed E0-E7 campaign summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "figures",
        help="Directory for README-ready PNG figures.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_summary(args.input)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )
    save_estimator_consistency(rows, args.output_dir / "e1_inertia_consistency.png")
    save_outlier_gating(rows, args.output_dir / "e4_outlier_gating.png")
    save_actuator_authority(rows, args.output_dir / "e6_control_authority.png")
    save_nominal_baseline(rows, args.output_dir / "e0_nominal_baseline.png")
    print(f"Wrote README figures to {args.output_dir}")


if __name__ == "__main__":
    main()
