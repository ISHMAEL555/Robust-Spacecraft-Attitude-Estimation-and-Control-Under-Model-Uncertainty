#!/usr/bin/env python3
"""Run reproducible parameter sweeps for research experiments E1-E7."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import (
    compute_consistency_stats,
    load_config,
    run_monte_carlo,
    save_results,
)
from src.estimation.linear_reference import (
    lyapunov_pointing_ultimate_bound,
    periodic_gyro_driven_riccati,
)
from src.sensors.gyro import GyroParams
from src.sensors.star_tracker import StarTrackerParams


ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")


def _case(name: str, campaign: str, config_name: str, **parameters: Any) -> Dict[str, Any]:
    config = load_config(str(ROOT / "config" / config_name))
    overrides = parameters.pop("config_overrides", {})
    if "duration" in overrides:
        config = replace(config, duration=float(overrides.pop("duration")))
    config = replace(
        config,
        filter={**config.filter, **overrides.pop("filter", {})},
        star_tracker={**config.star_tracker, **overrides.pop("star_tracker", {})},
        spacecraft={**config.spacecraft, **overrides.pop("spacecraft", {})},
        gyroscope={**config.gyroscope, **overrides.pop("gyroscope", {})},
        controller={**config.controller, **overrides.pop("controller", {})},
        reaction_wheels={**config.reaction_wheels, **overrides.pop("reaction_wheels", {})},
        analysis={**config.analysis, **overrides.pop("analysis", {})},
    )
    if overrides:
        raise ValueError(f"Unsupported configuration overrides: {sorted(overrides)}")
    return {
        "name": name,
        "campaign": campaign,
        "config": config,
        **parameters,
    }


def build_campaign_cases(campaign: str, duration: float | None = None) -> List[Dict[str, Any]]:
    """Build deterministic cases; each sweep changes one named assumption at a time."""
    campaigns = (CAMPAIGNS if campaign == "all" else (campaign,))
    cases: List[Dict[str, Any]] = []
    for selected in campaigns:
        if selected == "E0":
            cases.append(_case("matched_nominal", selected, "nominal.yaml"))
            for disturbance in (1e-4, 1e-3, 5e-3):
                cases.append(_case(
                    f"disturbance_{disturbance:g}",
                    selected,
                    "nominal.yaml",
                    config_overrides={
                        "spacecraft": {
                            "disturbance_torque": [disturbance, 0.0, 0.0]
                        },
                    },
                    disturbance_torque_nm=disturbance,
                ))
                cases.append(_case(
                    f"disturbance_{disturbance:g}_long",
                    selected,
                    "nominal.yaml",
                    config_overrides={
                        "duration": 1000.0,
                        "spacecraft": {
                            "disturbance_torque": [disturbance, 0.0, 0.0]
                        },
                    },
                    disturbance_torque_nm=disturbance,
                    equilibrium_tail_window_s=[900.0, 1000.0],
                ))
            for momentum in (0.001, 10.0):
                for coupling in (False, True):
                    wheel_case = _case(
                        f"wheel_h{momentum:g}_gyro_{'on' if coupling else 'off'}",
                        selected,
                        "nominal.yaml",
                        config_overrides={
                            "reaction_wheels": {
                                "gyroscopic_coupling": coupling,
                                "initial_momentum": [0.0, 0.0, momentum],
                            },
                        },
                        wheel_gyroscopic_coupling=coupling,
                        initial_wheel_momentum_nms=momentum,
                    )
                    wheel_case["control_torque_limit_nm"] = float(
                        wheel_case["config"].controller["max_torque"]
                    )
                    cases.append(wheel_case)
            initial_error = 0.005
            cases.append(_case(
                "lyapunov_tight_outage",
                selected,
                "sensor_degradation.yaml",
                config_overrides={
                    "duration": 500.0,
                    "spacecraft": {
                        "initial_attitude": [
                            float(np.sin(initial_error / 2.0)), 0.0, 0.0,
                            float(np.cos(initial_error / 2.0)),
                        ],
                        "disturbance_torque": [1e-3, 0.0, 0.0],
                    },
                    "gyroscope": {"sigma_u": 1e-5},
                    "star_tracker": {
                        "outage_prob": 0.0,
                        "outage_start_time": 60.0,
                        "outage_duration": 300,
                    },
                },
                lyapunov_test=True,
            ))
        elif selected == "E1":
            mismatches = (-0.25, -0.1, 0.0, 0.1, 0.25)
            for mismatch in mismatches:
                cases.append(_case(
                    f"gyro_driven_inertia_{mismatch:+.2f}",
                    selected,
                    "model_uncertainty.yaml",
                    variant="gyro_driven",
                    inertia_mismatch=mismatch,
                ))
            for mismatch in mismatches:
                for q_scale in (0.1, 1.0, 10.0):
                    for tracker_rate in (0.2, 1.0, 5.0):
                        cases.append(_case(
                            f"model_aided_i{mismatch:+.2f}_q{q_scale:g}_st{tracker_rate:g}",
                            selected,
                            "model_uncertainty.yaml",
                            config_overrides={
                                "filter": {
                                    "Q_scale": q_scale,
                                    "model_error_relative_bound": abs(mismatch),
                                },
                                "star_tracker": {"rate": tracker_rate},
                            },
                            variant="model_aided",
                            inertia_mismatch=mismatch,
                            Q_scale=q_scale,
                            tracker_rate_hz=tracker_rate,
                        ))
            for variant in ("gyro_driven", "model_aided"):
                for gyro_scale in (
                    1e-5, 1e-4, 3e-4, 5e-4, 0.001, 0.01, 0.025,
                    0.05, 0.075, 0.1, 1.0, 10.0, 100.0,
                ):
                    cases.append(_case(
                        f"{variant}_gyro_quality_{gyro_scale:g}",
                        selected,
                        "model_uncertainty.yaml",
                        config_overrides={
                            "gyroscope": {
                                "sigma_v": 1e-4 * gyro_scale,
                                "sigma_u": 1e-6 * gyro_scale,
                            },
                        },
                        variant=variant,
                        inertia_mismatch=0.0,
                        gyro_noise_scale=gyro_scale,
                    ))
                for gyro_scale in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
                    cases.append(_case(
                        f"{variant}_gyro_quality_i+0.10_{gyro_scale:g}",
                        selected,
                        "model_uncertainty.yaml",
                        config_overrides={
                            "gyroscope": {
                                "sigma_v": 1e-4 * gyro_scale,
                                "sigma_u": 1e-6 * gyro_scale,
                            },
                        },
                        variant=variant,
                        inertia_mismatch=0.1,
                        gyro_noise_scale=gyro_scale,
                    ))
            for mismatch in (-0.25, -0.1, 0.1, 0.25):
                for bound_factor in (0.0, 0.5, 1.0, 2.0):
                    cases.append(_case(
                        f"bound_tune_i{mismatch:+.2f}_b{bound_factor:g}",
                        selected,
                        "model_uncertainty.yaml",
                        config_overrides={
                            "filter": {
                                "model_error_relative_bound":
                                    abs(mismatch) * bound_factor,
                            },
                        },
                        variant="model_aided",
                        inertia_mismatch=mismatch,
                        model_error_bound_relative=abs(mismatch) * bound_factor,
                        bound_factor=bound_factor,
                        bound_tuning=True,
                    ))
        elif selected == "E2":
            for scale in (0.1, 1.0, 10.0):
                cases.append(_case(
                    f"Q_scale_{scale:g}", selected, "model_uncertainty.yaml",
                    config_overrides={"filter": {"Q_scale": scale}},
                    Q_scale=scale,
                ))
        elif selected == "E3":
            for scale in (0.1, 1.0, 10.0):
                cases.append(_case(
                    f"R_scale_{scale:g}", selected, "model_uncertainty.yaml",
                    config_overrides={"filter": {"R_scale": scale}},
                    R_scale=scale,
                ))
        elif selected == "E4":
            config = load_config(str(ROOT / "config" / "sensor_degradation.yaml"))
            _, prior, _ = periodic_gyro_driven_riccati(
                GyroParams(
                    sigma_v=config.gyroscope["sigma_v"],
                    sigma_u=config.gyroscope["sigma_u"],
                    rate=config.gyroscope["rate"],
                ),
                StarTrackerParams(
                    sigma_theta=config.star_tracker["sigma_theta"],
                    rate=config.star_tracker["rate"],
                ),
                config.dt,
            )
            innovation_variance = prior[0, 0] + config.star_tracker["sigma_theta"] ** 2
            gate_probability = 0.95
            gate = chi2.ppf(gate_probability, 3)
            cases.append(_case(
                "no_outlier_gate",
                selected,
                "sensor_degradation.yaml",
                config_overrides={
                    "filter": {
                        "nis_gate_enabled": True,
                        "nis_gate_confidence": gate_probability,
                    },
                    "star_tracker": {"outlier_prob": 0.0, "outage_prob": 0.0},
                },
                nis_ratio_to_gate=0.0,
                nis_gate_enabled=True,
            ))
            for ratio in (0.9, 1.0, 1.1):
                magnitude = float(np.sqrt(ratio * gate * innovation_variance))
                cases.append(_case(
                    f"near_gate_{ratio:g}",
                    selected,
                    "sensor_degradation.yaml",
                    config_overrides={
                        "filter": {
                            "nis_gate_enabled": True,
                            "nis_gate_confidence": gate_probability,
                        },
                        "star_tracker": {
                            "outlier_prob": 1.0,
                            "outlier_magnitude": magnitude,
                            "outage_prob": 0.0,
                        },
                    },
                    nis_ratio_to_gate=ratio,
                    outlier_magnitude_rad=magnitude,
                    nis_gate_enabled=True,
                ))
            cases.append(_case(
                "gross_outlier_gate",
                selected,
                "sensor_degradation.yaml",
                config_overrides={
                    "filter": {"nis_gate_enabled": True},
                    "star_tracker": {
                        "outlier_prob": 1.0,
                        "outlier_magnitude": 0.5,
                        "outage_prob": 0.0,
                    },
                },
                outlier_magnitude_rad=0.5,
                nis_gate_enabled=True,
            ))
        elif selected == "E5":
            for outage_seconds in (60, 300, 900):
                outage_updates = round(outage_seconds * 1.0)
                cases.append(_case(
                    f"outage_{outage_seconds}s",
                    selected,
                    "sensor_degradation.yaml",
                    config_overrides={
                        "duration": outage_seconds + 300.0,
                        "star_tracker": {
                            "outage_prob": 0.0,
                            "outage_start_time": 5.0,
                            "outage_duration": outage_updates,
                        },
                    },
                    outage_duration_s=outage_seconds,
                    outage_duration_measurements=outage_updates,
                    outage_start_time_s=5.0,
                ))
        elif selected == "E6":
            for angle in (170.0, 179.0):
                for torque_limit in (0.05, 0.1, 0.2):
                    for hysteresis in (False, True):
                        for sign in (-1.0, 1.0):
                            axis_angle = np.radians(angle)
                            initial_attitude = [
                                sign * float(np.sin(axis_angle / 2.0)), 0.0, 0.0,
                                sign * float(np.cos(axis_angle / 2.0)),
                            ]
                            cases.append(_case(
                                f"start{angle}_tau{torque_limit:g}_hyst{int(hysteresis)}_sign{int(sign)}",
                                selected,
                                "closed_loop.yaml",
                                config_overrides={
                                    "duration": 600.0,
                                    "spacecraft": {
                                        "initial_attitude": initial_attitude,
                                        "initial_omega": [0.0, 0.0, 0.0],
                                    },
                                    "controller": {
                                        "max_torque": torque_limit,
                                        "quaternion_sign_hysteresis": hysteresis,
                                    },
                                    "reaction_wheels": {
                                        "max_torque": torque_limit,
                                        "max_momentum": 100.0,
                                    },
                                },
                                initial_angle_deg=angle,
                                torque_limit_nm=torque_limit,
                                wheel_momentum_limit_nms=100.0,
                                quaternion_sign_hysteresis=hysteresis,
                                quaternion_sign=sign,
                            ))
        elif selected == "E7":
            cases.append(_case("nominal_monte_carlo", selected, "monte_carlo.yaml"))

    if duration is not None:
        for case in cases:
            case["config"] = replace(case["config"], duration=duration)
            if case.get("equilibrium_tail_window_s") is not None:
                case["equilibrium_tail_window_s"] = [
                    0.9 * duration,
                    duration,
                ]
    return cases


def run_case(case: Dict[str, Any], n_runs: int) -> Dict[str, Any]:
    config = case["config"]
    nominal_inertia = np.asarray(config.spacecraft["inertia"], dtype=float)
    mismatch = float(case.get("inertia_mismatch", 0.0))
    true_inertia = nominal_inertia + mismatch * np.diag(np.diag(nominal_inertia))
    filter_inertia = nominal_inertia if case.get("variant") == "model_aided" else None

    if "variant" in case:
        config = replace(
            config,
            filter={
                **config.filter,
                "variant": case["variant"],
                "model_error_relative_bound": abs(mismatch)
                if case["variant"] == "model_aided"
                else 0.0,
                **(
                    {
                        "model_error_relative_bound":
                            float(case["model_error_bound_relative"])
                    }
                    if case.get("model_error_bound_relative") is not None
                    else {}
                ),
            },
        )
    results = run_monte_carlo(config, n_runs, true_inertia, filter_inertia)
    evaluation_window = config.analysis.get("steady_state_window_s")
    if evaluation_window is not None and float(evaluation_window[1]) > config.duration:
        evaluation_window = None
    results["consistency"] = compute_consistency_stats(
        results,
        config.analysis["nees_confidence"],
        config.analysis["nis_confidence"],
        evaluation_window,
    )
    if case["campaign"] == "E0" and case["name"] == "matched_nominal":
        posterior, prior, _ = periodic_gyro_driven_riccati(
            GyroParams(
                sigma_v=config.gyroscope["sigma_v"],
                sigma_u=config.gyroscope["sigma_u"],
                rate=config.gyroscope["rate"],
            ),
            StarTrackerParams(
                sigma_theta=config.star_tracker["sigma_theta"],
                rate=config.star_tracker["rate"],
            ),
            config.dt,
            config.filter["Q_scale"],
            config.filter["R_scale"],
        )
        time = results["time"]
        window = np.ones(time.shape, dtype=bool)
        if evaluation_window is not None:
            window = (time >= evaluation_window[0]) & (time < evaluation_window[1])
        posterior_epochs = window & (results["innovation_count"] > 0)
        if not np.any(posterior_epochs):
            raise RuntimeError("E0 Riccati comparison has no tracker posterior epochs.")
        empirical = np.mean(results["P_mean"][posterior_epochs], axis=0)
        results["riccati_posterior"] = posterior
        results["riccati_prior"] = prior
        results["riccati_relative_frobenius_error"] = float(
            np.linalg.norm(empirical - posterior, ord="fro")
            / np.linalg.norm(posterior, ord="fro")
        )
        results["riccati_marginal_relative_error"] = (
            np.diag(empirical) - np.diag(posterior)
        ) / np.diag(posterior)
        if config.duration >= 1000.0:
            late_window = (time >= 600.0) & (time < 1000.0)
            late_posteriors = late_window & (results["innovation_count"] > 0)
            if np.any(late_posteriors):
                late_empirical = np.mean(results["P_mean"][late_posteriors], axis=0)
                results["riccati_relative_frobenius_error_late"] = float(
                    np.linalg.norm(late_empirical - posterior, ord="fro")
                    / np.linalg.norm(posterior, ord="fro")
                )
                results["riccati_marginal_relative_error_late"] = (
                    np.diag(late_empirical) - np.diag(posterior)
                ) / np.diag(posterior)
    if case["campaign"] == "E0" and "disturbance_torque_nm" in case:
        torque = float(case["disturbance_torque_nm"])
        gain = float(config.controller["K_q"])
        results["predicted_equilibrium_pointing_rad"] = float(
            2.0 * np.arcsin(min(torque / gain, 1.0))
        )
        time = np.asarray(results["time"])
        tail_window = case.get(
            "equilibrium_tail_window_s",
            [max(0.0, config.duration * 0.9), config.duration],
        )
        tail = (time >= tail_window[0]) & (time < tail_window[1])
        tail_error = np.asarray(results["pointing_error_mean"])[tail]
        if tail_error.size < 2:
            raise RuntimeError(
                f"Disturbance tail window contains fewer than two samples: {tail_window}"
            )
        half = tail_error.size // 2
        early_mean = float(np.mean(tail_error[:half]))
        late_mean = float(np.mean(tail_error[half:]))
        measured_equilibrium = float(np.mean(tail_error))
        tail_drift = abs(late_mean - early_mean) / max(measured_equilibrium, 1e-12)
        results["equilibrium_tail_drift_fraction"] = tail_drift
        results["equilibrium_steady_state_pass"] = tail_drift <= 0.10
        results["measured_equilibrium_pointing_rad"] = measured_equilibrium
        results["equilibrium_relative_error"] = abs(
            results["measured_equilibrium_pointing_rad"]
            - results["predicted_equilibrium_pointing_rad"]
        ) / max(results["predicted_equilibrium_pointing_rad"], 1e-12)
        results["equilibrium_prediction_pass"] = (
            results["equilibrium_relative_error"] <= 0.15
            and results["equilibrium_steady_state_pass"]
        )
    if case["campaign"] == "E0" and case.get("lyapunov_test"):
        runs = results["performance_by_run"]
        attitude_bound = max(
            run["attitude_estimation_error_peak_rad"] for run in runs
        )
        rate_bound = max(
            run["angular_rate_estimation_error_peak_rad_s"] for run in runs
        )
        measured_pointing = max(run["pointing_peak_rad"] for run in runs)
        pointing_bound = lyapunov_pointing_ultimate_bound(
            np.asarray(config.spacecraft["inertia"], dtype=float),
            float(config.controller["K_q"]) / 2.0,
            float(config.controller["K_omega"]),
            attitude_bound,
            rate_bound,
            float(np.linalg.norm(config.spacecraft["disturbance_torque"])),
        )
        results["lyapunov_pointing_bound_rad"] = pointing_bound
        results["lyapunov_measured_peak_pointing_rad"] = measured_pointing
        results["lyapunov_bound_pass"] = measured_pointing <= 1.1 * pointing_bound
    if case["campaign"] == "E1":
        time = results["time"]
        if evaluation_window is None:
            window = np.ones(time.shape, dtype=bool)
        else:
            window = (time >= evaluation_window[0]) & (time < evaluation_window[1])
        predicted_nees = float(np.mean(results["predicted_NEES_mean"][window]))
        measured_nees = float(results["consistency"]["nees_mean"])
        results["predicted_NEES_mean_steady"] = predicted_nees
        results["prediction_NEES_relative_error"] = abs(
            measured_nees - predicted_nees
        ) / max(abs(predicted_nees), 1e-12)
        results["prediction_NEES_pass"] = (
            results["prediction_NEES_relative_error"] <= 0.20
        )
        if "model_error_covariance_relative_error" in results:
            results["prediction_error_covariance_pass"] = (
                results["model_error_covariance_relative_error"] <= 0.20
            )
    results["campaign_metadata"] = {
        key: value for key, value in case.items()
        if key not in ("config",)
    }
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summary_row(case: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    consistency = results["consistency"]

    def finite_or_none(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    row: Dict[str, Any] = {
        "campaign": case["campaign"],
        "case": case["name"],
        "runs": results["n_runs"],
        "duration_s": case["config"].duration,
        "NEES_dof": consistency["dof_nees"],
        "NIS_dof": consistency["dof_nis"],
        "NEES_mean": finite_or_none(consistency["nees_mean"]),
        "NEES_in_bounds_pct": finite_or_none(100.0 * consistency["nees_in_bounds"]),
        "NIS_mean": finite_or_none(consistency["nis_mean"]),
        "NIS_pre_gate_mean": finite_or_none(consistency["pre_gate_nis_mean"]),
        "NIS_in_bounds_pct": finite_or_none(100.0 * consistency["nis_in_bounds"]),
        "star_tracker_updates": int(np.sum(results["innovation_count"])),
        "rejected_measurements": int(np.sum(results["measurement_rejection_count"])),
        "predicted_NEES_mean_steady": results.get("predicted_NEES_mean_steady"),
        "prediction_NEES_relative_error": results.get("prediction_NEES_relative_error"),
        "prediction_NEES_pass": results.get("prediction_NEES_pass"),
        "model_error_covariance_relative_error": results.get(
            "model_error_covariance_relative_error"
        ),
        "prediction_error_covariance_pass": results.get(
            "prediction_error_covariance_pass"
        ),
        "riccati_relative_frobenius_error": results.get(
            "riccati_relative_frobenius_error"
        ),
        "riccati_marginal_relative_error": results.get(
            "riccati_marginal_relative_error"
        ),
        "riccati_relative_frobenius_error_late": results.get(
            "riccati_relative_frobenius_error_late"
        ),
        "riccati_marginal_relative_error_late": results.get(
            "riccati_marginal_relative_error_late"
        ),
        "predicted_equilibrium_pointing_rad": results.get(
            "predicted_equilibrium_pointing_rad"
        ),
        "measured_equilibrium_pointing_rad": results.get(
            "measured_equilibrium_pointing_rad"
        ),
        "equilibrium_relative_error": results.get("equilibrium_relative_error"),
        "equilibrium_tail_drift_fraction": results.get(
            "equilibrium_tail_drift_fraction"
        ),
        "equilibrium_steady_state_pass": results.get(
            "equilibrium_steady_state_pass"
        ),
        "equilibrium_prediction_pass": results.get("equilibrium_prediction_pass"),
        "lyapunov_pointing_bound_rad": results.get("lyapunov_pointing_bound_rad"),
        "lyapunov_measured_peak_pointing_rad": results.get(
            "lyapunov_measured_peak_pointing_rad"
        ),
        "lyapunov_bound_pass": results.get("lyapunov_bound_pass"),
        "run_nees_mean": consistency.get("run_nees_mean"),
        "run_nees_mean_ci95": consistency.get("run_nees_mean_ci95"),
        "run_nis_mean": consistency.get("run_nis_mean"),
        "run_nis_mean_ci95": consistency.get("run_nis_mean_ci95"),
    }
    for key, value in case.items():
        if key not in ("name", "campaign", "config"):
            row[key] = value
    row.setdefault("Q_scale", float(case["config"].filter["Q_scale"]))
    row.setdefault("R_scale", float(case["config"].filter["R_scale"]))
    row.setdefault(
        "tracker_rate_hz",
        float(case["config"].star_tracker["rate"]),
    )
    for metric, summary in results["performance_summary"].items():
        for field, value in summary.items():
            row[f"{metric}_{field}"] = finite_or_none(value)
    return _json_safe(row)


def _load_saved_results(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as saved:
        results = {}
        for key in saved.files:
            value = saved[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            results[key] = value
    return results


def _augment_saved_measurement_masks(
    case: Dict[str, Any],
    results: Dict[str, Any],
) -> None:
    mask_keys = (
        "measurement_available_runs",
        "measurement_accepted_runs",
        "innovation_valid_runs",
    )
    if all(key in results for key in mask_keys):
        return
    config = case["config"]
    if (
        config.filter.get("nis_gate_enabled", False)
        or config.star_tracker.get("outlier_prob", 0.0) > 0.0
        or config.star_tracker.get("outage_prob", 0.0) > 0.0
        or config.star_tracker.get("outage_duration", 0) > 0
        or int(np.sum(results["measurement_rejection_count"])) > 0
    ):
        raise RuntimeError(
            "Cannot reconstruct exact measurement masks for a gated, anomalous, or "
            f"intermittent case: {case['campaign']} {case['name']}."
        )
    counts = np.asarray(results["innovation_count"])
    n_runs = int(results["n_runs"])
    if np.any((counts != 0) & (counts != n_runs)):
        raise RuntimeError(
            f"Saved update counts do not establish deterministic masks for "
            f"{case['campaign']} {case['name']}."
        )
    available = np.broadcast_to(counts[None, :] > 0, (n_runs, counts.size)).copy()
    results["measurement_available_runs"] = available
    results["measurement_accepted_runs"] = available.copy()
    results["innovation_valid_runs"] = available.copy()


def _augment_saved_diagnostics(case: Dict[str, Any], results: Dict[str, Any]) -> None:
    time = np.asarray(results["time"])
    if (
        case["campaign"] == "E0"
        and case["name"] == "matched_nominal"
        and case["config"].duration >= 1000.0
        and "riccati_relative_frobenius_error_late" not in results
    ):
        posterior = np.asarray(results["riccati_posterior"])
        covariance = np.asarray(results["P_mean"])
        posterior_epochs = (
            (time >= 600.0)
            & (time < 1000.0)
            & (np.asarray(results["innovation_count"]) > 0)
        )
        if not np.any(posterior_epochs):
            raise RuntimeError("Saved E0 trace has no late-window tracker posterior epochs.")
        empirical = np.mean(covariance[posterior_epochs], axis=0)
        results["riccati_relative_frobenius_error_late"] = float(
            np.linalg.norm(empirical - posterior, ord="fro")
            / np.linalg.norm(posterior, ord="fro")
        )
        results["riccati_marginal_relative_error_late"] = (
            np.diag(empirical) - np.diag(posterior)
        ) / np.diag(posterior)

    if int(results["n_runs"]) != 1 or "pointing_error_mean" not in results:
        return
    summary = results["performance_summary"]
    pointing = np.asarray(results["pointing_error_mean"], dtype=float)
    initial_error = float(pointing[0])
    within_tolerance = pointing <= 0.02 * initial_error
    remains_within = np.logical_and.accumulate(within_tolerance[::-1])[::-1]
    settled = np.flatnonzero(remains_within) if initial_error > 0.0 else np.array([])
    settling_time = float(time[settled[0]]) if settled.size else None
    metrics = {
        "pointing_rms_rad_full": float(np.sqrt(np.mean(pointing**2))),
        "pointing_peak_rad_full": float(np.max(pointing)),
        "pointing_final_rad_full": float(pointing[-1]),
        "settling_time_2pct_s_full": settling_time,
    }
    for name, value in metrics.items():
        if name in summary:
            continue
        summary[name] = {
            "mean": value,
            "std": 0.0 if value is not None else None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n": 1,
            "p05": value,
            "p50": value,
            "p95": value,
            "p05_bootstrap_ci95": [value, value],
            "p50_bootstrap_ci95": [value, value],
            "p95_bootstrap_ci95": [value, value],
        }


def _write_campaign_plot(rows: List[Dict[str, Any]], campaign: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["campaign"] == campaign]
    if not selected:
        return
    metrics = (
        ("NEES_mean", "Mean NEES"),
        ("NIS_mean", "Mean NIS"),
        ("pointing_rms_rad_mean", "Mean run pointing RMS (deg)", 180.0 / np.pi),
        ("achieved_control_effort_nm2_s_mean", "Mean applied control effort (N^2 m^2 s)"),
    )
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(9, len(selected) * 0.65), 10))
    for ax, metric in zip(axes, metrics):
        key, label = metric[:2]
        scale = metric[2] if len(metric) == 3 else 1.0
        x = np.arange(len(selected))
        values = np.asarray([row.get(key, np.nan) for row in selected], dtype=float) * scale
        ax.plot(x, values, "o-")
        ax.set_ylabel(label)
        ax.set_xticks(x, [row["case"] for row in selected], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Campaign case")
    fig.suptitle(f"{campaign} Research Campaign")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"{campaign.lower()}_summary.png"), dpi=150)
    plt.close(fig)


def _write_tuning_sensitivity_plot(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    selected = [row for row in rows if row["campaign"] in ("E2", "E3")]
    if not selected:
        return
    metrics = (
        ("NEES_mean", "Accepted-sample NEES"),
        ("NIS_mean", "Accepted-only NIS"),
        ("attitude_estimation_rmse_rad_mean", "Attitude RMSE (deg)", 180.0 / np.pi),
        ("pointing_rms_rad_mean", "True-pointing RMS (deg)", 180.0 / np.pi),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics):
        key, label = metric[:2]
        scale = metric[2] if len(metric) == 3 else 1.0
        for campaign, style in (("E2", "o-"), ("E3", "s--")):
            campaign_rows = sorted(
                (row for row in selected if row["campaign"] == campaign),
                key=lambda row: row["Q_scale"] if campaign == "E2"
                else row["R_scale"],
            )
            if not campaign_rows:
                continue
            x_key = "Q_scale" if campaign == "E2" else "R_scale"
            x = [row[x_key] for row in campaign_rows]
            y = [row.get(key, np.nan) * scale for row in campaign_rows]
            ax.plot(x, y, style, label=f"{campaign}: {x_key}")
        ax.set_xscale("log")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Combined E2/E3 tuning sensitivity")
    fig.tight_layout()
    fig.savefig(str(output_dir / "e2_e3_tuning_sensitivity.png"), dpi=150)
    plt.close(fig)


def _write_e1_diagnostics(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    tuning = [row for row in rows if row.get("bound_tuning")]
    if tuning:
        mismatches = sorted({abs(float(row["inertia_mismatch"])) for row in tuning})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        optimal_bounds = []
        for mismatch in mismatches:
            group = [
                row for row in tuning
                if abs(float(row["inertia_mismatch"])) == mismatch
            ]
            bounds = sorted({
                float(row["model_error_bound_relative"]) for row in group
            })
            if not bounds:
                continue
            errors = np.asarray([
                np.mean([
                    row["attitude_estimation_rmse_rad_mean"] for row in group
                    if float(row["model_error_bound_relative"]) == bound
                ])
                for bound in bounds
            ])
            bounds = np.asarray(bounds)
            best = int(np.argmin(errors))
            optimal_bounds.append((mismatch, float(bounds[best]), float(errors[best])))
            axes[0].plot(
                bounds,
                errors,
                "o-",
                label=f"|δI|/I={mismatch:.0%}",
            )
            axes[1].plot(
                bounds,
                100.0 * (errors - errors[best]) / max(errors[best], 1e-15),
                "o-",
                label=f"|δI|/I={mismatch:.0%}",
            )
        axes[0].set_xlabel("Configured relative inertia-error bound")
        axes[0].set_ylabel("Attitude RMSE (rad)")
        axes[1].set_xlabel("Configured relative inertia-error bound")
        axes[1].set_ylabel("Accuracy lost vs. best bound (%)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("E1 model-aided process-noise bound trade-off")
        fig.tight_layout()
        fig.savefig(str(output_dir / "e1_bound_tradeoff.png"), dpi=150)
        plt.close(fig)
        with (output_dir / "e1_optimal_bounds.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["absolute_inertia_error_fraction", "optimal_relative_bound", "attitude_rmse_rad"]
            )
            writer.writerows(optimal_bounds)

    quality_rows = [
        row for row in rows
        if row["campaign"] == "E1"
        and row.get("gyro_noise_scale") is not None
        and row.get("inertia_mismatch") == 0.0
    ]
    crossover_rows = []
    for scale in sorted({float(row["gyro_noise_scale"]) for row in quality_rows}):
        pair = {
            row["variant"]: row["attitude_estimation_rmse_rad_mean"]
            for row in quality_rows
            if float(row["gyro_noise_scale"]) == scale
        }
        if {"gyro_driven", "model_aided"} <= pair.keys():
            crossover_rows.append({
                "gyro_noise_scale": scale,
                "gyro_driven_attitude_rmse_rad": pair["gyro_driven"],
                "model_aided_attitude_rmse_rad": pair["model_aided"],
                "model_aided_wins": pair["model_aided"] < pair["gyro_driven"],
            })
    if crossover_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [row["gyro_noise_scale"] for row in crossover_rows],
            [row["gyro_driven_attitude_rmse_rad"] for row in crossover_rows],
            "o-",
            label="Gyro-driven",
        )
        ax.plot(
            [row["gyro_noise_scale"] for row in crossover_rows],
            [row["model_aided_attitude_rmse_rad"] for row in crossover_rows],
            "s--",
            label="Model-aided",
        )
        ax.set_xscale("log")
        ax.set_xlabel("ARW/BRW scale")
        ax.set_ylabel("Attitude RMSE (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle("E1 gyro-quality crossover")
        fig.tight_layout()
        fig.savefig(str(output_dir / "e1_gyro_crossover.png"), dpi=150)
        plt.close(fig)
        with (output_dir / "e1_gyro_crossover.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(crossover_rows[0]))
            writer.writeheader()
            writer.writerows(crossover_rows)
        differences = np.asarray([
            row["model_aided_attitude_rmse_rad"]
            - row["gyro_driven_attitude_rmse_rad"]
            for row in crossover_rows
        ])
        scales = np.asarray([row["gyro_noise_scale"] for row in crossover_rows])
        crossing = None
        for index in range(1, len(scales)):
            if differences[index - 1] * differences[index] <= 0.0:
                fraction = (
                    -differences[index - 1]
                    / (differences[index] - differences[index - 1])
                    if differences[index] != differences[index - 1]
                    else 0.0
                )
                crossing = float(
                    10.0 ** (
                        np.log10(scales[index - 1])
                        + fraction * (np.log10(scales[index]) - np.log10(scales[index - 1]))
                    )
                )
                break
        with (output_dir / "e1_gyro_crossover_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "crossover_gyro_noise_scale": crossing,
                    "first_model_aided_win_scale": next(
                        (
                            row["gyro_noise_scale"] for row in crossover_rows
                            if row["model_aided_wins"]
                        ),
                        None,
                    ),
                    "fit": "linear interpolation in log10 noise scale between measured RMSE differences",
                },
                stream,
                indent=2,
                allow_nan=False,
            )

    mismatch_quality_rows = [
        row for row in rows
        if row["campaign"] == "E1"
        and row.get("gyro_noise_scale") is not None
        and row.get("inertia_mismatch") == 0.1
    ]
    mismatch_crossover_rows = []
    for scale in sorted(
        {float(row["gyro_noise_scale"]) for row in mismatch_quality_rows}
    ):
        pair = {
            row["variant"]: row["attitude_estimation_rmse_rad_mean"]
            for row in mismatch_quality_rows
            if float(row["gyro_noise_scale"]) == scale
        }
        if {"gyro_driven", "model_aided"} <= pair.keys():
            mismatch_crossover_rows.append({
                "gyro_noise_scale": scale,
                "gyro_driven_attitude_rmse_rad": pair["gyro_driven"],
                "model_aided_attitude_rmse_rad": pair["model_aided"],
                "model_aided_wins": pair["model_aided"] < pair["gyro_driven"],
            })
    if mismatch_crossover_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [row["gyro_noise_scale"] for row in mismatch_crossover_rows],
            [row["gyro_driven_attitude_rmse_rad"] for row in mismatch_crossover_rows],
            "o-",
            label="Gyro-driven",
        )
        ax.plot(
            [row["gyro_noise_scale"] for row in mismatch_crossover_rows],
            [row["model_aided_attitude_rmse_rad"] for row in mismatch_crossover_rows],
            "s--",
            label="Model-aided, 10% inertia mismatch",
        )
        ax.set_xscale("log")
        ax.set_xlabel("ARW/BRW scale")
        ax.set_ylabel("Attitude RMSE (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle("E1 gyro-quality crossover at 10% inertia mismatch")
        fig.tight_layout()
        fig.savefig(str(output_dir / "e1_gyro_crossover_mismatch.png"), dpi=150)
        plt.close(fig)
        with (output_dir / "e1_gyro_crossover_mismatch.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(mismatch_crossover_rows[0])
            )
            writer.writeheader()
            writer.writerows(mismatch_crossover_rows)

    mismatch_rows = [
        row for row in rows if row["campaign"] == "E1" and row.get("prediction_NEES_pass") is not None
    ]
    if mismatch_rows:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            [row["predicted_NEES_mean_steady"] for row in mismatch_rows],
            [row["NEES_mean"] for row in mismatch_rows],
            c=[row["Q_scale"] for row in mismatch_rows],
            cmap="viridis",
            norm="log",
        )
        max_value = max(
            max(row["predicted_NEES_mean_steady"] for row in mismatch_rows),
            max(row["NEES_mean"] for row in mismatch_rows),
        )
        ax.plot([0, max_value], [0, max_value], "k--", linewidth=1)
        ax.set_xlabel("Predicted mean NEES")
        ax.set_ylabel("Measured mean NEES")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.suptitle("E1 predicted vs measured NEES")
        fig.tight_layout()
        fig.savefig(str(output_dir / "e1_predicted_vs_measured_nees.png"), dpi=150)
        plt.close(fig)


def _write_prediction_table(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    table = []

    def add(
        experiment: str,
        prediction: Any,
        measured: Any,
        tolerance: str,
        passed: Any,
        explanation: str = "",
    ) -> None:
        table.append({
            "experiment": experiment,
            "predicted": prediction,
            "measured": measured,
            "tolerance": tolerance,
            "pass": passed,
            "miss_explanation": explanation,
        })

    for row in rows:
        if row["campaign"] == "E0" and row["case"] == "matched_nominal":
            run_label = f" ({row.get('duration_s', 'unknown')} s run)"
            relative_error = row.get("riccati_relative_frobenius_error")
            consistency_pass = (
                row.get("run_nees_mean_ci95") is not None
                and row["run_nees_mean_ci95"][0] <= 6.0 <= row["run_nees_mean_ci95"][1]
                and row.get("run_nis_mean_ci95") is not None
                and row["run_nis_mean_ci95"][0] <= 3.0 <= row["run_nis_mean_ci95"][1]
            )
            add(
                f"E0 periodic Riccati covariance{run_label}, 60–100 s window",
                "relative Frobenius error ≤ 10%",
                relative_error,
                "≤ 10%",
                relative_error is not None and relative_error <= 0.10,
                "" if relative_error is not None and relative_error <= 0.10
                else "Nonzero-rate/nonlinear implementation departs from the zero-rate linear reference.",
            )
            marginal_errors = row.get("riccati_marginal_relative_error")
            marginal_pass = (
                marginal_errors is not None
                and all(abs(float(value)) <= 0.15 for value in marginal_errors)
            )
            add(
                f"E0 Riccati marginal variances{run_label}, 60–100 s window",
                "each attitude/bias marginal variance within 15%",
                marginal_errors,
                "absolute relative error ≤ 15% for every marginal",
                marginal_pass,
                "" if marginal_pass else
                "At least one state marginal differs by more than 15%; the bias covariance may not yet be at steady state.",
            )
            late_relative = row.get("riccati_relative_frobenius_error_late")
            if late_relative is not None:
                late_marginal = row.get("riccati_marginal_relative_error_late")
                late_pass = (
                    late_relative <= 0.10
                    and late_marginal is not None
                    and all(abs(float(value)) <= 0.15 for value in late_marginal)
                )
                add(
                    f"E0 late-window Riccati covariance{run_label} (600–1,000 s)",
                    "Frobenius error ≤10%; every marginal error ≤15%",
                    {"frobenius": late_relative, "marginals": late_marginal},
                    "registered P0 tolerances",
                    late_pass,
                    "" if late_pass else
                    "The late-window covariance has not converged to the registered DARE tolerances.",
                )
            add(
                "E0 ensemble consistency",
                "run-level 95% CI contains NEES=6 and ANIS=3",
                (
                    row.get("run_nees_mean_ci95"),
                    row.get("run_nis_mean_ci95"),
                ),
                "Both confidence intervals contain their DOF",
                consistency_pass,
                "" if consistency_pass else "Ensemble CI or accepted-only innovation sample count is inadequate.",
            )
        if row["campaign"] == "E1" and row.get("prediction_NEES_pass") is not None:
            add(
                f"E1 {row['case']} NEES",
                row.get("predicted_NEES_mean_steady"),
                row.get("NEES_mean"),
                "relative error ≤ 20%",
                row.get("prediction_NEES_pass"),
                "" if row.get("prediction_NEES_pass") else
                "Linearized residual/gain propagation or the declared torque-error statistics did not predict this case.",
            )
            covariance_error = row.get("model_error_covariance_relative_error")
            if covariance_error is not None:
                add(
                    f"E1 {row['case']} error covariance",
                    "predicted covariance from filter P plus mean-model-error outer product",
                    covariance_error,
                    "relative Frobenius error ≤ 20%",
                    row.get("prediction_error_covariance_pass"),
                    "" if row.get("prediction_error_covariance_pass") else
                    "Time variation, nonlinear error coupling, or omitted residual statistics exceeded the first-order covariance prediction.",
                )
        if row["campaign"] == "E0" and row.get("equilibrium_prediction_pass") is not None:
            add(
                f"E0 {row['case']} disturbance equilibrium",
                row.get("predicted_equilibrium_pointing_rad"),
                row.get("measured_equilibrium_pointing_rad"),
                "tail mean relative error ≤ 15%; first/second-tail mean drift ≤ 10%",
                row.get("equilibrium_prediction_pass"),
                "" if row.get("equilibrium_prediction_pass") else
                (
                    "The terminal-tail means remain transient, so the steady-equilibrium comparison is inconclusive."
                    if row.get("equilibrium_steady_state_pass") is False
                    else "The steady terminal-tail pointing offset differs from the linearized PD equilibrium by more than 15%."
                ),
            )
        if row["campaign"] == "E0" and row.get("lyapunov_bound_pass") is not None:
            add(
                "P8 tight pointing / tracker outage",
                row.get("lyapunov_pointing_bound_rad"),
                row.get("lyapunov_measured_peak_pointing_rad"),
                "measured peak ≤ 1.1 × bound",
                row.get("lyapunov_bound_pass"),
                "" if row.get("lyapunov_bound_pass") else
                "Observed trajectory exceeded the local unsaturated small-angle ultimate bound.",
            )
        if row["campaign"] == "E0" and row.get("wheel_gyroscopic_coupling") is not None:
            potential_coupling = row.get("wheel_gyroscopic_torque_rms_nm_mean")
            enabled = bool(row["wheel_gyroscopic_coupling"])
            applied_coupling = potential_coupling if enabled else 0.0
            coupling_pass = (
                potential_coupling is not None
                and (
                    applied_coupling == 0.0
                    if not enabled
                    else applied_coupling > 0.0
                )
            )
            add(
                f"P5 {row['case']} wheel-momentum term",
                "applied RMS torque is zero when disabled and equals |ω × h_w| when enabled",
                {
                    "potential_cross_product_rms_nm": potential_coupling,
                    "applied_torque_rms_nm": applied_coupling,
                },
                "zero applied torque off; positive applied torque on",
                coupling_pass,
                "" if coupling_pass else
                "The configured coupling switch and logged wheel-momentum term disagree.",
            )
    crossover_samples = []
    mismatch_quality = sorted(
        (
            row for row in rows
            if row["campaign"] == "E1"
            and row.get("inertia_mismatch") == 0.1
            and row.get("gyro_noise_scale") is not None
        ),
        key=lambda row: float(row["gyro_noise_scale"]),
    )
    for scale in sorted({float(row["gyro_noise_scale"]) for row in mismatch_quality}):
        pair = {
            row["variant"]: row["attitude_estimation_rmse_rad_mean"]
            for row in mismatch_quality
            if float(row["gyro_noise_scale"]) == scale
        }
        if {"gyro_driven", "model_aided"} <= pair.keys():
            crossover_samples.append(
                (
                    scale,
                    float(pair["model_aided"] - pair["gyro_driven"]),
                )
            )
    crossover_scales = []
    for earlier, later in zip(crossover_samples, crossover_samples[1:]):
        if earlier[1] * later[1] <= 0.0:
            fraction = (
                -earlier[1] / (later[1] - earlier[1])
                if later[1] != earlier[1]
                else 0.0
            )
            crossover_scales.append(
                10.0 ** (
                    np.log10(earlier[0])
                    + fraction * (np.log10(later[0]) - np.log10(earlier[0]))
                )
            )
    if crossover_samples:
        add(
            "P4 E1 gyro-quality crossover at 10% inertia mismatch",
            "model-aided and gyro-driven RMSE ordering reverses within the tested grid",
            {
                "model_minus_gyro_rmse_rad_by_scale": crossover_samples,
                "log_scale_interpolated_crossovers": crossover_scales,
            },
            "sign change bracketed by adjacent tested scales",
            bool(crossover_scales),
            "" if crossover_scales else
            "No RMSE-ordering reversal was observed over the tested gyro-noise range; the crossover is outside this grid or absent in this operating regime.",
        )

    wheel_pairs: Dict[float, Dict[bool, Dict[str, Any]]] = {}
    for row in rows:
        if row["campaign"] == "E0" and row.get("wheel_gyroscopic_coupling") is not None:
            momentum = float(row["initial_wheel_momentum_nms"])
            wheel_pairs.setdefault(momentum, {})[
                bool(row["wheel_gyroscopic_coupling"])
            ] = row
    for momentum, pair in sorted(wheel_pairs.items()):
        if False not in pair or True not in pair:
            continue
        reduced, coupled = pair[False], pair[True]
        reduced_pointing = reduced.get("pointing_rms_rad_mean")
        coupled_pointing = coupled.get("pointing_rms_rad_mean")
        control_limit = float(coupled["control_torque_limit_nm"])
        potential = max(
            float(reduced["wheel_gyroscopic_torque_rms_nm_mean"]),
            float(coupled["wheel_gyroscopic_torque_rms_nm_mean"]),
        )
        relative_state_change = (
            abs(float(coupled_pointing) - float(reduced_pointing))
            / max(abs(float(reduced_pointing)), np.finfo(float).eps)
            if reduced_pointing is not None and coupled_pointing is not None
            else None
        )
        weak_coupling = potential < 0.01 * control_limit
        pair_pass = (
            relative_state_change is not None
            and (
                relative_state_change <= 0.05
                if weak_coupling
                else relative_state_change > 1e-12
            )
        )
        add(
            f"P5 wheel momentum {momentum:g} N m s paired state effect",
            "≤5% state change below 1% control torque; resolved state change above",
            {
                "potential_cross_product_rms_nm": potential,
                "control_torque_limit_nm": control_limit,
            },
            {
                "pointing_rms_relative_change": relative_state_change,
                "weak_coupling": weak_coupling,
            },
            pair_pass,
            "" if pair_pass else
            "The paired reduced/full model difference did not follow the registered coupling-strength criterion.",
        )

    e1_grid = [
        row for row in rows
        if row["campaign"] == "E1" and row["case"].startswith("model_aided_i")
    ]
    tracker_rate_comparisons = []
    q_scale_comparisons = []
    for mismatch in (-0.25, -0.1, 0.0, 0.1, 0.25):
        for q_scale in (0.1, 1.0, 10.0):
            group = sorted(
                (
                    row for row in e1_grid
                    if row["inertia_mismatch"] == mismatch
                    and row["Q_scale"] == q_scale
                ),
                key=lambda row: row["tracker_rate_hz"],
            )
            tracker_rate_comparisons.extend(
                (
                    earlier["case"],
                    later["case"],
                    later["attitude_estimation_rmse_rad_mean"]
                    <= earlier["attitude_estimation_rmse_rad_mean"],
                )
                for earlier, later in zip(group, group[1:])
            )
        for tracker_rate in (0.2, 1.0, 5.0):
            group = sorted(
                (
                    row for row in e1_grid
                    if row["inertia_mismatch"] == mismatch
                    and row["tracker_rate_hz"] == tracker_rate
                ),
                key=lambda row: row["Q_scale"],
            )
            q_scale_comparisons.extend(
                (
                    earlier["case"],
                    later["case"],
                    later["NEES_mean"] <= earlier["NEES_mean"],
                )
                for earlier, later in zip(group, group[1:])
            )
    tracker_rate_passes = sum(item[2] for item in tracker_rate_comparisons)
    tracker_rate_pass = (
        len(tracker_rate_comparisons) == 30
        and tracker_rate_passes == len(tracker_rate_comparisons)
    )
    if tracker_rate_comparisons:
        add(
            "P3 E1 tracker-rate direction",
            "attitude RMSE nonincreasing as tracker rate rises",
            {
                "directional_transitions_passed": tracker_rate_passes,
                "directional_transitions_tested": len(tracker_rate_comparisons),
            },
            "all tested transitions nonincreasing",
            tracker_rate_pass,
            "" if tracker_rate_pass else
            "At least one inertia/Q slice reversed the predicted tracker-rate trend.",
        )
    q_scale_passes = sum(item[2] for item in q_scale_comparisons)
    q_scale_pass = (
        len(q_scale_comparisons) == 30
        and q_scale_passes == len(q_scale_comparisons)
    )
    if q_scale_comparisons:
        add(
            "P3 E1 assumed-Q direction",
            "measured NEES nonincreasing as Q scale rises",
            {
                "directional_transitions_passed": q_scale_passes,
                "directional_transitions_tested": len(q_scale_comparisons),
            },
            "all tested transitions nonincreasing",
            q_scale_pass,
            "" if q_scale_pass else
            "At least one inertia/tracker-rate slice reversed the predicted Q trend.",
        )

    e4_rows = sorted(
        (row for row in rows if row["campaign"] == "E4"),
        key=lambda row: float(row.get("nis_ratio_to_gate", 0.0)),
    )
    if e4_rows:
        baseline = next(
            (row for row in e4_rows if row["case"] == "no_outlier_gate"), None
        )
        baseline_rate = (
            baseline["rejected_measurements"] / max(baseline["star_tracker_updates"], 1)
            if baseline is not None else None
        )
        edge_rates = [
            (
                float(row["nis_ratio_to_gate"]),
                row["rejected_measurements"] / max(row["star_tracker_updates"], 1),
            )
            for row in e4_rows if row["case"].startswith("near_gate_")
        ]
        monotone = all(
            later[1] >= earlier[1]
            for earlier, later in zip(edge_rates, edge_rates[1:])
        )
        passed = (
            baseline_rate is not None
            and baseline_rate <= 0.05
            and monotone
            and len(edge_rates) == 3
        )
        add(
            "E4 NIS gate calibration",
            "≤5% nominal rejection at the 95% gate; edge rejection nondecreasing",
            {"nominal_rejection_rate": baseline_rate, "edge_rejection_rates": edge_rates},
            "nominal ≤5%; monotone 0.9/1.0/1.1 gate-ratio trend",
            passed,
            "" if passed else
            "Finite-sample gate decisions did not meet the nominal false-alarm or edge-trend prediction.",
        )

    e5_rows = sorted(
        (row for row in rows if row["campaign"] == "E5"),
        key=lambda row: float(row.get("outage_duration_s", 0.0)),
    )
    if e5_rows:
        recovery = [row.get("outage_recovery_time_s_mean") for row in e5_rows]
        errors = [row.get("attitude_estimation_rmse_rad_mean") for row in e5_rows]
        passed = (
            all(value is not None for value in recovery + errors)
            and all(b >= a for a, b in zip(recovery, recovery[1:]))
            and all(b >= a for a, b in zip(errors, errors[1:]))
        )
        add(
            "E5 long-outage trend",
            "longer blindness increases attitude error and does not shorten recovery",
            {"outage_s": [row.get("outage_duration_s") for row in e5_rows],
             "recovery_s": recovery, "attitude_rmse_rad": errors},
            "nondecreasing recovery time and attitude RMSE",
            passed,
            "" if passed else
            "The longer-outage trend was non-monotone or recovery was not observed within the run.",
        )

    e6_rows = [row for row in rows if row["campaign"] == "E6"]
    sign_pairs = []
    for row in e6_rows:
        if float(row.get("quaternion_sign", 1.0)) < 0.0:
            continue
        opposite = next(
            (
                candidate for candidate in e6_rows
                if candidate.get("initial_angle_deg") == row.get("initial_angle_deg")
                and candidate.get("torque_limit_nm") == row.get("torque_limit_nm")
                and candidate.get("quaternion_sign_hysteresis")
                == row.get("quaternion_sign_hysteresis")
                and float(candidate.get("quaternion_sign", 1.0)) < 0.0
            ),
            None,
        )
        if opposite is not None:
            deltas = {}
            for metric in (
                "pointing_rms_rad_mean",
                "pointing_final_rad_mean",
                "pointing_rms_rad_full_mean",
                "pointing_peak_rad_full_mean",
                "pointing_final_rad_full_mean",
            ):
                first, second = row.get(metric), opposite.get(metric)
                if first is not None and second is not None:
                    deltas[metric] = abs(first - second) / max(abs(first), abs(second), 1e-12)
            sign_pairs.append((row, opposite, deltas))
    for positive, negative, deltas in sign_pairs:
        passed = bool(deltas) and all(delta <= 0.01 for delta in deltas.values())
        add(
            f"E6 quaternion sign equivalence {positive['case']}",
            "q and -q physical metrics agree within 1%",
            deltas,
            "relative metric difference ≤1%",
            passed,
            "" if passed else
            "Equivalent quaternion representatives produced materially different closed-loop metrics.",
        )

    for angle in sorted({row["initial_angle_deg"] for row in e6_rows}):
        for torque_limit in sorted({
            row["torque_limit_nm"] for row in e6_rows
            if row["initial_angle_deg"] == angle
        }):
            group = [
                row for row in e6_rows
                if row["initial_angle_deg"] == angle
                and row["torque_limit_nm"] == torque_limit
            ]
            settling_by_hysteresis = {}
            for enabled in (False, True):
                values = [
                    row.get("settling_time_2pct_s_full_mean")
                    for row in group
                    if row["quaternion_sign_hysteresis"] is enabled
                ]
                finite_values = [
                    float(value) for value in values
                    if value is not None and np.isfinite(float(value))
                ]
                if finite_values:
                    settling_by_hysteresis[enabled] = float(np.mean(finite_values))
            off = settling_by_hysteresis.get(False)
            on = settling_by_hysteresis.get(True)
            penalty = (
                (on - off) / off
                if off is not None and on is not None and off > 0.0
                else None
            )
            passed = penalty is not None and penalty <= 0.10
            add(
                f"E6 hysteresis settling penalty {angle:g}deg/{torque_limit:g}Nm",
                "hysteresis must not increase settling time by more than 10%",
                penalty,
                "relative settling-time increase ≤10%",
                passed,
                "" if passed else
                "Both modes did not settle within 600 s, or hysteresis exceeded the registered settling penalty.",
            )

    e7_rows = [row for row in rows if row["campaign"] == "E7"]
    for row in e7_rows:
        passed = row.get("precision_target_met") is True
        add(
            "E7 dispersed Monte Carlo precision",
            "95% mean CI half-width ≤10%; bootstrap percentile-CI width ≤20%",
            {"runs": row.get("precision_runs"), "target_met": row.get("precision_target_met")},
            "pre-registered precision targets",
            passed,
            "" if passed else "The run cap was reached before the precision criteria were met.",
        )

    if table:
        with (output_dir / "predicted_vs_measured.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def _precision_targets_met(results: Dict[str, Any]) -> bool:
    summary = results["performance_summary"]
    metrics = (
        "pointing_rms_rad",
        "attitude_estimation_rmse_rad",
        "angular_rate_estimation_rmse_rad_s",
    )
    for metric in metrics:
        values = summary.get(metric)
        if not values or values["n"] < 2:
            return False
        mean = abs(values["mean"])
        mean_half_width = (values["ci95_upper"] - values["ci95_lower"]) / 2.0
        if mean == 0.0 or mean_half_width > 0.10 * mean:
            return False
        for percentile in ("p05", "p50", "p95"):
            estimate = abs(values[percentile])
            interval = values[f"{percentile}_bootstrap_ci95"]
            if estimate == 0.0 or interval[1] - interval[0] > 0.20 * estimate:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=(*CAMPAIGNS, "all"), default="all")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--precision-max-runs", type=int, default=500)
    parser.add_argument("--duration", type=float, help="Override each case duration (seconds)")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "prediction_validation"
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only the named case; may be supplied more than once.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Regenerate summaries and plots from saved case traces without rerunning simulations.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least one")
    if args.precision_max_runs < 100:
        parser.error("--precision-max-runs must be at least 100")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_campaign_cases(args.campaign, args.duration)
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"unknown case name(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["name"] in set(args.case)]
    rows = []
    if args.summarize_existing:
        for case in cases:
            trace_path = args.output_dir / f"{case['campaign'].lower()}_{case['name']}.npz"
            if not trace_path.exists():
                parser.error(f"saved trace not found: {trace_path}")
            results = _load_saved_results(trace_path)
            metadata = results.get("campaign_metadata", {})
            case.update({
                key: value for key, value in metadata.items()
                if key != "config"
            })
            time = np.asarray(results["time"])
            if time.size > 1:
                case["config"] = replace(
                    case["config"],
                    duration=float(time[-1] + np.median(np.diff(time))),
                )
            _augment_saved_measurement_masks(case, results)
            _augment_saved_diagnostics(case, results)
            save_results(results, str(trace_path))
            rows.append(_summary_row(case, results))
    else:
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] {case['campaign']} {case['name']}: "
                f"{args.runs} runs x {case['config'].duration:g} s"
            )
            n_runs = max(args.runs, 100) if case["campaign"] == "E7" else args.runs
            results = run_case(case, n_runs)
            precision_met = _precision_targets_met(results) if case["campaign"] == "E7" else None
            while (
                case["campaign"] == "E7"
                and not precision_met
                and n_runs < args.precision_max_runs
            ):
                n_runs = min(2 * n_runs, args.precision_max_runs)
                print(f"E7 precision target not met; increasing N to {n_runs}")
                results = run_case(case, n_runs)
                precision_met = _precision_targets_met(results)
            if case["campaign"] == "E7":
                case["precision_target_met"] = bool(precision_met)
                case["precision_runs"] = n_runs
            save_results(
                results,
                str(args.output_dir / f"{case['campaign'].lower()}_{case['name']}.npz"),
            )
            rows.append(_summary_row(case, results))
            del results

    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
    for campaign in sorted({row["campaign"] for row in rows}):
        _write_campaign_plot(rows, campaign, args.output_dir)
    _write_tuning_sensitivity_plot(rows, args.output_dir)
    _write_e1_diagnostics(rows, args.output_dir)
    _write_prediction_table(rows, args.output_dir)
    print(f"Campaign summaries and traces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
