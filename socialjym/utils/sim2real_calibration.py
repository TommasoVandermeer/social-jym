"""Offline system-identification helpers for JESSI-S2R domain randomization."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def _finite_array(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _robust_bounds(values, *, margin=0.20, positive=False, fallback=(1.0, 1.0)):
    values = _finite_array(values)
    if values.size == 0:
        return [float(fallback[0]), float(fallback[1])]
    low, high = np.percentile(values, [1.0, 99.0])
    span = max(high - low, abs(low) * 0.05, abs(high) * 0.05, 1e-6)
    low -= margin * span
    high += margin * span
    if positive:
        low = max(low, 1e-6)
    return [float(low), float(max(high, low))]


def _record_value(record, *names, default=np.nan):
    for name in names:
        if name in record:
            return record[name]
    return default


def calibrate_trajectories(trajectories, params, *, max_delay_steps=12, margin=0.20):
    """Estimate reproducible randomization bounds from recorded controller data.

    The estimator deliberately uses robust percentiles rather than a single
    best-fit model.  The resulting interval is intended for domain
    randomization, not as a claim that the physical robot is exactly first order.
    """
    records = [record for trajectory in trajectories for record in trajectory]
    if len(records) < 3:
        raise ValueError("At least three controller records are required")

    control_times = np.asarray([
        _record_value(r, "control_loop_timestamp", "command_timestamp") for r in records
    ], dtype=np.float64)
    valid_dt = np.diff(control_times)
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 1e-4) & (valid_dt < 5.0)]
    configured_dt = 1.0 / float(params.get("frequency", 4.0))
    nominal_dt = float(np.median(valid_dt)) if valid_dt.size else configured_dt

    commands = np.asarray([
        _record_value(r, "published_action", "policy_action", "action", default=[np.nan, np.nan])
        for r in records
    ], dtype=np.float64)
    measured = np.asarray([
        _record_value(r, "measured_twist", default=[np.nan, np.nan]) for r in records
    ], dtype=np.float64)

    best_lag, best_error = 0, np.inf
    for lag in range(min(max_delay_steps, len(records) - 2) + 1):
        command_slice = commands[: len(commands) - lag] if lag else commands
        measured_slice = measured[lag:]
        valid = np.all(np.isfinite(command_slice), axis=1) & np.all(np.isfinite(measured_slice), axis=1)
        valid &= np.linalg.norm(command_slice, axis=1) > 0.03
        if np.count_nonzero(valid) < 3:
            continue
        error = float(np.median(np.linalg.norm(measured_slice[valid] - command_slice[valid], axis=1)))
        if error < best_error:
            best_lag, best_error = lag, error

    delayed_commands = np.full_like(commands, np.nan)
    if best_lag:
        delayed_commands[best_lag:] = commands[:-best_lag]
    else:
        delayed_commands[:] = commands

    valid_gain = (
        np.isfinite(delayed_commands[:, 0])
        & np.isfinite(measured[:, 0])
        & (np.abs(delayed_commands[:, 0]) > 0.05)
    )
    gains = measured[valid_gain, 0] / delayed_commands[valid_gain, 0]
    gains = gains[(gains > 0.05) & (gains < 3.0)]

    wheel_distance = float(params.get("wheels_distance", 0.4736842105263158))
    measured_left = measured[:, 0] - 0.5 * wheel_distance * measured[:, 1]
    measured_right = measured[:, 0] + 0.5 * wheel_distance * measured[:, 1]
    dt_per_step = np.diff(control_times)
    valid_accel = np.isfinite(dt_per_step) & (dt_per_step > 1e-4) & (dt_per_step < 5.0)
    wheel_accels = np.concatenate([
        np.abs(np.diff(measured_left)[valid_accel] / dt_per_step[valid_accel]),
        np.abs(np.diff(measured_right)[valid_accel] / dt_per_step[valid_accel]),
    ])

    tau_samples = []
    for axis in range(2):
        previous = measured[:-1, axis]
        current = measured[1:, axis]
        target = delayed_commands[1:, axis]
        denominator = previous - target
        valid = np.isfinite(previous) & np.isfinite(current) & np.isfinite(target)
        valid &= np.abs(denominator) > 0.03
        alpha = (current[valid] - target[valid]) / denominator[valid]
        alpha = alpha[(alpha > 0.01) & (alpha < 0.99)]
        if alpha.size:
            tau_samples.extend((-nominal_dt / np.log(alpha)).tolist())

    # Prefer software-aligned timestamps. Raw ROS stamps can live on a different
    # clock and therefore make a harmless clock offset look like sensor latency.
    scan_latencies = np.asarray([
        _record_value(r, "control_loop_timestamp") - _record_value(r, "scan_timestamp", "raw_scan_timestamp")
        for r in records
    ])
    odom_latencies = np.asarray([
        _record_value(r, "control_loop_timestamp") - _record_value(r, "odom_timestamp", "raw_odom_timestamp")
        for r in records
    ])
    # Never differentiate across file/episode boundaries. Those gaps can be
    # minutes long and otherwise dominate a small real-world campaign.
    def trajectory_periods(primary, fallback):
        periods = []
        for trajectory in trajectories:
            timestamps = np.asarray([
                _record_value(record, primary, fallback) for record in trajectory
            ], dtype=np.float64)
            deltas = np.diff(timestamps)
            deltas = deltas[
                np.isfinite(deltas)
                & (deltas > 1e-4)
                & (deltas <= 2.0 * nominal_dt)
            ]
            periods.extend(deltas.tolist())
        return np.asarray(periods, dtype=np.float64)

    scan_periods = trajectory_periods("scan_timestamp", "raw_scan_timestamp")
    odom_periods = trajectory_periods("odom_timestamp", "raw_odom_timestamp")

    v_max = float(params.get("v_max", 0.45))
    radius = float(params.get("robot_radius", 0.3))
    delay = best_lag * nominal_dt
    delay_jitter = max(nominal_dt / 2.0, 1e-3)
    gain_bounds = _robust_bounds(gains, margin=margin, positive=True, fallback=(0.9, 1.1))
    accel_bounds = _robust_bounds(wheel_accels, margin=margin, positive=True, fallback=(0.5, 1.5))
    tau_bounds = _robust_bounds(tau_samples, margin=margin, positive=True, fallback=(0.0, 0.5))

    # Identification outliers should be visible in diagnostics, but feeding
    # near-immobile or multi-second actuators into PPO from update zero makes
    # the task needlessly unlearnable. These are conservative feasibility caps,
    # and the uncapped estimates remain reported below for auditability.
    raw_gain_bounds = gain_bounds.copy()
    raw_accel_bounds = accel_bounds.copy()
    raw_tau_bounds = tau_bounds.copy()
    gain_bounds = [max(0.5, gain_bounds[0]), min(1.5, gain_bounds[1])]
    accel_bounds = [max(0.1, accel_bounds[0]), accel_bounds[1]]
    tau_bounds = [max(0.0, tau_bounds[0]), min(4.0 * configured_dt, tau_bounds[1])]

    def bounded_sensor_interval(values, fallback):
        bounds = _robust_bounds(values, margin=margin, positive=True, fallback=fallback)
        return [max(1e-3, bounds[0]), min(configured_dt, bounds[1])]

    def bounded_latency(values):
        values = _finite_array(values)
        values = values[(values >= 0.0) & (values <= 4.0 * configured_dt)]
        bounds = _robust_bounds(values, margin=margin, positive=False, fallback=(0.0, configured_dt))
        return [max(0.0, bounds[0]), min(configured_dt, bounds[1])]

    return {
        "schema_version": 1,
        "estimator": {
            "percentiles": [1.0, 99.0],
            "margin": margin,
            "records": len(records),
            "best_delay_steps": best_lag,
            "best_delay_fit_error": None if not np.isfinite(best_error) else best_error,
            "uncapped_identification_bounds": {
                "actuation_gain": raw_gain_bounds,
                "wheel_accel_max": raw_accel_bounds,
                "tau": raw_tau_bounds,
            },
        },
        "robot_param_bounds": {
            "radius": [radius, radius],
            "v_max": [0.8 * v_max, 1.1 * v_max],
            "wheels_distance": [0.95 * wheel_distance, 1.05 * wheel_distance],
            "control_dt": [configured_dt, configured_dt],
            "wheel_accel_max": accel_bounds,
            "tau_linear": tau_bounds,
            "tau_angular": tau_bounds,
            "control_delay_mean": [max(0.0, delay - delay_jitter), delay + delay_jitter],
            "control_delay_std": [0.0, delay_jitter],
            "actuation_gain": gain_bounds,
            "slip_scale": [0.9, 1.1],
        },
        "env_param_bounds": {
            "lidar_period": bounded_sensor_interval(scan_periods, (configured_dt, configured_dt)),
            "odometry_period": bounded_sensor_interval(odom_periods, (configured_dt, configured_dt)),
            "lidar_latency": bounded_latency(scan_latencies),
            "odometry_latency": bounded_latency(odom_latencies),
        },
    }


def calibrate_controller_files(paths, *, max_delay_steps=12, margin=0.20):
    trajectories, merged_params = [], {}
    for path in paths:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        trajectory = payload.get("trajectory")
        if not isinstance(trajectory, list):
            raise ValueError(f"{path} does not contain a trajectory list")
        trajectories.append(trajectory)
        merged_params.update(payload.get("params", {}))
    result = calibrate_trajectories(
        trajectories,
        merged_params,
        max_delay_steps=max_delay_steps,
        margin=margin,
    )
    result["sources"] = [str(Path(path)) for path in paths]
    return result


def write_calibration_json(calibration, output_path):
    output_path = Path(output_path)
    output_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
