#!/usr/bin/env python3
"""Compute control-time real-world navigation metrics for one experiment run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import atomic_write_json, read_json, utc_now  # type: ignore
else:
    from .common import atomic_write_json, read_json, utc_now


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z),
    )


def interpolate_series(source_t, values, target_t, max_gap=0.5):
    source_t = np.asarray(source_t, dtype=float)
    values = np.asarray(values, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    result = np.full((len(target_t), *values.shape[1:]), np.nan, dtype=float)
    right = np.searchsorted(source_t, target_t, side="left")
    exact = (right < len(source_t)) & np.isclose(source_t[np.minimum(right, len(source_t) - 1)], target_t)
    if np.any(exact):
        result[exact] = values[right[exact]]
    candidates = (~exact) & (right > 0) & (right < len(source_t))
    indices = np.flatnonzero(candidates)
    left_indices, right_indices = right[indices] - 1, right[indices]
    gaps = source_t[right_indices] - source_t[left_indices]
    usable = (gaps > 0) & (gaps <= max_gap)
    indices = indices[usable]
    left_indices, right_indices, gaps = left_indices[usable], right_indices[usable], gaps[usable]
    if len(indices):
        ratio = (target_t[indices] - source_t[left_indices]) / gaps
        reshape = (len(ratio),) + (1,) * (values.ndim - 1)
        result[indices] = values[left_indices] + ratio.reshape(reshape) * (
            values[right_indices] - values[left_indices]
        )
    return result


def local_polynomial_motion(times, velocities, window=7, degree=2):
    """Timestamp-aware local fit returning smoothed v, acceleration, and jerk."""
    times = np.asarray(times, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if len(times) < degree + 1:
        shape = velocities.shape
        return velocities.copy(), np.full(shape, np.nan), np.full(shape, np.nan)
    window = min(window, len(times))
    if window % 2 == 0:
        window -= 1
    window = max(window, degree + 1 + ((degree + 1) % 2 == 0))
    smoothed = np.full_like(velocities, np.nan)
    acceleration = np.full_like(velocities, np.nan)
    jerk = np.full_like(velocities, np.nan)
    half = window // 2
    for index in range(len(times)):
        start = max(0, min(index - half, len(times) - window))
        stop = start + window
        local_t = times[start:stop] - times[index]
        for dimension in range(velocities.shape[1]):
            local_v = velocities[start:stop, dimension]
            finite = np.isfinite(local_t) & np.isfinite(local_v)
            if np.sum(finite) < degree + 1:
                continue
            coefficients = np.polynomial.polynomial.polyfit(
                local_t[finite], local_v[finite], degree
            )
            smoothed[index, dimension] = coefficients[0]
            acceleration[index, dimension] = coefficients[1]
            jerk[index, dimension] = 2.0 * coefficients[2]
    return smoothed, acceleration, jerk


def time_weighted_mean(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return float("nan")
    valid_intervals = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (np.diff(times) > 0)
    )
    if not np.any(valid_intervals):
        return float("nan")
    dt = np.diff(times)[valid_intervals]
    area = 0.5 * (values[:-1][valid_intervals] + values[1:][valid_intervals]) * dt
    return float(np.sum(area) / np.sum(dt))


def tracking_clearance_at_controls(track_data, control_t, robot_xy, config):
    track_t = track_data["timestamps"]
    states = track_data["states"]
    active = track_data["active"]
    valid = track_data["valid"]
    clearance = np.full(len(control_t), np.nan)
    known = np.zeros(len(control_t), dtype=bool)
    tracked_count = np.zeros(len(control_t), dtype=int)
    max_gap = float(config["tracking_max_gap_s"])
    combined_radius = float(config["robot_radius_m"] + config["human_radius_m"])
    for sample, timestamp in enumerate(control_t):
        right = int(np.searchsorted(track_t, timestamp, side="left"))
        exact_index = (
            right if right < len(track_t) and np.isclose(track_t[right], timestamp) else None
        )
        if exact_index is not None:
            usable = active[exact_index] & valid[exact_index]
            usable &= np.isfinite(states[exact_index, :, :2]).all(axis=1)
            tracked_count[sample] = int(np.sum(usable))
            if tracked_count[sample] == 0:
                clearance[sample], known[sample] = np.inf, True
            else:
                distances = np.linalg.norm(
                    states[exact_index, usable, :2] - robot_xy[sample], axis=1
                )
                clearance[sample] = float(np.min(distances) - combined_radius)
                known[sample] = True
            continue
        if right == 0 or right >= len(track_t):
            continue
        left = right - 1
        gap = track_t[right] - track_t[left]
        if gap <= 0 or gap > max_gap:
            continue
        usable = active[left] & active[right] & valid[left] & valid[right]
        usable &= np.isfinite(states[left, :, :2]).all(axis=1)
        usable &= np.isfinite(states[right, :, :2]).all(axis=1)
        tracked_count[sample] = int(np.sum(usable))
        if tracked_count[sample] == 0:
            clearance[sample], known[sample] = np.inf, True
            continue
        ratio = (timestamp - track_t[left]) / gap
        human_xy = states[left, usable, :2] + ratio * (
            states[right, usable, :2] - states[left, usable, :2]
        )
        distances = np.linalg.norm(human_xy - robot_xy[sample], axis=1)
        clearance[sample] = float(np.min(distances) - combined_radius)
        known[sample] = True
    return clearance, known, tracked_count


def json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compute_run(run_dir: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    config = manifest["configuration"]
    with (run_dir / "sensor_messages.pkl").open("rb") as stream:
        sensors = pickle.load(stream)
    controller_path = run_dir / "controller.pkl"
    if controller_path.exists():
        with controller_path.open("rb") as stream:
            controller = pickle.load(stream)
    else:
        controller = {"trajectory": [], "final_event": manifest.get("outcome_event")}
    trajectory = controller.get("trajectory", [])
    if len(trajectory) < 3 and len(sensors.get("cmd", [])) >= 3:
        trajectory = []
        for message in sensors["cmd"]:
            timestamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            action = np.array([message.twist.linear.x, message.twist.angular.z])
            trajectory.append(
                {
                    "command_timestamp": timestamp,
                    "action": action,
                    "published_action": action,
                }
            )
    if len(trajectory) < 3:
        raise ValueError("At least three published control samples are required")

    control_t = np.asarray([step["command_timestamp"] for step in trajectory], dtype=float)
    keep = np.r_[True, np.diff(control_t) > 0]
    trajectory = [step for step, use in zip(trajectory, keep) if use]
    control_t = control_t[keep]
    odom = sensors["odom"]
    odom_t = np.asarray(
        [msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for msg in odom]
    )
    odom_xy = np.asarray(
        [[msg.pose.pose.position.x, msg.pose.pose.position.y] for msg in odom]
    )
    odom_yaw = np.unwrap(
        np.asarray([yaw_from_quaternion(msg.pose.pose.orientation) for msg in odom])
    )
    odom_v = np.asarray([msg.twist.twist.linear.x for msg in odom])
    pose_xy = interpolate_series(odom_t, odom_xy, control_t, config["tracking_max_gap_s"])
    yaw = interpolate_series(odom_t, odom_yaw[:, None], control_t, config["tracking_max_gap_s"])[:, 0]
    body_v = interpolate_series(odom_t, odom_v[:, None], control_t, config["tracking_max_gap_s"])[:, 0]
    if not np.all(np.isfinite(pose_xy)) or not np.all(np.isfinite(yaw)):
        raise ValueError("Odometry does not cover every control timestamp within the maximum gap")
    world_v = np.column_stack((body_v * np.cos(yaw), body_v * np.sin(yaw)))
    smoothed_v, acceleration, jerk = local_polynomial_motion(
        control_t,
        world_v,
        int(config["jerk_window_samples"]),
        int(config["jerk_polynomial_degree"]),
    )
    speed = np.linalg.norm(smoothed_v, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    dt = np.diff(control_t)
    path_until_termination = float(np.sum(np.linalg.norm(np.diff(pose_xy, axis=0), axis=1)))

    track_path = run_dir / "human_tracks.npz"
    if track_path.exists():
        with np.load(track_path) as tracks:
            track_data = {key: tracks[key] for key in tracks.files}
        clearance, tracking_known, tracked_count = tracking_clearance_at_controls(
            track_data, control_t, pose_xy, config
        )
        compliant = np.where(
            tracking_known,
            clearance > float(config["personal_space_m"]),
            np.nan,
        )
        coverage = time_weighted_mean(tracking_known.astype(float), control_t)
        space_compliance = time_weighted_mean(compliant, control_t)
        finite_clearance = clearance[np.isfinite(clearance)]
        minimum_clearance = float(np.min(finite_clearance)) if len(finite_clearance) else float("nan")
    else:
        clearance = np.full(len(control_t), np.nan)
        tracking_known = np.zeros(len(control_t), dtype=bool)
        tracked_count = np.zeros(len(control_t), dtype=int)
        compliant = np.full(len(control_t), np.nan)
        coverage = space_compliance = minimum_clearance = float("nan")

    success = manifest["outcome"] == "success"
    event = manifest.get("outcome_event") or controller.get("final_event")
    goal_timestamp = event.get("timestamp") if success and event else None
    time_to_goal = float(goal_timestamp - control_t[0]) if goal_timestamp is not None else float("nan")
    tracking_threshold = float(config["minimum_tracking_coverage"])
    space_valid = bool(np.isfinite(coverage) and coverage >= tracking_threshold)
    alignment_path = run_dir / "timestamp_alignment.json"
    synchronization_valid = read_json(alignment_path).get("valid", False) if alignment_path.exists() else False

    metrics = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": manifest["run_id"],
        "policy": manifest["policy"],
        "policy_trial": manifest["policy_trial"],
        "outcome": manifest["outcome"],
        "success": success,
        "operator_collision": manifest["outcome"] == "collision",
        "timeout": manifest["outcome"] == "timeout",
        "control_samples": len(control_t),
        "duration_s": json_number(control_t[-1] - control_t[0]),
        "time_to_goal_s": json_number(time_to_goal),
        "path_length_m": json_number(path_until_termination if success else np.nan),
        "path_length_until_termination_m": path_until_termination,
        "average_speed_m_s": json_number(time_weighted_mean(speed, control_t)),
        "average_jerk_m_s3": json_number(time_weighted_mean(jerk_magnitude, control_t)),
        "space_compliance": json_number(space_compliance if space_valid else np.nan),
        "space_compliance_unqualified": json_number(space_compliance),
        "tracking_coverage": json_number(coverage),
        "tracking_valid_for_comparison": space_valid,
        "minimum_human_clearance_m": json_number(minimum_clearance),
        "collision_proxy": bool(np.isfinite(minimum_clearance) and minimum_clearance <= 0),
        "mean_control_dt_s": json_number(np.mean(dt)),
        "std_control_dt_s": json_number(np.std(dt)),
        "p95_control_dt_s": json_number(np.quantile(dt, 0.95)),
        "synchronization_valid": synchronization_valid,
        "jerk_estimator": {
            "source": "measured odometry body speed transformed to global velocity",
            "window_samples": int(config["jerk_window_samples"]),
            "polynomial_degree": int(config["jerk_polynomial_degree"]),
            "timestamp_aware": True,
        },
    }
    atomic_write_json(run_dir / "metrics.json", metrics)

    fieldnames = (
        "timestamp", "x_m", "y_m", "yaw_rad", "speed_m_s", "vx_m_s", "vy_m_s",
        "ax_m_s2", "ay_m_s2", "jerk_x_m_s3", "jerk_y_m_s3", "jerk_m_s3",
        "minimum_human_clearance_m", "space_compliant", "tracking_known",
        "tracked_humans", "command_v_m_s", "command_w_rad_s",
    )
    with (run_dir / "control_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(trajectory):
            command = step.get("published_action", step["action"])
            writer.writerow({
                "timestamp": control_t[index], "x_m": pose_xy[index, 0],
                "y_m": pose_xy[index, 1], "yaw_rad": yaw[index],
                "speed_m_s": speed[index], "vx_m_s": smoothed_v[index, 0],
                "vy_m_s": smoothed_v[index, 1], "ax_m_s2": acceleration[index, 0],
                "ay_m_s2": acceleration[index, 1], "jerk_x_m_s3": jerk[index, 0],
                "jerk_y_m_s3": jerk[index, 1], "jerk_m_s3": jerk_magnitude[index],
                "minimum_human_clearance_m": clearance[index],
                "space_compliant": compliant[index], "tracking_known": tracking_known[index],
                "tracked_humans": tracked_count[index], "command_v_m_s": command[0],
                "command_w_rad_s": command[1],
            })
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    metrics = compute_run(args.run_dir.resolve())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
