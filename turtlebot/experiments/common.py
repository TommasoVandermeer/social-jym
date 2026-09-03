"""Shared configuration and storage helpers for real-world experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULTS = {
    "control_frequency_hz": 4.0,
    "lidar_rays": 200,
    "goal_tolerance_m": 0.8,
    "timeout_s": 120.0,
    "trials_per_policy": 10,
    "engineering_filters": False,
    "interpolate_odometry": True,
    "align_goal": False,
    "pure_pursuit": False,
    "personal_space_m": 0.5,
    "robot_radius_m": 0.3,
    "human_radius_m": 0.3,
    "correction_interval_frames": 15,
    "label_preview_window_frames": 21,
    "tracking_max_gap_s": 0.5,
    "minimum_tracking_coverage": 0.9,
    "jerk_window_samples": 7,
    "jerk_polynomial_degree": 2,
    "schedule_seed": 20260831,
    "bootstrap_seed": 20260831,
    "bootstrap_samples": 10000,
    "network_selection": "best",
}

POLICIES = ("JESSI-S2R", "DWA")
ROS_TOPICS = (
    "/turtlebot1/scan",
    "/turtlebot1/odom",
    "/turtlebot1/cmd_vel",
    "/turtlebot1/cmd_vel_stamped",
    "/turtlebot1/hazard_detection",
    "/tf",
    "/tf_static",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def load_config(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    config = {**DEFAULTS, **raw}
    required = ("campaign_name", "goal", "jessi_network")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    if not isinstance(config["campaign_name"], str) or not config["campaign_name"].strip():
        raise ValueError("campaign_name must be a non-empty string")
    goal = config["goal"]
    if not isinstance(goal, list) or len(goal) != 2 or not all(
        isinstance(value, (int, float)) for value in goal
    ):
        raise ValueError("goal must be [x, y] in metres")
    positive = (
        "control_frequency_hz",
        "lidar_rays",
        "goal_tolerance_m",
        "timeout_s",
        "trials_per_policy",
        "personal_space_m",
        "robot_radius_m",
        "human_radius_m",
        "correction_interval_frames",
        "label_preview_window_frames",
        "tracking_max_gap_s",
        "jerk_window_samples",
        "bootstrap_samples",
    )
    for key in positive:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["jerk_window_samples"] % 2 == 0:
        raise ValueError("jerk_window_samples must be odd")
    if config["label_preview_window_frames"] % 2 == 0:
        raise ValueError("label_preview_window_frames must be odd")
    if config["jerk_window_samples"] <= config["jerk_polynomial_degree"]:
        raise ValueError("jerk_window_samples must exceed jerk_polynomial_degree")
    if not 0 <= config["minimum_tracking_coverage"] <= 1:
        raise ValueError("minimum_tracking_coverage must be in [0, 1]")
    if config["engineering_filters"]:
        raise ValueError(
            "This comparison protocol requires engineering_filters=false for both policies"
        )
    if config["network_selection"] not in ("best", "final"):
        raise ValueError("network_selection must be 'best' or 'final'")
    return config


def balanced_schedule(trials_per_policy: int, seed: int) -> list[dict[str, Any]]:
    """Generate a reproducible shuffle with runs of at most two equal policies."""
    labels = [policy for policy in POLICIES for _ in range(trials_per_policy)]
    rng = random.Random(seed)
    for _ in range(10000):
        rng.shuffle(labels)
        if all(
            not (labels[index] == labels[index - 1] == labels[index - 2])
            for index in range(2, len(labels))
        ):
            break
    else:
        raise RuntimeError("Could not construct a balanced policy schedule")
    per_policy = {policy: 0 for policy in POLICIES}
    schedule = []
    for ordinal, policy in enumerate(labels, start=1):
        per_policy[policy] += 1
        schedule.append(
            {
                "ordinal": ordinal,
                "policy": policy,
                "policy_trial": per_policy[policy],
                "status": "pending",
                "run_directory": None,
            }
        )
    return schedule


def run_directory_name(entry: dict[str, Any]) -> str:
    return (
        f"run_{entry['ordinal']:03d}_{entry['policy'].lower()}_"
        f"trial_{entry['policy_trial']:02d}"
    )


def campaign_root(config_path: Path, config: dict[str, Any]) -> Path:
    configured = config.get("data_root")
    if configured:
        root = Path(configured)
        if not root.is_absolute():
            root = config_path.parent / root
    else:
        root = config_path.parent / "data"
    return root.resolve() / config["campaign_name"]
