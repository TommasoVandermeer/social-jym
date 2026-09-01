#!/usr/bin/env python3
"""Convert a run's ROS bag to timestamp-aligned messages for offline analysis."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - only available in the ROS container
    raise SystemExit("Run this script inside the project ROS 2 Docker container") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import atomic_write_json, utc_now  # type: ignore
else:
    from .common import atomic_write_json, utc_now


TOPIC_KEYS = {
    "/turtlebot1/scan": "scan",
    "/turtlebot1/odom": "odom",
    "/turtlebot1/cmd_vel_stamped": "cmd",
    "/turtlebot1/hazard_detection": "hazard",
}


def stamp_seconds(message) -> float | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def set_stamp(message, timestamp: float) -> None:
    seconds = int(np.floor(timestamp))
    nanoseconds = int(round((timestamp - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = nanoseconds


def estimate_offset(receive_times: np.ndarray, header_times: np.ndarray) -> tuple[float, np.ndarray]:
    differences = receive_times - header_times
    median = float(np.median(differences))
    absolute_deviation = np.abs(differences - median)
    mad = float(np.median(absolute_deviation))
    threshold = max(0.005, 3.0 * 1.4826 * mad)
    inliers = absolute_deviation <= threshold
    if np.any(inliers):
        median = float(np.median(differences[inliers]))
    return median, inliers


def alignment_stats(receive, raw, aligned, inliers) -> dict:
    residuals = receive - aligned
    absolute = np.abs(residuals)
    gaps = np.diff(aligned)
    p95 = float(np.quantile(absolute, 0.95)) if len(absolute) else None
    return {
        "messages": int(len(aligned)),
        "inliers": int(np.sum(inliers)),
        "outliers": int(len(inliers) - np.sum(inliers)),
        "dropped_messages": None,
        "raw_monotonicity_violations": int(np.sum(np.diff(raw) <= 0)),
        "aligned_monotonicity_violations": int(np.sum(gaps <= 0)),
        "median_absolute_residual_s": float(np.median(absolute)),
        "p95_absolute_residual_s": p95,
        "maximum_absolute_residual_s": float(np.max(absolute)),
        "median_gap_s": float(np.median(gaps)) if len(gaps) else None,
        "maximum_gap_s": float(np.max(gaps)) if len(gaps) else None,
        "warning": bool(p95 is not None and p95 > 0.05),
        "valid": bool(p95 is not None and p95 <= 0.25 and np.all(gaps > 0)),
    }


def read_bag(bag_path: Path) -> tuple[dict[str, list], dict[str, str]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    selected = {topic: [] for topic in TOPIC_KEYS if topic in topic_types}
    message_types = {topic: get_message(topic_types[topic]) for topic in selected}
    while reader.has_next():
        topic, serialized, receive_ns = reader.read_next()
        if topic not in selected:
            continue
        message = deserialize_message(serialized, message_types[topic])
        selected[topic].append((message, receive_ns * 1e-9))
    return selected, topic_types


def convert_run(run_dir: Path) -> Path:
    bag_path = run_dir / "rosbag"
    if not bag_path.exists():
        raise FileNotFoundError(f"ROS bag not found: {bag_path}")
    records, topic_types = read_bag(bag_path)
    output = {
        "schema_version": 2,
        "created_at": utc_now(),
        "source_bag": str(bag_path),
        "scan": [],
        "odom": [],
        "cmd": [],
        "hazard": [],
        "raw_header_timestamps": {},
        "bag_receive_timestamps": {},
        "aligned_timestamps": {},
    }
    report = {"source_bag": str(bag_path), "topics": {}, "topic_types": topic_types}
    for topic, key in TOPIC_KEYS.items():
        entries = records.get(topic, [])
        if not entries:
            report["topics"][topic] = {"messages": 0, "valid": False, "missing": True}
            continue
        stamped = [(msg, receive, stamp_seconds(msg)) for msg, receive in entries]
        stamped = [(msg, receive, raw) for msg, receive, raw in stamped if raw is not None]
        receive = np.asarray([entry[1] for entry in stamped], dtype=float)
        raw = np.asarray([entry[2] for entry in stamped], dtype=float)
        offset, inliers = estimate_offset(receive, raw)
        aligned = raw + offset
        corrected = []
        for (message, _, _), timestamp in zip(stamped, aligned):
            message_copy = copy.deepcopy(message)
            set_stamp(message_copy, float(timestamp))
            corrected.append(message_copy)
        output[key] = corrected
        output["raw_header_timestamps"][key] = raw
        output["bag_receive_timestamps"][key] = receive
        output["aligned_timestamps"][key] = aligned
        stats = alignment_stats(receive, raw, aligned, inliers)
        stats["offset_s"] = offset
        report["topics"][topic] = stats
        level = "WARNING" if stats["warning"] else "OK"
        print(
            f"{level}: {topic}: {len(aligned)} messages, offset={offset:+.6f}s, "
            f"p95 residual={stats['p95_absolute_residual_s']:.4f}s"
        )

    required = ("/turtlebot1/scan", "/turtlebot1/odom", "/turtlebot1/cmd_vel_stamped")
    report["valid"] = all(report["topics"].get(topic, {}).get("valid", False) for topic in required)
    output_path = run_dir / "sensor_messages.pkl"
    temporary = output_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(output, stream)
    temporary.replace(output_path)
    atomic_write_json(run_dir / "timestamp_alignment.json", report)
    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(f"Wrote {convert_run(args.run_dir.resolve())}")


if __name__ == "__main__":
    main()
