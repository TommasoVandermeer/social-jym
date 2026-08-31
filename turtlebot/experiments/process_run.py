#!/usr/bin/env python3
"""Rebuild derived data, label pedestrians, and compute one run's metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bag_to_recording import convert_run  # type: ignore
    from common import atomic_write_json, campaign_root, load_config, read_json  # type: ignore
    from compute_metrics import compute_run  # type: ignore
else:
    from .bag_to_recording import convert_run
    from .common import atomic_write_json, campaign_root, load_config, read_json
    from .compute_metrics import compute_run


HERE = Path(__file__).resolve().parent
TRACKER = HERE.parent / "extract_humans_tracks_and_render.py"


def latest_campaign_run(campaign_dir: Path) -> Path:
    """Return the highest-ordinal recorded run from a campaign schedule."""
    schedule_path = campaign_dir / "schedule.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Campaign schedule not found: {schedule_path}")
    schedule = read_json(schedule_path)
    candidates = []
    for entry in schedule.get("runs", []):
        directory = entry.get("run_directory")
        if not directory:
            continue
        path = campaign_dir / directory
        if path.is_dir() and (path / "manifest.json").is_file():
            candidates.append((int(entry["ordinal"]), path))
    if not candidates:
        raise FileNotFoundError(f"No recorded runs found in campaign: {campaign_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def process(run_dir: Path, skip_tracking=False, save_animation=False) -> None:
    sensor_path = convert_run(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    manifest["timing_diagnostics"] = read_json(run_dir / "timestamp_alignment.json")
    atomic_write_json(run_dir / "manifest.json", manifest)
    config = manifest["configuration"]
    if not skip_tracking:
        command = [
            sys.executable,
            str(TRACKER),
            "--input",
            str(sensor_path),
            "--correction-interval",
            str(config["correction_interval_frames"]),
            "--label-preview-window",
            str(config.get("label_preview_window_frames", 21)),
            "--tracks-output",
            str(run_dir / "human_tracks.npz"),
            "--no-render",
        ]
        if save_animation:
            command.extend(("--save", str(run_dir / "human_tracking.gif")))
        subprocess.run(command, check=True, cwd=HERE.parent.parent)
    metrics = compute_run(run_dir)
    manifest["metrics_summary"] = {
        key: metrics[key]
        for key in (
            "time_to_goal_s", "path_length_m", "average_jerk_m_s3",
            "space_compliance", "tracking_coverage", "synchronization_valid",
        )
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    print(
        f"Computed {metrics['run_id']}: outcome={metrics['outcome']}, "
        f"jerk={metrics['average_jerk_m_s3']}, "
        f"space compliance={metrics['space_compliance']}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="process the latest recorded run resolved from --config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="campaign configuration (required with --latest)",
    )
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    args = parser.parse_args(argv)
    if args.latest:
        if args.run_dir is not None:
            parser.error("run_dir and --latest are mutually exclusive")
        if args.config is None:
            parser.error("--latest requires --config")
        config_path = args.config.resolve()
        campaign_dir = campaign_root(config_path, load_config(config_path))
        run_dir = latest_campaign_run(campaign_dir)
        print(f"Selected latest campaign run: {run_dir}")
    else:
        if args.run_dir is None:
            parser.error("provide run_dir or use --latest --config CONFIG")
        run_dir = args.run_dir.resolve()
    process(run_dir, args.skip_tracking, args.save_animation)


if __name__ == "__main__":
    main()
