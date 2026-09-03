#!/usr/bin/env python3
"""Initialize and execute reproducible JESSI-S2R-vs-DWA TurtleBot trials."""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        ROS_TOPICS,
        atomic_write_json,
        balanced_schedule,
        campaign_root,
        file_sha256,
        git_commit,
        load_config,
        read_json,
        run_directory_name,
        utc_now,
    )
else:
    from .common import (
        ROS_TOPICS,
        atomic_write_json,
        balanced_schedule,
        campaign_root,
        file_sha256,
        git_commit,
        load_config,
        read_json,
        run_directory_name,
        utc_now,
    )


HERE = Path(__file__).resolve().parent
TURTLEBOT_DIR = HERE.parent
REPO_ROOT = TURTLEBOT_DIR.parent
RETRYABLE_OUTCOMES = {"operator_abort", "controller_error"}


def resolve_network(config_path: Path, configured: str) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JESSI-S2R network not found: {path}")
    return path


def init_campaign(config_path: Path) -> Path:
    config = load_config(config_path)
    root = campaign_root(config_path, config)
    schedule_path = root / "schedule.json"
    if schedule_path.exists():
        raise FileExistsError(f"Campaign already initialized: {root}")
    network = resolve_network(config_path, config["jessi_network"])
    root.mkdir(parents=True, exist_ok=True)
    saved_config = dict(config)
    saved_config["jessi_network"] = str(network)
    saved_config["data_root"] = str(root.parent)
    atomic_write_json(root / "campaign_config.json", saved_config)
    schedule = {
        "campaign_name": config["campaign_name"],
        "created_at": utc_now(),
        "seed": config["schedule_seed"],
        "runs": balanced_schedule(config["trials_per_policy"], config["schedule_seed"]),
    }
    atomic_write_json(schedule_path, schedule)
    atomic_write_json(
        root / "campaign_manifest.json",
        {
            "created_at": utc_now(),
            "git_commit": git_commit(REPO_ROOT),
            "jessi_network": str(network),
            "jessi_network_sha256": file_sha256(network),
            "jessi_s2r_parameter_selection": config["network_selection"],
            "configuration": config,
        },
    )
    return root


def stop_process(process: subprocess.Popen | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def controller_outcome(run_dir: Path) -> tuple[str | None, dict | None]:
    path = run_dir / "controller.pkl"
    if not path.exists():
        return None, None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        event = payload.get("final_event")
        if event and event.get("reason") in ("goal_reached", "timeout"):
            return event["reason"], event
    except Exception:
        return None, None
    return None, None


def ask_interrupted_outcome() -> str:
    choices = {"c": "collision", "s": "safety_stop", "a": "operator_abort"}
    while True:
        answer = input(
            "Run interrupted: [c]ollision, [s]afety stop, or unrelated [a]bort? "
        ).strip().lower()
        if answer in choices:
            break
    return choices[answer]


def ask_run_context() -> tuple[int | None, str]:
    raw_count = input("Number of pedestrians present [5]: ").strip()
    try:
        pedestrian_count = int(raw_count) if raw_count else 5
    except ValueError:
        pedestrian_count = None
    notes = input("Optional run notes (unusual behavior, interventions, conditions): ").strip()
    return pedestrian_count, notes


def run_next(config_path: Path) -> Path:
    config = load_config(config_path)
    root = campaign_root(config_path, config)
    schedule_path = root / "schedule.json"
    if not schedule_path.exists():
        raise FileNotFoundError("Initialize the campaign first with the 'init' command")
    schedule = read_json(schedule_path)
    entry = next((run for run in schedule["runs"] if run["status"] == "pending"), None)
    if entry is None:
        raise RuntimeError("No pending trials remain")
    run_dir = root / run_directory_name(entry)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    entry["status"] = "running"
    entry["run_directory"] = run_dir.name
    atomic_write_json(schedule_path, schedule)

    network = resolve_network(config_path, config["jessi_network"])
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "ordinal": entry["ordinal"],
        "policy": entry["policy"],
        "policy_trial": entry["policy_trial"],
        "attempt_number": len(entry.get("previous_attempts", [])) + 1,
        "started_at": utc_now(),
        "finished_at": None,
        "outcome": "running",
        "outcome_event": None,
        "notes": "",
        "git_commit": git_commit(REPO_ROOT),
        "jessi_network_sha256": file_sha256(network),
        "jessi_s2r_parameter_selection": config["network_selection"],
        "configuration": config,
        "ros_topics": list(ROS_TOPICS),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    bag_log = (run_dir / "rosbag.log").open("w", encoding="utf-8")
    bag_command = ["ros2", "bag", "record", "-o", str(run_dir / "rosbag"), *ROS_TOPICS]
    controller_command = [
        sys.executable,
        str(TURTLEBOT_DIR / "turtlebot_controller.py"),
        "--planner",
        entry["policy"],
        "--goals",
        str(config["goal"][0]),
        str(config["goal"][1]),
        "--frequency",
        str(config["control_frequency_hz"]),
        "--lidar_rays",
        str(config["lidar_rays"]),
        "--network",
        str(network),
        "--network-selection",
        config["network_selection"],
        "--experiment-dir",
        str(run_dir),
        "--timeout",
        str(config["timeout_s"]),
        "--goal-tolerance",
        str(config["goal_tolerance_m"]),
        "--stop-on-goal",
    ]
    if config["interpolate_odometry"]:
        controller_command.append("--interp")
    if config["align_goal"]:
        controller_command.append("--align")
    if config["pure_pursuit"]:
        controller_command.append("--pure_pursuit")
    if config["engineering_filters"]:
        controller_command.append("--engineering_filters")

    bag_process = None
    controller_process = None
    interrupted = False
    execution_error = None
    try:
        bag_process = subprocess.Popen(
            bag_command,
            cwd=REPO_ROOT,
            stdout=bag_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1.0)
        if bag_process.poll() is not None:
            raise RuntimeError(f"ros2 bag exited early; see {run_dir / 'rosbag.log'}")
        with (run_dir / "controller.log").open("w", encoding="utf-8") as controller_log:
            controller_process = subprocess.Popen(
                controller_command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert controller_process.stdout is not None
            for line in controller_process.stdout:
                print(line, end="")
                controller_log.write(line)
                controller_log.flush()
            controller_process.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("\nOperator interruption received; stopping robot logging safely...")
    except Exception as exc:
        execution_error = exc
        print(f"Experiment process error: {exc}", file=sys.stderr)
    finally:
        stop_process(controller_process)
        stop_process(bag_process)
        bag_log.close()

    automatic_outcome, event = controller_outcome(run_dir)
    if automatic_outcome == "goal_reached":
        outcome = "success"
    elif automatic_outcome == "timeout":
        outcome = "timeout"
    elif interrupted:
        outcome = ask_interrupted_outcome()
    else:
        outcome = "controller_error"
    pedestrian_count, operator_notes = ask_run_context()
    error_note = (
        f"Controller exit code: {controller_process.returncode if controller_process else None}; "
        f"execution error: {execution_error}"
        if outcome == "controller_error" else ""
    )
    notes = "; ".join(note for note in (error_note, operator_notes) if note)

    manifest.update(
        {
            "finished_at": utc_now(),
            "outcome": outcome,
            "outcome_event": event,
            "notes": notes,
            "pedestrian_count": pedestrian_count,
            "controller_exit_code": (
                controller_process.returncode if controller_process is not None else None
            ),
            "rosbag_exit_code": bag_process.returncode if bag_process is not None else None,
        }
    )
    atomic_write_json(run_dir / "manifest.json", manifest)
    entry["status"] = "complete" if outcome != "controller_error" else "failed"
    entry["outcome"] = outcome
    atomic_write_json(schedule_path, schedule)
    return run_dir


def prepare_retry(campaign_dir: Path) -> tuple[Path, dict]:
    """Archive the latest retryable attempt and reset its schedule entry."""
    schedule_path = campaign_dir / "schedule.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Campaign schedule not found: {schedule_path}")
    schedule = read_json(schedule_path)
    attempted = [entry for entry in schedule.get("runs", []) if entry.get("run_directory")]
    if not attempted:
        raise RuntimeError("No recorded campaign run is available to retry")
    entry = max(attempted, key=lambda item: int(item["ordinal"]))
    outcome = entry.get("outcome")
    if outcome not in RETRYABLE_OUTCOMES:
        allowed = ", ".join(sorted(RETRYABLE_OUTCOMES))
        raise ValueError(
            f"Latest run outcome '{outcome}' is not retryable. "
            f"Only {allowed} attempts may be retried."
        )

    run_dir = campaign_dir / entry["run_directory"]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    archive_root = campaign_dir / "aborted_attempts"
    archive_root.mkdir(exist_ok=True)
    attempt_number = 1
    while True:
        archive_dir = archive_root / f"{run_dir.name}_attempt_{attempt_number:02d}"
        if not archive_dir.exists():
            break
        attempt_number += 1

    manifest = read_json(manifest_path)
    archived_at = utc_now()
    manifest["archived_for_retry_at"] = archived_at
    manifest["archived_attempt_number"] = attempt_number
    atomic_write_json(manifest_path, manifest)
    run_dir.rename(archive_dir)

    archived_relative = str(archive_dir.relative_to(campaign_dir))
    entry.setdefault("previous_attempts", []).append(
        {
            "attempt": attempt_number,
            "outcome": outcome,
            "archived_at": archived_at,
            "directory": archived_relative,
        }
    )
    entry["status"] = "pending"
    entry["run_directory"] = None
    entry.pop("outcome", None)
    atomic_write_json(schedule_path, schedule)
    return archive_dir, entry


def print_status(config_path: Path) -> None:
    config = load_config(config_path)
    root = campaign_root(config_path, config)
    schedule = read_json(root / "schedule.json")
    for run in schedule["runs"]:
        archived_attempts = len(run.get("previous_attempts", []))
        retry_text = f"  archived attempts {archived_attempts}" if archived_attempts else ""
        print(
            f"{run['ordinal']:02d}  {run['policy']:<5}  policy trial "
            f"{run['policy_trial']:02d}  {run['status']:<8}  {run.get('outcome', '')}"
            f"{retry_text}"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("init", "run-next", "retry-last", "status")
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    if args.command == "init":
        print(f"Initialized campaign at {init_campaign(config_path)}")
    elif args.command == "run-next":
        print(f"Completed run at {run_next(config_path)}")
    elif args.command == "retry-last":
        config = load_config(config_path)
        root = campaign_root(config_path, config)
        archive_dir, entry = prepare_retry(root)
        print(
            f"Archived aborted attempt at {archive_dir}\n"
            f"Reset run {entry['ordinal']:03d} ({entry['policy']} policy trial "
            f"{entry['policy_trial']:02d}) to pending. Use 'run-next' to repeat it."
        )
    else:
        print_status(config_path)


if __name__ == "__main__":
    main()
