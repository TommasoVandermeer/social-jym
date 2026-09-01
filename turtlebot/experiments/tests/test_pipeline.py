from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from turtlebot.experiments.bag_to_recording import estimate_offset
from turtlebot.experiments.common import balanced_schedule, load_config, run_directory_name
from turtlebot.experiments.compute_metrics import (
    compute_run,
    interpolate_series,
    local_polynomial_motion,
    tracking_clearance_at_controls,
)
from turtlebot.experiments.run_experiment import controller_outcome, prepare_retry
from turtlebot.experiments.process_run import latest_campaign_run
from turtlebot.extract_humans_tracks_and_render import (
    add_track_positions_to_preview,
    preview_frame_bounds,
)


def stamp(timestamp):
    seconds = int(timestamp)
    return SimpleNamespace(sec=seconds, nanosec=int((timestamp - seconds) * 1e9))


def odometry(timestamp, x, y, yaw, speed):
    orientation = SimpleNamespace(
        x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2)
    )
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp(timestamp)),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y), orientation=orientation
            )
        ),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=speed))),
    )


class ScheduleTests(unittest.TestCase):
    def test_balanced_schedule(self):
        schedule = balanced_schedule(10, 42)
        labels = [entry["policy"] for entry in schedule]
        self.assertEqual(labels.count("JESSI"), 10)
        self.assertEqual(labels.count("DWA"), 10)
        self.assertFalse(
            any(labels[index] == labels[index - 1] == labels[index - 2] for index in range(2, 20))
        )
        self.assertEqual(
            run_directory_name(schedule[2]),
            f"run_003_{schedule[2]['policy'].lower()}_trial_{schedule[2]['policy_trial']:02d}",
        )

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"campaign_name": "test", "goal": [2, 0], "jessi_network": "net.pkl"}))
            config = load_config(path)
            self.assertEqual(config["timeout_s"], 120.0)
            self.assertFalse(config["engineering_filters"])

    def test_controller_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.pkl"
            with path.open("wb") as stream:
                pickle.dump({"final_event": {"reason": "goal_reached", "timestamp": 4.0}}, stream)
            reason, event = controller_outcome(Path(directory))
            self.assertEqual(reason, "goal_reached")
            self.assertEqual(event["timestamp"], 4.0)

    def test_retry_archives_aborted_attempt_and_resets_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run_002_jessi_trial_01"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "outcome": "operator_abort"
            }))
            schedule = {"runs": [
                {
                    "ordinal": 1, "policy": "DWA", "policy_trial": 1,
                    "status": "complete", "outcome": "success",
                    "run_directory": "run_001_dwa_trial_01",
                },
                {
                    "ordinal": 2, "policy": "JESSI", "policy_trial": 1,
                    "status": "complete", "outcome": "operator_abort",
                    "run_directory": run_dir.name,
                },
            ]}
            (root / "schedule.json").write_text(json.dumps(schedule))
            archive_dir, entry = prepare_retry(root)
            self.assertFalse(run_dir.exists())
            self.assertTrue((archive_dir / "manifest.json").is_file())
            self.assertEqual(entry["status"], "pending")
            self.assertIsNone(entry["run_directory"])
            self.assertNotIn("outcome", entry)
            self.assertEqual(
                entry["previous_attempts"][0]["directory"],
                "aborted_attempts/run_002_jessi_trial_01_attempt_01",
            )
            saved = json.loads((root / "schedule.json").read_text())
            self.assertEqual(saved["runs"][1]["status"], "pending")

    def test_retry_rejects_meaningful_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run_001_dwa_trial_01"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "outcome": "collision"
            }))
            (root / "schedule.json").write_text(json.dumps({"runs": [{
                "ordinal": 1, "policy": "DWA", "policy_trial": 1,
                "status": "complete", "outcome": "collision",
                "run_directory": run_dir.name,
            }]}))
            with self.assertRaises(ValueError):
                prepare_retry(root)
            self.assertTrue(run_dir.is_dir())

    def test_latest_campaign_run_uses_schedule_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "run_001_jessi_trial_01"
            latest = root / "run_003_dwa_trial_02"
            for run_dir in (older, latest):
                run_dir.mkdir()
                (run_dir / "manifest.json").write_text("{}")
            (root / "schedule.json").write_text(json.dumps({"runs": [
                {"ordinal": 3, "run_directory": latest.name},
                {"ordinal": 1, "run_directory": older.name},
                {"ordinal": 4, "run_directory": None},
            ]}))
            self.assertEqual(latest_campaign_run(root), latest)

    def test_preview_window_is_centered_and_clipped(self):
        self.assertEqual(preview_frame_bounds(100, 50, 21), (40, 61))
        self.assertEqual(preview_frame_bounds(100, 0, 21), (0, 21))
        self.assertEqual(preview_frame_bounds(100, 99, 21), (79, 100))
        self.assertEqual(preview_frame_bounds(5, 2, 21), (0, 5))

    def test_preview_tracks_only_exist_in_processed_frames(self):
        preview = [
            {"frame_number": index + 1} for index in range(5)
        ]
        track = SimpleNamespace(
            track_id=7,
            start_frame=0,
            active=True,
            missed_frames=0,
            filtered_x=[
                np.array([float(index), 1.0, 0.0, 0.0]) for index in range(3)
            ],
            active_history=[True, True, True],
            valid_history=[True, False, True],
            x=np.array([3.0, 1.0, 0.0, 0.0]),
        )
        add_track_positions_to_preview(preview, [track], current_frame_index=3)
        self.assertEqual(preview[0]["track_ids"], [7])
        self.assertEqual(preview[1]["track_ids"], [])
        self.assertEqual(preview[2]["track_ids"], [7])
        self.assertEqual(preview[3]["track_ids"], [7])
        self.assertFalse(preview[4]["track_context_available"])
        self.assertEqual(preview[4]["track_ids"], [])


class TimestampAndMetricTests(unittest.TestCase):
    def test_offset_rejects_outlier(self):
        header = np.arange(20.0)
        receive = header + 4.5
        receive[-1] += 2.0
        offset, inliers = estimate_offset(receive, header)
        self.assertAlmostEqual(offset, 4.5)
        self.assertFalse(inliers[-1])

    def test_interpolation_rejects_large_gap(self):
        result = interpolate_series([0, 0.1, 2.0], [[0], [1], [2]], [0.05, 1.0], max_gap=0.5)
        self.assertAlmostEqual(result[0, 0], 0.5)
        self.assertTrue(np.isnan(result[1, 0]))

    def test_irregular_polynomial_jerk(self):
        times = np.array([0.0, .21, .49, .76, 1.02, 1.31, 1.57, 1.83, 2.1])
        velocity = np.column_stack((1 + 2 * times + 3 * times**2, np.zeros_like(times)))
        _, acceleration, jerk = local_polynomial_motion(times, velocity, 7, 2)
        np.testing.assert_allclose(acceleration[:, 0], 2 + 6 * times, atol=1e-10)
        np.testing.assert_allclose(jerk[:, 0], 6.0, atol=1e-10)

    def test_space_compliance_ignores_invalid_tracks(self):
        track_data = {
            "timestamps": np.array([0.0, 1.0]),
            "states": np.array([[[1.2, 0, 0, 0]], [[1.2, 0, 0, 0]]]),
            "active": np.array([[True], [True]]),
            "valid": np.array([[True], [True]]),
        }
        config = {"tracking_max_gap_s": 1.5, "robot_radius_m": .3, "human_radius_m": .3}
        clearance, known, _ = tracking_clearance_at_controls(
            track_data, np.array([.5]), np.array([[0., 0.]]), config
        )
        self.assertAlmostEqual(clearance[0], .6)
        self.assertTrue(known[0])
        track_data["valid"][1, 0] = False
        clearance, known, count = tracking_clearance_at_controls(
            track_data, np.array([.5]), np.array([[0., 0.]]), config
        )
        self.assertTrue(known[0])
        self.assertTrue(np.isinf(clearance[0]))
        self.assertEqual(count[0], 0)

        track_data = {
            "timestamps": np.array([0.0]),
            "states": np.array([[[.8, 0, 0, 0], [2.0, 0, 0, 0]]]),
            "active": np.array([[True, True]]),
            "valid": np.array([[False, True]]),
        }
        clearance, known, count = tracking_clearance_at_controls(
            track_data, np.array([0.0]), np.array([[0., 0.]]), config
        )
        self.assertTrue(known[0])
        self.assertAlmostEqual(clearance[0], 1.4)
        self.assertEqual(count[0], 1)

    def test_no_pedestrians_is_known_and_compliant(self):
        track_data = {
            "timestamps": np.array([0.0, 1.0]),
            "states": np.empty((2, 0, 4)),
            "active": np.empty((2, 0), dtype=bool),
            "valid": np.empty((2, 0), dtype=bool),
        }
        config = {"tracking_max_gap_s": 1.5, "robot_radius_m": .3, "human_radius_m": .3}
        clearance, known, count = tracking_clearance_at_controls(
            track_data, np.array([0.0, .5, 1.0]), np.zeros((3, 2)), config
        )
        self.assertTrue(np.all(known))
        self.assertTrue(np.all(np.isinf(clearance)))
        self.assertTrue(np.all(count == 0))

    def test_end_to_end_run_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {
                "tracking_max_gap_s": .5, "jerk_window_samples": 3,
                "jerk_polynomial_degree": 2, "robot_radius_m": .3,
                "human_radius_m": .3, "personal_space_m": .5,
                "minimum_tracking_coverage": .9,
            }
            manifest = {
                "run_id": "run_001_jessi_trial_01", "policy": "JESSI",
                "policy_trial": 1, "outcome": "success", "configuration": config,
                "outcome_event": {"reason": "goal_reached", "timestamp": 3.0},
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            trajectory = [
                {"command_timestamp": t, "action": np.array([1., 0.]), "published_action": np.array([1., 0.])}
                for t in (0., 1., 2.)
            ]
            with (run_dir / "controller.pkl").open("wb") as stream:
                pickle.dump({"trajectory": trajectory, "final_event": manifest["outcome_event"]}, stream)
            odom = [odometry(t, t, 0, 0, 1) for t in (0., .5, 1., 1.5, 2.)]
            with (run_dir / "sensor_messages.pkl").open("wb") as stream:
                pickle.dump({"odom": odom}, stream)
            np.savez_compressed(
                run_dir / "human_tracks.npz",
                timestamps=np.array([0., 1., 2.]), track_ids=np.array([1]),
                states=np.array([[[2., 0, 0, 0]], [[3., 0, 0, 0]], [[4., 0, 0, 0]]]),
                active=np.ones((3, 1), bool), valid=np.ones((3, 1), bool),
            )
            (run_dir / "timestamp_alignment.json").write_text(json.dumps({"valid": True}))
            metrics = compute_run(run_dir)
            self.assertAlmostEqual(metrics["time_to_goal_s"], 3.0)
            self.assertAlmostEqual(metrics["path_length_m"], 2.0)
            self.assertAlmostEqual(metrics["average_jerk_m_s3"], 0.0)
            self.assertTrue(metrics["synchronization_valid"])


if __name__ == "__main__":
    unittest.main()
