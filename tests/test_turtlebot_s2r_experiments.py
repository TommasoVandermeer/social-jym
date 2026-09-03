import json
import tempfile
import unittest
from pathlib import Path

from turtlebot.experiments.aggregate_results import aggregate
from turtlebot.experiments.common import POLICIES, balanced_schedule, load_config


class TurtleBotS2RExperimentTests(unittest.TestCase):
    def test_default_campaign_uses_s2r_policy_and_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "campaign_name": "s2r",
                "goal": [3.0, 0.0],
                "jessi_network": "network.pkl",
            }))
            config = load_config(config_path)
        self.assertEqual(POLICIES, ("JESSI-S2R", "DWA"))
        self.assertEqual(config["lidar_rays"], 200)
        self.assertEqual(config["network_selection"], "best")
        schedule = balanced_schedule(3, 7)
        self.assertEqual(sum(run["policy"] == "JESSI-S2R" for run in schedule), 3)

    def test_aggregate_uses_s2r_comparison_key(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "campaign_config.json").write_text(json.dumps({
                "bootstrap_seed": 1, "bootstrap_samples": 20
            }))
            for index, policy in enumerate(POLICIES, start=1):
                run_dir = campaign / f"run_{index:03d}"
                run_dir.mkdir()
                (run_dir / "metrics.json").write_text(json.dumps({
                    "policy": policy,
                    "success": True,
                    "operator_collision": False,
                    "timeout": False,
                    "synchronization_valid": True,
                    "time_to_goal_s": 10.0 + index,
                    "path_length_m": 4.0 + index,
                    "average_jerk_m_s3": 0.2 * index,
                    "space_compliance": 0.9,
                    "minimum_human_clearance_m": 0.6,
                }))
            result = aggregate(campaign)
        self.assertIn("jessi_s2r_minus_dwa", result)
        self.assertNotIn("jessi_minus_dwa", result)


if __name__ == "__main__":
    unittest.main()
