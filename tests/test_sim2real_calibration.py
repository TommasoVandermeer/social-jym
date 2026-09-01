import unittest

import numpy as np

from socialjym.utils.sim2real_calibration import calibrate_trajectories


class Sim2RealCalibrationTests(unittest.TestCase):
    def test_synthetic_delayed_first_order_log_produces_finite_bounds(self):
        dt = 0.25
        commands = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.2, 0.0]] * 8)
        measured = np.zeros_like(commands)
        for i in range(1, len(commands)):
            delayed = commands[max(i - 1, 0)]
            measured[i] = 0.6 * measured[i - 1] + 0.4 * delayed
        trajectory = []
        for i, (command, twist) in enumerate(zip(commands, measured)):
            t = i * dt
            trajectory.append({
                "control_loop_timestamp": t,
                "raw_scan_timestamp": t - 0.05,
                "raw_odom_timestamp": t - 0.02,
                "published_action": command,
                "measured_twist": twist,
            })
        result = calibrate_trajectories(
            [trajectory],
            {"v_max": 0.45, "wheels_distance": 0.47, "robot_radius": 0.3},
        )
        self.assertGreaterEqual(result["estimator"]["best_delay_steps"], 0)
        for bounds in result["robot_param_bounds"].values():
            self.assertTrue(np.all(np.isfinite(bounds)))
            self.assertLessEqual(bounds[0], bounds[1])


if __name__ == "__main__":
    unittest.main()
