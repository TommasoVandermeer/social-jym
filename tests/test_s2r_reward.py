import unittest

import jax.numpy as jnp

from socialjym.utils.rewards.lasernav_rewards.s2r_reward import S2RReward


def _state(human_xy, robot_xy=(0.0, 0.0), human_velocity=(0.0, 0.0), robot_v=0.0):
    return jnp.array(
        [
            [human_xy[0], human_xy[1], human_velocity[0], human_velocity[1], 0.0, 0.0],
            [robot_xy[0], robot_xy[1], robot_v, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )


class S2RRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = S2RReward(robot_radius=0.3, v_max=0.45)
        self.info = {
            "humans_parameters": jnp.array([[0.3]], dtype=jnp.float32),
            "static_obstacles": jnp.array(
                [[[[[100.0, 100.0], [101.0, 100.0]]]]], dtype=jnp.float32
            ),
            "robot_goal": jnp.array([10.0, 0.0], dtype=jnp.float32),
            "time": jnp.array(0.0, dtype=jnp.float32),
        }

    def test_teleport_across_robot_is_not_swept_collision(self):
        initial = _state((-2.0, 0.0))
        history = _state((2.0, 0.0))[None, ...]
        _, outcome, _ = self.reward.evaluate_transition(
            initial, jnp.zeros(2), self.info, 0.25, history
        )
        self.assertFalse(bool(outcome["collision_with_human"]))

    def test_intermediate_overlap_is_detected(self):
        initial = _state((-2.0, 0.0))
        history = jnp.stack((_state((0.2, 0.0)), _state((2.0, 0.0))))
        reward, outcome, _ = self.reward.evaluate_transition(
            initial, jnp.zeros(2), self.info, 0.25, history
        )
        self.assertTrue(bool(outcome["collision_with_human"]))
        self.assertLess(float(reward), -4.0)

    def test_yielding_is_better_than_driving_into_oncoming_human(self):
        n_substeps = 25
        human_x = jnp.linspace(0.8, 0.675, n_substeps)
        stopped_history = jnp.stack(
            [_state((float(x), 0.0), human_velocity=(-0.5, 0.0)) for x in human_x]
        )
        moving_history = jnp.stack(
            [
                _state(
                    (float(x), 0.0),
                    robot_xy=(float(0.1 * (i + 1) / n_substeps), 0.0),
                    human_velocity=(-0.5, 0.0),
                    robot_v=0.4,
                )
                for i, x in enumerate(human_x)
            ]
        )
        initial = _state((0.8, 0.0), human_velocity=(-0.5, 0.0))
        stopped_reward, _, _ = self.reward.evaluate_transition(
            initial, jnp.zeros(2), self.info, 0.25, stopped_history
        )
        moving_reward, _, _ = self.reward.evaluate_transition(
            initial, jnp.array([0.4, 0.0]), self.info, 0.25, moving_history
        )
        self.assertGreater(float(stopped_reward), float(moving_reward))


if __name__ == "__main__":
    unittest.main()
