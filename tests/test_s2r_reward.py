import unittest

import jax.numpy as jnp

from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1


def make_state(robot_position, human_position=(10.0, 10.0)):
    human = jnp.array([human_position[0], human_position[1], 0.0, 0.0, 0.0, 0.0])
    robot = jnp.array([robot_position[0], robot_position[1], 0.0, 0.0, 0.0, 0.0])
    return jnp.stack([human, robot])


class TransitionRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = Reward1(
            robot_radius=0.3,
            v_max=0.45,
            goal_reward=5.0,
            collision_with_humans_penalty=-2.0,
            collision_with_obstacles_penalty=-2.0,
            timeout_penalty_reward=True,
            timeout_penalty=-0.5,
            progress_to_goal_weight=0.3,
            high_rotation_penalty_reward=False,
        )

    def info(self, goal=(10.0, 0.0), time=0.0, obstacle=None):
        if obstacle is None:
            obstacle = jnp.full((1, 1, 2, 2), jnp.nan)
        return {
            "robot_goal": jnp.array(goal),
            "humans_parameters": jnp.array([[0.3] + [0.0] * 18]),
            "static_obstacles": jnp.stack([obstacle, obstacle]),
            "time": jnp.asarray(time),
        }

    def transition(self, old_state, new_state, info):
        return self.reward.transition(
            old_state,
            new_state,
            new_state[None, ...],
            jnp.array([0.0, 0.0]),
            info,
            0.25,
        )

    def test_progress_uses_actual_endpoint(self):
        old_state = make_state((0.0, 0.0))
        new_state = make_state((0.1, 0.0))
        reward, outcome, _ = self.transition(old_state, new_state, self.info())
        self.assertAlmostEqual(float(reward), 0.03, places=5)
        self.assertTrue(bool(outcome["nothing"]))

    def test_swept_human_collision_dominates_goal(self):
        old_state = make_state((-1.0, 0.0), human_position=(0.0, 0.0))
        new_state = make_state((1.0, 0.0), human_position=(0.0, 0.0))
        reward, outcome, _ = self.transition(old_state, new_state, self.info(goal=(1.0, 0.0)))
        self.assertTrue(bool(outcome["collision_with_human"]))
        self.assertFalse(bool(outcome["success"]))
        self.assertLess(float(reward), -1.9)

    def test_swept_obstacle_collision_detects_segment_crossing(self):
        obstacle = jnp.array([[[[0.0, -1.0], [0.0, 1.0]]]])
        old_state = make_state((-1.0, 0.0))
        new_state = make_state((1.0, 0.0))
        _, outcome, _ = self.transition(old_state, new_state, self.info(obstacle=obstacle))
        self.assertTrue(bool(outcome["collision_with_obstacle"]))

    def test_timeout_uses_end_of_transition(self):
        state = make_state((0.0, 0.0))
        reward, outcome, _ = self.transition(state, state, self.info(time=49.9))
        self.assertTrue(bool(outcome["timeout"]))
        self.assertAlmostEqual(float(reward), -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
