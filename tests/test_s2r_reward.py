import unittest

import jax.numpy as jnp
import numpy as np

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

    def test_leg_dynamics_uses_swept_feet_with_effective_radius(self):
        reward_fn = Reward1(
            robot_radius=0.3,
            use_leg_collisions=True,
            effective_foot_radius=0.2,
            high_rotation_penalty_reward=False,
        )
        state = make_state((0.0, 0.0), human_position=(2.0, 2.0))
        # The body is safely away, while the left foot overlaps the robot's
        # effective collision circle.  The physical leg radius is only 0.12 m.
        legs = jnp.array([[0.45, 0.0, 0.0, 0.9, 0.0, 0.0]])
        info = self.info()
        info["humans_leg_state"] = legs
        _, outcome, _ = reward_fn.transition(
            state,
            state,
            state[None],
            jnp.zeros(2),
            info,
            0.25,
            intermediate_leg_states=legs[None],
            leg_dynamics=True,
        )
        self.assertTrue(bool(outcome["collision_with_human"]))

    def test_body_collision_is_retained_without_leg_dynamics(self):
        reward_fn = Reward1(
            robot_radius=0.3,
            use_leg_collisions=True,
            effective_foot_radius=0.2,
            high_rotation_penalty_reward=False,
        )
        state = make_state((0.0, 0.0), human_position=(2.0, 2.0))
        legs = jnp.array([[0.45, 0.0, 0.0, 0.9, 0.0, 0.0]])
        info = self.info()
        info["humans_leg_state"] = legs
        _, outcome, _ = reward_fn.transition(
            state,
            state,
            state[None],
            jnp.zeros(2),
            info,
            0.25,
            intermediate_leg_states=legs[None],
            leg_dynamics=False,
        )
        self.assertFalse(bool(outcome["collision_with_human"]))

    def test_respawn_teleport_is_not_a_swept_collision(self):
        reward_fn = Reward1(
            robot_radius=0.3,
            use_leg_collisions=False,
            high_rotation_penalty_reward=False,
        )
        old_state = make_state((0.0, 0.0), human_position=(-7.0, 0.0))
        post_respawn_state = make_state((0.0, 0.0), human_position=(7.0, 0.0))
        physical_endpoint = jnp.array([[[-6.9, 0.0]]])
        _, outcome, _ = reward_fn.transition(
            old_state,
            post_respawn_state,
            post_respawn_state[None],
            jnp.zeros(2),
            self.info(),
            0.25,
            intermediate_human_end_positions=physical_endpoint,
            intermediate_human_respawns=jnp.array([[True]]),
        )
        self.assertFalse(bool(outcome["collision_with_human"]))

    def test_anticipatory_reward_prefers_steering_out_of_head_on_path(self):
        reward_fn = Reward1(
            robot_radius=0.3,
            v_max=0.45,
            discomfort_penalty_reward=True,
            progress_to_goal_reward=False,
            high_rotation_penalty_reward=False,
            anticipatory_avoidance_reward=True,
            avoidance_distance=1.0,
            avoidance_horizon=1.5,
            avoidance_penalty_weight=0.15,
            avoidance_improvement_weight=0.2,
            head_on_risk_multiplier=1.0,
        )
        old_state = jnp.stack((
            jnp.array([1.5, 0.0, 0.5, 0.0, jnp.pi, 0.0]),
            jnp.array([0.0, 0.0, 0.45, 0.0, 0.0, 0.0]),
        ))
        head_on_state = old_state.at[0, 0].set(1.25).at[1, 0].set(0.1)
        steered_state = head_on_state.at[-1, 4].set(jnp.pi / 2)
        info = self.info()
        head_on_reward, _, _ = reward_fn.transition(
            old_state, head_on_state, head_on_state[None], jnp.zeros(2), info, 0.25
        )
        steered_reward, _, _ = reward_fn.transition(
            old_state, steered_state, steered_state[None], jnp.zeros(2), info, 0.25
        )
        self.assertGreater(float(steered_reward), float(head_on_reward))

    def test_escape_reward_values_clearance_gain_not_rotation_itself(self):
        reward_fn = Reward1(
            robot_radius=0.3,
            v_max=0.45,
            progress_to_goal_reward=True,
            high_rotation_penalty_reward=False,
            local_minimum_escape_reward=True,
            escape_clearance_weight=0.05,
            stalled_rotation_penalty_weight=0.01,
        )
        obstacle = jnp.array([[[[0.6, -1.0], [0.6, 1.0]]]])
        info = self.info(obstacle=obstacle)
        info["action_history"] = jnp.array([[0.0, 0.8], [0.0, 0.8]])
        old_state = make_state((0.0, 0.0))
        productive_turn = old_state.at[-1, 4].set(jnp.pi / 2)
        stalled_turn = old_state.at[-1, 4].set(0.1)
        productive_reward, _, _ = reward_fn.transition(
            old_state,
            productive_turn,
            productive_turn[None],
            jnp.array([0.0, 0.8]),
            info,
            0.25,
        )
        stalled_reward, _, _ = reward_fn.transition(
            old_state,
            stalled_turn,
            stalled_turn[None],
            jnp.array([0.0, 0.8]),
            info,
            0.25,
        )
        self.assertGreater(float(productive_reward), 0.0)
        self.assertLess(float(stalled_reward), 0.0)
        self.assertGreater(float(productive_reward), float(stalled_reward))
        self.assertTrue(np.isfinite(float(productive_reward)))


if __name__ == "__main__":
    unittest.main()
