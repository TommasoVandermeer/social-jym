import unittest

import jax
import jax.numpy as jnp

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1


class TrafficRespawnTests(unittest.TestCase):
    def test_crowd_chasing_translates_feet_and_preserves_phases(self):
        env = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.05,
            scenario="crowd_chasing",
            n_humans=1,
            n_obstacles=0,
            reward_function=Reward1(
                robot_radius=0.3, high_rotation_penalty_reward=False
            ),
            leg_dynamics=True,
            lidar_num_rays=10,
            n_stack=2,
        )
        state, _, _, initial_info, _ = env.reset(
            jax.random.PRNGKey(0), visibility_chance=0.5
        )
        for flipped in (False, True):
            info = dict(initial_info)
            info["is_x_flipped"] = jnp.asarray(flipped)
            info["humans_goal"] = state[:-1, :2]
            old_body = state[:-1, :2]
            old_feet = info["humans_leg_state"][:, [0, 1, 3, 4]]
            old_phases = info["humans_leg_state"][:, [2, 5]]

            new_info, new_state = env._scenario_based_state_post_update(state, info)
            displacement = new_state[:-1, :2] - old_body
            foot_displacement = (
                new_info["humans_leg_state"][:, [0, 1, 3, 4]] - old_feet
            )

            self.assertTrue(bool(new_info["humans_respawned"][0]))
            self.assertTrue(bool(jnp.allclose(foot_displacement[:, :2], displacement)))
            self.assertTrue(bool(jnp.allclose(foot_displacement[:, 2:], displacement)))
            self.assertTrue(
                bool(jnp.array_equal(new_info["humans_leg_state"][:, [2, 5]], old_phases))
            )


if __name__ == "__main__":
    unittest.main()
