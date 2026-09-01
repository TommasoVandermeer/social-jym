import unittest

import jax.numpy as jnp
from jax import random

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1


class SocialNavContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SocialNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="circular_crossing",
            n_humans=2,
            reward_function=Reward1(v_max=1.0, time_limit=1.0, kinematics="holonomic"),
            kinematics="holonomic",
        )

    def test_legacy_contract_is_unchanged(self):
        state, key, obs, info, outcome = self.env.reset(random.PRNGKey(0))
        self.assertEqual(obs.shape, (3, 6))
        result = self.env.step(
            state,
            info,
            jnp.array([0.1, 0.0]),
            reset_key=key,
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(result[1].shape, obs.shape)
        self.assertNotIn("_robot_params", info)
        self.assertEqual(set(outcome), {"success", "failure", "timeout", "nothing"})

    def test_parameterized_contract_is_additive(self):
        state, key, obs, info, robot_params, env_params, _ = self.env.reset_with_params(
            random.PRNGKey(1),
            robot_param_bounds={"v_max": (0.2, 0.2), "radius": (0.35, 0.35)},
            env_param_bounds={"robot_visibility_probability": (1.0, 1.0)},
        )
        self.assertAlmostEqual(float(obs[-1, 4]), 0.35, places=6)
        result = self.env.step_with_params(
            state,
            info,
            robot_params,
            env_params,
            jnp.array([2.0, 0.0]),
            reset_key=key,
        )
        self.assertEqual(len(result), 8)
        self.assertLessEqual(float(jnp.linalg.norm(result[0][-1, 2:4])), 0.2 + 1e-5)


if __name__ == "__main__":
    unittest.main()
