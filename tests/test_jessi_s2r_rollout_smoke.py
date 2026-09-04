import unittest

import jax
import jax.numpy as jnp

from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_s2r_rollouts import (
    collect_rollout_step,
    process_buffer_and_gae,
)


class JessiS2RRolloutSmokeTests(unittest.TestCase):
    def test_float32_rollout_and_critic_are_consistent(self):
        subset = jnp.array([0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16])
        probabilities = jnp.ones((len(subset),)) / len(subset)
        reward = Reward1(
            robot_radius=0.3,
            v_max=0.45,
            timeout_penalty_reward=True,
            timeout_penalty=-0.5,
        )
        environment = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="hybrid_scenario",
            hybrid_scenario_subset=subset,
            n_humans=4,
            n_obstacles=5,
            reward_function=reward,
            robot_visible=None,
            wheels_distance=0.4736842105,
            wheels_max_linear_acceleration=0.87,
            lidar_num_rays=12,
        )
        policy = JESSI_S2R(
            robot_radius=0.3,
            v_max=0.45,
            wheels_distance=0.4736842105,
            lidar_num_rays=12,
            humans_prediction_horizon=2,
            humans_trajectory_noise_std=0.0,
        )
        perception_params, actor_params, critic_params, _ = policy.init_nns(jax.random.PRNGKey(0))
        network_params = policy.merge_nns_params(perception_params, actor_params)

        reset_keys = jax.random.split(jax.random.PRNGKey(1), 1)
        states, reset_keys, observations, infos, outcomes = environment.batch_reset(
            reset_keys,
            scenarios_prob=probabilities,
            visibility_chance=1.0,
        )
        self.assertTrue(bool(jnp.all(infos["visibility"][:, :-1, -1])))
        policy_keys = jax.random.split(jax.random.PRNGKey(2), 1)
        environment_keys = jax.random.split(jax.random.PRNGKey(3), 1)

        result = collect_rollout_step(
            network_params,
            critic_params,
            (states, observations, infos, outcomes),
            policy_keys,
            reset_keys,
            environment_keys,
            outcomes,
            policy,
            environment,
            1,
            probabilities,
            1.0,
        )
        environment_state, policy_keys, _, _, history, outcome_sums, *_ = result
        self.assertEqual(history["actions"].shape[-1], 2)
        self.assertEqual(history["env_params"]["humans_visibility"].shape[-2:], (4, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(history["values"]))))

        # The buffer stores the two Gaussian latents, while LaserNav receives
        # their mapped point inside the three-vertex feasible triangle.
        _, _, _, actor_distribution, _, _, _, _, _ = policy.e2e.apply(
            network_params,
            None,
            history["inputs0"][0],
            history["inputs1"][0],
        )
        environment_actions = policy.action_distribution.batch_to_env_action(
            actor_distribution, history["actions"][0]
        )
        self.assertTrue(
            bool(jnp.all(policy.action_distribution.batch_is_in_support(
                actor_distribution, environment_actions
            )))
        )

        # Re-evaluating the unchanged float32 policy must reproduce old log-probabilities.
        new_neglogp = policy.action_distribution.batch_neglogp(
            actor_distribution, history["actions"][0]
        )
        self.assertTrue(bool(jnp.allclose(new_neglogp, history["neglogpdfs"][0], atol=1e-5)))

        critic_keys = jax.random.split(jax.random.PRNGKey(4), 1)
        buffer = process_buffer_and_gae(
            critic_params,
            critic_keys,
            environment_state[0],
            environment_state[1],
            environment_state[2],
            ~environment_state[3]["nothing"],
            history,
            policy,
            environment,
            reward.gamma,
            policy.dt,
            policy.v_max,
            0.95,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(buffer["advantages"]))))
        self.assertEqual(buffer["env_params"]["humans_visibility"].shape[-2:], (4, 4))
        self.assertGreaterEqual(int(jnp.sum(outcome_sums["terminal"])), 0)


if __name__ == "__main__":
    unittest.main()
