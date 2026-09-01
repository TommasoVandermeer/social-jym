import unittest

import jax.numpy as jnp
from jax import random, tree_util
import optax

from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_s2r_rollouts import jessi_s2r_rl_rollout


class JessiS2RTrainingSmoke(unittest.TestCase):
    def test_one_context_aware_ppo_update_is_finite(self):
        reward = Reward1(robot_radius=0.3, v_max=0.45, time_limit=0.5)
        env = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="hybrid_scenario",
            hybrid_scenario_subset=jnp.array([0]),
            n_humans=1,
            n_obstacles=1,
            reward_function=reward,
            wheels_distance=0.47,
            n_stack=2,
            lidar_num_rays=10,
        )
        policy = JESSI_S2R(
            v_max=0.45,
            wheels_distance=0.47,
            n_stack=2,
            n_actions_history=2,
            lidar_num_rays=10,
            n_sectors=2,
            n_detectable_humans=2,
            embedding_dim=8,
            humans_prediction_horizon=2,
        )
        perception_params, actor_params, critic_params, _ = policy.init_nns(
            random.PRNGKey(0)
        )
        network_params = policy.merge_nns_params(perception_params, actor_params)
        result = jessi_s2r_rl_rollout(
            initial_actor_parameters=network_params,
            initial_critic_parameters=critic_params,
            n_parallel_envs=2,
            train_updates=1,
            random_seed=1,
            actor_network_optimizer=optax.adam(1e-4),
            critic_network_optimizer=optax.adam(1e-4),
            total_batch_size=4,
            mini_batch_size=4,
            micro_batch_size=4,
            policy=policy,
            env=env,
            clip_range=0.2,
            n_epochs=1,
            beta_entropy=1e-3,
            lambda_gae=0.95,
            training_type="policy",
            target_kl=None,
            safety_loss=True,
            robot_param_bounds={
                "v_max": (0.40, 0.48),
                "wheels_distance": (0.45, 0.50),
            },
            env_param_bounds={
                "lidar_period": (0.10, 0.25),
                "lidar_latency": (0.0, 0.10),
            },
        )
        leaves = tree_util.tree_leaves(result[:4])
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in leaves))
        logs = result[-1]
        for key in ("losses", "critic_losses", "grad_norm", "approx_kl"):
            self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(logs[key])))))


if __name__ == "__main__":
    unittest.main()
