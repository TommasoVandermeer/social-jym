import unittest
import os
import pickle
import tempfile

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
        checkpoint_dir = tempfile.TemporaryDirectory()
        self.addCleanup(checkpoint_dir.cleanup)
        rollout_args = dict(
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
            checkpoint_dir=checkpoint_dir.name,
            checkpoint_every=1,
            evaluation_every=1,
            evaluation_episodes=1,
        )
        result = jessi_s2r_rl_rollout(**rollout_args)
        checkpoint_path = os.path.join(
            checkpoint_dir.name, "update_000001.pkl"
        )
        self.assertTrue(os.path.exists(
            checkpoint_path
        ))
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self.assertEqual(checkpoint["schema_version"], 2)
        saved_state = checkpoint["state"]
        self.assertIn("curriculum", saved_state)
        self.assertIn("phase", saved_state["curriculum"])
        self.assertIn("best_curriculum_level", saved_state)
        self.assertIn("nominal_best_params", saved_state)
        self.assertIn("robust_best_params", saved_state)
        resumed = jessi_s2r_rl_rollout(
            **(rollout_args | {"checkpoint_dir": None, "resume_from": checkpoint_path})
        )
        self.assertTrue(all(
            jnp.array_equal(a, b)
            for a, b in zip(
                tree_util.tree_leaves(result[1]),
                tree_util.tree_leaves(resumed[1]),
            )
        ))
        leaves = tree_util.tree_leaves(result[:4])
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in leaves))
        logs = result[-1]
        for key in ("losses", "critic_losses", "grad_norm", "approx_kl"):
            self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(logs[key])))))


if __name__ == "__main__":
    unittest.main()
