import tempfile
import unittest

import jax
import jax.numpy as jnp
import optax

from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_s2r_rollouts import jessi_s2r_rl_rollout
from socialjym.utils.training_artifacts import ArtifactStore


class JessiS2RTrainingSmokeTests(unittest.TestCase):
    def test_one_ppo_update_and_checkpoint_reload(self):
        reward = Reward1(
            robot_radius=0.3,
            v_max=0.45,
            time_limit=0.2,
            goal_reward=5.0,
            collision_with_humans_penalty=-2.0,
            collision_with_obstacles_penalty=-2.0,
            timeout_penalty_reward=True,
            timeout_penalty=-0.5,
            progress_to_goal_weight=0.3,
            high_rotation_penalty_reward=False,
        )
        environment = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="hybrid_scenario",
            hybrid_scenario_subset=jnp.array([0]),
            n_humans=1,
            n_obstacles=1,
            reward_function=reward,
            wheels_distance=0.47,
            wheels_max_linear_acceleration=0.87,
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
            humans_trajectory_noise_std=0.0,
        )
        perception, actor, critic, _ = policy.init_nns(jax.random.PRNGKey(0))
        result = jessi_s2r_rl_rollout(
            initial_actor_parameters=policy.merge_nns_params(perception, actor),
            initial_critic_parameters=critic,
            n_parallel_envs=1,
            train_updates=1,
            random_seed=1,
            actor_network_optimizer=optax.adam(1e-5),
            critic_network_optimizer=optax.adam(1e-5),
            total_batch_size=1,
            mini_batch_size=1,
            micro_batch_size=1,
            policy=policy,
            env=environment,
            clip_range=0.2,
            n_epochs=1,
            beta_entropy=0.007,
            lambda_gae=0.95,
            initial_visibility=0.5,
            training_type="policy",
        )
        self.assertEqual(
            set(result),
            {
                "best_actor_params",
                "final_actor_params",
                "best_critic_params",
                "final_critic_params",
                "metrics",
            },
        )
        self.assertTrue(bool(jnp.isfinite(result["metrics"]["losses"][0])))
        self.assertGreater(result["metrics"]["weight_entropies"][0], 0.0)
        self.assertEqual(result["metrics"]["visibilities"], [0.5])

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"smoke": True})
            store.save("dataset", {"size": 1})
            store.save("perception", perception, dependencies=("dataset",))
            store.save("actor", actor, dependencies=("dataset", "perception"))
            store.save("critic", critic, dependencies=("dataset",))
            dependencies = ("perception", "actor", "critic")
            store.save("rl", result, dependencies=dependencies)
            reloaded = store.load("rl", dependencies=dependencies)
            self.assertEqual(set(reloaded), set(result))


if __name__ == "__main__":
    unittest.main()
