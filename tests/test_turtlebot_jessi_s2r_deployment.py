import pickle
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
from jax import random, tree_util

from socialjym.policies.jessi_s2r import JESSI_S2R
from turtlebot.jessi_s2r_runtime import (
    load_jessi_s2r_parameters,
    relative_sensor_timing,
    validate_parameter_shapes,
)


class TurtleBotJessiS2RDeploymentTests(unittest.TestCase):
    def test_runtime_helpers_extract_and_preserve_relative_timing(self):
        best = {"x": jnp.array([1.0])}
        final = {"x": jnp.array([2.0])}
        with tempfile.NamedTemporaryFile(suffix=".pkl") as stream:
            pickle.dump((best, final, {}, {}, {}), stream)
            stream.flush()
            loaded_best, best_source = load_jessi_s2r_parameters(stream.name)
            loaded_final, final_source = load_jessi_s2r_parameters(
                stream.name, selection="final"
            )
        self.assertEqual(best_source, "rollout:best")
        self.assertEqual(final_source, "rollout:final")
        self.assertTrue(jnp.array_equal(loaded_best["x"], best["x"]))
        self.assertTrue(jnp.array_equal(loaded_final["x"], final["x"]))

        observations = jnp.zeros((2, 12)).at[:, 8:11].set(jnp.array([
            [100.1, 100.2, 100.3], [99.8, 99.9, 100.0]
        ]))
        timing = relative_sensor_timing(observations, 100.3)
        self.assertTrue(jnp.allclose(
            timing,
            jnp.array([[-0.2, -0.1, 0.0], [-0.5, -0.4, -0.3]]),
            atol=1e-5,
        ))

    def test_turtlebot_pickle_matches_policy_and_produces_finite_action(self):
        network_path = Path(__file__).parents[1] / "turtlebot" / "jessi_s2r_v4_multitask_rl_out_32.pkl"
        self.assertTrue(network_path.is_file())
        parameters, _ = load_jessi_s2r_parameters(network_path, selection="best")
        policy = JESSI_S2R(
            robot_radius=0.3,
            v_max=0.45,
            wheels_distance=2 * 0.45 / 1.9,
            dt=0.25,
            n_stack=5,
            n_actions_history=5,
            lidar_num_rays=200,
            lidar_angular_range=2 * jnp.pi,
            lidar_max_dist=10.0,
            n_detectable_humans=10,
            embedding_dim=32,
            n_sectors=60,
            n_stack_for_action_space_bounding=1,
            beam_dropout_rate=0.2,
            humans_trajectory_noise_std=0.0,
        )
        expected = policy.init_nns(random.PRNGKey(0))[3]
        validate_parameter_shapes(parameters, expected)

        obs = jnp.zeros((5, 211))
        obs = obs.at[:, 8:11].set(jnp.array([
            [-0.03, -0.02, 0.0], [-0.28, -0.27, -0.25],
            [-0.53, -0.52, -0.50], [-0.78, -0.77, -0.75],
            [-1.03, -1.02, -1.0],
        ]))
        obs = obs.at[:, 11:].set(10.0)
        info = {
            "robot_goal": jnp.array([3.0, 0.0]),
            "_sensor_timing": obs[:, 8:11],
        }
        robot_params = {
            "v_max": jnp.array(0.45),
            "radius": jnp.array(0.3),
            "wheels_distance": jnp.array(2 * 0.45 / 1.9),
        }
        output = policy.act_with_params(
            random.PRNGKey(1), obs, info, robot_params, parameters, sample=False
        )
        self.assertEqual(output[0].shape, (2,))
        self.assertTrue(all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in tree_util.tree_leaves(output)
        ))


if __name__ == "__main__":
    unittest.main()
