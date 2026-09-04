import unittest

import haiku as hk
import jax
import jax.numpy as jnp
from jhsfm.utils import get_standard_humans_parameters

from socialjym.policies.jessi import AngularLocalCrossAttention
from socialjym.policies.jessi_s2r import (
    JESSI_S2R,
    pack_humans_trajectory,
    wrap_relative_angles,
)
from socialjym.utils.distributions.logistic_normal import LogisticNormal
from socialjym.utils.rollouts.jessi_s2r_rollouts import entropy_coefficient, update_ema


class LogisticNormalTests(unittest.TestCase):
    def setUp(self):
        self.distribution = LogisticNormal()
        self.vertices = jnp.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]])

    def test_two_dimensional_latent_is_finite(self):
        parameters = {
            "locs": jnp.array([1_000.0, -1_000.0]),
            "log_scales": jnp.array([-1.0, -1.0]),
            "vertices": self.vertices,
        }
        latent = self.distribution.sample(parameters, jax.random.PRNGKey(0))
        action = self.distribution.to_env_action(parameters, latent)
        self.assertEqual(latent.shape, (2,))
        self.assertEqual(action.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(action))))
        self.assertTrue(bool(jnp.isfinite(self.distribution.weight_entropy(parameters))))
        self.assertTrue(bool(jnp.isfinite(self.distribution.neglogp(parameters, latent))))
        self.assertAlmostEqual(float(jnp.sum(self.distribution.weights(parameters, latent))), 1.0, places=6)

    def test_projection_is_inside_triangle(self):
        parameters = {"vertices": self.vertices}
        projected = self.distribution.project_to_support(parameters, jnp.array([2.0, 2.0]))
        self.assertTrue(bool(self.distribution.is_in_support(parameters, projected)))


class ScheduleAndEmaTests(unittest.TestCase):
    def test_entropy_schedule_uses_training_length(self):
        initial = 0.007
        self.assertAlmostEqual(float(entropy_coefficient(initial, 0, 1_000)), initial, places=7)
        self.assertGreater(float(entropy_coefficient(initial, 1, 1_000)), 0.006)
        self.assertAlmostEqual(
            float(entropy_coefficient(initial, 600, 1_000)),
            initial * float(jnp.exp(-1.0)),
            places=7,
        )

    def test_ema_always_returns_pair_and_masks_unseen_scenarios(self):
        global_ema, scenario_ema = update_ema(
            None,
            jnp.nan,
            None,
            jnp.array([1.0, jnp.nan]),
            batch_valid=False,
            scenario_batch_valid=jnp.array([True, False]),
        )
        self.assertAlmostEqual(float(global_ema), 0.5)
        self.assertAlmostEqual(float(scenario_ema[0]), 0.54)
        self.assertAlmostEqual(float(scenario_ema[1]), 0.5)


class PolicyArchitectureTests(unittest.TestCase):
    def test_time_major_trajectories_are_packed_per_human(self):
        trajectory = jnp.array(
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[3.0, 4.0], [30.0, 40.0]],
            ]
        )
        packed = pack_humans_trajectory(trajectory)
        self.assertTrue(bool(jnp.array_equal(packed[0], jnp.array([1.0, 2.0, 3.0, 4.0]))))
        self.assertTrue(bool(jnp.array_equal(packed[1], jnp.array([10.0, 20.0, 30.0, 40.0]))))

    def test_current_and_predicted_relative_angles_are_wrapped(self):
        headings = jnp.array([[3.5 * jnp.pi, -3.5 * jnp.pi], [5.0 * jnp.pi, -5.0 * jnp.pi]])
        wrapped = wrap_relative_angles(headings, jnp.pi / 2)
        self.assertTrue(bool(jnp.all(wrapped >= -jnp.pi)))
        self.assertTrue(bool(jnp.all(wrapped <= jnp.pi)))
        shifted = wrap_relative_angles(headings + 4 * jnp.pi, jnp.pi / 2)
        self.assertTrue(bool(jnp.allclose(jnp.sin(wrapped), jnp.sin(shifted), atol=2e-6)))
        self.assertTrue(bool(jnp.allclose(jnp.cos(wrapped), jnp.cos(shifted), atol=2e-6)))

    def test_s2r_initializes_four_networks_with_two_latents(self):
        policy = JESSI_S2R(
            v_max=0.45,
            wheels_distance=0.47,
            lidar_num_rays=12,
            humans_prediction_horizon=2,
            humans_trajectory_noise_std=0.0,
        )
        perception, actor, critic, e2e = policy.init_nns(jax.random.PRNGKey(0))
        self.assertTrue(perception and actor and critic and e2e)
        raw_scales = next(
            value
            for module in actor.values()
            for name, value in module.items()
            if name == "raw_logscales"
        )
        self.assertEqual(raw_scales.shape, (2,))

    def test_trajectory_noise_does_not_modify_heading(self):
        policy = JESSI_S2R(lidar_num_rays=12, humans_trajectory_noise_std=0.5)
        state = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        noisy, clean, _ = policy.next_humans_state(
            jax.random.PRNGKey(0),
            state,
            jnp.array([[False]]),
            jnp.array([[5.0, 0.0]]),
            get_standard_humans_parameters(1),
            jnp.full((1, 1, 1, 2, 2), jnp.nan),
        )
        self.assertTrue(bool(jnp.allclose(noisy[:, 4], clean[:, 4])))
        self.assertFalse(bool(jnp.allclose(noisy[:, 5], clean[:, 5])))

    def test_beam_dropout_is_training_only_and_finite(self):
        def network(x, sectors, dropout_key=None):
            return AngularLocalCrossAttention(8, 4, beam_dropout_rate=0.75)(
                x, sectors, key=dropout_key
            )[0]

        transformed = hk.transform(network)
        x = jnp.arange(1 * 16 * 8, dtype=jnp.float32).reshape((1, 16, 8)) / 100.0
        sectors = jnp.tile(jnp.arange(4), 4).reshape((1, 16, 1))
        params = transformed.init(jax.random.PRNGKey(0), x, sectors, None)
        deterministic_a = transformed.apply(params, None, x, sectors, None)
        deterministic_b = transformed.apply(params, None, x, sectors, None)
        dropped = transformed.apply(params, None, x, sectors, jax.random.PRNGKey(1))
        self.assertTrue(bool(jnp.allclose(deterministic_a, deterministic_b)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(dropped))))
        self.assertFalse(bool(jnp.allclose(deterministic_a, dropped)))


if __name__ == "__main__":
    unittest.main()
