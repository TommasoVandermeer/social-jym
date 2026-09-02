import math
import os
import tempfile
import unittest
import pickle
from unittest.mock import patch

import jax.numpy as jnp
from jax import random, tree_util
from jhsfm.hsfm import step as hsfm_step
from jhsfm.utils import get_standard_humans_parameters

from socialjym.envs.lasernav import LaserNav
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.utils.distributions.logistic_normal import LogisticNormal
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1
from socialjym.utils.rollouts.jessi_s2r_rollouts import (
    CURRICULUM_STAGES,
    evaluate_at_curriculum_stage,
    get_v4_scenario_probabilities,
    group_normalize_advantages,
    group_weighted_mean,
    initial_v4_curriculum,
    interpolated_bounds,
    load_warm_start_candidates,
    prepare_numeric_metrics,
    get_social_curriculum_probabilities,
    load_training_checkpoint,
    save_training_checkpoint,
    scenario_curriculum_arrays,
    tree_all_finite,
    tree_select,
    update_ema,
    update_difficulty_curriculum,
    update_v4_curriculum,
)


def _all_finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x))) for x in tree_util.tree_leaves(tree))


class JessiS2RUtilityContracts(unittest.TestCase):
    def test_v4_stages_change_one_axis_and_reach_zero_visibility(self):
        self.assertEqual(CURRICULUM_STAGES[0], (0.0, 1.0))
        self.assertEqual(CURRICULUM_STAGES[-1], (1.0, 0.0))
        for previous, current in zip(CURRICULUM_STAGES, CURRICULUM_STAGES[1:]):
            changed = sum(a != b for a, b in zip(previous, current))
            self.assertEqual(changed, 1)
        self.assertEqual(initial_v4_curriculum(0)["phase"], "domain_ramp")
        self.assertEqual(initial_v4_curriculum(6)["phase"], "joint_alternation")
        self.assertEqual(initial_v4_curriculum(16)["phase"], "visibility_ramp")

    def test_stage_evaluation_receives_exact_bounds_and_visibility(self):
        captured = {}

        def fake_evaluation(*args, **kwargs):
            captured.update(kwargs)
            return {
                "per_scenario": {
                    0: {
                        "success": 1.0,
                        "collision_with_human": 0.0,
                        "collision_with_obstacle": 0.0,
                        "timeout": 0.0,
                    }
                }
            }

        nominal = {"x": jnp.array(2.0)}
        lower = {"x": jnp.array(0.0)}
        upper = {"x": jnp.array(6.0)}
        with patch(
            "socialjym.utils.rollouts.jessi_s2r_rollouts.evaluate_jessi_s2r_policy",
            side_effect=fake_evaluation,
        ):
            result = evaluate_at_curriculum_stage(
                None, None, None, (0,), (0.5, 0.7),
                nominal, lower, upper, nominal, lower, upper,
            )
        self.assertAlmostEqual(float(captured["robot_param_bounds"]["x"][0]), 1.0)
        self.assertAlmostEqual(float(captured["robot_param_bounds"]["x"][1]), 4.0)
        self.assertAlmostEqual(float(captured["env_param_bounds"]["x"][0]), 1.0)
        self.assertEqual(captured["visibility"], 0.7)
        self.assertEqual(result["domain_fraction"], 0.5)

    def test_v4_curriculum_promotes_and_regresses_with_streaks(self):
        good = {
            "social_macro_success": 0.9,
            "social_worst_success": 0.7,
            "social_human_collision_rate": 0.05,
            "navigation_macro_success": 0.9,
            "navigation_worst_success": 0.6,
            "navigation_present": True,
        }
        state = initial_v4_curriculum()
        for update in (25, 50):
            state = update_v4_curriculum(state, good, update)
            self.assertEqual(state["level"], 0)
        state = update_v4_curriculum(state, good, 75)
        self.assertEqual(state["level"], 1)
        bad = good | {"social_macro_success": 0.5}
        state = update_v4_curriculum(state, bad, 100)
        state = update_v4_curriculum(state, bad, 125)
        self.assertEqual(state["level"], 0)

    def test_v4_curriculum_can_traverse_complete_ordered_sequence(self):
        passing = {
            "social_macro_success": 0.9,
            "social_worst_success": 0.7,
            "social_human_collision_rate": 0.05,
            "navigation_macro_success": 0.9,
            "navigation_worst_success": 0.6,
            "navigation_present": True,
        }
        state = initial_v4_curriculum()
        update = 0
        visited = [state["level"]]
        while state["level"] < len(CURRICULUM_STAGES) - 1:
            previous_level = state["level"]
            for _ in range(3):
                update += 25
                state = update_v4_curriculum(state, passing, update)
            self.assertEqual(state["level"], previous_level + 1)
            visited.append(state["level"])
        self.assertEqual(visited, list(range(len(CURRICULUM_STAGES))))
        self.assertEqual((state["domain_fraction"], state["visibility"]), (1.0, 0.0))

    def test_v4_sampling_budgets_and_group_statistics(self):
        keys = (0, 1, 10, 11)
        recovery = get_v4_scenario_probabilities(keys, social_mastered=False)
        mastered = get_v4_scenario_probabilities(keys, social_mastered=True)
        self.assertAlmostEqual(float(jnp.sum(recovery[:2])), 0.9, places=6)
        self.assertAlmostEqual(float(jnp.sum(mastered[:2])), 0.8, places=6)
        equal_metrics = {
            "per_scenario": {
                key: {
                    "success": 0.5,
                    "collision_with_human": 0.2,
                    "collision_with_obstacle": 0.1,
                    "timeout": 0.1,
                }
                for key in keys
            }
        }
        adaptive = get_v4_scenario_probabilities(keys, equal_metrics, False)
        self.assertGreater(float(adaptive[1]), float(adaptive[0]))
        self.assertTrue(bool(jnp.all(adaptive > 0.0)))
        advantages = jnp.array([1.0, 3.0, 10.0, 14.0])
        scenarios = jnp.array(keys)
        normalized = group_normalize_advantages(advantages, scenarios)
        self.assertAlmostEqual(float(jnp.mean(normalized[:2])), 0.0, places=6)
        self.assertAlmostEqual(float(jnp.mean(normalized[2:])), 0.0, places=6)
        only_social = group_weighted_mean(
            jnp.array([1.0, 3.0]), jnp.array([0, 1]), 0.8
        )
        self.assertAlmostEqual(float(only_social), 2.0, places=6)

    def test_interpolated_bounds_and_structured_metrics(self):
        nominal = {"x": jnp.array(1.0)}
        bounds = interpolated_bounds(
            nominal, {"x": jnp.array(0.0)}, {"x": jnp.array(3.0)}, 0.5
        )
        self.assertAlmostEqual(float(bounds["x"][0]), 0.5)
        self.assertAlmostEqual(float(bounds["x"][1]), 2.0)
        structured = [{"macro": 0.5, "per_scenario": {0: {"success": 1.0}}}]
        processed = prepare_numeric_metrics({
            "losses": [1.0, 2.0], "stage_evaluations": structured
        })
        self.assertTrue(jnp.array_equal(processed["losses"], jnp.array([1.0, 2.0])))
        self.assertIs(processed["stage_evaluations"], structured)

    def test_legacy_checkpoint_is_accepted_only_as_warm_start(self):
        payload = {
            "schema_version": 1,
            "state": {
                "params": {"x": jnp.array([1.0])},
                "critic_params": {"x": jnp.array([2.0])},
                "best_params": {"x": jnp.array([3.0])},
                "best_critic_params": {"x": jnp.array([4.0])},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl") as stream:
            pickle.dump(payload, stream)
            stream.flush()
            candidates = load_warm_start_candidates(stream.name)
        self.assertTrue(jnp.array_equal(candidates["final"][0]["x"], jnp.array([1.0])))
        self.assertTrue(jnp.array_equal(candidates["best"][0]["x"], jnp.array([3.0])))

    def test_social_curriculum_preserves_group_budgets(self):
        keys = (0, 1, 10, 11)
        initial = get_social_curriculum_probabilities(keys, update=0)
        final = get_social_curriculum_probabilities(keys, update=500)
        self.assertAlmostEqual(float(jnp.sum(initial[:2])), 0.9, places=6)
        self.assertAlmostEqual(float(jnp.sum(final[:2])), 0.7, places=6)
        self.assertAlmostEqual(float(jnp.sum(initial)), 1.0, places=6)

    def test_difficulty_curriculum_requires_two_promotions(self):
        keys = (0, 1, 10)
        rates = jnp.array([0.9, 0.8, 1.0])
        first = update_difficulty_curriculum(0.0, 1.0, keys, rates, 0.1)
        second = update_difficulty_curriculum(
            first[0], first[1], keys, rates, 0.1, first[2], first[3]
        )
        self.assertEqual(first[0], 0.0)
        self.assertAlmostEqual(second[0], 0.1)
        self.assertAlmostEqual(second[1], 0.9)

    def test_checkpoint_roundtrip_and_config_validation(self):
        config = {"devices": 1, "scenarios": [0, 1]}
        state = {"params": {"x": jnp.array([1.0, 2.0])}, "logs": [0.5]}
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_checkpoint(directory, 7, state, config)
            loaded = load_training_checkpoint(path, config)
            self.assertEqual(loaded["next_update"], 7)
            self.assertTrue(jnp.array_equal(
                loaded["state"]["params"]["x"], state["params"]["x"]
            ))
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_training_checkpoint(path, {"devices": 2, "scenarios": [0, 1]})
            self.assertTrue(os.path.exists(path))

    def test_scenario_curriculum_arrays_use_subset_order(self):
        rates = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        episodes = {0: 1, 1: 0, 2: 4, 3: 9, 4: 12}

        scenario_rates, observed = scenario_curriculum_arrays(
            (3, 1, 2), rates, episodes
        )

        self.assertTrue(jnp.allclose(scenario_rates, jnp.array([0.4, 0.2, 0.3])))
        self.assertTrue(jnp.array_equal(observed, jnp.array([True, False, True])))
        self.assertEqual(scenario_rates.shape, observed.shape)

    def test_update_ema_always_returns_both_emas(self):
        overall, scenarios = update_ema(
            None,
            0.5,
            None,
            jnp.array([0.25, 0.75]),
        )
        self.assertAlmostEqual(float(overall), 0.5)
        self.assertTrue(jnp.allclose(scenarios, jnp.array([0.25, 0.75])))

        overall, scenarios = update_ema(
            overall,
            1.0,
            scenarios,
            jnp.array([1.0, 0.0]),
        )
        self.assertTrue(math.isfinite(float(overall)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(scenarios))))

        _, masked_scenarios = update_ema(
            overall,
            0.0,
            scenarios,
            jnp.array([0.0, 0.0]),
            scenario_observed=jnp.array([True, False]),
        )
        self.assertAlmostEqual(
            float(masked_scenarios[1]), float(scenarios[1]), places=6
        )

    def test_logistic_normal_extreme_inputs_are_finite(self):
        distribution = {
            "locs": jnp.array([1000.0, -1000.0, 0.0]),
            "log_scales": jnp.array([-20.0, 2.0, -20.0]),
            "vertices": jnp.array([[0.0, 2.0], [0.0, -2.0], [1.0, 0.0]]),
        }
        dist = LogisticNormal()
        latent = dist.sample(distribution, random.PRNGKey(0))
        values = (
            latent,
            dist.mean(distribution),
            dist.to_env_action(distribution, latent),
            dist.weight_entropy(distribution),
            dist.neglogp(distribution, latent),
            dist.var(distribution),
        )
        self.assertTrue(_all_finite(values))

    def test_nonfinite_transaction_selects_last_good_tree(self):
        old = {"x": jnp.array([1.0, 2.0])}
        candidate = {"x": jnp.array([jnp.nan, 3.0])}
        finite = tree_all_finite(candidate)
        committed = tree_select(finite, candidate, old)
        self.assertFalse(bool(finite))
        self.assertTrue(jnp.array_equal(committed["x"], old["x"]))

    def test_hsfm_overlap_does_not_generate_nan(self):
        states = jnp.zeros((2, 6))
        goals = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
        params = get_standard_humans_parameters(2)
        obstacles = jnp.full((2, 1, 1, 2, 2), jnp.nan)
        visibility = jnp.array([[False, True], [True, False]])
        next_states = hsfm_step(
            states, visibility, goals, params, obstacles, 0.01
        )
        self.assertTrue(_all_finite(next_states))


class LaserNavLegacyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward = Reward1(robot_radius=0.3, v_max=0.45, time_limit=1.0)
        cls.env = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="circular_crossing",
            n_humans=2,
            n_obstacles=0,
            reward_function=cls.reward,
            wheels_distance=0.47,
            n_stack=2,
            lidar_num_rays=16,
        )

    def test_legacy_reset_step_and_observation_contract(self):
        state, reset_key, obs, info, outcome = self.env.reset(random.PRNGKey(0))
        self.assertEqual(obs.shape, (2, 16 + 11))
        self.assertEqual(set(outcome), {
            "success", "collision_with_human", "collision_with_obstacle", "timeout", "nothing"
        })
        next_state, next_obs, next_info, reward, next_outcome, keys = self.env.step(
            state,
            info,
            jnp.array([0.1, 0.0]),
            reset_key=reset_key,
            env_key=random.PRNGKey(1),
        )
        self.assertEqual(next_state.shape, state.shape)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(len(reward), 2)
        self.assertEqual(len(keys), 2)
        self.assertTrue(_all_finite((next_state, next_obs, reward[0])))
        self.assertEqual(set(next_info), set(info))
        self.assertEqual(set(next_outcome), set(outcome))

    def test_jessi_s2r_legacy_act_contract(self):
        _, _, obs, info, _ = self.env.reset(random.PRNGKey(2))
        policy = JESSI_S2R(
            v_max=0.45,
            wheels_distance=0.47,
            n_stack=2,
            n_actions_history=2,
            lidar_num_rays=16,
            n_sectors=4,
            n_detectable_humans=2,
            embedding_dim=8,
            humans_prediction_horizon=2,
        )
        _, _, _, params = policy.init_nns(random.PRNGKey(3))
        outputs = policy.act(random.PRNGKey(4), obs, info, params, sample=False)
        self.assertEqual(len(outputs), 12)
        self.assertEqual(outputs[0].shape, (2,))
        self.assertTrue(_all_finite(outputs))

    def test_parameterized_api_is_additive_and_uses_runtime_limits(self):
        legacy_info_keys = set(self.env.reset(random.PRNGKey(5))[3])
        result = self.env.reset_with_params(
            random.PRNGKey(6),
            robot_param_bounds={
                "v_max": (0.2, 0.2),
                "wheels_distance": (0.5, 0.5),
                "actuation_gain": (0.5, 0.5),
            },
            env_param_bounds={
                "robot_visibility_probability": (1.0, 1.0),
                "lidar_period": (0.1, 0.1),
                "lidar_latency": (0.05, 0.05),
                "lidar_range_scale": (0.5, 0.5),
            },
        )
        state, reset_key, obs, info, robot_params, env_params, outcome = result
        self.assertEqual(obs.shape, (2, 27))
        self.assertEqual(set(info) - legacy_info_keys, {"_robot_params", "_env_params"})
        self.assertAlmostEqual(float(robot_params["v_max"]), 0.2, places=6)
        self.assertAlmostEqual(float(robot_params["wheels_distance"]), 0.5, places=6)
        self.assertAlmostEqual(float(env_params["robot_visibility_probability"]), 1.0, places=6)
        self.assertLessEqual(float(jnp.max(obs[:, 11:])), 5.0 + 1e-5)
        self.assertGreaterEqual(float(obs[0, 10] - obs[0, 8]), 0.05 - 1e-5)

        step_result = self.env.step_with_params(
            state,
            info,
            robot_params,
            env_params,
            jnp.array([1.0, 10.0]),
            reset_key=reset_key,
            env_key=random.PRNGKey(7),
        )
        next_state, next_obs, _, next_robot_params, _, reward, next_outcome, keys = step_result
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(len(keys), 2)
        self.assertLessEqual(float(next_state[-1, 2]), 0.2 + 1e-5)
        self.assertAlmostEqual(float(next_robot_params["v_max"]), 0.2, places=6)
        self.assertTrue(_all_finite((next_state, next_obs, reward[0], next_outcome)))

        policy = JESSI_S2R(
            v_max=0.45,
            wheels_distance=0.47,
            n_stack=2,
            n_actions_history=2,
            lidar_num_rays=16,
            n_sectors=4,
            n_detectable_humans=2,
            embedding_dim=8,
            humans_prediction_horizon=2,
        )
        _, _, _, params = policy.init_nns(random.PRNGKey(8))
        policy_outputs = policy.act_with_params(
            random.PRNGKey(9), obs, info, robot_params, params, sample=False
        )
        vertices = policy_outputs[7]["vertices"]
        self.assertLessEqual(float(jnp.max(vertices[:, 0])), 0.2 + 1e-5)
        self.assertLessEqual(float(jnp.max(jnp.abs(vertices[:, 1]))), 0.8 + 1e-5)
        self.assertTrue(_all_finite(policy_outputs))


if __name__ == "__main__":
    unittest.main()
