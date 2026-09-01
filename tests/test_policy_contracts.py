import unittest

import jax.numpy as jnp
from jax import random

from socialjym.envs.lasernav import LaserNav
from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.dra_mppi import DRAMPPI
from socialjym.policies.dwa import DWA
from socialjym.policies.jessi import JESSI
from socialjym.policies.jessi_sa import JESSI_SA
from socialjym.policies.jessi_s2r import JESSI_S2R
from socialjym.policies.mppi import MPPI
from socialjym.policies.sarl import SARL
from socialjym.policies.sarl_ppo import SARLPPO
from socialjym.policies.sarl_star import SARLStar
from socialjym.policies.vanilla_e2e import VanillaE2E
from socialjym.utils.rewards.lasernav_rewards.reward1 import Reward1 as LaserReward
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1 as SocialReward


class PolicyContracts(unittest.TestCase):
    """One-step public inference contracts for every shipped policy family."""

    @classmethod
    def setUpClass(cls):
        cls.v_max = 0.45
        cls.wheels_distance = 0.47
        cls.social_reward = SocialReward(
            v_max=cls.v_max, time_limit=0.25, kinematics="unicycle"
        )
        cls.social_env = SocialNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="circular_crossing",
            n_humans=1,
            reward_function=cls.social_reward,
            kinematics="unicycle",
            lidar_num_rays=10,
        )
        cls.laser_env = LaserNav(
            robot_radius=0.3,
            robot_dt=0.25,
            humans_dt=0.01,
            scenario="circular_crossing",
            n_humans=1,
            n_obstacles=0,
            reward_function=LaserReward(
                robot_radius=0.3, v_max=cls.v_max, time_limit=0.25
            ),
            wheels_distance=cls.wheels_distance,
            n_stack=2,
            lidar_num_rays=10,
        )
        _, _, cls.social_obs, cls.social_info, _ = cls.social_env.reset(
            random.PRNGKey(0)
        )
        _, _, cls.laser_obs, cls.laser_info, _ = cls.laser_env.reset(
            random.PRNGKey(1)
        )

    def assert_action(self, action):
        self.assertEqual(action.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(action))))

    def test_socialnav_policy_act_contracts(self):
        policies = [
            CADRL(
                self.social_reward,
                v_max=self.v_max,
                dt=0.25,
                wheels_distance=self.wheels_distance,
                kinematics="unicycle",
            ),
            SARL(
                self.social_reward,
                v_max=self.v_max,
                dt=0.25,
                wheels_distance=self.wheels_distance,
                kinematics="unicycle",
            ),
            SARLStar(
                self.social_reward,
                grid_size=jnp.array([20, 20]),
                use_planner=False,
                v_max=self.v_max,
                dt=0.25,
                wheels_distance=self.wheels_distance,
                kinematics="unicycle",
            ),
        ]
        for index, policy in enumerate(policies):
            with self.subTest(policy=policy.name):
                params = policy.model.init(
                    random.PRNGKey(10 + index),
                    jnp.zeros((self.social_env.n_humans, policy.vnet_input_size)),
                )
                outputs = policy.act(
                    random.PRNGKey(20 + index),
                    self.social_obs,
                    self.social_info,
                    params,
                    epsilon=0.0,
                )
                action = outputs[0]
                self.assert_action(action)

        actor_policies = [
            SARLPPO(
                self.social_reward,
                v_max=self.v_max,
                dt=0.25,
                wheels_distance=self.wheels_distance,
                kinematics="unicycle",
            ),
            DIRSAFE(
                self.social_reward,
                v_max=self.v_max,
                dt=0.25,
                wheels_distance=self.wheels_distance,
            ),
        ]
        for index, policy in enumerate(actor_policies):
            with self.subTest(policy=policy.name):
                actor_params, _ = policy.init_nns(
                    random.PRNGKey(30 + index), self.social_obs, self.social_info
                )
                action = policy.act(
                    random.PRNGKey(40 + index),
                    self.social_obs,
                    self.social_info,
                    actor_params,
                    sample=False,
                )[0]
                self.assert_action(action)

        dra_mppi = DRAMPPI(
            use_halton_spline=False,
            num_samples=4,
            horizon=2,
            monte_carlo_risk_estimation_samples=4,
            robot_radius=0.3,
            v_max=self.v_max,
            dt=0.25,
            wheels_distance=self.wheels_distance,
            n_stack=2,
            lidar_num_rays=10,
        )
        u_mean, beta = dra_mppi.init_u_mean_and_beta()
        action = dra_mppi.act(
            self.social_obs,
            self.social_info,
            u_mean,
            beta,
            random.PRNGKey(50),
        )[0]
        self.assert_action(action)

    def test_lasernav_policy_act_contracts(self):
        dwa = DWA(
            actions_discretization=3,
            predict_time_horizon=0.25,
            robot_radius=0.3,
            v_max=self.v_max,
            dt=0.25,
            wheels_distance=self.wheels_distance,
            n_stack=2,
            lidar_num_rays=10,
        )
        self.assert_action(dwa.act(self.laser_obs, self.laser_info)[0])

        mppi = MPPI(
            num_samples=4,
            horizon=2,
            robot_radius=0.3,
            v_max=self.v_max,
            dt=0.25,
            wheels_distance=self.wheels_distance,
            n_stack=2,
            lidar_num_rays=10,
        )
        self.assert_action(
            mppi.act(
                self.laser_obs,
                self.laser_info,
                mppi.init_u_mean(),
                random.PRNGKey(60),
            )[0]
        )

        vanilla = VanillaE2E(
            robot_radius=0.3,
            v_max=self.v_max,
            dt=0.25,
            wheels_distance=self.wheels_distance,
            n_stack=2,
            lidar_num_rays=10,
        )
        vanilla_params = vanilla.init_nn(random.PRNGKey(61))
        self.assert_action(
            vanilla.act(
                random.PRNGKey(62),
                self.laser_obs,
                self.laser_info,
                vanilla_params,
                sample=False,
            )[0]
        )

        jessi_policies = [
            JESSI(
                v_max=self.v_max,
                wheels_distance=self.wheels_distance,
                n_stack=2,
                lidar_num_rays=10,
                n_sectors=2,
                n_detectable_humans=2,
                embedding_dim=8,
            ),
            JESSI_SA(
                v_max=self.v_max,
                wheels_distance=self.wheels_distance,
                n_stack=2,
                n_actions_history=2,
                lidar_num_rays=10,
                n_sectors=2,
                n_detectable_humans=2,
                embedding_dim=8,
            ),
            JESSI_S2R(
                v_max=self.v_max,
                wheels_distance=self.wheels_distance,
                n_stack=2,
                n_actions_history=2,
                lidar_num_rays=10,
                n_sectors=2,
                n_detectable_humans=2,
                embedding_dim=8,
                humans_prediction_horizon=2,
            ),
        ]
        for index, policy in enumerate(jessi_policies):
            with self.subTest(policy=type(policy).__name__):
                initialized = policy.init_nns(random.PRNGKey(70 + index))
                e2e_params = initialized[-1]
                action = policy.act(
                    random.PRNGKey(80 + index),
                    self.laser_obs,
                    self.laser_info,
                    e2e_params,
                    sample=False,
                )[0]
                self.assert_action(action)

    def test_all_public_evaluate_methods_complete(self):
        """Every concrete policy must complete its public one-episode evaluator."""
        social_value_policies = [
            CADRL(self.social_reward, v_max=self.v_max, dt=0.25,
                  wheels_distance=self.wheels_distance, kinematics="unicycle"),
            SARL(self.social_reward, v_max=self.v_max, dt=0.25,
                 wheels_distance=self.wheels_distance, kinematics="unicycle"),
            SARLStar(self.social_reward, grid_size=jnp.array([20, 20]),
                     use_planner=False, v_max=self.v_max, dt=0.25,
                     wheels_distance=self.wheels_distance, kinematics="unicycle"),
        ]
        for index, policy in enumerate(social_value_policies):
            with self.subTest(evaluate=policy.name):
                params = policy.model.init(
                    random.PRNGKey(100 + index),
                    jnp.zeros((self.social_env.n_humans, policy.vnet_input_size)),
                )
                self.assertIsInstance(
                    policy.evaluate(1, 200 + index, self.social_env, params), dict
                )

        social_actor_policies = [
            SARLPPO(self.social_reward, v_max=self.v_max, dt=0.25,
                    wheels_distance=self.wheels_distance, kinematics="unicycle"),
            DIRSAFE(self.social_reward, v_max=self.v_max, dt=0.25,
                    wheels_distance=self.wheels_distance),
        ]
        for index, policy in enumerate(social_actor_policies):
            with self.subTest(evaluate=policy.name):
                actor_params, _ = policy.init_nns(
                    random.PRNGKey(110 + index), self.social_obs, self.social_info
                )
                self.assertIsInstance(
                    policy.evaluate(1, 210 + index, self.social_env, actor_params), dict
                )

        dra_mppi = DRAMPPI(
            use_halton_spline=False, num_samples=4, horizon=2,
            monte_carlo_risk_estimation_samples=4, robot_radius=0.3,
            v_max=self.v_max, dt=0.25, wheels_distance=self.wheels_distance,
            n_stack=2, lidar_num_rays=10,
        )
        self.assertIsInstance(dra_mppi.evaluate(1, 220, self.social_env), dict)

        laser_classical = [
            DWA(actions_discretization=3, predict_time_horizon=0.25,
                robot_radius=0.3, v_max=self.v_max, dt=0.25,
                wheels_distance=self.wheels_distance, n_stack=2,
                lidar_num_rays=10),
            MPPI(num_samples=4, horizon=2, robot_radius=0.3,
                 v_max=self.v_max, dt=0.25,
                 wheels_distance=self.wheels_distance, n_stack=2,
                 lidar_num_rays=10),
        ]
        for index, policy in enumerate(laser_classical):
            with self.subTest(evaluate=policy.name):
                self.assertIsInstance(
                    policy.evaluate(1, 230 + index, self.laser_env), dict
                )

        vanilla = VanillaE2E(robot_radius=0.3, v_max=self.v_max, dt=0.25,
                             wheels_distance=self.wheels_distance, n_stack=2,
                             lidar_num_rays=10)
        self.assertIsInstance(
            vanilla.evaluate(1, 240, self.laser_env,
                             vanilla.init_nn(random.PRNGKey(140))), dict
        )

        jessi_policies = [
            JESSI(v_max=self.v_max, wheels_distance=self.wheels_distance,
                  n_stack=2, lidar_num_rays=10, n_sectors=2,
                  n_detectable_humans=2, embedding_dim=8),
            JESSI_SA(v_max=self.v_max, wheels_distance=self.wheels_distance,
                     n_stack=2, n_actions_history=2, lidar_num_rays=10,
                     n_sectors=2, n_detectable_humans=2, embedding_dim=8),
            JESSI_S2R(v_max=self.v_max, wheels_distance=self.wheels_distance,
                      n_stack=2, n_actions_history=2, lidar_num_rays=10,
                      n_sectors=2, n_detectable_humans=2, embedding_dim=8,
                      humans_prediction_horizon=2),
        ]
        for index, policy in enumerate(jessi_policies):
            with self.subTest(evaluate=type(policy).__name__):
                params = policy.init_nns(random.PRNGKey(150 + index))[-1]
                self.assertIsInstance(
                    policy.evaluate(1, 250 + index, self.laser_env, params), dict
                )


if __name__ == "__main__":
    unittest.main()
