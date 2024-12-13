from jax import random
from jax.tree_util import tree_map
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import pickle
from datetime import date
import math
import pandas as pd

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import test_k_trials, decimal_to_binary, load_socialjym_policy
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

random_seed = 0
n_il_epochs = 50
n_rl_episodes = 30_000
n_test_trials = 1000
test_n_humans = [5,15,25]
humans_policy = 'hsfm'
scenario = 'hybrid_scenario'
# Reward terms params
reward_terms = ['progress_to_goal', 'time_penalty', 'high_rotation_penalty']
ds = 0.2 # Discomfort distance
wp = 0.03 # Progress to goal weight
wt = 0.005 # Time penalty weight
wr = 0.07 # High rotation penalty weight
w_bound = 2. # Rotation bound

# List of trained policies. It's len() must be 2**len(reward_terms)
policies = [
    'sarl_hsfm_unicycle_reward_0_circular_crossing_09_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_1_circular_crossing_09_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_2_circular_crossing_10_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_3_circular_crossing_10_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_4_circular_crossing_10_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_5_circular_crossing_10_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_6_circular_crossing_10_12_2024.pkl',
    'sarl_hsfm_unicycle_reward_7_circular_crossing_11_12_2024.pkl',
]

all_metrics_after_rl = {
    "successes": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "collisions": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "timeouts": jnp.empty((2**len(reward_terms),len(test_n_humans))), 
    "returns": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "times_to_goal": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_speed": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_acceleration": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "average_angular_jerk": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "min_distance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "space_compliance": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "episodic_spl": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials)),
    "path_length": jnp.empty((2**len(reward_terms),len(test_n_humans), n_test_trials))
}

# TESTING LOOP FOR EACH REWARD FUNCTION
for reward_type_decimal in range(2**(len(reward_terms))):
    print(f"\n#### REWARD {reward_type_decimal} ####\n")
    binary_reward = decimal_to_binary(reward_type_decimal, len(reward_terms))
    ### INITIALIZE REWARD FUNCTION
    reward_function = Reward2(
        target_reached_reward = True,
        collision_penalty_reward = True,
        discomfort_penalty_reward = True,
        progress_to_goal_reward = binary_reward[0],
        time_penalty_reward = binary_reward[1],
        high_rotation_penalty_reward = binary_reward[2],
        discomfort_distance=ds,
        progress_to_goal_weight=wp,
        time_penalty=wt,
        angular_speed_bound=w_bound,
        angular_speed_penalty_weight=wr
    )
    # Initialize robot policy and vnet params
    policy = SARL(reward_function, dt=0.25, kinematics='unicycle')
    rl_model_params = load_socialjym_policy(
        os.path.join(os.path.expanduser("~"),f"Repos/social-jym/trained_policies/socialjym_policies/{policies[reward_type_decimal]}")
    )
    # Execute tests to evaluate return after RL
    for test, n_humans in enumerate(test_n_humans):
        test_env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': 0.25,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': scenario,
            'hybrid_scenario_subset': jnp.array([0,1,2,3], dtype=jnp.int32),
            'humans_policy': humans_policy,
            'circle_radius': 7,
            'reward_function': reward_function,
            'kinematics': 'unicycle',
        }
        test_env = SocialNav(**test_env_params)
        metrics_after_rl = test_k_trials(
            n_test_trials, 
            32_000 + n_test_trials, 
            test_env, 
            policy, 
            rl_model_params, 
            reward_function.time_limit)
        all_metrics_after_rl = tree_map(lambda x, y: x.at[reward_type_decimal,test].set(y), all_metrics_after_rl, metrics_after_rl)

# Save results
with open(os.path.join(os.path.dirname(__file__),"metrics_after_rl_ablation_study.pkl"), 'wb') as f:
    pickle.dump(all_metrics_after_rl, f)