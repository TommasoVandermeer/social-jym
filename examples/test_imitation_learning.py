import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.aux_functions import test_k_trials

### Hyperparameters
imitation_policy = "hsfm"
random_seed = 0 
n_humans = 5
n_obstacles = 2
n_trials = 1000
reward_function = Reward2(
    target_reached_reward = True,
    collision_penalty_reward = True,
    discomfort_penalty_reward = True,
    v_max = 1.,
    progress_to_goal_reward = True,
    progress_to_goal_weight = 0.03,
    high_rotation_penalty_reward=True,
    angular_speed_bound=1.,
    angular_speed_penalty_weight=0.0075,
)

test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans,
    'n_obstacles': n_obstacles,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'humans_policy': imitation_policy,
    'reward_function': reward_function,
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 0,
}
test_env = SocialNav(**test_env_params)
metrics = test_k_trials(
    n_trials, 
    random_seed, 
    test_env, 
    "imitation_learning", 
    {}, 
    reward_function.time_limit
)