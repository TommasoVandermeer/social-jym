from jax import random, vmap, debug
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory, load_crowdnav_policy, test_k_trials, load_socialjym_policy

### Hyperparameters
random_seed = 13_000 # Usually we train with 3_000 IL episodes and 10_000 RL episodes
n_episodes = 50
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
}
reward_function = Reward1(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 25,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing',
    'humans_policy': 'sfm',
    'reward_function': reward_function,
    'kinematics': 'unicycle'
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy
policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics='unicycle')

### Plot action space
plt.scatter(policy.action_space[:,0], policy.action_space[:,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
