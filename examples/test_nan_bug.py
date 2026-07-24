from jax import random, vmap, jit, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import os
import pickle
import matplotlib.pyplot as plt

from socialjym.envs.lasernav import LaserNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.lasernav_rewards.reward4 import Reward4 as Reward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 3
robot_vmax = 1
robot_wheel_distance = 0.7
time_limit = 50
n_episodes = 100
n_steps = 300
kinematics = 'unicycle'

# Load actor inputs
with open(os.path.join(os.path.dirname(__file__), f'controller_training_dataset_200.pkl'), 'rb') as f:
    controller_dataset = pickle.load(f)
print(controller_dataset.keys())
print("Obs nan: ", jnp.any(jnp.isnan(controller_dataset['observations'])))
print("Robot goals nan: ", jnp.any(jnp.isnan(controller_dataset['rc_robot_goals'])))
print("Actions nan: ", jnp.any(jnp.isnan(controller_dataset['actor_actions'])))
print("Returns nan: ", jnp.any(jnp.isnan(controller_dataset['returns'])))
print("Max return: ", jnp.max(controller_dataset['returns']))
print("Min return: ", jnp.min(controller_dataset['returns']))
print("Max robot action: ", jnp.max(controller_dataset['actor_actions'], axis=(0,)))
print("Min robot action: ", jnp.min(controller_dataset['actor_actions'], axis=(0,)))

for scenario in SCENARIOS[:-1]:
    for lidar_rays in [50,75,100,200,300,400,500]:
        env_params = {
            'n_stack': 5,
            'lidar_num_rays': lidar_rays,
            'lidar_angular_range': jnp.pi * 2,
            'lidar_max_dist': 10.,
            'wheels_distance': robot_wheel_distance,
            'n_humans': 3,
            'n_obstacles': 0,
            'robot_radius': 0.3,
            'robot_dt': 0.25,
            'humans_dt': 0.01,      
            'robot_visible': True,
            'scenario': scenario, 
            'reward_function': Reward(robot_radius=0.3, time_limit=time_limit, v_max=robot_vmax),
            'kinematics': kinematics,
            'lidar_noise': True,
            # 'leg_dynamics': True,
        }
        env = LaserNav(**env_params)
        # Simulate some episodes
        obs_nan_count = 0
        reward_nan_count = 0
        for i in range(n_episodes):
            policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) 
            state, reset_key, obs, info, outcome = env.reset(reset_key)
            if jnp.any(jnp.isnan(obs[:,9:])):
                obs_nan_count += 1
            for j in range(n_steps):
                state, obs, info, (reward, _), outcome, (_, env_key) = env.step(state,info,jnp.array([1.,0.]),test=True,env_key=env_key)
                if jnp.isnan(reward):
                    reward_nan_count +=1
        print(f"Scenario: {scenario} - Lidar rays: {lidar_rays} - Obs Nan count: {obs_nan_count} - Reward Nan count: {reward_nan_count}")