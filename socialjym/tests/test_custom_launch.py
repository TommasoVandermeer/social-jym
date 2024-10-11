from jax import random, debug, vmap, device_get
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory

# Hyperparameters
custom_data_file = "custom_episodes_1.pkl"
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
}
reward_function = Reward1(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'hybrid_scenario',
    'humans_policy': 'hsfm',
    'reward_function': reward_function
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARL(env.reward_function, dt=env_params['robot_dt'])
initial_vnet_params = policy.model.init(random.key(1), jnp.zeros((env_params["n_humans"],policy.vnet_input_size)))

# Load custom episodes data
custom_data_dir = os.path.join(os.path.expanduser("~"),"Repos/social-jym/custom_episodes/",custom_data_file)
with open(custom_data_dir, 'rb') as f:
    custom_episodes = pickle.load(f)
n_episodes = len(custom_episodes)

# Simulate some episodes
episode_simulation_times = np.empty((n_episodes,))
for i in range(n_episodes):
    policy_key = random.PRNGKey(0)
    outcome = {"nothing": True, "success": False, "failure": False, "timeout": False}
    state, obs, info = env.reset_custom_episode(custom_episodes[f"{i}"])
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, initial_vnet_params, 0.)
        state, obs, info, reward, outcome = env.step(state,info,action,test=True)
        all_states = np.vstack((all_states, [state]))
    all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting
    ## Plot episode trajectory
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
        plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], info['current_scenario'], info["humans_parameters"][:,0], env.robot_radius)
    # plot last state
    plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius)
    plt.show()
    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'])