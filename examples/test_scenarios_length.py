import jax.numpy as jnp
from jax import random, vmap, lax, jit
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.aux_functions import animate_trajectory, plot_state, plot_trajectory, load_socialjym_policy, test_k_trials, initialize_metrics_dict
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2

random_seed = 0
n_episodes = 1000
n_humans_tests = [5,15,25]
robot_policy = 'sarl'
robot_dt = 0.25
# Scenarios parameters
circle_radius = 7.
traffic_height = 3.
traffic_length = 14.
crowding_square_side = 14.
max_cc_delay = 5.

metrics_dims = (len(SCENARIOS)-1,len(n_humans_tests))
all_metrics = initialize_metrics_dict(n_episodes, metrics_dims)

# Initialize reward function
reward_function = Reward1(kinematics='unicycle')

# Initialize robot policy and vnet params
policy = SARL(reward_function, dt=robot_dt, kinematics='unicycle')
vnet_params = load_socialjym_policy(os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/sarl_after_RL_hsfm_unicycle_reward_0_hybrid_scenario_12_12_2024.pkl"))

# Execute tests
for i, scenario in enumerate(SCENARIOS[:-1]):
    for j, n_humans in enumerate(n_humans_tests):
        # Initialize and reset environment
        env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': robot_dt,
            'humans_dt': 0.01,
            'robot_visible': True,
            'scenario': scenario,
            'humans_policy': 'hsfm',
            'reward_function': reward_function,
            'kinematics': 'unicycle',
            'circle_radius': circle_radius,
            'traffic_height': traffic_height,
            'traffic_length': traffic_length,
            'crowding_square_side': crowding_square_side,
            'max_cc_delay': max_cc_delay,
        }
        env = SocialNav(**env_params)
        metrics = test_k_trials(n_episodes, random_seed, env, policy, vnet_params, env_params['reward_function'].time_limit)
        # Save data
        all_metrics = tree_map(lambda x, y: x.at[i,j].set(y), all_metrics, metrics)

# Save results
with open(os.path.join(os.path.dirname(__file__),"scenarios_length_test_data.pkl"), 'wb') as f:
    pickle.dump(all_metrics, f)

# Load results
with open(os.path.join(os.path.dirname(__file__),"scenarios_length_test_data.pkl"), "rb") as f:
    all_metrics = pickle.load(f)

# Plot results
scenarios_labels = ["CC", "PaT", "PeT", "RC", "DCC"]
time_to_goal = all_metrics["times_to_goal"]
path_length = all_metrics["path_length"]
figure, ax = plt.subplots(2, 3, figsize=(10, 6))
figure.subplots_adjust(hspace=0.5, wspace=0.5, top=0.8)
figure.suptitle(f"Scenarios tuning\n{circle_radius}m circle radius - {traffic_height}m traffic height - {traffic_length}m traffic length - {crowding_square_side}m crowding square side - {max_cc_delay}s max CC delay")
for i, n_humans in enumerate(n_humans_tests):
    ax[0, i].grid()
    ax[0, i].set_title(f"{n_humans} humans")
    ax[0, i].set_xlabel("Scenario")
    ax[0, i].set_ylabel("Time to goal (s)")
    ax[1, i].grid()
    ax[1, i].set_title(f"{n_humans} humans")
    ax[1, i].set_xlabel("Scenario")
    ax[1, i].set_ylabel("Path length (m)")
    for j, scenario in enumerate(SCENARIOS[:-1]):
        time_data = pd.DataFrame(time_to_goal[j,i])
        time_data = time_data.dropna()
        ax[0, i].boxplot(
            time_data, widths=0.4, patch_artist=True, 
            positions = [j],
            boxprops=dict(facecolor="lightblue", edgecolor="lightblue"),
            whiskerprops=dict(color="blue"),
            capprops=dict(color="blue"),
            medianprops=dict(color="blue"),
            meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"), 
            showfliers=False,
            showmeans=True)
        length_data = pd.DataFrame(path_length[j,i])
        length_data = length_data.dropna()
        ax[1, i].boxplot(
            length_data, widths=0.4, patch_artist=True,
            positions = [j], 
            boxprops=dict(facecolor="lightcoral", edgecolor="lightcoral"),
            whiskerprops=dict(color="coral"),
            capprops=dict(color="coral"),
            medianprops=dict(color="red"),
            meanprops=dict(markerfacecolor="red", markeredgecolor="red"), 
            showfliers=False,
            showmeans=True)
    ax[0, i].set_xticklabels(scenarios_labels)
    ax[1, i].set_xticklabels(scenarios_labels)
# Save figure
figure.savefig(os.path.join(os.path.dirname(__file__),f"scenarios_tuning.eps"), format='eps')