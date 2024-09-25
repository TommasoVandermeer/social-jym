from jax import random, debug
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.reward1 import generate_reward_done_function
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory, load_crowdnav_policy, test_k_trials

# Hyperparameters
random_seed = 1
n_episodes = 50
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
}
reward_function = generate_reward_done_function(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing',
    'humans_policy': 'hsfm',
    'reward_function': reward_function
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
# Load Social-Navigation-PyEnvs policy vnet params
vnet_params = load_crowdnav_policy(
    "sarl", 
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_on_hsfm_new_guo/rl_model.pth"))
policy = SARL(env.reward_function, dt=env_params['robot_dt'])
# vnet_params = load_crowdnav_policy(
#     "cadrl", 
#     os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/cadrl_on_hsfm_new_guo/rl_model.pth"))
# policy = CADRL(env.reward_function, dt=env_params['robot_dt'])

# Test Social-Navigation-PyEnvs policy
test_k_trials(100, 10, env, policy, vnet_params, success_reward=reward_params['goal_reward'], failure_reward=reward_params['collision_penalty'])

# Initialize random keys
reset_key = random.key(random_seed)
policy_key = random.key(random_seed)

# Simulate some episodes
for i in range(n_episodes):
    done = False
    episode_start_time = time.time()
    state, reset_key, obs, info = env.reset(reset_key)
    all_states = np.array([state])
    while not done:
        action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
        state, obs, info, reward, done = env.step(state,info,action,test=True) 
        all_states = np.vstack((all_states, [state]))

    ## Plot episode trajectory
    # figure, ax = plt.subplots(figsize=(10,10))
    # ax.axis('equal')
    # plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    # for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
    #     plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius)
    # # plot last state
    # plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius)
    # plt.show()

    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'])