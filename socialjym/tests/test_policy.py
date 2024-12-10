from jax import random, vmap, debug
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory, load_crowdnav_policy, test_k_trials, load_socialjym_policy

### Hyperparameters
random_seed = 13_000 # Usually we train with 3_000 IL episodes and 10_000 RL episodes
n_episodes = 50
kinematics = 'unicycle'
reward_params = {
    'goal_reward': 1.,
    'collision_penalty': -0.25,
    'discomfort_distance': 0.2,
    'time_limit': 50.,
    'kinematics': kinematics,
}
reward_function = Reward1(**reward_params)
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing',
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy

## Load Social-Navigation-PyEnvs policy vnet params
# vnet_params = load_crowdnav_policy(
#     "sarl", 
#     os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_hybrid_scenario/rl_model.pth"))
# policy = SARL(env.reward_function, dt=env_params['robot_dt'])
# vnet_params = load_crowdnav_policy(
#     "cadrl", 
#     os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/crowdnav_policies/cadrl_1_sfm_hybrid_scenario/rl_model.pth"))
# policy = CADRL(env.reward_function, dt=env_params['robot_dt'])

## Load social-jym policy
# vnet_params = load_socialjym_policy(
#     os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/cadrl_k1_nh1_hp1_s4_r1_20_11_2024.pkl"))
# policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)

vnet_params = load_socialjym_policy(
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/sarl_hsfm_unicycle_reward_1_circular_crossing_06_12_2024.pkl"))
policy = SARL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)

### Test Social-Navigation-PyEnvs policy
# metrics = test_k_trials(100, random_seed, env, policy, vnet_params, reward_params["time_limit"])

### Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    outcome = {"success": 0, "failure": 0, "timeout": 0, "nothing": 1}
    episode_start_time = time.time()
    state, reset_key, obs, info = env.reset(reset_key)
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
        state, obs, info, reward, outcome = env.step(state,info,action,test=True) 
        all_states = np.vstack((all_states, [state]))

    ## Plot episode trajectory
    # figure, ax = plt.subplots(figsize=(10,10))
    # ax.axis('equal')
    # plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    # for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
    #     plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    # # plot last state
    # plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    # plt.show()

    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'],
        kinematics=env_params['kinematics'])
    