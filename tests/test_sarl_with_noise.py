from jax import random, vmap, debug, lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.sarl import SARL
from socialjym.policies.cadrl import CADRL
from socialjym.utils.aux_functions import animate_trajectory, load_socialjym_policy, plot_state, plot_trajectory

### Hyperparameters
noise_sigma_percentage = 0.025
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
    'humans_policy': 'sfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy
policy = SARL(
    env.reward_function, 
    dt = env_params['robot_dt'], 
    kinematics = kinematics, 
    noise = True, 
    noise_sigma_percentage = noise_sigma_percentage)
# vnet_params = policy.model.init(random.key(random_seed), jnp.zeros((env_params["n_humans"],policy.vnet_input_size)))
vnet_params = load_socialjym_policy(
    os.path.join(os.path.expanduser("~"),"Repos/social-jym/trained_policies/socialjym_policies/sarl_after_RL_hsfm_unicycle_reward_0_hybrid_scenario_10_01_2025.pkl"))

# ### Preliminary noise test on initial conditions
samples = 1000
policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed)
state, reset_key, obs, info, outcome = env.reset(reset_key)
noisy_obs = vmap(policy._batch_add_noise_to_human_obs, in_axes=(None,0))(obs, random.split(policy_key, samples))
# Plot initial positions without noise
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('equal')
ax.set_title('Initial Positions with and without Noise')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
# Plot robot position
ax.plot(obs[-1, 0], obs[-1, 1], 'r^', label='Robot position')
# Plot humans with noise
for i in range(env_params['n_humans']):
    ax.plot(noisy_obs[:, i, 0], noisy_obs[:, i, 1], 'bo', alpha=0.5, label='Human with noise' if i == 0 else "")
# Plot humans without noise
for i in range(env_params['n_humans']):
    ax.plot(obs[i, 0], obs[i, 1], 'ro', label='Human without noise' if i == 0 else "")
ax.legend()
plt.show()

### Secondary noise test on another random configuration (with humans' velocities different from 0)
steps = 28
samples = 100
for i in range(steps):
    action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
    state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True) 
noisy_obs = vmap(policy._batch_add_noise_to_human_obs, in_axes=(None,0))(obs, random.split(policy_key, samples))
## Plot positions with and without noise
# Plot positions without noise
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('equal')
ax.set_title("Positions with and without Noise")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
# Plot robot position
ax.plot(obs[-1, 0], obs[-1, 1], 'r^', label='Robot position')
# Plot humans with noise
for i in range(env_params['n_humans']):
    ax.plot(noisy_obs[:, i, 0], noisy_obs[:, i, 1], 'bo', alpha=0.5, label='Human with noise' if i == 0 else "")
# Plot humans without noise
for i in range(env_params['n_humans']):
    ax.plot(obs[i, 0], obs[i, 1], 'ro', label='Human without noise' if i == 0 else "")
ax.legend()
plt.show()
## Plot humans' velocities with and without noise
# Plot humans' velocities without noise
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('equal')
ax.set_title("Velocities with and without Noise")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
# Plot robot position
ax.plot(obs[-1, 0], obs[-1, 1], 'r^', label='Robot without noise')
# Plot humans without noise
for i in range(env_params['n_humans']):
    ax.plot(obs[i, 0], obs[i, 1], 'ro', label='Humans position' if i == 0 else "")
# Plot humans' velocities with noise
for i in range(env_params['n_humans']):
    for j in range(samples):
        ax.arrow(obs[i, 0], obs[i, 1], noisy_obs[j, i, 2], noisy_obs[j, i, 3], color='b', alpha=0.5, head_width=0.005)
# Plot humans' velocities without noise
for i in range(env_params['n_humans']):
    ax.arrow(obs[i, 0], obs[i, 1], obs[i, 2], obs[i, 3], color='r', label='Velocity without noise' if i == 0 else "", head_width=0.005)
ax.legend()
plt.show()

### Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, vnet_params, 0.)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True) 
        all_states = np.vstack((all_states, [state]))
    ## Plot episode trajectory
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/env_params['robot_dt'])):
        plot_state(ax, k*env_params['robot_dt'], all_states[k], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    # plot last state
    plot_state(ax, (len(all_states)-1)*env_params['robot_dt'], all_states[len(all_states)-1], env_params['humans_policy'], env_params['scenario'], info["humans_parameters"][:,0], env.robot_radius, kinematics=env_params['kinematics'])
    plt.show()
    # Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'],
        kinematics=env_params['kinematics'])
    