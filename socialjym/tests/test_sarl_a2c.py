from jax import random, vmap, debug
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.policies.sarl_a2c import SARLA2C
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory

### Hyperparameters
random_seed = 0
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
    'scenario': 'hybrid_scenario',
    'humans_policy': 'sfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize policy
policy = SARLA2C(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)
critic_params = policy.critic.init(random_seed, jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))
actor_params = policy.actor.init(random_seed, jnp.zeros((env_params['n_humans'], policy.vnet_input_size)))

### Watch n_samples action sampled from actor output at the initial state
n_samples = 10_000
state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
keys = random.split(random.PRNGKey(random_seed), n_samples)
actions, _, _ = vmap(policy.act, in_axes=(0, None, None, None, None))(keys, obs, info, actor_params, True)
figure, ax = plt.subplots(figsize=(10,10))
figure.suptitle(f'{n_samples} actor outputs at the initial state')
ax.axis('equal')
ax.plot(actions[:,0], actions[:,1], 'o')
if kinematics == 'unicycle':
    ax.set_xlabel('v ($m/s$)')
    ax.set_ylabel('$\omega$ $(rad/s)$')
else:
    ax.set_xlabel('vx ($m/s$)')
    ax.set_ylabel('vy $(m/s)$')
mean_action, _, _ = policy.act(random.PRNGKey(random_seed), obs, info, actor_params, False)
ax.plot(mean_action[0], mean_action[1], 'ro')
plt.show()

### Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i)
    episode_start_time = time.time()
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    while outcome["nothing"]:
        action, policy_key, _ = policy.act(policy_key, obs, info, actor_params, True)
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
    