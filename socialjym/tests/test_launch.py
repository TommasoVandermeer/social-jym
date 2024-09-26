from jax import random, debug, device_get
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.reward1 import generate_reward_done_function
from socialjym.policies.cadrl import CADRL
from socialjym.policies.sarl import SARL
from socialjym.utils.aux_functions import plot_state, plot_trajectory, animate_trajectory

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
    'n_humans': 15,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'robot_crowding',
    'humans_policy': 'hsfm',
    'reward_function': reward_function
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARL(env.reward_function, dt=env_params['robot_dt'])
initial_vnet_params = policy.model.init(random.key(random_seed), jnp.zeros((env_params["n_humans"],policy.vnet_input_size)))

# Warm up the environment and policy - Dummy step and act to jit compile the functions 
# (this way, computation time will only reflect execution and not compilation)
state, _, _, info = env.reset(random.key(0))
_, obs, _, _, _ = env.step(state,info,jnp.zeros((2,)))
_ = env.imitation_learning_step(state,info)
_ = env.get_lidar_measurements(obs[-1,:2], jnp.atan2(*jnp.flip(obs[-1,2:4])), obs[:-1,:2], info["humans_parameters"][:,0])
_ = policy.act(random.key(0), obs, info, initial_vnet_params, 0.1)

# Initialize random keys
reset_key = random.key(random_seed)
policy_key = random.key(random_seed)

# Simulate some episodes
episode_simulation_times = np.empty((n_episodes,))
for i in range(n_episodes):
    outcome = {"nothing": True, "success": False, "failure": False, "timeout": False}
    episode_start_time = time.time()
    state, reset_key, obs, info = env.reset(reset_key)
    lidar_measurements = env.get_lidar_measurements(obs[-1,:2], jnp.atan2(*jnp.flip(obs[-1,2:4])), obs[:-1,:2], info["humans_parameters"][:,0])

    info["humans_parameters"] = info["humans_parameters"].at[:,18].set(jnp.ones((env.n_humans,)) * 0.1) # Set humans' safety space to 0.1

    all_states = np.array([state])
    all_lidar_measurements = np.array([lidar_measurements])
    while outcome["nothing"]:

        # action, policy_key, _ = policy.act(policy_key, obs, info, initial_vnet_params, 0.)
        # state, obs, info, reward, outcome = env.step(state,info,action,test=True)

        state, obs, info, reward, outcome = env.imitation_learning_step(state,info)

        lidar_measurements = env.get_lidar_measurements(obs[-1,:2], jnp.atan2(*jnp.flip(obs[-1,2:4])), obs[:-1,:2], info["humans_parameters"][:,0])
        all_lidar_measurements = np.vstack((all_lidar_measurements, [lidar_measurements]))
        all_states = np.vstack((all_states, [state]))
    episode_simulation_times[i] = round(time.time() - episode_start_time,2)
    all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting
    all_lidar_measurements = device_get(all_lidar_measurements) # Transfer data from GPU to CPU for plotting
    print(f"Episode {i} ended - Execution time {episode_simulation_times[i]} seconds - Plotting trajectory...")
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
        robot_dt=env_params['robot_dt'],
        lidar_measurements=all_lidar_measurements)
# Print simulation times
print(f"Average time per episode: {round(np.mean(episode_simulation_times),2)} seconds")
print(f"Total time for {n_episodes} episodes: {round(np.sum(episode_simulation_times),2)} seconds")