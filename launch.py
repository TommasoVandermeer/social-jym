from jax import random, debug, device_get
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import plot_state, plot_trajectory

# Hyperparameters
n_humans = 5
n_episodes = 5
robot_dt = 0.25
humans_dt = 0.01
random_seed = 1

# Initialize and reset environment
env = SocialNav(robot_radius=0.3, robot_dt=robot_dt, humans_dt=humans_dt, scenario='circular_crossing', n_humans=n_humans, robot_visible=False)
env_state, obs, info = env.reset(random.key(random_seed))

# Simulate some episodes
episode_simulation_times = np.empty((n_episodes,))
for i in range(n_episodes):
    all_states = np.array([env_state[0]])
    done = False
    episode_start_time = time.time()
    while not done:
        action = jnp.array([0.,1.])
        env_state, obs, info, reward, done = env.step(env_state,info,action)
        if not done: all_states = np.vstack((all_states, [env_state[0]]))
        key = env_state[1]
    all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting
    episode_simulation_times[i] = round(time.time() - episode_start_time,2)
    ## Plot episode trajectory
    print(f"Episode {i} ended - Simulation time {episode_simulation_times[i]} seconds - Plotting trajectory...")
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/robot_dt)):
        plot_state(ax, k*robot_dt, all_states[k], 'hsfm', 'circular_crossing', 0.3 * np.ones((n_humans,)), env.robot_radius)
    plt.show()
print(f"Average time per episode: {round(np.mean(episode_simulation_times),2)} seconds")