from jax import random, debug, device_get
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from socialjym.envs.socialnav import SocialNav
from socialjym.policies.cadrl import CADRL
from socialjym.utils.aux_functions import plot_state, plot_trajectory

# Hyperparameters
n_humans = 5
n_episodes = 5
robot_dt = 0.25
humans_dt = 0.01
random_seed = 1
robot_visible = False
scenario = 'circular_crossing'
humans_policy = 'hsfm'

# Initialize and reset environment
env = SocialNav(robot_radius=0.3, robot_dt=robot_dt, humans_dt=humans_dt, humans_policy=humans_policy, scenario=scenario, n_humans=n_humans, robot_visible=robot_visible)

# Warm up the environment - Dummy step to jit compile the step function (this way, computation time will only reflect execution and not compilation)
env_state, _, info = env.reset(random.key(random_seed))
_ = env.step(env_state,info,jnp.zeros((2,)))
key = random.key(random_seed)

# Simulate some episodes
episode_simulation_times = np.empty((n_episodes,))
for i in range(n_episodes):
    done = False
    episode_start_time = time.time()
    env_state, obs, info = env.reset(key)
    all_states = np.array([env_state[0]])
    while not done:
        action = jnp.array([0.,1.])
        env_state, obs, info, reward, done = env.step(env_state,info,action)
        all_states = np.vstack((all_states, [env_state[0]]))
        key = env_state[1]
    all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting
    episode_simulation_times[i] = round(time.time() - episode_start_time,2)
    ## Plot episode trajectory
    print(f"Episode {i} ended - Execution time {episode_simulation_times[i]} seconds - Plotting trajectory...")
    figure, ax = plt.subplots(figsize=(10,10))
    ax.axis('equal')
    plot_trajectory(ax, all_states, info['humans_goal'], info['robot_goal'])
    for k in range(0,len(all_states),int(3/robot_dt)):
        plot_state(ax, k*robot_dt, all_states[k], humans_policy, scenario, info["humans_parameters"][:,0], env.robot_radius)
    # plot last state
    plot_state(ax, len(all_states)*robot_dt, all_states[len(all_states)-1], humans_policy, scenario, info["humans_parameters"][:,0], env.robot_radius)
    plt.show()
print(f"Average time per episode: {round(np.mean(episode_simulation_times),2)} seconds")