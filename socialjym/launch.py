from jax import random, debug, device_get
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from envs.socialnav import SocialNav
from socialjym.utils.aux_functions import plot_state, plot_trajectory

# Hyperparameters
n_humans = 5
steps = 1500
robot_dt = 0.01
humans_dt = 0.01
random_seed = 1

# Initialize and reset environment
env = SocialNav(robot_radius=0.3, robot_dt=robot_dt, humans_dt=humans_dt, scenario='circular_crossing', n_humans=n_humans)
env_state, obs, info = env.reset(random.key(random_seed))

# Simulate some steps
all_states = np.empty((steps+1, n_humans+1, 6), np.float32)
all_states[0] = env_state[0]
for i in range(steps): 
    env_state, _, info, _, _ = env.step(env_state,info,jnp.zeros((2,)))
    all_states[i+1] = env_state[0]
all_states = device_get(all_states) # Transfer data from GPU to CPU for plotting

# Plot simulation
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
plot_trajectory(ax, all_states, info['humans_goal'], env.robot_goal)
for k in range(0,steps+1,int(3/robot_dt)):
    plot_state(ax, k*robot_dt, all_states[k], 'hsfm', 'circular_crossing', 0.3 * np.ones((n_humans,)), env.robot_radius)
plt.show()