import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax, jit, vmap, debug, random
from functools import partial
import time

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.aux_functions import plot_state

### Hyperparameters
reward_function = Reward2()
env_params = {
    'robot_radius': 0.3,
    'n_humans': 15,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'perpendicular_traffic',
    'hybrid_scenario_subset': jnp.array([0, 1, 2, 3, 4]),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': 'unicycle',
}
env = SocialNav(**env_params)
state, _, obs, info, _ = env.reset(random.PRNGKey(0))

# TODO: I want to define a function that takes in input the agents' position (and their radiuses), some box bounds, a number of obstacles, a number 
# of segments per obstacle and the minimum passage available for the robot to pass through, and returns a list of the specified number of obstacles
# that are randomly generated within the box bounds, avoiding collisions with humans and ensuring the minimum passage.
def generate_random_obstacles(agents_pos, agents_radiuses, box_bounds, n_obstacles, n_segments, min_passage):
    x_min, x_max, y_min, y_max = box_bounds
    obstacles = []
    pass

obstacles = generate_random_obstacles(
    agents_pos=state[:,:2],
    agents_radiuses=jnp.concatenate((info['humans_parameters'][:,0],jnp.array([env_params['robot_radius']])), axis=0),
    box_bounds=(-10, 10, -10, 10),
    n_obstacles=5,
    n_segments=3,
    min_passage=1.0,
)

# Plot the generated obstacles
fig, ax = plt.subplots()
for obs in obstacles:
    ax.plot(obs[:, 0], obs[:, 1], marker='o')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title('Generated Obstacles')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.grid()
plt.show()