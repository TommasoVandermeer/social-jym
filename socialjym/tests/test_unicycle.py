from jax import random, vmap, debug, lax, jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
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
    'n_humans': 25,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing',
    'humans_policy': 'sfm',
    'reward_function': reward_function,
    'kinematics': kinematics
}


### Initialize and reset environment
env = SocialNav(**env_params)

### Initialize robot policy
policy = CADRL(env.reward_function, dt=env_params['robot_dt'], kinematics=kinematics)

### Plot action space
# Plot (v,w) action space
plt.scatter(policy.action_space[:,0], policy.action_space[:,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"Action space (V,w) - Wheelbase: {policy.wheels_distance/2} - Vmax: {policy.v_max}")
plt.xlabel("V (m/s)")
plt.ylabel("w (r/s)")
plt.show()
# Plot (px,py,thetha) for each action starting from (0,0,0) integrating at robot dt
pxy_action_space = policy.action_space[:,0:2] * jnp.array([jnp.cos(0), jnp.sin(0)])
theta_action_space = policy.action_space[:,1] * env_params['robot_dt']
orientations = jnp.ones((len(policy.action_space), 2)) * 0.15 * jnp.array([jnp.cos(theta_action_space), jnp.sin(theta_action_space)]).T
plt.scatter(pxy_action_space[:,0], pxy_action_space[:,1])
for i, orientation in enumerate(orientations):
    plt.arrow(pxy_action_space[i,0], pxy_action_space[i,1], orientation[0], orientation[1], color='black', head_width=0.02, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("All robot positions and orientations applying action space (V,w) starting from P=(0,0), theta=0 and integrating at robot dt")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# Plot (px,py,thetha) for each action starting from (0,0,0) integrating at humans dt
@jit
def integrate_action_space(i:int, val:jnp.ndarray) -> jnp.ndarray:
    x, action_space = val
    x = x.at[:,0:2].set(x[:,0:2] + (action_space[:,0] * jnp.array([jnp.cos(x[:,2]), jnp.sin(x[:,2])]) * env_params['humans_dt']).T)
    x = x.at[:,2].set(x[:,2] + action_space[:,1] * env_params['humans_dt'])
    return x, action_space
pxy_theta, action_space = lax.fori_loop(
    0, 
    int(env_params['robot_dt']/env_params['humans_dt']), 
    integrate_action_space, 
    (jnp.zeros((len(policy.action_space),3)), policy.action_space))
orientations = jnp.ones((len(policy.action_space), 2)) * 0.05 * jnp.array([jnp.cos(pxy_theta[:,2]), jnp.sin(pxy_theta[:,2])]).T
plt.scatter(pxy_theta[:,0], pxy_theta[:,1])
for i, orientation in enumerate(orientations):
    plt.arrow(pxy_theta[i,0], pxy_theta[i,1], orientation[0], orientation[1], color='black', head_width=0.005, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("All robot positions and orientations applying action space (V,w) starting from P=(0,0), theta=0 and integrating at humans dt")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
