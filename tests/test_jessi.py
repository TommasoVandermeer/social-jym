from jax import random, vmap
import jax.numpy as jnp
import numpy as np
import os
import pickle

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
from socialjym.policies.jessi import JESSI
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 1
n_episodes = 50
kinematics = 'unicycle'
env_params = {
    'n_stack': 5,
    'lidar_num_rays': 100,
    'lidar_angular_range': 2*jnp.pi,
    'lidar_max_dist': 10.,
    'n_humans': 7,
    'n_obstacles': 5,
    'robot_radius': 0.3,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'hybrid_scenario',
    'reward_function': DummyReward(robot_radius=0.3),
    'kinematics': kinematics,
}

# Initialize the environment
env = LaserNav(**env_params)

# Initialize the policy
policy = JESSI()
with open(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl'), 'rb') as f:
    encoder_params = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'rb') as f:
    actor_params = pickle.load(f)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    all_observations = np.array([obs])
    while outcome["nothing"]:
        # Compute action from trained JESSI
        action, _, _, _, encoder_distrs, actor_distr = policy.act(random.PRNGKey(0), obs, info, encoder_params, actor_params, sample=False)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
        all_states = np.vstack((all_states, [state]))
        all_observations = np.vstack((all_observations, [obs]))
    print(outcome)
    ## Animate trajectory
    angles = vmap(lambda robot_yaw: jnp.linspace(robot_yaw - env.lidar_angular_range/2, robot_yaw + env.lidar_angular_range/2, env.lidar_num_rays))(all_states[:,-1,4])
    lidar_measurements = vmap(lambda mes, ang: jnp.stack((mes, ang), axis=-1))(all_observations[:,0,6:], angles)
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        'hsfm',
        info['robot_goal'],
        info['current_scenario'],
        static_obstacles=info['static_obstacles'][-1],
        robot_dt=env_params['robot_dt'],
        lidar_measurements=lidar_measurements,
        kinematics=kinematics,
    )