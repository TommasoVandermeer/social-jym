from jax import random, vmap
import jax.numpy as jnp
import numpy as np

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
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

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    all_states = np.array([state])
    all_observations = np.array([obs])
    while outcome["nothing"]:
        state, obs, info, reward, outcome, (_, env_key) = env.step(state,info,jnp.array([1.,0.]),test=True,env_key=env_key)
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