import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from functools import partial
from jax import random, lax, jit, vmap
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.aux_functions import animate_trajectory, interpolate_humans_boundaries, interpolate_obstacle_segments

### Hyperparameters
policy = 'dir-safe' # 'dir-safe' or 'dwa'
trial = 15 # 23
n_humans = 9
n_obstacles = 5
scenario = 'circular_crossing_with_static_obstacles'
reward_function = Reward2(
    target_reached_reward = True,
    collision_penalty_reward = True,
    discomfort_penalty_reward = True,
    v_max = 1.,
    progress_to_goal_reward = True,
    progress_to_goal_weight = 0.03,
    high_rotation_penalty_reward=True,
    angular_speed_bound=1.,
    angular_speed_penalty_weight=0.0075,
)

### Initialize environment
test_env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans,
    'n_obstacles': n_obstacles,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': scenario,
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 5,
}
test_env = SocialNav(**test_env_params)

### Run custom episodes
if policy == 'dir-safe':
    ## Initialize robot policy
    actor = DIRSAFE(reward_function, v_max=1., dt=0.25)
    with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
        actor_params = pickle.load(f)['actor_params']
    ## Reset the environment
    state, _, obs, info, outcome = test_env.reset(random.PRNGKey(trial))
    ## Compute obstacles if in "circular_crossing_with_static_obstacles" scenario (squares circumscribing humans disks)
    if scenario == 'circular_crossing_with_static_obstacles':
        static_humans_positions = state[0:test_env_params['ccso_n_static_humans'],0:2]
        static_humans_radii = info['humans_parameters'][0:test_env_params['ccso_n_static_humans'],0]
        static_obstacles = jnp.array([actor.batch_compute_disk_circumscribing_n_agon(static_humans_positions, static_humans_radii, 10)])
        nan_obstacles = jnp.full((test_env_params['n_humans'],) + static_obstacles.shape[1:], jnp.nan)
        static_obstacles = jnp.vstack((nan_obstacles, static_obstacles))
    ## Run the episode
    all_states = jnp.array([state])
    all_robot_goals = jnp.array([info['robot_goal']])
    while outcome['nothing']:
        # Overwrite obstacles if in "circular_crossing_with_static_obstacles" scenario
        aux_info = info.copy()
        aux_obs = obs.copy()
        if scenario == 'circular_crossing_with_static_obstacles':
            aux_info['static_obstacles'] = static_obstacles # Set obstacles as squares circumscribing static humans
            aux_obs = obs[test_env_params['ccso_n_static_humans']:, :] # Remove static humans from observations (so they are not considered as humans by the policy, but only as obstacles)
        # Step the environment
        action, _, _, _, _ = actor.act(random.PRNGKey(0), aux_obs, aux_info, actor_params, sample=False)
        state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
        # Save the state
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
elif policy == 'dwa':
    try:
        import dwa
    except ImportError:
        raise ImportError("DWA package is not installed. Please install it to use this function.\nYou can install it with 'pip3 install dynamic-window-approach'.\n Checkout https://github.com/goktug97/DynamicWindowApproach")
    ## Initialize DWA config
    dwa_config = dwa.Config(
        max_speed=1,
        min_speed=0.0,
        max_yawrate=2/0.7,
        dt = test_env.robot_dt,
        max_accel=4,
        max_dyawrate=4,
        predict_time = .5,
        velocity_resolution = 0.1, # Discretization of the velocity space
        yawrate_resolution = np.radians(1.0), # Discretization of the yawrate space
        heading = 0.2,
        clearance = 0.2,
        velocity = 0.2,
        base=[-test_env.robot_radius, -test_env.robot_radius, test_env.robot_radius, test_env.robot_radius],  # [x_min, y_min, x_max, y_max] in meters
    )
    ## Reset the environment
    state, _, obs, info, outcome = test_env.reset(random.PRNGKey(trial))
    ## Construct point cloud
    obstacles_point_cloud = interpolate_obstacle_segments(info["static_obstacles"][-1])
    ## Run the episode
    all_states = jnp.array([state])
    all_robot_goals = jnp.array([info['robot_goal']])
    while outcome['nothing']:
        # Construct point cloud
        humans_point_cloud = interpolate_humans_boundaries(obs[:-1,:2], info['humans_parameters'][:,0])
        point_cloud = jnp.concatenate((obstacles_point_cloud, humans_point_cloud), axis=0)
        # Step the environment
        action = jnp.array(dwa.planning(tuple(map(float, np.append(obs[-1,:2],obs[-1,5]))), tuple(map(float, obs[-1,2:4])), tuple(map(float, info['robot_goal'])), np.array(point_cloud, dtype=np.float32), dwa_config))
        state, obs, info, _, outcome, _ = test_env.step(state,info,action,test=True)
        # Save the state
        all_states = jnp.vstack((all_states, jnp.array([state])))
        all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
else:
    raise ValueError(f'Policy {policy} not available.')

### Animate trajectory
animate_trajectory(
    all_states, 
    info['humans_parameters'][:,0], 
    test_env.robot_radius, 
    test_env_params['humans_policy'],
    all_robot_goals,
    SCENARIOS.index(scenario),
    robot_dt=test_env_params['robot_dt'],
    static_obstacles=static_obstacles[-1] if ((scenario == 'circular_crossing_with_static_obstacles') and (policy == 'dir-safe')) else info['static_obstacles'][-1],
    kinematics='unicycle',
    vmax=1.,
    wheels_distance=0.7,
    figsize= (11, 6.6),
)