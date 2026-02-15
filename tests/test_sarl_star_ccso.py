from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.sarl_star import SARLStar
from socialjym.utils.aux_functions import animate_trajectory, load_socialjym_policy

# Hyperparameters
random_seed = 50
n_episodes = 50
kinematics = 'unicycle'
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
env_params = {
    'robot_radius': 0.3,
    'n_humans': 10,
    'n_obstacles': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'circular_crossing_with_static_obstacles',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5,6,7], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
    'grid_map_computation': True, # Enable grid map computation for global planning
    'ccso_n_static_humans': 6,
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Initialize robot policy
policy = SARLStar(
    reward_function, 
    env.get_grid_size(), 
    planner="A*", 
    v_max=1.0, 
    dt=0.25, 
    kinematics='unicycle', 
    wheels_distance=0.7
)
policy_params = load_socialjym_policy(os.path.join(os.path.dirname(__file__), 'best_sarl.pkl'))

# Simulate some episodes
for i in range(n_episodes):
    reset_key = random.PRNGKey(random_seed + i) # We don't care if we generate two identical keys, they operate differently
    episode_start_time = time.time()
    state, reset_key, obs, info, outcome = env.reset(reset_key)
    if info['current_scenario'] == SCENARIOS.index('circular_crossing_with_static_obstacles'):
        # Compute static obstacles as n-agons circumscribing static humans
        static_humans_positions = state[0:env.ccso_n_static_humans,0:2]
        static_humans_radii = info['humans_parameters'][0:env.ccso_n_static_humans,0]
        static_obstacles = jnp.array([policy.batch_compute_disk_circumscribing_n_agon(
            static_humans_positions, 
            static_humans_radii, 
            n_edges=10
        )])
        nan_obstacles = jnp.full((env.n_humans,) + static_obstacles.shape[1:], jnp.nan)
        static_obstacles = jnp.vstack((nan_obstacles, static_obstacles))
        aux_info = info.copy()
        aux_info['static_obstacles'] = static_obstacles # Set obstacles as n-agons circumscribing static humans
        info['grid_cells'], info['occupancy_grid'] = env.build_grid_map_and_occupancy(state, aux_info)
        # Find start and end nodes
        robot_cell_idx, robot_cell_center, robot_cell_occupancy = policy.planner._query_grid_map(info['grid_cells'], info['occupancy_grid'], state[-1,:2])
        goal_cell_idx, goal_cell_center, goal_cell_occupancy = policy.planner._query_grid_map(info['grid_cells'], info['occupancy_grid'], info['robot_goal'])
        # Plot occupancy grid, obstacles and shortest path
        for i, row in enumerate(info['grid_cells']):
            for j, coord in enumerate(row):
                cell_size = 0.9
                facecolor = 'red' if info['occupancy_grid'][i,j] else 'none'
                facecolor = 'grey' if jnp.array_equal(jnp.array([i,j]), robot_cell_idx) or jnp.array_equal(jnp.array([i,j]), goal_cell_idx) else facecolor
                rect = plt.Rectangle((coord[0]-cell_size/2, coord[1]-cell_size/2), cell_size, cell_size, facecolor=facecolor, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
                plt.gca().add_patch(rect)
        for o in aux_info['static_obstacles'][-1]: plt.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
        plt.plot(state[-1,0], state[-1,1], marker='o', color='b', markersize=10, label='Robot start')
        plt.plot(info['robot_goal'][0], info['robot_goal'][1], marker='*', color='g', markersize=15, label='Robot goal')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    all_states = jnp.array([state])
    while outcome["nothing"]:
        if info['current_scenario'] == SCENARIOS.index('circular_crossing_with_static_obstacles'):
            aux_info = info.copy()
            aux_info['static_obstacles'] = static_obstacles[env.ccso_n_static_humans:] # Set obstacles as n-agons circumscribing static humans
            aux_obs = obs[env.ccso_n_static_humans:, :] # Remove static humans from observations (so they are not considered as humans by the policy, but only as obstacles)
            action, _, _, _ = policy.act(random.PRNGKey(0), aux_obs, aux_info, policy_params, 0.)
        else:
            action, _, _, _ = policy.act(random.PRNGKey(0), obs, info, policy_params, 0.)
        state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True)
        # print(f"Return in steps [0,{info['step']}):", info["return"], f" - time : {info['time']}")
        all_states = jnp.vstack((all_states, jnp.array([state])))
    ## Animate trajectory
    animate_trajectory(
        all_states, 
        info['humans_parameters'][:,0], 
        env.robot_radius, 
        env_params['humans_policy'],
        info['robot_goal'],
        info['current_scenario'],
        robot_dt=env_params['robot_dt'],
        kinematics=kinematics,
        static_obstacles=info['static_obstacles'][-1] if not info['current_scenario'] == SCENARIOS.index('circular_crossing_with_static_obstacles') else static_obstacles[-1],
    )