from jax import random, debug, vmap, lax, jit
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

from socialjym.envs.socialnav import SocialNav
from socialjym.envs.base_env import SCENARIOS
from socialjym.utils.rewards.socialnav_rewards.reward1 import Reward1
from socialjym.utils.global_planners.dijkstra import DijkstraPlanner
from socialjym.utils.global_planners.a_star import AStarPlanner
from socialjym.utils.aux_functions import animate_trajectory

# Hyperparameters
random_seed = 1
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
    'n_humans': 15,
    'n_obstacles': 5,
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': False,
    'scenario': 'perpendicular_traffic',
    'hybrid_scenario_subset': jnp.array([0,1,2,3,4,5,6,7], dtype=jnp.int32),
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': kinematics,
}

# Initialize and reset environment
env = SocialNav(**env_params)

# Warm up the environment and policy - Dummy step and act to jit compile the functions 
# (this way, computation time will only reflect execution and not compilation)
state, _, _, info, _ = env.reset(random.key(0))
_, obs, _, _, _, _ = env.step(state,info,jnp.zeros((2,)))
_ = env.imitation_learning_step(state,info)

# Simulate some episodes
for i in range(n_episodes):
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + random_seed + i) # We don't care if we generate two identical keys, they operate differently
    state, reset_key, obs, info, outcome = env.reset(reset_key)

    ## Grid map building and visualization
    print("Scenario: ", SCENARIOS[info['current_scenario']])
    grid_cells, occupancy_grid = env.build_grid_map_and_occupancy(state, info)
    # Initialize global planner
    global_planner = AStarPlanner(jnp.array([grid_cells.shape[0], grid_cells.shape[1]])) # DijkstraPlanner(jnp.array([grid_cells.shape[0], grid_cells.shape[1]]))
    # Query starting position of the robot, its goal and a point outside the grid
    outside_point = jnp.array([grid_cells[-1,-1,0] + 1., grid_cells[-1,-1,1] + 1.])
    robot_cell_idx, robot_cell_center, robot_cell_occupancy = global_planner._query_grid_map(grid_cells, occupancy_grid, state[-1,:2])
    goal_cell_idx, goal_cell_center, goal_cell_occupancy = global_planner._query_grid_map(grid_cells, occupancy_grid, info['robot_goal'])
    assert robot_cell_occupancy == 0, "Robot is initialized in an occupied cell!"
    assert goal_cell_occupancy == 0, "Robot goal is in an occupied cell!"
    outside_cell_idx, outside_cell_center, outside_cell_occupancy = global_planner._query_grid_map(grid_cells, occupancy_grid, outside_point)
    print(f"Robot position: {state[-1,:2]}, in cell {robot_cell_idx} centered at {robot_cell_center}, occupied: {robot_cell_occupancy}")
    print(f"Robot goal: {info['robot_goal']}, in cell {goal_cell_idx} centered at {goal_cell_center}, occupied: {goal_cell_occupancy}")
    # print(f"Outside point: {outside_point}, in cell {outside_cell_idx} centered at {outside_cell_center}, occupied: {outside_cell_occupancy}")
    # Compute shortest path
    path, path_length = global_planner.find_path(state[-1,:2], info['robot_goal'], grid_cells, occupancy_grid)
    print(f"Computed path length: {path_length}\n")
    # Plot occupancy grid, obstacles and shortest path
    for i, row in enumerate(grid_cells):
        for j, coord in enumerate(row):
            cell_size = 0.9
            facecolor = 'red' if occupancy_grid[i,j] else 'none'
            facecolor = 'grey' if jnp.array_equal(jnp.array([i,j]), robot_cell_idx) or jnp.array_equal(jnp.array([i,j]), goal_cell_idx) else facecolor
            rect = plt.Rectangle((coord[0]-cell_size/2, coord[1]-cell_size/2), cell_size, cell_size, facecolor=facecolor, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
            plt.gca().add_patch(rect)
    if info['static_obstacles'][-1].shape[1] > 1: # Polygon obstacles
        for o in info['static_obstacles'][-1]: plt.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in info['static_obstacles'][-1]: plt.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    plt.plot(state[-1,0], state[-1,1], marker='o', color='b', markersize=10, label='Robot start')
    plt.plot(info['robot_goal'][0], info['robot_goal'][1], marker='*', color='g', markersize=15, label='Robot goal')
    plt.plot(path[:, 0], path[:, 1], color='blue', linewidth=2, label='Computed path', zorder=3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    ## Simulation
    info["humans_parameters"] = info["humans_parameters"].at[:,18].set(jnp.ones((env.n_humans,)) * 0.1) # Set humans' safety space to 0.1
    all_states = jnp.array([state])
    while outcome["nothing"]:
        state, obs, info, reward, outcome = env.imitation_learning_step(state,info)
        all_states = jnp.vstack((all_states, jnp.array([state])))
    # # Animate trajectory
    # animate_trajectory(
    #     all_states, 
    #     info['humans_parameters'][:,0], 
    #     env.robot_radius, 
    #     env_params['humans_policy'],
    #     info['robot_goal'],
    #     info['current_scenario'],
    #     robot_dt=env_params['robot_dt'],
    #     kinematics=kinematics,
    #     static_obstacles=info['static_obstacles'][-1],
    # )