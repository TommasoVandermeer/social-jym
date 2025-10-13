import jax.numpy as jnp
from jax import random, jit, vmap
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from jax.tree_util import tree_map
import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.utils.cell_decompositions.quadtree import decompose, query_map
from socialjym.utils.global_planners.a_star import AStarPlanner

### Hyperparameters
n_trials = 100
random_seed = 0
n_humans = 30
min_cell_size = 0.9
robot_vmax = 1.0
robot_wheels_distance = 0.7
reward_function = Reward2(
    target_reached_reward = True,
    collision_penalty_reward = True,
    discomfort_penalty_reward = True,
    v_max = robot_vmax,
    progress_to_goal_reward = True,
    progress_to_goal_weight = 0.03,
    high_rotation_penalty_reward=True,
    angular_speed_bound=1.,
    angular_speed_penalty_weight=0.0075,
)

### Start points and corresponding final goals for the robot
start_positions = jnp.array([
    [13., -8.5],
    [-12.5, -2.5],
    [7.5, -7.],
    [-13.5, 1.],
    [-7.5, 7.5],
])
goal_positions = jnp.array([
    [-13.5, 8.5],
    [14., 9.],
    [-5., 7.5],
    [10., 0.],
    [7., -8.],
])
assert len(start_positions) == len(goal_positions), "Number of start positions must be equal to number of goal positions"
n_paths = len(start_positions)

### Obstacles
obstacles = jnp.array([
    [[[-15.,10.], [-15.,-10.]]],
    [[[-15.,10.], [15.,10.]]],
    [[[15.,10.], [15.,-10.]]],
    [[[15.,-10.], [-15.,-10.]]],
    [[[-15.,7.], [-11.,7.]]],
    [[[-11.,7.], [-11.,4.]]],
    [[[-11.,4.], [-13.,4.]]],
    [[[-15.,2.], [-8.,2.]]],
    [[[-9.,10.], [-9.,4.]]],
    [[[-8.,0.], [-8.,-8.]]],
    [[[-6.,10.], [-6.,7.]]],
    [[[-2.,10.], [-2.,7.]]],
    [[[2.,10.], [2.,6.]]],
    [[[5.,10.], [5.,6.]]],
    [[[7.,10.], [7.,4.]]],
    [[[11.,10.], [11.,4.]]],
    [[[13.,7.], [15.,7.]]],
    [[[-15.,0.], [-8.,0.]]],
    [[[-6.,4.], [-6.,-4.]]],
    [[[-6.,4.], [5.,4.]]],
    [[[5.,4.], [5.,-4.]]],
    [[[5.,-4.], [3.,-4.]]],
    [[[-6.,-7.], [-7.,-7.]]],
    [[[7.,-5.], [7.,-4.]]],
    [[[9.,-6.], [9.,-10.]]],
    [[[11.,-6.], [11.,-8.]]],
    [[[13.,-6.], [15.,-6.]]],
    [[[9.,-4.], [13.,-4.]]],
    [[[7.,-2.], [8.,-2.]]],
    [[[7.,-2.], [7.,2.]]],
    [[[7.,2.], [13.,2.]]],
    [[[13.,2.], [13.,-2.]]],
    [[[11.,-2.], [13.,-2.]]],
])

### Generate robot waypoints
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'path_to_goal.pkl')):
    # Environment parameters
    env_params = {
        'robot_radius': 0.3,
        'n_humans': n_humans,
        'n_obstacles': len(obstacles),
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': True,
        'scenario': None, # Custom scenario
        'humans_policy': 'hsfm',
        'reward_function': reward_function,
        'kinematics': 'unicycle',
        'ccso_n_static_humans': 0,
    }
    # Initialize environment
    env = SocialNav(**env_params)
    # Get initial state and info
    state, _, _, info, _ = env.reset(random.PRNGKey(random_seed))
    # Decompose the environment
    free_cells, occupied_cells, edges = decompose(
        min_cell_size,
        jnp.array([30., 20.]),
        jnp.array([0., 0.]),
        obstacles
    )
    # Plot decomposition
    fig, ax = plt.subplots(figsize=(8,8))
    if obstacles.shape[1] > 1: # Polygon obstacles
        for o in obstacles: plt.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in obstacles: plt.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    plt.scatter(free_cells[:,0], free_cells[:,1], color='green', s=10, label='Free cells', zorder=4, alpha=0.7)
    for cell in free_cells:
        cell_size = cell[2:]
        cell_center = cell[:2]
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='green', linewidth=0.5, alpha=0.5, zorder=1)
        ax.add_patch(rect)
    for cell in occupied_cells:
        cell_size = cell[2:]
        cell_center = cell[:2]
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='red', edgecolor='red', linewidth=0.5, alpha=0.5, zorder=2)
        ax.add_patch(rect)
    for x, from_cell in enumerate(free_cells):
        for y, to_cell in enumerate(free_cells):
            if edges[x, y] > 0:
                plt.plot([from_cell[0], to_cell[0]], [from_cell[1], to_cell[1]], color='blue', linewidth=0.5, zorder=5)
    ax.set
    plt.show()
    # Initialize A* planner
    # TODO: Redesign planners to take in input the 1D vector of cells instead of a 2D grid and the edges matrix
    planner = AStarPlanner(jnp.array([30., 20.]))
    ## Main loop
    for path in range(n_paths):
        print(f"Generating waypoints for path {path+1}/{n_paths}...")
        # Set robot start position and goal
        info['robot_goal'] = goal_positions[path]
        state = state.at[-1, :2].set(start_positions[path])
        # Set obstacles
        info['static_obstacles'] = jnp.repeat(jnp.array([obstacles]), env.n_humans+1, axis=0)
        # Query map for start and goal positions
        start_cell = query_map(free_cells, start_positions[path])
        goal_cell = query_map(free_cells, goal_positions[path])
        if start_cell is None or goal_cell is None:
            print(f"Start or goal position is not in a free cell for path {path+1}/{n_paths}.")
