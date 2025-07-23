import jax.numpy as jnp
import numpy as np
from jax import jit, lax
import heapq
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.soappo import SOAPPO
from socialjym.utils.aux_functions import test_k_custom_trials, test_k_custom_trials_dwa, plot_state

### Hyperparameters
random_seed = 0 
n_trials = 100
n_humans = 20
time_limit = 100.
reward_function = Reward2(
    time_limit=time_limit,
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

### Set numpy seed
np.random.seed(random_seed)

### Environment parameters
### Parameters
grid_cell_size = .95
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
start_pos = jnp.array([13., -8.5])  # Starting position of the robot
goal_pos = jnp.array([-13.5, 8.5])  # Goal position of the robot

### Generate robot waypoints
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'path_to_goal.pkl')):
    @jit
    def segment_rectangle_intersection(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
        dx = x2 - x1
        dy = y2 - y1
        p = jnp.array([-dx, dx, -dy, dy])
        q = jnp.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1])
        @jit
        def loop_body(i, tup):
            t, p, q = tup
            t0, t1 = t
            t0, t1 = lax.switch(
                (jnp.sign(p[i])+1).astype(jnp.int32),
                [
                    lambda t: lax.cond(q[i]/p[i] > t[1], lambda _: (2.,1.), lambda x: (jnp.max(jnp.array([x[0],q[i]/p[i]])), x[1]), t),  # p[i] < 0
                    lambda t: lax.cond(q[i] < 0, lambda _: (2.,1.), lambda x: x, t),  # p[i] == 0
                    lambda t: lax.cond(q[i]/p[i] < t[0], lambda _: (2.,1.), lambda x: (x[0], jnp.min(jnp.array([x[1],q[i]/p[i]]))), t),  # p[i] > 0
                ],
                (t0, t1),
            )
            # debug.print("t0: {x}, t1: {y}, switch_case: {z}", x=t0, y=t1, z=(jnp.sign(p[i])+1).astype(jnp.int32))
            return ((t0, t1), p ,q)
        t, p, q = lax.fori_loop(
            0, 
            4,
            loop_body,
            ((0., 1.), p, q),
        )
        t0, t1 = t
        inside_or_intersects = ~(t0 > t1)
        return inside_or_intersects
    def grid_cell_obstacle_occupancy(static_obstacles:jnp.ndarray, cell_size:float, distance_threshold:int, epsilon:float=1e-5):
        """
        Returns a grid cell occupancy map for the static obstacles in the simulation.

        args:
        - static_obstacles: jnp.ndarray of shape (n_obstacles, n_edges, 2, 2) - Static obstacles in the simulation.
        - cell_size: float - Resolution of the grid cells.
        - distance_threshold: int - Distance threshold (in cells) to consider a cell occupied by an obstacle.

        outputs:
        - static_obstacles_for_each_cell: jnp.ndarray of booleans of shape (n+distance_threshold,n+distance_threshold,max_static_obstacles) 
                            where n is the max number of cells necessary to cover all obstacles 
                            in the x and y direction - Grid cell occupancy map for the static obstacles.
        - new_static_obstacles: jnp.ndarray of shape (n_obstacles+1, n_edges, 2, 2) - Static obstacles in the simulation. WARNING: Last row is a dummy nan obstacle.
        - grid_cell_coords: jnp.ndarray of shape (n+distance_threshold,n+distance_threshold,2) - Coordinates of the min point of grid cells.
        """
        # Flatten all obstacle points
        obstacle_points = static_obstacles.reshape(-1, 2)
        # Find bounds
        min_xy = jnp.floor((jnp.nanmin(obstacle_points, axis=0) / cell_size) - epsilon).astype(int)
        max_xy = jnp.ceil((jnp.nanmax(obstacle_points, axis=0) / cell_size) + epsilon).astype(int)
        # Grid size (add distance_threshold padding)
        grid_shape = (max_xy - min_xy) + 2 * distance_threshold
        # Initialize grid
        grid = jnp.zeros((grid_shape[0], grid_shape[1], len(static_obstacles)), dtype=bool)
        # Mark occupied cells within distance_threshold for each obstacle
        for i, row in enumerate(grid):
            for j, _ in enumerate(row):
                # Check each obstacle
                for obs_idx, obs in enumerate(static_obstacles):
                    if jnp.isnan(obs).any():
                        continue
                    # Check if the obstacle is within the distance threshold from the cell center
                    obs_edges = obs
                    for edge in obs_edges:
                        p0, p1 = edge
                        if jnp.isnan(p0).any() or jnp.isnan(p1).any():
                            continue
                        # Check if the edge intersects with the cell
                        inside = segment_rectangle_intersection(
                            p0[0], 
                            p0[1], 
                            p1[0], 
                            p1[1],
                            ((min_xy[0] - 2*distance_threshold) + i) * cell_size - epsilon,
                            ((min_xy[0]) + (i + 1)) * cell_size + epsilon,
                            ((min_xy[1] - 2*distance_threshold) + j) * cell_size - epsilon,
                            ((min_xy[1]) + (j + 1)) * cell_size + epsilon,
                        )
                        if inside:
                            grid = grid.at[i,j,obs_idx].set(inside)
                            break

        # Compute max number of obstacles in the cells
        max_obstacles_per_cell = int(jnp.max(jnp.sum(grid, axis=2)))
        # Compute indices of obstacles for each cell
        static_obstacles_for_each_cell = jnp.full(
            (grid_shape[0], grid_shape[1], max_obstacles_per_cell),
            jnp.nan,
        )
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                cell_occupancy = grid[i, j]
                obs_indices = jnp.where(cell_occupancy)[0]
                selected_obstacles = obs_indices
                n_selected = selected_obstacles.shape[0]
                # Pad if needed
                if n_selected < max_obstacles_per_cell:
                    pad_shape = (max_obstacles_per_cell - n_selected)
                    pad = jnp.full(pad_shape, len(static_obstacles), dtype=static_obstacles.dtype)
                    selected_obstacles = jnp.concatenate([selected_obstacles, pad], axis=0)
                static_obstacles_for_each_cell = static_obstacles_for_each_cell.at[i, j].set(selected_obstacles)
        # Compute the min coordinate of each cell in the grid
        # grid shape: (nx, ny, n_obstacles)
        nx, ny = grid_shape
        x_coords = jnp.arange(nx) * cell_size + (min_xy[0] - distance_threshold) * cell_size
        y_coords = jnp.arange(ny) * cell_size + (min_xy[1] - distance_threshold) * cell_size
        # Create a meshgrid of cell min coordinates (nx, ny, 2)
        grid_cell_coords = jnp.stack(jnp.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
        # Append nan_obstacle to the static_obstacles array
        nan_obstacle = jnp.full((1, static_obstacles.shape[1], 2, 2), jnp.nan)
        new_static_obstacles = jnp.concatenate([static_obstacles, nan_obstacle], axis=0)
        return static_obstacles_for_each_cell, new_static_obstacles, grid_cell_coords
    static_obstacles_for_each_cell, new_static_obstacles, grid_coordinates = grid_cell_obstacle_occupancy(obstacles,grid_cell_size,0)
    print(f"Grid size: {grid_coordinates.shape[0]} x {grid_coordinates.shape[1]}")
    print(f"Max obstacles per cell: {static_obstacles_for_each_cell.shape[2]}")
    occupancy_grid = jnp.zeros((static_obstacles_for_each_cell.shape[0], static_obstacles_for_each_cell.shape[1]), dtype=bool)
    for idx, coord in enumerate(grid_coordinates.reshape(-1,2)):
        i = idx // static_obstacles_for_each_cell.shape[1]
        j = idx % static_obstacles_for_each_cell.shape[1]
        # Check if the cell is occupied by any obstacle
        occupancy_grid = occupancy_grid.at[i, j].set(jnp.any(static_obstacles_for_each_cell[i, j] != len(obstacles)))
    def find_a_star_path(occupancy_grid, start_pos, goal_pos, grid_coordinates, grid_cell_size, epsilon:float=1e-5):
        """
        A* pathfinding algorithm implementation.
        
        Args:
        - occupancy_grid: jnp.ndarray of shape (n, m) - Occupancy grid where True indicates occupied cells.
        - start_pos: jnp.ndarray of shape (2,) - Starting position of the robot.
        - goal_pos: jnp.ndarray of shape (2,) - Goal position of the robot.
        - grid_coordinates: jnp.ndarray of shape (n, m, 2) - Coordinates of the grid cells.
        - grid_cell_size: float - Size of each grid cell.
        
        Returns:
        - path: jnp.ndarray of shape (k, 2) - Path from start to goal position.
        """
        ### Initial data generation
        start_node = jnp.floor((start_pos - grid_coordinates[0, 0]) / grid_cell_size).astype(int)
        goal_node = jnp.floor((goal_pos - grid_coordinates[0, 0]) / grid_cell_size).astype(int)
        assert occupancy_grid[start_node[0],start_node[1]] == False, "Start position is occupied by an obstacle."
        assert occupancy_grid[goal_node[0], goal_node[1]] == False, "Goal position is occupied by an obstacle."
        nodes_data = {
            'position': grid_coordinates + grid_cell_size / 2,
            'g': jnp.full((grid_coordinates.shape[0], grid_coordinates.shape[1]), jnp.inf),  # Cost to reach each node
            'h': jnp.full((grid_coordinates.shape[0], grid_coordinates.shape[1]), jnp.inf),  # Heuristic cost to goal
            'f': jnp.full((grid_coordinates.shape[0], grid_coordinates.shape[1]), jnp.inf),  # Total cost
            'parent': jnp.full((grid_coordinates.shape[0], grid_coordinates.shape[1], 2), -1, dtype=int),  # Parent node index
        }
        ### Functions definitions
        def create_node(nodes_data, node, g, h, parent):
            i, j = node
            nodes_data = tree_map(
                lambda x, y: x.at[i, j].set(y), 
                nodes_data, 
                {'position': nodes_data['position'][i,j], 'g': g, 'h': h, 'f': g+h, 'parent': parent}
            )
            return nodes_data
        def heuristic(pos1, pos2):
            return jnp.linalg.norm(pos1 - pos2) # Euclidean distance
        def get_neighbors(node, occupancy_grid):
            i, j = node
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if (di == 0 and dj == 0):
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < occupancy_grid.shape[0] and 0 <= nj < occupancy_grid.shape[1]:
                        if not occupancy_grid[ni, nj]:
                            neighbors.append((ni, nj))
            return neighbors
        def downsize_path(path):
            if path.shape[0] < 3:
                return path
            downsized = [path[0]]
            for i in range(1, path.shape[0] - 1):
                prev = path[i - 1]
                curr = path[i]
                next_ = path[i + 1]
                v1 = curr - prev
                v2 = next_ - curr
                # Check if vectors are colinear (cross product close to zero)
                if jnp.abs(jnp.cross(v1, v2)) > epsilon:
                    downsized.append(curr)
            downsized.append(path[-1])
            return jnp.stack(downsized)
        def reconstruct_path(nodes_data, goal_node):
            path = []
            current_node = goal_node
            while not jnp.all(current_node == -1):
                path.append(nodes_data['position'][current_node[0], current_node[1]])
                current_node = nodes_data['parent'][current_node[0], current_node[1]]
            return downsize_path(jnp.array(path[::-1]))
        ### Start A* algorithm
        # Initialize the start node
        nodes_data = create_node(
            nodes_data, 
            start_node, 
            g=0.0, 
            h=heuristic(nodes_data['position'][start_node[0], start_node[1]], nodes_data['position'][goal_node[0], goal_node[1]]), 
            parent=jnp.array([-1, -1], dtype=int)  # No parent for the start node
        )
        # Initialize the open set (priority queue) and closed set
        open_set = [(nodes_data['f'][start_node[0], start_node[1]], (int(start_node[0]), int(start_node[1])))]
        closed_set = set()
        while open_set:
            # Get the node with the lowest f value
            _, current_node = heapq.heappop(open_set)
            if jnp.array_equal(current_node, goal_node):
                return reconstruct_path(nodes_data, goal_node)
            closed_set.add((current_node[0], current_node[1]))
            # Get neighbors
            for neighbor in get_neighbors(current_node, occupancy_grid):
                if tuple(neighbor) in closed_set:
                    continue
                tentative_g = nodes_data['g'][current_node[0], current_node[1]] + heuristic(
                    nodes_data['position'][current_node[0], current_node[1]], 
                    nodes_data['position'][neighbor[0], neighbor[1]]
                )
                if tentative_g < nodes_data['g'][neighbor[0], neighbor[1]]:
                    nodes_data = create_node(
                        nodes_data, 
                        neighbor, 
                        g=tentative_g, 
                        h=heuristic(nodes_data['position'][neighbor[0], neighbor[1]], nodes_data['position'][goal_node[0], goal_node[1]]), 
                        parent=current_node
                    )
                    if (nodes_data['f'][neighbor[0], neighbor[1]], neighbor) not in open_set:
                        heapq.heappush(open_set, (nodes_data['f'][neighbor[0], neighbor[1]], (int(neighbor[0]), int(neighbor[1]))))
        return jnp.array([])  # No path found
    path_to_goal = find_a_star_path(occupancy_grid, start_pos, goal_pos, grid_coordinates, grid_cell_size)
    print(f"Path to goal: {path_to_goal.shape[0]-1} waypoints")
    # Plot obstacles and cell decomposition
    fig, ax = plt.subplots(figsize=(8, 8))
    if obstacles.shape[1] > 1: # Polygon obstacles
        for o in obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    for idx, coord in enumerate(grid_coordinates.reshape(-1,2)):
        i = idx // static_obstacles_for_each_cell.shape[1]
        j = idx % static_obstacles_for_each_cell.shape[1]
        facecolor = 'red' if occupancy_grid[i,j] else 'none'
        rect = plt.Rectangle((coord[0], coord[1]), grid_cell_size, grid_cell_size, facecolor=facecolor, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
        ax.add_patch(rect)
    ax.scatter(start_pos[0], start_pos[1], color='green', label='Start Position', zorder=4, marker='o', s=100)
    ax.scatter(goal_pos[0], goal_pos[1], color='red', label='Goal Position', zorder=4, marker='*', s=100)
    if path_to_goal.shape[0] > 0:
        ax.plot(path_to_goal[:, 0], path_to_goal[:, 1], color='blue', linewidth=2, label='A* Path', zorder=3)
        ax.scatter(path_to_goal[1:-1, 0], path_to_goal[1:-1, 1], color='blue', s=10, zorder=3)
    else:
        print("No path found.")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    # Save path to goal
    with open(os.path.join(os.path.dirname(__file__), 'path_to_goal.pkl'), 'wb') as f:
        pickle.dump(path_to_goal, f)
else:
    # Load path to goal
    with open(os.path.join(os.path.dirname(__file__), 'path_to_goal.pkl'), 'rb') as f:
        path_to_goal = pickle.load(f)

### Generate humans initial positions and goals
if not os.path.exists(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans}_humans.pkl')):
    def segment_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Check if two line segments (x1, y1) to (x2, y2) and (x3, y3) to (x4, y4) intersect.
        Returns True if they intersect, False otherwise.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        A = jnp.array([x1, y1])
        B = jnp.array([x2, y2])
        C = jnp.array([x3, y3])
        D = jnp.array([x4, y4])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    humans_initial_states = jnp.zeros((n_trials, n_humans, 6))  # (px, py, bvx, bvy, theta, omega)
    humans_goals = jnp.zeros((n_trials, n_humans, 2))  # (goal_x, goal_y)
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}: generating humans initial states and goals...")
        covered_positions = [start_pos.tolist(), goal_pos.tolist()]  # Start and goal positions are covered
        for human in range(n_humans):
            repeat = True
            while repeat:
                repeat = False
                ## Human position
                human_pos = jnp.array([
                    np.random.uniform(-15+0.05, 15-0.05),
                    np.random.uniform(-10+0.05, 10-0.05),
                ])
                # Ensure the human is not too close to the robot start position, robot goal position or already covered positions
                while jnp.any(jnp.abs(human_pos - jnp.array(covered_positions)) < 0.6):
                    human_pos = jnp.array([
                        np.random.uniform(-15+0.05, 15-0.05),
                        np.random.uniform(-10+0.05, 10-0.05),
                    ])
                covered_positions.append(human_pos)
                humans_initial_states = humans_initial_states.at[trial, human, :2].set(human_pos)
                ## Human goal
                distance = np.random.uniform(1.0, 7.0)  # Random distance from the human position
                angle = np.random.uniform(0, 2 * np.pi)  # Random angle
                human_goal = jnp.clip(human_pos + jnp.array([distance * jnp.cos(angle), distance * jnp.sin(angle)]), jnp.array([-15+0.1, -10+0.1]), jnp.array([15-0.1, 10-0.1]))
                # Ensure the human goal is not too close to the robot start position, robot goal position or already covered positions and ensure it does not intersect with obstacles
                iteration = 0
                while jnp.any(jnp.abs(human_goal - jnp.array(covered_positions)) < 0.3) \
                    or jnp.any(jnp.array([segment_segment_intersection(human_pos[0], human_pos[1], human_goal[0], human_goal[1], obs[0, 0, 0], obs[0, 0, 1], obs[0, 1, 0], obs[0, 1, 1]) for obs in obstacles])):
                    distance = np.random.uniform(1.0, 7.0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    human_goal = jnp.clip(human_pos + jnp.array([distance * jnp.cos(angle), distance * jnp.sin(angle)]), jnp.array([-15+0.1, -10+0.1]), jnp.array([15-0.1, 10-0.1]))
                    iteration += 1
                    if iteration > 100:  # Prevent infinite loop
                        # print(f"Warning: Unable to find a valid goal for human {human+1} in trial {trial+1}.")
                        covered_positions.pop()  # Remove the last position
                        repeat = True
                        break
                humans_goals = humans_goals.at[trial, human].set(human_goal) 
    ## Generate custom_episodes dictionary
    full_initial_states = jnp.zeros((n_trials, n_humans+1, 6))
    full_initial_states = full_initial_states.at[:, -1, :2].set(start_pos)  # Robot initial state
    full_initial_states = full_initial_states.at[:, :-1, :].set(humans_initial_states[:, :, :])  # Humans initial states
    custom_episodes = {
        "full_state": full_initial_states,
        "humans_goal": humans_goals,
        "robot_goals": jnp.tile(path_to_goal[1:], (n_trials, 1, 1)),  # Robot goals (excluding start position)
        "static_obstacles": jnp.tile(obstacles, (n_trials, n_humans+1, 1, 1, 1, 1)),  # Static obstacles
        "humans_radius": jnp.full((n_trials, n_humans), 0.3),
        "humans_speed": jnp.full((n_trials, n_humans), 1.0),
    }
    # Save custom episodes
    with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans}_humans.pkl'), 'wb') as f:
        pickle.dump(custom_episodes, f)
else:
    # Load custom episodes
    with open(os.path.join(os.path.dirname(__file__), f'custom_episodes_{n_humans}_humans.pkl'), 'rb') as f:
        custom_episodes = pickle.load(f)

### Visualize some custom episodes
# colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.TABLEAU_COLORS.values())
# xlims = [jnp.nanmin(obstacles[:,:,:,0]), jnp.nanmax(obstacles[:,:,:,0])]
# ylims = [jnp.nanmin(obstacles[:,:,:,1]), jnp.nanmax(obstacles[:,:,:,1])]
# for i in range(5):
#     trial = np.random.randint(0, n_trials)
#     figure, ax = plt.subplots(1,1)
#     plot_state(
#         ax, 
#         0.,
#         custom_episodes['full_state'][trial, :, :],
#         'hsfm',
#         None,
#         custom_episodes['humans_radius'][trial, :],
#         0.3,
#         kinematics='unicycle',
#         xlims=xlims,
#         ylims=ylims,
#     )
#     for o in obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
#     ax.scatter(custom_episodes['humans_goal'][trial, :, 0], custom_episodes['humans_goal'][trial, :, 1], marker="*", color=colors[:n_humans], zorder=2)
#     ax.set_aspect('equal')
#     plt.show()

### Initialize robot policy
policy = SOAPPO(reward_function, v_max=1., dt=0.25)
with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']

### Initialize output data structure
empty_trials_metrics_array = jnp.zeros((n_trials))
all_metrics = {
    "successes": jnp.zeros((1,), dtype=jnp.int32), 
    "collisions": jnp.zeros((1,), dtype=jnp.int32),
    "timeouts": jnp.zeros((1,), dtype=jnp.int32), 
    "returns": empty_trials_metrics_array,
    "times_to_goal": empty_trials_metrics_array,
    "average_speed": empty_trials_metrics_array,
    "average_acceleration": empty_trials_metrics_array,
    "average_jerk": empty_trials_metrics_array,
    "average_angular_speed": empty_trials_metrics_array,
    "average_angular_acceleration": empty_trials_metrics_array,
    "average_angular_jerk": empty_trials_metrics_array,
    "min_distance": empty_trials_metrics_array,
    "space_compliance": empty_trials_metrics_array,
    "episodic_spl": empty_trials_metrics_array,
    "path_length": empty_trials_metrics_array,
    "scenario": jnp.zeros((n_trials), dtype=jnp.int32),
    "waypoint_reached": empty_trials_metrics_array,
}
all_metrics_dwa = all_metrics.copy()

### Test policies
if not os.path.exists(os.path.join(os.path.dirname(__file__),f"dir_safe_tests_long_nav_{n_humans}_humans.pkl")):
    print(f"\n## Testing DIR-SAFE ###")
    test_env_params = {
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
    test_env = SocialNav(**test_env_params)
    ## Test trained policy
    metrics = test_k_custom_trials(
        n_trials, 
        random_seed, 
        test_env, 
        policy, 
        actor_params, 
        reward_function.time_limit,
        custom_episodes=custom_episodes,
    )
    all_metrics = tree_map(lambda x, y: x.at[:].set(y), all_metrics, metrics)
    ## Save results
    with open(os.path.join(os.path.dirname(__file__),f"dir_safe_tests_long_nav_{n_humans}_humans.pkl"), 'wb') as f:
        pickle.dump(all_metrics, f)
else:
    ### Load results
    with open(os.path.join(os.path.dirname(__file__),f"dir_safe_tests_long_nav_{n_humans}_humans.pkl"), 'rb') as f:
        all_metrics = pickle.load(f)
if not os.path.exists(os.path.join(os.path.dirname(__file__),f"dwa_tests_long_nav_{n_humans}_humans.pkl")):
    print(f"\n## Testing DWA ###")
    test_env_params = {
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
    test_env = SocialNav(**test_env_params)
    # Test DWA policy
    metrics_dwa = test_k_custom_trials_dwa(
        n_trials, 
        random_seed, 
        test_env, 
        reward_function.time_limit,
        robot_vmax=policy.v_max,
        robot_wmax=2*policy.v_max/policy.wheels_distance, 
        custom_episodes=custom_episodes,
    )
    all_metrics_dwa = tree_map(lambda x, y: x.at[:].set(y), all_metrics_dwa, metrics_dwa)

    ## Save results
    with open(os.path.join(os.path.dirname(__file__),f"dwa_tests_long_nav_{n_humans}_humans.pkl"), 'wb') as f:
        pickle.dump(all_metrics_dwa, f)
else:
    ## Load results
    with open(os.path.join(os.path.dirname(__file__),f"dwa_tests_long_nav_{n_humans}_humans.pkl"), 'rb') as f:
        all_metrics_dwa = pickle.load(f)

### Plot results
# Matplotlib font
from matplotlib import rc
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
metrics = {
    "successes": {"label": "Success Rate", "episodic": False}, 
    "collisions": {"label": "Collision Rate", "episodic": False}, 
    "timeouts": {"label": "Timeout Rate", "episodic": False}, 
    "returns": {"label": "Return ($\gamma = 0.9$)", "episodic": True},
    "times_to_goal": {"label": "Time to goal ($s$)", "episodic": True},
    "average_speed": {"label": "Lin. speed ($m/s$)", "episodic": True},
    "average_acceleration": {"label": "Lin. accel. ($m/s^2$)", "episodic": True},
    "average_jerk": {"label": "Lin. jerk ($m/s^3$)", "episodic": True},
    "average_angular_speed": {"label": "Ang. speed ($rad/s$)", "episodic": True},
    "average_angular_acceleration": {"label": "Ang. accel. ($rad/s^2$)", "episodic": True},
    "average_angular_jerk": {"label": "Ang. jerk ($rad/s^3$)", "episodic": True},
    "min_distance": {"label": "Minimum distance to humans ($m$)", "episodic": True},
    "space_compliance": {"label": "Space compliance", "episodic": True},
    "episodic_spl": {"label": "Episodic SPL", "episodic": True},
    "path_length": {"label": "Path length ($m$)", "episodic": True},
    "waypoint_reached": {"label": "Last Waypoint Reached", "episodic": True},
}