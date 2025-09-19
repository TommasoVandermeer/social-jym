import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import heapq
import os 
import pickle

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.reward2 import Reward2
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.aux_functions import animate_trajectory

### Parameters
time_limit = 120.
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
    # [[[-6.,-4.], [-3.,-4.]]],
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
humans_pose = jnp.array([
    [10, -9, jnp.pi/2],
    [12, -3.5, jnp.pi],
    [6, -6, jnp.pi/2],
    [0., 0., -jnp.pi/2],
    [-7.25, -5, jnp.pi/2],
    [-14., 3., 0.],
    [-6.5, -8.5, 0.],
    [-13.5, 1., 0.],
])
humans_goal = jnp.array([
    [10, -5.],
    [6, -3.5],
    [6, 4],
    [0., -7.5],
    [-7.25, 5.],
    [-7., 3.],
    [3.5, 0.],
    [-7., 1.],
])

### Enlarge obstacles to prevent the robot getting too close from them
def enlarge_obstacles(obstacles, enlargement_size):
    """
    For each obstacle (single segment), enlarge it to a rectangle (four segments) with the given enlargement size.
    Returns a new array of obstacles, each as a rectangle (4 segments).
    """
    enlarged = []
    for obs in obstacles:
        # obs shape: (1, 2, 2) -> one segment, two endpoints, two coordinates
        p1 = obs[0, 0]
        p2 = obs[0, 1]
        # Direction vector
        d = p2 - p1
        d_norm = d / (jnp.linalg.norm(d) + 1e-8)
        # Perpendicular vector
        perp = jnp.array([-d_norm[1], d_norm[0]])
        # Parallel vector
        parallel = d_norm
        # Offset points (enlarge in both perpendicular and parallel directions)
        p1a = p1 + perp * enlargement_size / 2 - parallel * enlargement_size / 2
        p1b = p1 - perp * enlargement_size / 2 - parallel * enlargement_size / 2
        p2a = p2 + perp * enlargement_size / 2 + parallel * enlargement_size / 2
        p2b = p2 - perp * enlargement_size / 2 + parallel * enlargement_size / 2
        # Rectangle segments: [p1a-p2a], [p2a-p2b], [p2b-p1b], [p1b-p1a]
        rect = jnp.array([
            [p1a, p2a],
            [p2a, p2b],
            [p2b, p1b],
            [p1b, p1a]
        ])
        enlarged.append(rect[None, ...])  # shape (1, 4, 2, 2)
    return jnp.concatenate(enlarged, axis=0)
enlarged_obstacles = enlarge_obstacles(obstacles, enlargement_size=0.2)

### Computations
# Generate grid coordinates
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

### Plotting
# Plot obstacles and cell decomposition
from matplotlib import rc
font = {
    'weight' : 'regular',
    'size'   : 12
}
rc('font', **font)
fig, ax = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(6, 4.2))
fig2.subplots_adjust(right=0.99, left=0.12, top=0.99, bottom=0.14)
try:
    with open(os.path.join(os.path.dirname(__file__), 'custom_episodes_30_humans.pkl'), 'rb') as f:
        custom_episodes = pickle.load(f)
        full_state = custom_episodes['full_state'][1, :-1]
        for h in range(full_state.shape[0]):
            head = plt.Circle((full_state[h,0] + jnp.cos(full_state[h,4]) * 0.3, full_state[h,1] + jnp.sin(full_state[h,4]) * 0.3), 0.1, color='purple', zorder=10)
            ax2.add_patch(head)
            circle = plt.Circle((full_state[h,0],full_state[h,1]),0.3, edgecolor='purple', facecolor="white", fill=True, zorder=10)
            ax2.add_patch(circle)
except:
    pass
if obstacles.shape[1] > 1: # Polygon obstacles
    for o in obstacles: 
        ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
        ax2.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
else: # One segment obstacles
    for o in obstacles: 
        ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
        ax2.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
for idx, coord in enumerate(grid_coordinates.reshape(-1,2)):
    i = idx // static_obstacles_for_each_cell.shape[1]
    j = idx % static_obstacles_for_each_cell.shape[1]
    facecolor = 'red' if occupancy_grid[i,j] else 'none'
    rect = plt.Rectangle((coord[0], coord[1]), grid_cell_size, grid_cell_size, facecolor=facecolor, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
    ax.add_patch(rect)
ax.scatter(start_pos[0], start_pos[1], color='green', label='Start Position', zorder=4, marker='o', s=100)
ax.scatter(goal_pos[0], goal_pos[1], color='red', label='Goal Position', zorder=4, marker='*', s=100)
ax2.scatter(start_pos[0], start_pos[1], color='green', label='Start Position', zorder=4, marker='o', s=100)
ax2.scatter(goal_pos[0], goal_pos[1], color='red', label='Goal Position', zorder=4, marker='*', s=100)
if path_to_goal.shape[0] > 0:
    path_to_goal_to_plot = path_to_goal.at[0].set(start_pos)
    path_to_goal_to_plot = path_to_goal_to_plot.at[-1].set(goal_pos)
    ax.plot(path_to_goal_to_plot[:, 0], path_to_goal_to_plot[:, 1], color='blue', linewidth=2, label='A* Path', zorder=3)
    ax.scatter(path_to_goal_to_plot[1:-1, 0], path_to_goal_to_plot[1:-1, 1], color='blue', s=10, zorder=3)
    ax2.plot(path_to_goal_to_plot[:, 0], path_to_goal_to_plot[:, 1], color='blue', linewidth=2, label='A* path to goal', zorder=3)
else:
    print("No path found.")
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)', labelpad=-10)
ax2.set_xticks(jnp.arange(-15, 16, 5))
# h, l = ax2.get_legend_handles_labels()
# h.append(plt.Line2D([0], [0], color='black', lw=2, label='Obstacles'))
# l.append('Obstacles')
# ax2.legend(h, l, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_aspect('equal', adjustable='box')
fig2.savefig(os.path.join(os.path.dirname(__file__), 'long_nav_snapshot.eps'), format='eps')
plt.show()

### Simulate DIRSAFE to navigate the computed path on the given map
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
env_params = {
    'robot_radius': 0.3,
    'n_humans': len(humans_pose),
    'n_obstacles': len(enlarged_obstacles),
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': None,
    'hybrid_scenario_subset': jnp.array([0, 1, 2, 3, 4, 6]), # All scenarios but circular_crossing_with_static_obstacles
    'humans_policy': 'hsfm',
    'reward_function': reward_function,
    'kinematics': 'unicycle',
}
env = SocialNav(**env_params)
policy = DIRSAFE(env.reward_function, v_max=1., dt=env_params['robot_dt'])
with open(os.path.join(os.path.dirname(__file__), 'rl_out.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']
# Simulate a custom episode
policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int))
state, reset_key, obs, info, outcome = env.reset_custom_episode(
    reset_key, 
    {
        "full_state": jnp.array([[h[0], h[1], 0., 0., h[2], 0.] for h in jnp.append(humans_pose, jnp.array([[path_to_goal[0,0], path_to_goal[0,1], 0.0]]), axis=0)]),
        "humans_goal": humans_goal,
        "robot_goal": path_to_goal[1],  # Start at the first waypoint
        "humans_radius": jnp.ones(env_params['n_humans']) * env_params['robot_radius'],
        "humans_speed": jnp.ones(env_params['n_humans']),
        "static_obstacles": jnp.repeat(enlarged_obstacles[None, :, :, :], env_params['n_humans']+1, axis=0),
        "scenario": -1,  # Custom scenario
    }
)
all_states = jnp.array([state])
all_robot_goals = jnp.array([info['robot_goal']])
all_action_space_params = []
# Humans and robot goals indexing
waypoint_idx = 1  # Start at the first waypoint
humans_chase_goal = jnp.ones(env_params['n_humans'], dtype=bool)  # All humans chase their goals initially
while outcome["nothing"]:
    # Environment step
    action, policy_key, _, _, distr = policy.act(policy_key, obs, info, actor_params, sample=True)
    action_space_params = [distr["vertices"][2,0]/policy.v_max,distr["vertices"][0,1]/(2*policy.v_max/policy.wheels_distance), distr["vertices"][1,1]/(-2*policy.v_max/policy.wheels_distance)]
    state, obs, info, reward, outcome, _ = env.step(state,info,action,test=True) 
    # Update robot goal
    if (waypoint_idx < path_to_goal.shape[0] - 1) and (jnp.linalg.norm(state[-1,:2]-info['robot_goal']) < env.robot_radius*2):
        print(f"Waypoint {waypoint_idx} reached! at time {info['time']:.2f}s")
        waypoint_idx += 1
        info['robot_goal'] = info["robot_goal"].at[:].set(path_to_goal[waypoint_idx])
    # Update humans goals
    for i in range(len(humans_pose)):
        if jnp.linalg.norm(state[i,:2] - info['humans_goal'][i]) < info['humans_parameters'][i,0]:
            humans_chase_goal = humans_chase_goal.at[i].set(not humans_chase_goal[i])  # Toggle chasing goal
            info['humans_goal'] = info['humans_goal'].at[i].set(humans_goal[i] if humans_chase_goal[i] else humans_pose[i,:2])   
    all_states = jnp.vstack((all_states, jnp.array([state])))
    all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
    all_action_space_params.append(action_space_params)
# Animate trajectory
animate_trajectory(
    all_states, 
    info['humans_parameters'][:,0], 
    env.robot_radius, 
    env_params['humans_policy'],
    all_robot_goals,
    None, # Custom scenario
    robot_dt=env_params['robot_dt'],
    static_obstacles=new_static_obstacles, #enlarged_obstacles, 
    kinematics='unicycle',
    # action_space_params=jnp.array(all_action_space_params),
    vmax=1.,
    wheels_distance=policy.wheels_distance,
    save=True,
    save_path=os.path.join(os.path.dirname(__file__), 'icar25_astar_local_planner.mp4'),
    figsize= (11, 6.6),
)