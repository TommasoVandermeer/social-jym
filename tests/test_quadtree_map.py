from jax import jit, vmap, lax, random, debug
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward

### Parameters
scenario = 'robot_crowding'

### Functions
@jit
def _edge_intersects_cell(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
    @jit
    def _not_nan_obs(val:tuple):
        x1, y1, x2, y2, xmin, xmax, ymin, ymax = val
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
    @jit
    def _nan_obs(val:tuple):
        # If the obstacle is NaN, it means it doesn't exist, so it cannot intersect the cell
        return False
    return lax.cond(
        jnp.any(jnp.isnan(jnp.array([x1, y1, x2, y2]))), 
        _nan_obs,
        _not_nan_obs, 
        (x1, y1, x2, y2, xmin, xmax, ymin, ymax)
    )
@jit
def _obstacle_intersects_cell(obstacle, xmin, xmax, ymin, ymax):
    return jnp.any(vmap(_edge_intersects_cell, in_axes=(0,0,0,0,None,None,None,None))(obstacle[:,0,0], obstacle[:,0,1], obstacle[:,1,0], obstacle[:,1,1], xmin, xmax, ymin, ymax))
@jit
def _is_cell_occupied(obstacles, xmin, xmax, ymin, ymax):
    return jnp.any(vmap(_obstacle_intersects_cell, in_axes=(0, None, None, None, None))(obstacles, xmin, xmax, ymin, ymax))
@jit
def _batch_is_cell_occupied(obstacles, cell_centers, cell_sizes):
    half_sizes = cell_sizes / 2.
    xmins = cell_centers[:,0] - half_sizes[:,0]
    xmaxs = cell_centers[:,0] + half_sizes[:,0]
    ymins = cell_centers[:,1] - half_sizes[:,1]
    ymaxs = cell_centers[:,1] + half_sizes[:,1]
    return vmap(_is_cell_occupied, in_axes=(None, 0, 0, 0, 0))(obstacles, xmins, xmaxs, ymins, ymaxs)
@jit
def _build_quadtree_branch(cell_center, cell_size):
    half_size = cell_size / 2.
    quarter_size = cell_size / 4.
    # Define the 4 quadrants: top-left, top-right, bottom-left, bottom-right
    quadrants = jnp.array([
        [cell_center[0] - quarter_size[0], cell_center[1] + quarter_size[1]], # Top-left
        [cell_center[0] + quarter_size[0], cell_center[1] + quarter_size[1]], # Top-right
        [cell_center[0] - quarter_size[0], cell_center[1] - quarter_size[1]], # Bottom-left
        [cell_center[0] + quarter_size[0], cell_center[1] - quarter_size[1]], # Bottom-right
    ])
    return half_size, quadrants
@jit
def _are_adjacent(cell1, cell2, eps=1e-5):
    center1, size1 = cell1[:2], cell1[2:]
    center2, size2 = cell2[:2], cell2[2:]
    half_size1 = size1 / 2.
    half_size2 = size2 / 2.
    dx = jnp.abs(center1[0] - center2[0])
    dy = jnp.abs(center1[1] - center2[1])
    adjacent_x = (dy + eps < (half_size1[1] + half_size2[1])) & jnp.allclose(dx, half_size1[0] + half_size2[0])
    adjacent_y = (dx + eps < (half_size1[0] + half_size2[0])) & jnp.allclose(dy, half_size1[1] + half_size2[1])
    return lax.cond(
        ~jnp.array_equal(cell1, cell2) & (adjacent_x | adjacent_y),
        lambda _: jnp.sqrt(dx**2 + dy**2),
        lambda _: 0.,
        None
    )
@jit
def _cell_adjacency(cell1, cells):
    return vmap(_are_adjacent, in_axes=(None, 0))(cell1, cells)
@jit
def cost_matrix(cells):
    return vmap(_cell_adjacency, in_axes=(0, None))(cells, cells)

### Build quadtree map 
env_params = {
    'robot_radius': 0.3,
    'n_humans': 5,
    'n_obstacles': 5, # n_obstacles is not used in this scenario
    'robot_dt': 0.25,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': scenario,
    'humans_policy': 'hsfm',
    'reward_function': DummyReward(kinematics='unicycle'),
    'kinematics': 'unicycle',
    'ccso_n_static_humans': 0,
}
env = SocialNav(**env_params)
state, _, _, info, _ = env.reset(random.PRNGKey(0))
map_size = jnp.array([env.get_grid_size()[0] * env.grid_cell_size, env.get_grid_size()[1] * env.grid_cell_size])
map_center = env.get_grid_map_center(state, info)
min_cell_size = env.grid_cell_size
print("Map size:", map_size)
print("Map center:", map_center)
print("Min cell size:", min_cell_size)
stop = False
open_cells = jnp.array([[*map_center, *map_size]]) # List of tuples (cell_size, cell_center)
free_cells = []
occupied_cells = []
while not stop:
    # Check if open cells are occupied
    occupied = _batch_is_cell_occupied(info['static_obstacles'][-1], open_cells[:,:2], open_cells[:,2:])
    # Add free cells to free_cells list
    free_cells += [open_cells[i] for i in range(len(open_cells)) if not occupied[i]]
    # Subdivide occupied cells if they are larger than min_cell_size, else add them to occupied_cells list
    new_open_cells = []
    for i in range(len(open_cells)):
        if occupied[i]:
            if jnp.all(open_cells[i,2:] > min_cell_size):
                new_size, quadrants = _build_quadtree_branch(open_cells[i,:2], open_cells[i,2:])
                for q in quadrants:
                    new_open_cells.append(jnp.array([*q, *new_size]))
            else:
                occupied_cells.append(open_cells[i])
    open_cells = jnp.array(new_open_cells)
    # Stop if no open cells left
    if len(open_cells) == 0:
        stop = True
free_cells = jnp.array(free_cells)
occupied_cells = jnp.array(occupied_cells)

### Compute adjacency matrix
edges = cost_matrix(free_cells)

### Print stats
_, normal_occupancy = env.build_grid_map_and_occupancy(state, info) # Build grid map to get number of cells in normal decomposition
print(f"\nNumber of cells in normal decomposition: {jnp.prod(jnp.array(normal_occupancy.shape))} - free: {jnp.sum(normal_occupancy==0)} - occupied: {jnp.sum(normal_occupancy==1)} - unknown: {jnp.sum(normal_occupancy==-1)}")
print(f"Number of cells in quadtree decomposition: {len(free_cells) + len(occupied_cells)} - free: {len(free_cells)} - occupied: {len(occupied_cells)} - adjacency matrix shape: {edges.shape}")
print(f"Max edge cost: {jnp.max(edges)} - Min edge cost (non-zero): {jnp.min(edges[edges>0])} - Average edge cost (non-zero): {jnp.mean(edges[edges>0])}\n")
### Plot map
fig, ax = plt.subplots(figsize=(8,8))
# Plot obstacles
if info['static_obstacles'][-1].shape[1] > 1: # Polygon obstacles
    for o in info['static_obstacles'][-1]: plt.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
else: # One segment obstacles
    for o in info['static_obstacles'][-1]: plt.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
# Plot free cells
plt.scatter(free_cells[:,0], free_cells[:,1], color='green', s=10, label='Free cells', zorder=4, alpha=0.7)
for cell in free_cells:
    cell_size = cell[2:]
    cell_center = cell[:2]
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='green', linewidth=0.5, alpha=0.5, zorder=1)
    ax.add_patch(rect)
# Plot occupied cells
for cell in occupied_cells:
    cell_size = cell[2:]
    cell_center = cell[:2]
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='red', edgecolor='red', linewidth=0.5, alpha=0.5, zorder=2)
    ax.add_patch(rect)
# Plot adjacency edges
for x, from_cell in enumerate(free_cells):
    for y, to_cell in enumerate(free_cells):
        if edges[x, y] > 0:
            plt.plot([from_cell[0], to_cell[0]], [from_cell[1], to_cell[1]], color='blue', linewidth=0.5, zorder=5)
plt.show()