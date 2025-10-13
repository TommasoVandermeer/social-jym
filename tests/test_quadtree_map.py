from jax import jit, vmap, lax, random, debug
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.cell_decompositions.quadtree import decompose
from socialjym.utils.cell_decompositions.utils import get_grid_map_center

### Parameters
scenario = 'robot_crowding'

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
min_cell_size = env.grid_cell_size
map_size = jnp.array([env.get_grid_size()[0] * min_cell_size, env.get_grid_size()[1] * min_cell_size])

### Execute quadtree decomposition
free_cells, occupied_cells, edges = decompose(
    min_cell_size, 
    map_size, 
    get_grid_map_center(state, info, jnp.reshape(env.static_obstacles_per_scenario[info['current_scenario']], (10,-1))),
    info['static_obstacles'][-1],
)

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