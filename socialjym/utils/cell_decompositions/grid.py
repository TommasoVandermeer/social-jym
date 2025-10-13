from jax import lax, jit, vmap, debug
import jax.numpy as jnp
from functools import partial

from socialjym.utils.cell_decompositions.utils import get_grid_map_center, is_cell_occupied

@partial(jit, static_argnames=("cell_size", "min_grid_size"))
def decompose(cell_size, min_grid_size, state, info, obstacle_points, epsilon=1e-5):
    """
    Builds a square grid map centered around the robot and computes the occupancy grid based on static obstacles.

    parameters:
    - cell_size: Size of each grid cell (in meters)
    - min_grid_size: Minimum size of the grid (in meters)
    - state: Current state of the environment (robot + humans)
    - info: Additional information from the environment
    - obstacle_points: Array of shape (n_obstacle_points, 2) containing the (x, y) coordinates of static obstacle points. WARNING: Only used to compute the grid map center, not for occupancy

    returns:
    - cells: Array of shape (n_cells, 2) containing the center of each grid cell
    - edges: Array of shape (n_cells, n_cells) representing the edges matrix for pathfinding
    - grid_info: Dictionary containing:
        - grid_cells: Array of shape (n_x, n_y, 2) containing the (x, y) coordinates of each grid cell center. n_x and n_y depend on the fixed grid size defined by cell_size and min_grid_size.
        - grid_occupancy: Boolean array of shape (n_x, n_y), where True indicates an occupied cell
    """
    center = get_grid_map_center(state, info, obstacle_points)
    dists_vector = jnp.concatenate([-jnp.arange(0, min_grid_size/2 + cell_size, cell_size)[::-1][:-1],jnp.arange(0, min_grid_size/2 + cell_size, cell_size)])
    grid_center_x, grid_center_y = jnp.meshgrid(dists_vector + center[0], dists_vector + center[1])
    n_x = grid_center_x.shape[0]
    n_y = grid_center_y.shape[1]
    cells = jnp.array(jnp.vstack((grid_center_x.flatten(), grid_center_y.flatten())).T)
    # Compute occupancy grid
    occupancy_vector = vmap(is_cell_occupied, in_axes=(None, 0, 0, 0, 0))(
        info['static_obstacles'][-1],
        cells[:,0] - cell_size/2 - epsilon,
        cells[:,0] + cell_size/2 + epsilon,
        cells[:,1] - cell_size/2 - epsilon,
        cells[:,1] + cell_size/2 + epsilon,
    )
    grid_cells = jnp.stack((grid_center_x, grid_center_y), axis=-1)
    grid_occupancy = jnp.reshape(occupancy_vector, (n_x, n_y))
    # Compute edges matrix
    @jit
    def _are_adjacent(idx1, cell1, idx2, cell2, occupancy_vector):
        dx = jnp.abs(cell1[0] - cell2[0])
        dy = jnp.abs(cell1[1] - cell2[1])
        adjacent_x = (jnp.allclose(dy, 0.) & jnp.allclose(dx, cell_size))
        adjacent_y = (jnp.allclose(dx, 0.) & jnp.allclose(dy, cell_size))
        adjacent = adjacent_x | adjacent_y
        return lax.cond(
            (~jnp.all(cell1 == cell2)) & (adjacent) & (~occupancy_vector[idx1]) & (~occupancy_vector[idx2]),
            lambda _: 1.,
            lambda _: jnp.inf,
            None,
        )
    @jit
    def _cell_adjacency(idx1, cell1, cells, occupancy_vector):
        return vmap(_are_adjacent, in_axes=(None, None, 0, 0, None))(idx1, cell1, jnp.arange((n_x * n_y)), cells, occupancy_vector)
    @jit
    def cost_matrix(cells, occupancy_vector):
        return vmap(_cell_adjacency, in_axes=(0, 0, None, None))(jnp.arange((n_x * n_y)), cells, cells, occupancy_vector)
    edges = cost_matrix(cells, occupancy_vector)
    return cells, edges, {"grid_cells": grid_cells, "grid_occupancy": grid_occupancy}