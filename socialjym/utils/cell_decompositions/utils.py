from jax import random, lax, jit, vmap
import jax.numpy as jnp
from functools import partial

@partial(jit, static_argnames=("cell_size", "min_grid_size"))
def get_grid_size(cell_size, min_grid_size):
    """
    Computes the size of the grid map based on the cell size and minimum grid size.

    parameters:
    - cell_size: Size of each grid cell (in meters)
    - min_grid_size: Minimum size of the grid (in meters)

    returns:
    - n_x: Number of cells in the x direction
    - n_y: Number of cells in the y direction
    """
    dists_vector = jnp.concatenate([-jnp.arange(0, min_grid_size/2 + cell_size, cell_size)[::-1][:-1],jnp.arange(0, min_grid_size/2 + cell_size, cell_size)])
    n_x = dists_vector.shape[0]
    n_y = dists_vector.shape[0]
    return n_x, n_y

@jit
def get_grid_map_center(state, info, obstacle_points):
    """
    Computes the center of the grid map based on the current state and info of the environment.

    parameters:
    - state: Current state of the environment (robot + humans)
    - info: Additional information from the environment
    - obstacle_points: Array of shape (n_obstacle_points, 2) containing the (x, y) coordinates of static obstacle points

    returns:
    - center: Array of shape (2,) containing the (x, y) coordinates of the grid map center
    """
    center = jnp.nanmean(jnp.vstack((obstacle_points, state[-1,:2], info['robot_goal'])), axis=0)
    return center

@jit
def edge_intersects_cell(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
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
def obstacle_intersects_cell(obstacle, xmin, xmax, ymin, ymax):
    return jnp.any(vmap(edge_intersects_cell, in_axes=(0,0,0,0,None,None,None,None))(obstacle[:,0,0], obstacle[:,0,1], obstacle[:,1,0], obstacle[:,1,1], xmin, xmax, ymin, ymax))

@jit
def is_cell_occupied(obstacles, xmin, xmax, ymin, ymax):
    """
    Checks if a grid cell is occupied by any of the obstacles.

    parameters:
    - obstacles: Array of shape (n_obstacles, n_edges, 2, 2) representing the line segments of the obstacles
    - xmin, xmax, ymin, ymax: Boundaries of the grid cell   

    returns:
    - occupied: Boolean indicating whether the cell is occupied (True) or free (False)
    """
    return jnp.any(vmap(obstacle_intersects_cell, in_axes=(0, None, None, None, None))(obstacles, xmin, xmax, ymin, ymax))