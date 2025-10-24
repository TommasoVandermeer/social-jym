from jax import lax, jit, vmap, debug
import jax.numpy as jnp
from functools import partial

from socialjym.utils.cell_decompositions.utils import get_grid_map_center, is_cell_occupied, edge_intersects_cell

def decompose(min_cell_size, map_size, map_center, obstacles):
    """
    Builds a square grid map centered around the robot and computes the occupancy grid based on static obstacles.

    parameters:
    - min_cell_size: Minimum size of each grid cell (in meters)
    - map_size: Size of the entire map (in meters)
    - map_center: (x, y) coordinates of the map center (usually the robot's position)
    - obstacles: Array of shape (n_obstacles, n_edges, n_points, 2) containing the (x, y) coordinates of each obstacle point

    returns:
    - cells: Array of shape (n_cells, 2) containing the center of each grid cell
    - edges: Array of shape (n_cells, n_cells) representing the edges matrix for pathfinding
    - grid_info: Dictionary containing:
        - grid_cells: Array of shape (n_x, n_y, 2) containing the (x, y) coordinates of each grid cell center. n_x and n_y depend on the fixed grid size defined by cell_size and min_grid_size.
        - grid_occupancy: Boolean array of shape (n_x, n_y), where True indicates an occupied cell
    """
    # Compute occupancy grid
    @jit
    def _batch_is_cell_occupied(obstacles, cell_centers, cell_sizes):
        half_sizes = cell_sizes / 2.
        xmins = cell_centers[:,0] - half_sizes[:,0]
        xmaxs = cell_centers[:,0] + half_sizes[:,0]
        ymins = cell_centers[:,1] - half_sizes[:,1]
        ymaxs = cell_centers[:,1] + half_sizes[:,1]
        return vmap(is_cell_occupied, in_axes=(None, 0, 0, 0, 0))(obstacles, xmins, xmaxs, ymins, ymaxs)
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
    stop = False
    open_cells = jnp.array([[*map_center, *map_size]]) # List of tuples (cell_size, cell_center)
    free_cells = []
    occupied_cells = []
    while not stop:
        # Check if open cells are occupied
        occupied = _batch_is_cell_occupied(obstacles, open_cells[:,:2], open_cells[:,2:])
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
    # Compute edges matrix
    @jit
    def _are_adjacent(cell1, cell2, eps=1e-5):
        ### Place edge only on adjacent cells
        # center1, size1 = cell1[:2], cell1[2:]
        # center2, size2 = cell2[:2], cell2[2:]
        # half_size1 = size1 / 2.
        # half_size2 = size2 / 2.
        # dx = jnp.abs(center1[0] - center2[0])
        # dy = jnp.abs(center1[1] - center2[1])
        # adjacent_x = (dy + eps < (half_size1[1] + half_size2[1])) & jnp.allclose(dx, half_size1[0] + half_size2[0])
        # adjacent_y = (dx + eps < (half_size1[0] + half_size2[0])) & jnp.allclose(dy, half_size1[1] + half_size2[1])
        # return lax.cond(
        #     ~jnp.array_equal(cell1, cell2) & (adjacent_x | adjacent_y),
        #     lambda _: jnp.sqrt(dx**2 + dy**2),
        #     lambda _: jnp.inf,
        #     None
        # )
        ### Place edge between all cells whose centers connection does not cross occupied cells
        center1, _ = cell1[:2], cell1[2:]
        center2, _ = cell2[:2], cell2[2:]
        is_intersecting = jnp.any(vmap(edge_intersects_cell, in_axes=(None, None, None, None, 0,0,0,0))(
            center1[0], 
            center1[1], 
            center2[0], 
            center2[1],
            occupied_cells[:,0] - occupied_cells[:,2]/2.,
            occupied_cells[:,0] + occupied_cells[:,2]/2.,
            occupied_cells[:,1] - occupied_cells[:,3]/2.,
            occupied_cells[:,1] + occupied_cells[:,3]/2.,
        ))
        return lax.cond(
            ~jnp.array_equal(cell1, cell2) & (~is_intersecting),
            lambda _: jnp.linalg.norm(center1 - center2),
            lambda _: jnp.inf,
            None
        )
    @jit
    def _cell_adjacency(cell1, cells):
        return vmap(_are_adjacent, in_axes=(None, 0))(cell1, cells)
    @jit
    def cost_matrix(cells):
        return vmap(_cell_adjacency, in_axes=(0, None))(cells, cells)
    edges = cost_matrix(free_cells)
    return free_cells, occupied_cells, edges

def query_map(free_cells, point):
    """
    Query the quadtree map to find the cell containing the given point.

    parameters:
    - free_cells: Array of shape (n_cells, 4) containing the (x, y) coordinates of each grid cell center and its size (width, height)
    - point: (x, y) coordinates of the point to query

    returns:
    - cell: Array of shape (4,) containing the (x, y) coordinates of the cell center and its size (width, height) if the point is inside a free cell, None otherwise
    """
    @jit
    def _is_point_in_cell(cell, point):
        half_size = cell[2:] / 2.
        return (point[0] >= cell[0] - half_size[0]) & (point[0] <= cell[0] + half_size[0]) & (point[1] >= cell[1] - half_size[1]) & (point[1] <= cell[1] + half_size[1])
    def _find_cell(cells, point):
        mask = vmap(_is_point_in_cell, in_axes=(0, None))(cells, point)
        if jnp.sum(mask) == 0:
            return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
        else:
            return cells[jnp.argwhere(mask)[0][0]]
    return _find_cell(free_cells, point)