from jax import jit, lax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from functools import partial

GLOBAL_PLANNERS = ["A*", "Dijkstra"]

class BaseGlobalPlanner(ABC):
    def __init__(self, grid_size:jnp.ndarray) -> None:
        self.moves = jnp.array([
            [1, 0],  # Up
            [-1, 0],   # Down
            [0, -1],  # Left
            [0, 1],   # Right
        ])
        self.grid_size_x = int(grid_size[0])
        self.grid_size_y = int(grid_size[1])
        self.n_cells = int(grid_size[0] * grid_size[1])
        self.n_neighbors = int(self.moves.shape[0])

    # --- Private methods ---
    
    @partial(jit, static_argnames=("self"))
    def _query_grid_map(
        self,
        grid_cells:jnp.ndarray, 
        occupancy_grid:jnp.ndarray, 
        point:jnp.ndarray,
    ) -> tuple:
        """
        Query the occupancy grid to check to which grid cell a point belongs to.
        WARNING: If the point is not contained in any grid cell, the function will return the closest cell anyway.

        parameters:
        - grid_cells: jnp.ndarray of shape (n_rows, n_cols, 2), the coordinates of the center of each grid cell
        - occupancy_grid: jnp.ndarray of shape (n_rows, n_cols), the occupancy grid (1 for occupied, 0 for free)
        - point: jnp.ndarray of shape (2,), the (x, y) coordinates of the point to query
        """
        cell_indices = jnp.argmin(jnp.linalg.norm(grid_cells - point, axis=2))
        row, col = jnp.unravel_index(cell_indices, occupancy_grid.shape)
        return jnp.array([row, col]), grid_cells[row, col], occupancy_grid[row, col]

    # --- Public methods ---

    @abstractmethod
    def find_path(
        self, 
        start:jnp.ndarray, 
        goal:jnp.ndarray, 
        grid_cells:jnp.ndarray, 
        occupancy_grid:jnp.ndarray,
        edges:jnp.ndarray,
    ) -> tuple:
        pass