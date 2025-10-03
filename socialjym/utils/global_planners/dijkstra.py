from jax import jit, lax, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
from functools import partial

from socialjym.utils.global_planners.base_global_planner import BaseGlobalPlanner

class DijkstraPlanner(BaseGlobalPlanner):
    def __init__(self, grid_size:jnp.ndarray) -> None:
        super().__init__(
            grid_size=grid_size
        )
        self.name = "Dijkstra"

    # --- Private methods ---

    @partial(jit, static_argnames=('self'))
    def _heuristic(self, pos1, pos2) -> float:
        return 0.

    # --- Public methods ---

    @partial(jit, static_argnames=("self"))
    def find_path(
        self,
        start:jnp.ndarray,
        goal:jnp.ndarray,
        grid_cells:jnp.ndarray,
        occupancy_grid:jnp.ndarray,
    ) -> tuple:
        n_rows, n_cols = grid_cells.shape[:2]
        ## Find start and goal nodes
        start_node, _, start_cell_occupancy = self._query_grid_map(grid_cells, occupancy_grid, start)
        goal_node, _, goal_cell_occupancy = self._query_grid_map(grid_cells, occupancy_grid, goal)
        ## Initialize node data
        nodes_data = {
            'position': grid_cells,
            'g': jnp.full((self.grid_size_x, self.grid_size_y), jnp.inf),  # Cost to reach each node
            'h': jnp.full((self.grid_size_x, self.grid_size_y), jnp.inf),  # Heuristic cost to goal
            'f': jnp.full((self.grid_size_x, self.grid_size_y), jnp.inf),  # Total cost
            'parent': jnp.full((self.grid_size_x, self.grid_size_y, 2), -1, dtype=jnp.int32),  # Parent node index
        }
        ## Utility functions
        @jit
        def create_node(nodes_data, node, g, h, parent):
            i, j = node
            nodes_data = tree_map(
                lambda x, y: x.at[i, j].set(y), 
                nodes_data, 
                {
                    'position': nodes_data['position'][i,j], 
                    'g': g, 
                    'h': h, 
                    'f': g+h, 
                    'parent': parent
                }
            )
            return nodes_data
        @jit
        def is_valid_neighbor(node, occupancy_grid):
            i, j = node
            return \
                (i >= 0) & \
                (i < self.grid_size_x) & \
                (j >= 0) & \
                (j < self.grid_size_y) & \
                (~(occupancy_grid[i, j]))
        @jit
        def get_neighbors(node, occupancy_grid):
            i, j = node
            neighbors = jnp.full((self.n_neighbors, 2), jnp.nan, dtype=jnp.int32)
            neighbors = lax.fori_loop(
                0,
                self.n_neighbors,
                lambda idx, arr: lax.cond(
                    is_valid_neighbor((i + self.moves[idx,0], j + self.moves[idx,1]), occupancy_grid),
                    lambda a: a.at[idx].set(jnp.array([i, j]) + self.moves[idx]),
                    lambda a: a,
                    arr
                ),
                neighbors
            )
            return neighbors
        @jit
        def reconstruct_path(nodes_data, goal_node):
            inverted_path = jnp.full((self.n_cells, 2), jnp.nan)
            _, inverted_path, path_length = lax.while_loop(
                lambda x: jnp.all(x[0] != -1),
                lambda x: (
                    nodes_data['parent'][x[0][0], x[0][1]],
                    x[1].at[x[2]].set(nodes_data['position'][x[0][0], x[0][1]]),
                    x[2] + 1
                ),
                (goal_node, inverted_path, 0)
            )
            path = lax.fori_loop(
                0,
                self.n_cells,
                lambda idx, arr: lax.cond(
                    idx < path_length,
                    lambda x: x.at[idx].set(inverted_path[path_length - idx - 1]),
                    lambda x: x,
                    arr
                ),
                inverted_path
            )
            return path, path_length
        ## Initialize the start node
        nodes_data = create_node(
            nodes_data, 
            start_node, 
            g=0.0, 
            h=self._heuristic(nodes_data['position'][start_node[0], start_node[1]], nodes_data['position'][goal_node[0], goal_node[1]]), 
            parent=jnp.array([-1, -1], dtype=int)  # No parent for the start node
        )
        ## Initialize the open set and closed set
        open_set = jnp.zeros((n_rows, n_cols), dtype=bool).at[start_node[0], start_node[1]].set(True) # Nodes to visit
        closed_set = jnp.zeros((n_rows, n_cols), dtype=bool) # Visited nodes
        ## Main loop
        @jit
        def _loop_body(state):
            nodes_data, open_set, closed_set, goal_node = state
            # Get the node with the lowest f value
            current_node = jnp.unravel_index(jnp.argmin(jnp.where(open_set, nodes_data['f'], jnp.inf)), (n_rows, n_cols))
            open_set = open_set.at[current_node[0], current_node[1]].set(False)
            closed_set = closed_set.at[current_node[0], current_node[1]].set(True)
            # For each neighbor of the current node
            neighbors = get_neighbors(current_node, occupancy_grid)
            @jit
            def _update_neighbor(closed_set, nodes_data, neighbor, current_node, goal_node):
                @jit
                def _not_visited_neighbor(args):
                    nodes_data, neighbor, current_node, goal_node = args
                    tentative_g = nodes_data['g'][current_node[0], current_node[1]] + 1.0 # Assume cost between nodes is 1
                    better_path = tentative_g < nodes_data['g'][neighbor[0], neighbor[1]]
                    h = lax.cond(
                        better_path,
                        lambda x: self._heuristic(nodes_data['position'][x[0], x[1]], nodes_data['position'][goal_node[0], goal_node[1]]),
                        lambda x: nodes_data['h'][x[0], x[1]],
                        neighbor
                    )
                    return tentative_g, h, better_path
                g, h, better = lax.cond(
                    closed_set[neighbor[0], neighbor[1]] | (jnp.any(jnp.isnan(neighbor))),
                    lambda x: (x[0]['g'][neighbor[0], neighbor[1]], x[0]['h'][neighbor[0], neighbor[1]], False),
                    lambda x: _not_visited_neighbor(x),
                    (nodes_data, neighbor, current_node, goal_node)
                )
                return g, h, better
            gs, hs, betters = vmap(_update_neighbor, in_axes=(None, None, 0, None, None))(
                closed_set, 
                nodes_data, 
                neighbors, 
                current_node, 
                goal_node
            )
            open_set, nodes_data = lax.fori_loop(
                0,
                self.n_neighbors,
                lambda idx, state: lax.cond(
                    betters[idx],
                    lambda x: (
                        x[0].at[neighbors[idx,0], neighbors[idx,1]].set(True),
                        create_node(
                            x[1], 
                            neighbors[idx], 
                            g=gs[idx], 
                            h=hs[idx], 
                            parent=jnp.array(current_node, dtype=int)
                        )
                    ),
                    lambda x: x,
                    state
                ),
                (open_set, nodes_data)
            )
            return (nodes_data, open_set, closed_set, goal_node)
        nodes_data, open_set, closed_set, _ = lax.while_loop(
            lambda x: (jnp.any(x[1]) & (~x[2][goal_node[0], goal_node[1]])),
            _loop_body,
            (nodes_data, open_set, closed_set, goal_node)
        )
        ## Reconstruct path
        return reconstruct_path(nodes_data, goal_node)