from jax import jit, lax, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
from functools import partial

from socialjym.utils.global_planners.dijkstra import DijkstraPlanner

class AStarPlanner(DijkstraPlanner):
    def __init__(self, grid_size:jnp.ndarray) -> None:
        super().__init__(
            grid_size=grid_size
        )
        self.name = "A*"

    # --- Private methods ---

    @partial(jit, static_argnames=('self'))
    def _heuristic(self, pos1, pos2) -> float:
        return jnp.linalg.norm(pos1 - pos2)

    # --- Public methods ---