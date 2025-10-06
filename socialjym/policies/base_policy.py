from abc import ABC, abstractmethod
from jax import jit, lax, vmap
import jax.numpy as jnp
from functools import partial

class BasePolicy(ABC):
    def __init__(self, discount) -> None:
        self.gamma = discount
        pass

    # Private methods

    @partial(jit, static_argnames=("self","n_edges"))
    def _compute_disk_circumscribing_n_agon(self, pos, rad, n_edges):
        shape_radius = rad / jnp.cos(jnp.pi / n_edges)  # Circumscribed radius
        angles = jnp.linspace(0, 2 * jnp.pi, n_edges+1)[:-1] + jnp.pi / n_edges  # Start at pi/n_edges for horizontal side
        shape_vertices = jnp.array([jnp.cos(angles), jnp.sin(angles)]).T * shape_radius + jnp.array(pos)
        shape_edges = lax.fori_loop(
            0,
            n_edges,
            lambda j, edges: edges.at[j].set(jnp.array([shape_vertices[j], shape_vertices[(j + 1) % n_edges]])),
            jnp.zeros((n_edges, 2, 2))
        )
        return jnp.array(shape_edges)

    # Public methods

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @partial(jit, static_argnames=("self","n_edges"))
    def batch_compute_disk_circumscribing_n_agon(self, positions, radii, n_edges=10):
        return vmap(BasePolicy._compute_disk_circumscribing_n_agon, in_axes=(None, 0, 0, None))(
            self, 
            positions, 
            radii, 
            n_edges
        )