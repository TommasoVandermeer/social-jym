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

    @partial(jit, static_argnames=("self"))
    def _segment_rectangle_intersection(self, x1, y1, x2, y2, xmin, xmax, ymin, ymax):
        """
        This is the Liang-Barsky algorithm for line clipping.
        """
        @jit
        def _nan_segment(val):
            return False, jnp.array([jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan])
        @jit
        def _not_nan_segment(val):
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
                return ((t0, t1), p ,q)
            t, p, q = lax.fori_loop(
                0, 
                4,
                loop_body,
                ((0., 1.), p, q),
            )
            t0, t1 = t
            inside_or_intersects = ~(t0 > t1)
            intersection_point_0 = lax.switch(
                jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t0 == 0), (inside_or_intersects) & (t0 > 0)])),
                [
                    lambda _: jnp.array([jnp.nan, jnp.nan]),
                    lambda _: jnp.array([x1, y1]),
                    lambda _: jnp.array([x1 + t0 * dx, y1 + t0 * dy]),
                ],
                None,
            )
            intersection_point_1 = lax.switch(
                jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t1 == 1), (inside_or_intersects) & (t1 < 1)])),
                [
                    lambda _: jnp.array([jnp.nan, jnp.nan]),
                    lambda _: jnp.array([x2, y2]),
                    lambda _: jnp.array([x1 + t1 * dx, y1 + t1 * dy]),
                ],
                None,
            )
            return inside_or_intersects, intersection_point_0, intersection_point_1
        return lax.cond(
            jnp.any(jnp.isnan(jnp.array([x1, y1, x2, y2]))),
            _nan_segment,
            _not_nan_segment,
            (x1, y1, x2, y2, xmin, xmax, ymin, ymax),
        )

    @partial(jit, static_argnames=("self"))
    def _batch_segment_rectangle_intersection(self, x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax):
        return vmap(BasePolicy._segment_rectangle_intersection, in_axes=(None,0,0,0,0,None,None,None,None))(self, x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax)
    

    # Public methods

    @abstractmethod
    def act(self):
        pass

    @partial(jit, static_argnames=("self","n_edges"))
    def batch_compute_disk_circumscribing_n_agon(self, positions, radii, n_edges=10):
        return vmap(BasePolicy._compute_disk_circumscribing_n_agon, in_axes=(None, 0, 0, None))(
            self, 
            positions, 
            radii, 
            n_edges
        )