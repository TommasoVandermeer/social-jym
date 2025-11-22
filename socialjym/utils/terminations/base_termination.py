from jax import jit, vmap, lax
import jax.numpy as jnp
from abc import ABC, abstractmethod

TERMINATIONS = [
    'instant_robot_human_collision',
    'interval_robot_human_collision',
    'instant_robot_obstacle_collision',
    'robot_reached_goal',
    'timeout',
]

@jit
def point_to_line_distance(point:jnp.ndarray, line_start:jnp.ndarray, line_end:jnp.ndarray) -> float:
    """
    Computes the distance between a point and a line defined by two points.

    args:
    - point: jnp.ndarray, shape=(2,), dtype=jnp.float32. The point to compute the distance from.
    - line_start: jnp.ndarray, shape=(2,), dtype=jnp.float32. The starting point of the line.
    - line_end: jnp.ndarray, shape=(2,), dtype=jnp.float32. The ending point of the line.

    output:
    - distance: float. The distance between the point and the line.
    """
    x = point[0]
    y = point[1]
    x1 = line_start[0]
    y1 = line_start[1]
    x2 = line_end[0]
    y2 = line_end[1]
    dx = x2 - x1
    dy = y2 - y1

    u = lax.cond(
        jnp.all(jnp.array([dx == 0, dy == 0])), 
        lambda _: 0., 
        lambda _: jnp.squeeze(((x - x1) * dx + (y - y1) * dy) / jnp.linalg.norm(line_end - line_start)**2), 
        None)

    # Clamp u to [0,1]
    u = lax.cond(u < 0, lambda _: 0., lambda x: x, u)
    u = lax.cond(u > 1, lambda _: 1., lambda x: x, u)

    closest_point = jnp.array([x1 + u * dx, y1 + u * dy])
    closest_distance = jnp.linalg.norm(closest_point - point)

    return closest_distance

@jit
def batch_point_to_line_distance(points:jnp.ndarray, line_starts:jnp.ndarray, line_ends:jnp.ndarray) -> jnp.ndarray:
    return vmap(point_to_line_distance, in_axes=(0, 0, 0))(points, line_starts, line_ends)


class BaseTermination(ABC):
    def __init__(self, name) -> None:
        self.termination_id = TERMINATIONS.index(name)

    # --- Private methods ---

    @abstractmethod
    def __call__(self, state, action) -> tuple:
        pass
