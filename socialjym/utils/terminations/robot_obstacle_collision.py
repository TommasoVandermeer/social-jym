from jax import jit, debug
import jax.numpy as jnp
from functools import partial

from socialjym.utils.terminations.base_termination import BaseTermination
from jhsfm.hsfm import vectorized_compute_obstacle_closest_point

class InstantRobotObstacleCollision(BaseTermination):
    """
    Termination condition based on robot-obstacle collisions.
    The episode ends if the robot collides with any obstacle at a specific instant.
    """
    def __init__(self):
        super().__init__('instant_robot_obstacle_collision')

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        robot_pos:jnp.ndarray, 
        robot_radius:jnp.ndarray,
        obstacles:jnp.ndarray,
    ) -> tuple[bool, dict]:
        """
        Computes whether a collision between the robot and any obstacle has occurred at a specific instant.

        args:
        - robot_pos: jnp.ndarray, shape=(2,), dtype=jnp.float32. The position of the robot.
        - robot_radius: jnp.ndarray, shape=(). The radius of the robot.
        - obstacles: jnp.ndarray, shape=(M,E,2,2), dtype=jnp.float32. The positions of the obstacles.
        WARNING: Some obstacles may be NaN.
        
        output:
        - collision: bool. True if a collision has occurred, False otherwise.
        - info: dict. Additional information about the collision status.
        """
        closest_points = vectorized_compute_obstacle_closest_point(robot_pos, obstacles)
        distances = jnp.linalg.norm(closest_points - robot_pos, axis=-1) - robot_radius
        min_distance = jnp.nanmax(jnp.array([jnp.nanmin(distances),0]))
        collision = jnp.any(distances < 0)
        return collision, {'min_distance': min_distance}
