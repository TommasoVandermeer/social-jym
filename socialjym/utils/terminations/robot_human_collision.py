from jax import jit
import jax.numpy as jnp
from functools import partial

from socialjym.utils.terminations.base_termination import BaseTermination, batch_point_to_line_distance

class InstantRobotHumanCollision(BaseTermination):
    """
    Termination condition based on robot-human collisions.
    The episode ends if the robot collides with any human at a specific instant.
    """
    def __init__(self):
        super().__init__('instant_robot_human_collision')

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        robot_pos:jnp.ndarray, 
        robot_radius:jnp.ndarray,
        human_poses:jnp.ndarray,
        human_radii:jnp.ndarray
    ) -> tuple[bool, dict]:
        """
        Computes whether a collision between the robot and any human has occurred at a specific instant.

        args:
        - robot_pos: jnp.ndarray, shape=(2,), dtype=jnp.float32. The position of the robot.
        - robot_radius: jnp.ndarray, shape=(). The radius of the robot.
        - human_poses: jnp.ndarray, shape=(N,2), dtype=jnp.float32. The positions of the humans.
        - human_radii: jnp.ndarray, shape=(N,), dtype=jnp.float32. The radii of the humans.

        output:
        - collision: bool. True if a collision has occurred, False otherwise.
        - info: dict. Additional information about the collision status.
        """
        distances = jnp.linalg.norm(human_poses - robot_pos, axis=1) - (human_radii + robot_radius)
        min_distance = jnp.max(jnp.array([jnp.min(distances),0.]))
        collision = jnp.any(distances < 0)
        return collision, {'min_distance': min_distance}
    
class IntervalRobotHumanCollision(BaseTermination):
    """
    Termination condition based on robot-human collisions.
    The episode ends if the robot collides with any human within an interval,
    ASSUMING ALL AGENTS move in a straight line.
    """
    def __init__(self):
        super().__init__('interval_robot_human_collision')

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        robot_pos:jnp.ndarray, 
        next_robot_pos:jnp.ndarray,
        robot_radius:jnp.ndarray,
        human_poses:jnp.ndarray,
        next_human_poses:jnp.ndarray,
        human_radii:jnp.ndarray
    ) -> tuple[bool, dict]:
        """
        Computes whether a collision between the robot and any human has occurred at a specific instant.

        args:
        - robot_pos: jnp.ndarray, shape=(2,), dtype=jnp.float32. The position of the robot.
        - next_robot_pos: jnp.ndarray, shape=(2,), dtype=jnp.float32. The next position of the robot.
        - robot_radius: jnp.ndarray, shape=(). The radius of the robot.
        - human_poses: jnp.ndarray, shape=(N,2), dtype=jnp.float32. The positions of the humans.
        - next_human_poses: jnp.ndarray, shape=(N,2), dtype=jnp.float32. The next positions of the humans.
        - human_radii: jnp.ndarray, shape=(N,), dtype=jnp.float32. The radii of the humans.

        output:
        - collision: bool. True if a collision has occurred, False otherwise.
        - info: dict. Additional information about the collision status.
        """
        distances = batch_point_to_line_distance(jnp.zeros((len(human_poses),2)), human_poses - robot_pos, next_human_poses - next_robot_pos) - (human_radii + robot_radius)
        min_distance = jnp.max(jnp.array([jnp.min(distances),0]))
        collision = jnp.any(distances < 0)
        return collision, {'min_distance': min_distance}