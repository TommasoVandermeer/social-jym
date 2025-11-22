from jax import jit
import jax.numpy as jnp
from functools import partial

from socialjym.utils.terminations.base_termination import BaseTermination

class RobotReachedGoal(BaseTermination):
    """
    Termination condition based on the robot reaching its goal.
    The episode ends if the robot reaches its goal position.
    """
    def __init__(self):
        super().__init__('robot_reached_goal')

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        robot_pos:jnp.ndarray, 
        robot_radius:jnp.ndarray,
        robot_goal:jnp.ndarray
    ) -> tuple[bool, dict]:
        """
        Computes whether a collision between the robot and any human has occurred at a specific instant.

        args:
        - robot_pos: jnp.ndarray, shape=(2,), dtype=jnp.float32. The position of the robot.
        - robot_radius: jnp.ndarray, shape=(). The radius of the robot.
        - robot_goal: jnp.ndarray, shape=(2,), dtype=jnp.float32. The goal position of the robot.

        output:
        - reached_goal: bool. True if the robot has reached its goal, False otherwise.
        - info: dict. Additional information about the goal reaching status.
        """
        distance = jnp.linalg.norm(robot_goal - robot_pos) - robot_radius
        reached_goal = distance < 0
        return reached_goal, {'distance': jnp.max(jnp.array([distance,0.]))}