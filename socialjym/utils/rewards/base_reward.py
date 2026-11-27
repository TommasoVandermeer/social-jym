from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
    from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision
    from socialjym.utils.terminations.robot_obstacle_collision import InstantRobotObstacleCollision

class BaseReward(ABC):
    # Attributes set by the reward functions (declared here for type checking)
    gamma: Any # A float, or one discount per reward term for multi-discount rewards
    time_limit: float
    v_max: float
    kinematics: int
    unique_gammas: tuple
    goal_reached_termination: "RobotReachedGoal"
    instant_collision_termination: "InstantRobotHumanCollision"
    instant_human_collision_termination: "InstantRobotHumanCollision"
    instant_obstacle_collision_termination: "InstantRobotObstacleCollision"

    def __init__(self, gamma:Union[float, list, tuple, jnp.ndarray] = 0.9,) -> None:
        self.gamma = gamma

    # --- Private methods ---

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple:
        pass

    # --- Public methods ---

    def get_parameters(self) -> dict:
        """
        This function returns the parameters of the reward function as a dictionary.

        output:
        - params: dictionary containing the parameters of the reward functions.
        """
        return self.__dict__.copy()