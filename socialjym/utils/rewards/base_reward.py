from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Union

class BaseReward(ABC):
    def __init__(self, gamma:Union[float, list, tuple, jnp.ndarray] = 0.9,) -> None:
        self.gamma = gamma

    # --- Private methods ---

    @abstractmethod
    def __call__(self, state, action) -> tuple:
        pass

    def transition(self, old_state, new_state, intermediate_states, action, info, dt):
        """Evaluate an executed transition.

        Reward implementations that do not yet provide transition-aware
        semantics retain their historical predictive behavior.
        """
        return self(old_state, action, info, dt)

    # --- Public methods ---

    def get_parameters(self) -> tuple:
        """
        This function returns the parameters of the reward function as a dictionary.

        output:
        - params: dictionary containing the parameters of the reward functions.
        """
        return self.__dict__.copy()
