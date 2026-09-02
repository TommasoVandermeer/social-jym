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

    def evaluate_transition(self, state, action, info, dt, state_history=None):
        """Evaluate an executed transition while preserving legacy rewards.

        Environments with a finer internal integration step may provide the
        executed ``state_history``.  Existing rewards intentionally ignore it
        and retain their historical semantics; transition-aware rewards can
        override this method without changing the public environment API.
        """
        del state_history
        return self(state, action, info, dt)

    # --- Public methods ---

    def get_parameters(self) -> tuple:
        """
        This function returns the parameters of the reward function as a dictionary.

        output:
        - params: dictionary containing the parameters of the reward functions.
        """
        return self.__dict__.copy()
