from abc import ABC, abstractmethod

class BaseReward(ABC):
    def __init__(self) -> None:
        pass

    # --- Private methods ---

    @abstractmethod
    def __call__(self, state, action) -> tuple:
        pass

    # --- Public methods ---

    def get_parameters(self) -> tuple:
        """
        This function returns the parameters of the reward function as a dictionary.

        output:
        - params: dictionary containing the parameters of the reward functions.
        """
        return self.__dict__.copy()