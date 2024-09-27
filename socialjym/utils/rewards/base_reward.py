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
        return self.__dict__