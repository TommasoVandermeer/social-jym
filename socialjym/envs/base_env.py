from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self) -> None:
        pass

    # --- Private methods ---

    @abstractmethod
    def _get_obs(self, state):
        pass

    @abstractmethod
    def _reset(self, key):
        pass

    # --- Public methods ---

    @abstractmethod
    def reset(self, key):
        pass

    @abstractmethod
    def step(self, env_state, action):
        pass