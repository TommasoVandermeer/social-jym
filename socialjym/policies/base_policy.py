from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    def __init__(self, discount) -> None:
        self.gamma = discount
        pass

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass