from abc import ABC, abstractmethod

DISTRIBUTIONS = [
    "gaussian",
    "dirichlet-bernoulli",
]

class BaseDistribution(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def neglogp(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def entropy(self):
        pass
