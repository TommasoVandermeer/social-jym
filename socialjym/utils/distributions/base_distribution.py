from abc import ABC, abstractmethod
from jax import random
import jax.numpy as jnp

DISTRIBUTIONS = [
    "gaussian",
    "dirichlet-bernoulli",
]

class BaseDistribution(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def neglogp(self, distribution:dict, action:jnp.ndarray):
        pass

    @abstractmethod
    def sample(self, distribution:dict, key:random.PRNGKey):
        pass

    @abstractmethod
    def entropy(self, distribution:dict):
        pass

    @abstractmethod
    def mean(self, distribution:dict):
        pass