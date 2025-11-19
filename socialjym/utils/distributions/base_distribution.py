from abc import ABC, abstractmethod
from jax import random
import jax.numpy as jnp
from functools import partial
from jax import jit

DISTRIBUTIONS = [
    "gaussian",
    "GMM",
    "dirichlet",
    "BivariateGMM",
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

    @abstractmethod
    def var(self, distribution:dict):
        pass

    @partial(jit, static_argnames=("self"))
    def std(self, distribution:dict) -> jnp.ndarray:
        return jnp.sqrt(self.var(distribution))