from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, jit, vmap, nn, debug, lax
from functools import partial

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
