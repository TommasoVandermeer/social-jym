from abc import ABC, abstractmethod
from functools import partial
from jax import jit

class BaseVNetReplayBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0))
    def add(
        self,
        buffer_state: dict,
        experience: tuple,
        idx: int,
    ):
        vnet_input, target = experience
        filling_idx = idx % self.buffer_size

        buffer_state["vnet_inputs"] = buffer_state["vnet_inputs"].at[filling_idx].set(vnet_input)
        buffer_state["targets"] = buffer_state["targets"].at[filling_idx].set(target)

        return buffer_state

    @abstractmethod
    def sample(self):
        pass