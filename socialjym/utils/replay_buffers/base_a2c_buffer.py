from functools import partial
from jax import jit

from .base_vnet_replay_buffer import BaseVNetReplayBuffer

class BaseA2CBuffer(BaseVNetReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    # Overwrite the add method to include the right experience tuple
    @partial(jit, static_argnames=("self"))
    def add(
        self,
        buffer_state: dict,
        experience: tuple,
        idx: int,
    ):
        input, critic_target, sample_action = experience
        filling_idx = idx % self.buffer_size

        buffer_state["inputs"] = buffer_state["inputs"].at[filling_idx].set(input)
        buffer_state["critic_targets"] = buffer_state["critic_targets"].at[filling_idx].set(critic_target)
        buffer_state["sample_actions"] = buffer_state["sample_actions"].at[filling_idx].set(sample_action)

        return buffer_state
    
    def sample(self):
        pass