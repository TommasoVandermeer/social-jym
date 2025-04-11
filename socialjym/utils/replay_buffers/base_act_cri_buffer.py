from functools import partial
from jax import jit, vmap
from jax import numpy as jnp

from .base_vnet_replay_buffer import BaseVNetReplayBuffer

class BaseACBuffer(BaseVNetReplayBuffer):
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
        input: jnp.ndarray,
        critic_target: jnp.ndarray,
        sample_action: jnp.ndarray,
        idx: int,
    ):
        filling_idx = idx % self.buffer_size

        buffer_state["inputs"] = buffer_state["inputs"].at[filling_idx].set(input)
        buffer_state["critic_targets"] = buffer_state["critic_targets"].at[filling_idx].set(critic_target)
        buffer_state["sample_actions"] = buffer_state["sample_actions"].at[filling_idx].set(sample_action)

        return buffer_state
    
    @partial(jit, static_argnames=("self"))
    def batch_add(
        self,
        buffer_state: dict,
        inputs: jnp.ndarray,
        critic_targets: jnp.ndarray,
        sample_actions: jnp.ndarray,
        idxs: jnp.ndarray,
    ):
        filling_idxs = idxs % self.buffer_size

        buffer_state["inputs"] = buffer_state["inputs"].at[filling_idxs].set(inputs)
        buffer_state["critic_targets"] = buffer_state["critic_targets"].at[filling_idxs].set(critic_targets)
        buffer_state["sample_actions"] = buffer_state["sample_actions"].at[filling_idxs].set(sample_actions)

        return buffer_state
    
    @partial(jit, static_argnames=("self"))
    def empty(
        self,
        buffer_state: dict,
    ):
        buffer_state["inputs"] = buffer_state["inputs"].at[:,:,:].set(0)
        buffer_state["critic_targets"] = buffer_state["critic_targets"].at[:].set(0)
        buffer_state["sample_actions"] = buffer_state["sample_actions"].at[:,:].set(0)

        return buffer_state