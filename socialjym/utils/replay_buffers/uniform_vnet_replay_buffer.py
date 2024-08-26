from jax import random, jit, vmap, tree_map, debug
from functools import partial

from .base_vnet_replay_buffer import BaseVNetReplayBuffer

class UniformVNetReplayBuffer(BaseVNetReplayBuffer):
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        super().__init__(buffer_size, batch_size)
        
    @partial(jit, static_argnames=("self"))
    def sample(self, key:random.PRNGKey, buffer_state:dict, current_buffer_size:int) -> tuple:

        @partial(vmap, in_axes=(0, None))
        def sample_batch(indexes, buffer):
            return tree_map(lambda x: x[indexes], buffer)

        key, subkey = random.split(key)
        indexes = random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=current_buffer_size)
        experiences = sample_batch(indexes, buffer_state)

        return experiences, subkey