from abc import ABC, abstractmethod
from functools import partial
from jax import jit, vmap, tree_map, random, lax, debug
import jax.numpy as jnp

class BaseVNetReplayBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @partial(jit, static_argnames=("self"))
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

    @partial(jit, static_argnames=("self"))
    def iterate(self, buffer_state: dict, current_buffer_size: int, iteration:int) -> dict:
        
        @partial(vmap, in_axes=(0, None))
        def iterate_batch(indexes, buffer):
            return tree_map(lambda x: x[indexes], buffer)
        
        # If the indexes exceed the buffer size, the buffer will be iterated from the beginning
        indexes = lax.fori_loop(
            0, 
            self.batch_size, 
            lambda i, idx: idx.at[i].set(((iteration * self.batch_size + i) % current_buffer_size).astype(int)), 
            jnp.zeros((self.batch_size,), dtype=jnp.int32))

        experiences = iterate_batch(indexes, buffer_state)
        return experiences
    
    @partial(jit, static_argnames=("self"))
    def batch_iterate(self, buffer_state:dict, current_buffer_size:int, iterations:int) -> dict:
        return vmap(BaseVNetReplayBuffer.iterate, in_axes=(None, None, None, 0))(self, buffer_state, current_buffer_size, iterations)

    @partial(jit, static_argnames=("self"))
    def shuffle(self, buffer_state: dict, key: random.PRNGKey, times: int = 1) -> dict:
            
            @partial(vmap, in_axes=(0, None))
            def shuffle_batch(indexes, buffer):
                return tree_map(lambda x: x[indexes], buffer)

            @jit
            def _fori_body(i:int, val:tuple):
                indexes, key = val
                key, subkey = random.split(key)
                new_indexes = random.permutation(subkey, indexes)
                return new_indexes, key
            
            indexes = jnp.arange(self.buffer_size)
            val_end = lax.fori_loop(0, times, _fori_body, (indexes, key))
            indexes, key = val_end
            shuffled_buffer_state = shuffle_batch(indexes, buffer_state)
            
            return shuffled_buffer_state, key

    @abstractmethod
    def sample(self):
        pass