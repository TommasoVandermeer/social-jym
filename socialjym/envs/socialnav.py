import jax.numpy as jnp
from jax import random, jit
from functools import partial

class SocialNav:
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL.
    """
    def __init__(self):
        pass

    @partial(jit, static_argnums=(0,))
    def step(self, state, action):
        pass