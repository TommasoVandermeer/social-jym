import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial

from socialjym.utils.distributions.base_distribution import BaseDistribution

class Gaussian(BaseDistribution):
    def __init__(self) -> None:
        "This is a diagonal Gaussian distribution."
        pass
    
    @partial(jit, static_argnames=("self"))
    def entropy(self, logsigmas:jnp.ndarray):
        return jnp.log(2*jnp.pi*jnp.exp(1)) + jnp.sum(logsigmas)

    @partial(jit, static_argnames=("self"))
    def sample(self, means:jnp.ndarray, sigmas:jnp.ndarray, key:random.PRNGKey):
        key1, key2 = random.split(key)
        return means + sigmas * jnp.array([random.normal(key1), random.normal(key2)])

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, means:jnp.ndarray, sigmas:jnp.ndarray, keys:jnp.ndarray):
        return vmap(Gaussian.sample, in_axes=(None, None, None, 0))(self, means, sigmas, keys)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, means:jnp.ndarray, logsigmas:jnp.ndarray, action:jnp.ndarray):
        return 0.5 * jnp.sum((action - means)**2 / jnp.exp(logsigmas)**2) + jnp.sum(logsigmas) + jnp.log(2 * jnp.pi)

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, means:jnp.ndarray, logsigmas:jnp.ndarray, actions:jnp.ndarray):
        return vmap(Gaussian.neglogp, in_axes=(None, None, None, 0))(self, means, logsigmas, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, means:jnp.ndarray, logsigmas:jnp.ndarray, action:jnp.ndarray):
        return -self.neglogp(means, logsigmas, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, means:jnp.ndarray, logsigmas:jnp.ndarray, actions:jnp.ndarray):
        return vmap(Gaussian.logp, in_axes=(None, None, None, 0))(self, means, logsigmas, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, means:jnp.ndarray, logsigmas:jnp.ndarray, action:jnp.ndarray):
        return jnp.exp(self.logp(means, logsigmas, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, means:jnp.ndarray, logsigmas:jnp.ndarray, actions:jnp.ndarray):
        return vmap(Gaussian.p, in_axes=(None, None, None, 0))(self, means, logsigmas, actions)