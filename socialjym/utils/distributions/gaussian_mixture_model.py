import jax.numpy as jnp
from jax import random, jit, vmap, lax
from jax.scipy.special import logsumexp
from functools import partial

from socialjym.utils.distributions.base_distribution import BaseDistribution

class GMM(BaseDistribution):
    def __init__(self, n_dimensions:int, n_components:int, epsilon=1e-6) -> None:
        """
        This is a Gaussian Mixture Model (GMM). When calling any method of this class, the distribution
        dict must contain the following keys: ["means", "variances", "weights"].
        WARNING: Variances must be positive. They are the diagonal of the covariance matrices.
        WARNING: Weights must be positive and sum to 1.

        parameters:
        - n_dimensions: Number of dimensions of each Gaussian component
        - n_components: Number of Gaussian components in the mixture
        - epsilon: Small value to add to variances for numerical stability
        """
        self.name="GMM"
        self.n_dimensions = n_dimensions
        self.n_components = n_components
        self.epsilon = epsilon

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        means = distribution["means"]
        weights = distribution["weights"]
        return jnp.sum(means * weights[:, None], axis=0)

    @partial(jit, static_argnames=("self"))
    def var(self, distribution:dict) -> jnp.ndarray:
        pass

    @partial(jit, static_argnames=("self"))
    def entropy(self, distribution:dict) -> float:
        pass

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        means = distribution["means"]
        variances = jnp.exp(distribution["variances"])
        weights = distribution["weights"]
        key1, key2 = random.split(key)
        component = random.categorical(key1, jnp.log(weights))
        return means[component] + lax.sqrt(variances[component]) * random.normal(key2, shape=(self.n_dimensions,))

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(GMM.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, sample:jnp.ndarray):
        means = distribution["means"]  # shape: (n_components, n_dimensions)
        variances = jnp.exp(distribution["variances"])  # shape: (n_components, n_dimensions)
        weights = distribution["weights"]  # shape: (n_components,)
        @jit
        def _component_logp(mean, variance, weight):
            # log N(x | mean, cov) + log(weight)
            log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance + self.epsilon))
            log_prob += -0.5 * jnp.sum((sample - mean) ** 2 / (variance + self.epsilon))
            log_prob += jnp.log(weight + self.epsilon)
            return log_prob

        logps = vmap(_component_logp)(means, variances, weights)
        return -logsumexp(logps)

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(GMM.neglogp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(GMM.logp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(GMM.p, in_axes=(None, None, 0))(self, distribution, actions)