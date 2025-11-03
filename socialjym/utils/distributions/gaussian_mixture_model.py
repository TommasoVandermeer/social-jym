import jax.numpy as jnp
from jax import random, jit, vmap, lax
from jax.scipy.special import logsumexp
from functools import partial

from socialjym.utils.distributions.base_distribution import BaseDistribution

class GMM(BaseDistribution):
    def __init__(self, n_dimensions:int, n_components:int, epsilon=1e-6) -> None:
        """
        This is a general n-dimensional Gaussian Mixture Model (GMM). When calling any method of this class, the distribution
        dict must contain the following keys: ["means", "logvariances", "weights"].
        WARNING: logvariances are the log of the diagonal of the covariance matrices.
        WARNING: Weights must be positive and sum to 1.

        parameters:
        - n_dimensions: Number of dimensions of each Gaussian component
        - n_components: Number of Gaussian components in the mixture
        - epsilon: Small value to add to variance for numerical stability
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
        variance = jnp.exp(distribution["logvariances"])
        weights = distribution["weights"]
        key1, key2 = random.split(key)
        component = random.categorical(key1, jnp.log(weights))
        return means[component] + lax.sqrt(variance[component]) * random.normal(key2, shape=(self.n_dimensions,))

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(GMM.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, sample:jnp.ndarray):
        means = distribution["means"]  # shape: (n_components, n_dimensions)
        variance = jnp.exp(distribution["logvariances"])  # shape: (n_components, n_dimensions)
        weights = distribution["weights"]  # shape: (n_components,)
        @jit
        def _component_logp(mean, variance, weight):
            # log N(x | mean, cov) + log(weight)
            log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance + self.epsilon))
            log_prob += -0.5 * jnp.sum((sample - mean) ** 2 / (variance + self.epsilon))
            log_prob += jnp.log(weight + self.epsilon)
            return log_prob

        logps = vmap(_component_logp)(means, variance, weights)
        return -logsumexp(logps)

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, samples:jnp.ndarray):
        return vmap(GMM.neglogp, in_axes=(None, None, 0))(self, distribution, samples)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, sample:jnp.ndarray):
        return -self.neglogp(distribution, sample)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, samples:jnp.ndarray):
        return vmap(GMM.logp, in_axes=(None, None, 0))(self, distribution, samples)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, sample:jnp.ndarray):
        return jnp.exp(self.logp(distribution, sample))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, samples:jnp.ndarray):
        return vmap(GMM.p, in_axes=(None, None, 0))(self, distribution, samples)
    
    @partial(jit, static_argnames=("self"))
    def neglogp_single_component(self, distribution:dict, sample:jnp.ndarray, component_idx:int):
        mean = distribution["means"][component_idx]
        variance = jnp.exp(distribution["logvariances"][component_idx])
        log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance + self.epsilon))
        log_prob += -0.5 * jnp.sum((sample - mean) ** 2 / (variance + self.epsilon))
        return -log_prob
    
    @partial(jit, static_argnames=("self"))
    def batch_neglogp_single_component(self, distribution:dict, sample:jnp.ndarray):
        """
        Compute the negative log probabilities of the sample under each single component of the GMM.
        Returns an array of shape (n_components,).
        """
        return vmap(GMM.neglogp_single_component, in_axes=(None, None, None, 0))(self, distribution, sample, jnp.arange(self.n_components))
    
    @partial(jit, static_argnames=("self"))
    def logp_single_component(self, distribution:dict, sample:jnp.ndarray, component_idx:int):
        return -self.neglogp_single_component(distribution, sample, component_idx)
    
    @partial(jit, static_argnames=("self"))
    def batch_logp_single_component(self, distribution:dict, sample:jnp.ndarray):
        """
        Compute the log probabilities of the sample under each single component of the GMM.
        Returns an array of shape (n_components,).
        """
        return vmap(GMM.logp_single_component, in_axes=(None, None, None, 0))(self, distribution, sample, jnp.arange(self.n_components))
    
    @partial(jit, static_argnames=("self"))
    def p_single_component(self, distribution:dict, sample:jnp.ndarray, component_idx:int):
        return jnp.exp(self.logp_single_component(distribution, sample, component_idx))

    @partial(jit, static_argnames=("self"))
    def batch_p_single_component(self, distribution:dict, sample:jnp.ndarray):
        """
        Compute the probabilities of the sample under each single component of the GMM.
        Returns an array of shape (n_components,).
        """
        return vmap(GMM.p_single_component, in_axes=(None, None, None, 0))(self, distribution, sample, jnp.arange(self.n_components))
    
    @partial(jit, static_argnames=("self"))
    def batch_samples_batch_p_single_component(self, distribution:dict, samples:jnp.ndarray):
        """
        Compute the probabilities of each sample under each single component of the GMM.
        Returns an array of shape (n_samples, n_components).
        """
        return vmap(GMM.batch_p_single_component, in_axes=(None, None, 0))(self, distribution, samples)
    
class BivariateGMM(BaseDistribution):
    def __init__(self, n_components:int, epsilon=1e-6) -> None:
        """
        This is a bivariate Gaussian Mixture Model (GMM). When calling any method of this class, the distribution
        dict must contain the following keys: ["means", "logsigmas", "correlations", "weights"].
        WARNING: Weights must be positive and sum to 1.

        parameters:
        - n_components: Number of Gaussian components in the mixture
        - epsilon: Small value to add to variance for numerical stability
        """
        self.name="BivariateGMM"
        self.n_dimensions = 2
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
    def covariances(self, distribution:dict) -> float:
        logsigmas = distribution["logsigmas"]  + self.epsilon # shape: (n_components, 2)   
        correlations = distribution["correlations"]  # shape: (n_components,)
        return vmap(
            lambda sigmas, corr: jnp.array([[sigmas[0] ** 2, corr * sigmas[0] * sigmas[1]], [corr * sigmas[0] * sigmas[1], sigmas[1] ** 2]]), 
            in_axes=(0,0))(jnp.exp(logsigmas), correlations)  # shape: (n_components, 2, 2)

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        means = distribution["means"]
        covariances = self.covariances(distribution)
        weights = distribution["weights"]
        key1, key2 = random.split(key)
        component = random.categorical(key1, jnp.log(weights))
        return random.multivariate_normal(key2, means[component], covariances[component])

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(BivariateGMM.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, sample:jnp.ndarray):
        means = distribution["means"]  # shape: (n_components, 2)
        covariances = self.covariances(distribution)  # shape: (n_components, 2, 2)
        weights = distribution["weights"]  # shape: (n_components,)
        @jit
        def _component_logp(mean, covariance, weight):
            # log N(x | mean, cov) + log(weight)
            inv_cov = jnp.linalg.inv(covariance)
            det_cov = jnp.linalg.det(covariance)
            diff = sample - mean
            log_prob = -0.5 * jnp.log((2 * jnp.pi) ** 2 * det_cov + self.epsilon)
            log_prob += -0.5 * diff.T @ inv_cov @ diff
            log_prob += jnp.log(weight + self.epsilon)
            return log_prob

        logps = vmap(_component_logp)(means, covariances, weights)
        return -logsumexp(logps)

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, samples:jnp.ndarray):
        return vmap(BivariateGMM.neglogp, in_axes=(None, None, 0))(self, distribution, samples)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, sample:jnp.ndarray):
        return -self.neglogp(distribution, sample)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, samples:jnp.ndarray):
        return vmap(BivariateGMM.logp, in_axes=(None, None, 0))(self, distribution, samples)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, sample:jnp.ndarray):
        return jnp.exp(self.logp(distribution, sample))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, samples:jnp.ndarray):
        return vmap(BivariateGMM.p, in_axes=(None, None, 0))(self, distribution, samples)