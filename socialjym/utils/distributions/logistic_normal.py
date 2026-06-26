import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from jax.scipy.stats import norm

from socialjym.utils.distributions.base_distribution import BaseDistribution

class LogisticNormal(BaseDistribution):
    def __init__(self, epsilon:float=1e-6) -> None:
        """
        Logistic-Normal (Softmax-Gaussian) distribution to sample actions directly
        from a convex feasible space with an arbitrary number of vertices.
        
        The distribution dict must contain: ["locs", "log_scales", "vertices"].
        
        args:
        - epsilon (float): small value to avoid math overflow.
        """
        self.name = "logistic_normal"
        self.epsilon = epsilon
    
    @partial(jit, static_argnames=("self"))
    def entropy(self, distribution:dict) -> float:
        """
        Computes the entropy of the latent Gaussian distribution as a surrogate 
        for the Logistic-Normal entropy, which has no closed-form solution.
        This provides a highly stable regularization signal for RL.
        """
        log_scales = distribution["log_scales"]
        # Gaussian entropy: 0.5 * sum(1 + ln(2*pi*sigma^2))
        return 0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2.0 * log_scales)

    @partial(jit, static_argnames=("self"))
    def batch_entropy(self, distributions:dict) -> jnp.ndarray:
        return vmap(LogisticNormal.entropy, in_axes=(None, 0))(self, distributions)

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        vertices = distribution["vertices"]
        # Reparameterization trick for latent Gaussian
        z = locs + scales * random.normal(key, locs.shape)
        # Softmax projection to the simplex
        weights = jax.nn.softmax(z)
        # Map to the feasible region
        return jnp.dot(weights, vertices)

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(LogisticNormal.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        locs = distribution["locs"]
        vertices = distribution["vertices"]
        # Deterministic approximation using the mode of the latents
        mean_weights = jax.nn.softmax(locs)
        return jnp.dot(mean_weights, vertices)

    @partial(jit, static_argnames=("self"))
    def var(self, distribution:dict) -> jnp.ndarray:
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        vertices = distribution["vertices"]
        # Delta method approximation for the variance mapped through the softmax
        weights = jax.nn.softmax(locs)
        # Jacobian of softmax (diagonal minus outer product)
        J = jnp.diag(weights) - jnp.outer(weights, weights)
        # Variance of latents
        Sigma = jnp.diag(scales**2)
        # Covariance of weights
        cov_weights = J @ Sigma @ J.T
        # Covariance of final action
        cov_action = vertices.T @ cov_weights @ vertices
        return jnp.diag(cov_action)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, action:jnp.ndarray):
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        vertices = distribution["vertices"]
        # Construct the system M * w = y
        M = jnp.vstack((vertices.T, jnp.ones((len(vertices),))))
        y = jnp.append(action, 1.0)
        # Pseudo-inverse to support arbitrary number of vertices (V > D + 1)
        # Finds the minimum-norm weight vector solving the underdetermined system
        M_pinv = jnp.linalg.pinv(M)
        w_raw = jnp.dot(M_pinv, y)
        # Clip and normalize to ensure strict simplex bounds and avoid log(0)
        w_safe = jnp.clip(w_raw, self.epsilon, 1.0)
        w_safe = w_safe / jnp.sum(w_safe)
        # Inverse of softmax (up to a constant shift, setting c=0)
        z = jnp.log(w_safe)
        # Log probability of the latent Gaussian
        log_prob_z = jnp.sum(norm.logpdf(z, loc=locs, scale=scales))
        # Log determinant of the Jacobian for the softmax transformation
        log_det_jacobian = jnp.sum(z) 
        # log pi(a) = log p(z) - log|det(J)|
        log_prob_action = log_prob_z - log_det_jacobian
        return -log_prob_action

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.neglogp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.logp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.p, in_axes=(None, None, 0))(self, distribution, actions)
    
    @partial(jit, static_argnames=("self"))
    def batch_std(self, distributions:dict) -> jnp.ndarray:
        return jnp.sqrt(vmap(LogisticNormal.var, in_axes=(None, 0))(self, distributions))

    @partial(jit, static_argnames=("self"))
    def is_in_support(self, distribution:dict, action:jnp.ndarray) -> bool:
        vertices = distribution["vertices"]
        M = jnp.vstack((vertices.T, jnp.ones((len(vertices),))))
        y = jnp.append(action, 1.0)
        w_raw = jnp.dot(jnp.linalg.pinv(M), y)
        # Check if the pseudo-inverse solution represents a valid convex combination
        # Tolerance added for floating point inaccuracies
        return jnp.all(w_raw >= -1e-4) & jnp.isclose(jnp.sum(w_raw), 1.0, atol=1e-4)
    
    @partial(jit, static_argnames=("self"))
    def batch_is_in_support(self, distribution:dict, actions:jnp.ndarray) -> jnp.ndarray:
        return vmap(LogisticNormal.is_in_support, in_axes=(None, None, 0))(self, distribution, actions)