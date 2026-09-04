import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from jax.scipy.stats import norm
from jax.scipy.special import xlogy

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

    def _contrast_basis(self, latent_dim:int, n_vertices:int, dtype) -> jnp.ndarray:
        """Map identifiable latent coordinates to zero-sum simplex logits.

        JESSI uses a triangle, so two Gaussian coordinates are sufficient for
        its three convex weights.  The legacy identity case is retained for
        other policies until they migrate their actor heads.
        """
        if latent_dim == n_vertices:
            return jnp.eye(n_vertices, dtype=dtype)
        if latent_dim == 2 and n_vertices == 3:
            return jnp.array(
                [
                    [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0)],
                    [-1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0)],
                    [0.0, -2.0 / jnp.sqrt(6.0)],
                ],
                dtype=dtype,
            )
        raise ValueError(
            f"Unsupported LogisticNormal dimensions: {latent_dim} latent "
            f"coordinates for {n_vertices} vertices"
        )

    def latent_to_logits(self, distribution:dict, latent:jnp.ndarray) -> jnp.ndarray:
        vertices = distribution["vertices"]
        basis = self._contrast_basis(latent.shape[-1], vertices.shape[-2], latent.dtype)
        return latent @ basis.T

    def weights(self, distribution:dict, latent:jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softmax(self.latent_to_logits(distribution, latent), axis=-1)

    def mean_weights(self, distribution:dict) -> jnp.ndarray:
        return self.weights(distribution, distribution["locs"])
    
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
    def weight_entropy(self, distribution:dict) -> float:
        """
        Computes the entropy of the latent Gaussian distribution as a surrogate 
        for the Logistic-Normal entropy, which has no closed-form solution.
        This provides a highly stable regularization signal for RL.
        """
        weights = self.mean_weights(distribution)
        return -jnp.sum(xlogy(weights, weights))

    @partial(jit, static_argnames=("self"))
    def batch_weight_entropy(self, distributions:dict) -> jnp.ndarray:
        return vmap(LogisticNormal.weight_entropy, in_axes=(None, 0))(self, distributions)

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        """
        Returns LATENT z. 
        """
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        return locs + scales * random.normal(key, locs.shape)

    @partial(jit, static_argnames=("self"))
    def to_env_action(self, distribution:dict, latent_action:jnp.ndarray) -> jnp.ndarray:
        """
        Maps latent z to action (v,w)
        """
        vertices = distribution["vertices"]
        weights = self.weights(distribution, latent_action)
        return jnp.dot(weights, vertices)

    @partial(jit, static_argnames=("self"))
    def batch_to_env_action(self, distributions:dict, latent_actions:jnp.ndarray) -> jnp.ndarray:
        return vmap(LogisticNormal.to_env_action, in_axes=(None, 0, 0))(
            self, distributions, latent_actions
        )

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(LogisticNormal.sample, in_axes=(None, 0, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        locs = distribution["locs"]
        vertices = distribution["vertices"]
        # Deterministic approximation using the mode of the latents
        mean_weights = self.weights(distribution, locs)
        return jnp.dot(mean_weights, vertices)

    @partial(jit, static_argnames=("self"))
    def var(self, distribution:dict) -> jnp.ndarray:
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        vertices = distribution["vertices"]
        # Delta method approximation for the variance mapped through the softmax
        basis = self._contrast_basis(locs.shape[-1], vertices.shape[-2], locs.dtype)
        weights = self.weights(distribution, locs)
        # Jacobian of softmax (diagonal minus outer product)
        J = (jnp.diag(weights) - jnp.outer(weights, weights)) @ basis
        # Variance of latents
        Sigma = jnp.diag(scales**2)
        # Covariance of weights
        cov_weights = J @ Sigma @ J.T
        # Covariance of final action
        cov_action = vertices.T @ cov_weights @ vertices
        return jnp.diag(cov_action)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, latent_action:jnp.ndarray):
        """
        Computes log-probability from latent z. 
        """
        locs = distribution["locs"]
        scales = jnp.exp(distribution["log_scales"])
        return -jnp.sum(norm.logpdf(latent_action, loc=locs, scale=scales))

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.neglogp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.logp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(LogisticNormal.p, in_axes=(None, 0, 0))(self, distribution, actions)
    
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
        return vmap(LogisticNormal.is_in_support, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def project_to_support(self, distribution:dict, action:jnp.ndarray) -> jnp.ndarray:
        """Project an action onto the closest point of a triangular support."""
        vertices = distribution["vertices"]
        edge_starts = vertices
        edge_ends = jnp.roll(vertices, shift=-1, axis=0)
        edge_vectors = edge_ends - edge_starts
        denominators = jnp.maximum(jnp.sum(edge_vectors**2, axis=-1), self.epsilon)
        coefficients = jnp.clip(
            jnp.sum((action - edge_starts) * edge_vectors, axis=-1) / denominators,
            0.0,
            1.0,
        )
        candidates = edge_starts + coefficients[:, None] * edge_vectors
        nearest = candidates[jnp.argmin(jnp.sum((candidates - action) ** 2, axis=-1))]
        return jnp.where(self.is_in_support(distribution, action), action, nearest)

    @partial(jit, static_argnames=("self"))
    def batch_project_to_support(self, distributions:dict, actions:jnp.ndarray) -> jnp.ndarray:
        return vmap(LogisticNormal.project_to_support, in_axes=(None, 0, 0))(
            self, distributions, actions
        )
