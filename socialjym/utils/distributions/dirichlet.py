import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug
from functools import partial
from jax.scipy.special import digamma, gammaln
from jax.scipy.stats.dirichlet import logpdf

from socialjym.utils.distributions.base_distribution import BaseDistribution

class Dirichlet(BaseDistribution):
    def __init__(self, vmax:float, wheels_distance:float, epsilon:float=1e-6) -> None:
        """"
        This is a Dirichlet distribution combined with a to sample actions directly
        from the feasible (v,w) action space. When calling any method of this class, the distribution
        dict must contain the following keys: ["alphas"].

        args:
        - vmax (float): max linear velocity of the robot.
        - wheels_distance (float): distance between the wheels of the robot.
        - epsilon (float): small value to avoid math overflow.
        """
        self.name = "dirichlet"
        self.vmax = vmax
        self.wheels_distance = wheels_distance
        self.epsilon = epsilon
        self.v = jnp.array([[0,2*self.vmax/self.wheels_distance],[0,-2*self.vmax/self.wheels_distance],[self.vmax,0]])
    
    @partial(jit, static_argnames=("self"))
    def entropy(self, distribution:dict) -> float:
        alphas = distribution["alphas"]
        concentration = jnp.sum(alphas)
        lnB = jnp.sum(gammaln(alphas)) - gammaln(jnp.sum(alphas))
        dirichlet_entropy = lnB + (concentration - len(alphas)) * digamma(concentration) - jnp.sum((alphas - 1) * digamma(alphas))
        return dirichlet_entropy

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        alphas = distribution["alphas"]
        # Sample from dirichlet distribution
        sample = random.dirichlet(key, alphas)
        # Map to the feasible region
        return jnp.dot(sample, self.v)

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(Dirichlet.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        alphas = distribution["alphas"]
        mean_xi = alphas / jnp.sum(alphas)
        return jnp.dot(mean_xi, self.v)

    @partial(jit, static_argnames=("self"))
    def var(self, distribution:dict) -> jnp.ndarray:
        alphas = distribution["alphas"]
        concentration = jnp.sum(alphas)
        covar_xi = jnp.array([
            [alphas[0] * (concentration - alphas[0]), -alphas[0] * alphas[1], -alphas[0] * alphas[2]],
            [-alphas[1] * alphas[0], alphas[1] * (concentration - alphas[1]), -alphas[1] * alphas[2]],
            [-alphas[2] * alphas[0], -alphas[2] * alphas[1], alphas[2] * (concentration - alphas[2])]
        ]) / (concentration**2 * (concentration + 1))
        return jnp.diag(self.v.T @ covar_xi @ self.v)

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, action:jnp.ndarray):
        alphas = distribution["alphas"]
        sample = jnp.linalg.solve(jnp.vstack((self.v.T,jnp.ones((len(self.v),)))), jnp.append(action, 1.))
        # Avoid inf computation
        sample = jnp.clip(sample, 0.+self.epsilon, 1.-self.epsilon) / jnp.sum(jnp.clip(sample, 0.+self.epsilon, 1.-self.epsilon))
        return - logpdf(sample, alphas)

    @partial(jit, static_argnames=("self"))
    def normalize_alphas(self, alphas:jnp.ndarray, concentration:float) -> jnp.ndarray:
        return alphas / jnp.sum(alphas) * concentration

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        """
        Compute the negative log pdf value of a batch of actions and distirbutions.
        Vectorized over distributions and actions!!!
        """
        return vmap(Dirichlet.neglogp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(Dirichlet.logp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(Dirichlet.p, in_axes=(None, None, 0))(self, distribution, actions)
    
    @partial(jit, static_argnames=("self"))
    def batch_std(self, distributions:dict) -> jnp.ndarray:
        """
        Compute the standard deviations of a batch of Dirichlet distributions.
        """
        return vmap(Dirichlet.std, in_axes=(None,0))(self, distributions)