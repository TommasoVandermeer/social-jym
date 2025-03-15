import jax.numpy as jnp
from jax import random, jit, vmap, lax
from functools import partial
from jax.scipy.special import gamma, digamma, gammaln

from socialjym.utils.distributions.base_distribution import BaseDistribution

class DirichletBernoulli(BaseDistribution):
    def __init__(self, vmax:float, wheels_distance:float, epsilon:float=1e-6) -> None:
        """"
        This is a Dirichlet distribution combined with a bernoulli to sample actions directly
        from the feasible (v,w) action space. When calling any method of this class, the distribution
        dict must contain the following keys: ["alphas", "p"].

        args:
        - vmax (float): max linear velocity of the robot.
        - wheels_distance (float): distance between the wheels of the robot.
        - epsilon (float): small value to avoid math overflow.
        """
        self.name = "dirichlet-bernoulli"
        self.vmax = vmax
        self.wheels_distance = wheels_distance
        self.epsilon = epsilon
    
    @partial(jit, static_argnames=("self"))
    def entropy(self, distribution:dict) -> float:
        alphas = distribution["alphas"]
        p = distribution["p"]
        p = jnp.clip(p, 0., 1.) # Just in case, we clip p to its feasible values
        dirichlet_entropy = jnp.log(jnp.prod(gamma(alphas))+self.epsilon) - jnp.log(gamma(jnp.sum(alphas))+self.epsilon) + (jnp.sum(alphas) - 3) * digamma(jnp.sum(alphas)) - jnp.sum((alphas - 1) * digamma(alphas))
        binomial_entropy = -p * jnp.log(p + self.epsilon) - (1-p) * jnp.log(1-p + self.epsilon)
        return dirichlet_entropy + binomial_entropy

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        alphas = distribution["alphas"]
        p = distribution["p"]
        p = jnp.clip(p, 0., 1.) # Just in case, we clip p to its feasible values
        key1, key2 = random.split(key)
        # Sample from dirichlet distribution
        sample = random.dirichlet(key1, alphas)
        # Scale sample to correct range
        sample = sample.at[0].set(sample[0] * self.vmax)
        sample = sample.at[1].set(sample[1] * self.vmax * 2 / self.wheels_distance)
        # Sample w sign
        sign = random.binomial(key2, n=1, p=p) * 2 -1
        sample = sample.at[1].set(sample[1] * sign)
        return sample[:2] # We only consider the first two values of the sample, the other will be (1 - v - w)

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(DirichletBernoulli.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        alphas = distribution["alphas"]
        p = distribution["p"]
        mean_v = (alphas[0] / jnp.sum(alphas)) * self.vmax
        mean_w = (((alphas[1] / jnp.sum(alphas)) * self.vmax * 2) / self.wheels_distance) * (2 * p - 1)
        return jnp.array([mean_v, mean_w])

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, action:jnp.ndarray):
        alphas = distribution["alphas"]
        p = distribution["p"]
        descaled_v = action[0] / self.vmax
        descaled_w = jnp.abs(action[1] * self.wheels_distance / (self.vmax * 2)) # We consider only positive values of y (the binomial changes the sign)
        realization = jnp.array([descaled_v, descaled_w, 1-(descaled_v+descaled_w)])
        log_pdf_value_dirichlet = jnp.log(gamma(jnp.sum(alphas))+self.epsilon) - jnp.sum(jnp.log(gamma(alphas)+self.epsilon)) + jnp.sum((alphas - 1) * jnp.log(realization + self.epsilon))
        # log_pdf_value_dirichlet = gammaln(jnp.sum(alphas)) - jnp.sum(gammaln(alphas)) + jnp.sum((alphas - 1) * jnp.log(realization + self.epsilon))
        case = jnp.argmax(jnp.array([action[1] > 0, action[1] < 0, action[1] == 0], dtype=jnp.int32))
        log_pdf_value_binomial = lax.switch(
            case, 
            [
                lambda _: jnp.log(p + self.epsilon), 
                lambda _: jnp.log(1-p + self.epsilon),
                lambda _: 0.,
            ],
            None,
        )
        neg_log_pdf_value = lax.cond(
            (action[0] > self.vmax) | (action[0] < 0),# | (jnp.abs(action[1]) > (2*self.vmax/self.wheels_distance -2*action[0]/self.wheels_distance)),
            lambda _: jnp.inf,
            lambda _: -log_pdf_value_dirichlet - log_pdf_value_binomial,
            None,
        )
        return neg_log_pdf_value

    @partial(jit, static_argnames=("self"))
    def normalize_alphas(self, alphas:jnp.ndarray, concentration:float) -> jnp.ndarray:
        return alphas / jnp.sum(alphas) * concentration

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        """
        Compute the negative log pdf value of a batch of actions and distirbutions.
        Vectorized over distributions and actions!!!
        """
        return vmap(DirichletBernoulli.neglogp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(DirichletBernoulli.logp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(DirichletBernoulli.p, in_axes=(None, None, 0))(self, distribution, actions)