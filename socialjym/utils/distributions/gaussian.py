import jax.numpy as jnp
from jax import random, jit, vmap, lax
from functools import partial

from socialjym.utils.distributions.base_distribution import BaseDistribution
from socialjym.envs.base_env import ROBOT_KINEMATICS

class Gaussian(BaseDistribution):
    def __init__(self) -> None:
        """
        This is a diagonal multivariate Gaussian distribution. When calling any method of this class, the distribution
        dict must contain the following keys: ["means", "logsigmas"]."
        """
        self.name="gaussian"
    
    @partial(jit, static_argnames=("self"))
    def entropy(self, distribution:dict) -> float:
        logsigmas = distribution["logsigmas"]
        return .5 * jnp.log(2*jnp.pi*jnp.exp(1)) * len(logsigmas)  + jnp.sum(logsigmas)

    @partial(jit, static_argnames=("self"))
    def sample(self, distribution:dict, key:random.PRNGKey):
        means = distribution["means"]
        sigmas = jnp.exp(distribution["logsigmas"])
        return means + sigmas * random.normal(key, shape=(len(means),))

    @partial(jit, static_argnames=("self"))
    def batch_sample(self, distribution:dict, keys:jnp.ndarray):
        return vmap(Gaussian.sample, in_axes=(None, None, 0))(self, distribution, keys)

    @partial(jit, static_argnames=("self"))
    def mean(self, distribution:dict) -> jnp.ndarray:
        return distribution["means"]

    @partial(jit, static_argnames=("self"))
    def var(self, distribution:dict) -> jnp.ndarray:
        return jnp.exp(2 * distribution["logsigmas"])

    @partial(jit, static_argnames=("self"))
    def neglogp(self, distribution:dict, action:jnp.ndarray):
        means = distribution["means"]
        logsigmas = distribution["logsigmas"]
        return 0.5 * jnp.sum((action - means)**2 / jnp.exp(logsigmas)**2) + jnp.sum(logsigmas) + jnp.log(2 * jnp.pi)

    @partial(jit, static_argnames=("self"))
    def batch_neglogp(self, distribution:dict, actions:jnp.ndarray):
        """
        Compute the negative log pdf value of a batch of actions and distirbutions.
        Vectorized over distributions and actions!!!
        """
        return vmap(Gaussian.neglogp, in_axes=(None, 0, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def logp(self, distribution:dict, action:jnp.ndarray):
        return -self.neglogp(distribution, action)

    @partial(jit, static_argnames=("self"))
    def batch_logp(self, distribution:dict, actions:jnp.ndarray):
        return vmap(Gaussian.logp, in_axes=(None, None, 0))(self, distribution, actions)

    @partial(jit, static_argnames=("self"))
    def p(self, distribution:dict, action:jnp.ndarray):
        return jnp.exp(self.logp(distribution, action))

    @partial(jit, static_argnames=("self"))
    def batch_p(self, distribution:dict, actions:jnp.ndarray):
        return vmap(Gaussian.p, in_axes=(None, None, 0))(self, distribution, actions)
    
    @partial(jit, static_argnames=("self"))
    def bound_action(
        self,
        sampled_action:jnp.ndarray,
        kinematics:int,
        v_max:float,
        wheels_distance:float,
    ) -> jnp.ndarray:
        """
        This function is only used to limit the values of a 2D Diagonal Gaussian within a given range. Defined by the kinematics of the robot.

        args:
        - sampled_action (jnp.ndarray): sampled action from the Gaussian distribution.
        - kinematics (int): index of the robot kinematics.
        - v_max (float): max linear velocity of the robot.
        - wheels_distance (float): distance between the wheels of the robot.

        returns:
        - constrained_action (jnp.ndarray): action bounded by the robot kinematics.
        """
        @jit
        def _unicycle_action(sampled_action, v_max, wheels_distance):
            ## Bound the final action with HARD CLIPPING
            # v, w = sampled_action
            # v = jnp.clip(v, 0, v_max)
            # w_max = (2 * (v_max - v)) / wheels_distance
            # w = jnp.clip(w, -w_max, w_max)
            # return jnp.array([v, w])
            ## Bound the final action with DISTANCE TO ORIGIN CLIPPING
            x, y = sampled_action
            cases = jnp.array([
                x <= 0,
                (y == 0) & (x > 0),
                (abs(y) <= 2*(v_max - x)/wheels_distance) & (x > 0),
                (y >= 0) & (y > 2*(v_max - x)/wheels_distance) & (x > 0),
                (y < 0) & (y < 2*(x - v_max)/wheels_distance) & (x > 0)
            ], dtype=jnp.int32)
            bounded_action = lax.switch(
                jnp.argmax(cases),
                [
                    lambda _: jnp.array([0., jnp.clip(y, -2*v_max/wheels_distance, 2*v_max/wheels_distance)]),
                    lambda _: jnp.array([jnp.min(jnp.array([v_max,x])), 0.]),
                    lambda _: jnp.array([x, y]),
                    lambda _: jnp.array([2*v_max*x/(y*wheels_distance+2*x), 2*v_max*y/(y*wheels_distance+2*x)]),
                    lambda _: jnp.array([-2*v_max*x/(y*wheels_distance-2*x), -2*v_max*y/(y*wheels_distance-2*x)]),
                ],
                None,
            )
            return bounded_action
        
        @jit
        def _holonomic_action(sampled_action, v_max):
            vx, vy = sampled_action
            norm = jnp.linalg.norm(jnp.array([vx, vy]))
            ## Bound the norm of the velocity with HARD CLIPPING
            scaling_factor = lax.cond(
                norm != 0,
                lambda _: jnp.minimum(norm, v_max) / norm,
                lambda _: 1.,
                None,
            )
            vx = vx * scaling_factor
            vy = vy * scaling_factor
            return jnp.array([vx, vy])

        constrained_action = lax.switch(
            kinematics,
            [ # Make sure these are coherent with ROBOT_KINEMATICS order (defined in socialjym.envs.base_env)
                lambda _: _holonomic_action(sampled_action, v_max),
                lambda _: _unicycle_action(sampled_action, v_max, wheels_distance),
            ],
            None,
        )
        return constrained_action
    
    @partial(jit, static_argnames=("self"))
    def batch_std(self, distributions:dict) -> jnp.ndarray:
        """
        Compute the standard deviations of a batch of Gaussian distributions.
        """
        return jnp.exp(distributions["logsigmas"])