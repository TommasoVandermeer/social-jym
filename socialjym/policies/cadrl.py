import jax.numpy as jnp
from jax import random, jit, lax, debug, nn
from functools import partial
import haiku as hk

from .base_policy import BasePolicy

VN_PARAMS = {
    "output_sizes": [150, 100, 100, 1],
    "activation": nn.relu,
    "activate_final": False
}

@hk.transform
def value_network(x):
    mlp = hk.nets.MLP(**VN_PARAMS)
    return mlp(x)

class CADRL(BasePolicy):
    def __init__(self, v_max=1., gamma=0.9, dt=0.25, speed_samples=5, rotation_samples=16) -> None:
        # Configurable attributes
        super().__init__(gamma)
        self.v_max = v_max
        self.dt = dt
        self.speed_samples = speed_samples
        self.rotation_samples = rotation_samples
        self.action_space = self._build_action_space(v_max, self.speed_samples, self.rotation_samples)
        # Default attributes
        self.vnet_input_size = 13
        self.model = value_network

    @partial(jit, static_argnames=("self"))
    def _build_action_space(self, v_max:float, speed_samples:int, rotation_samples:int) -> jnp.ndarray:
        speeds = jnp.array([(jnp.exp((i + 1) / speed_samples) - 1) / (jnp.e - 1) * v_max for i in range(speed_samples)])
        rotations = jnp.linspace(0, 2 * jnp.pi, rotation_samples, endpoint=False)
        action_space = jnp.empty((speed_samples * rotation_samples + 1,2))
        action_space = action_space.at[0].set(jnp.array([0, 0])) # First action is to stay still
        action_space = lax.fori_loop(0,
                                     len(action_space), 
                                     lambda i, acts: acts.at[i].set(jnp.array([speeds[i // rotation_samples] * jnp.cos(rotations[i % rotation_samples]),
                                                                               speeds[i // rotation_samples] * jnp.sin(rotations[i % rotation_samples])])),
                                     action_space)
        return action_space

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, 
            online_vnet_params:dict, epsilon:float, reward_fun:function) -> jnp.ndarray:
        
        def _random_action(key):
            key, subkey = random.split(key)
            return random.choice(subkey, jnp.arange(self.n_actions)), key
        
        def _forward_pass(key):
            # Propagate humans state
            next_obs = lax.fori_loop(0,
                                     obs.shape[0]-1,
                                     lambda i, obs: obs.at[i,0:2].set(obs[i,0:2] + obs[i,2:4] * self.dt),
                                     obs)
            pass
            # return action, key
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        action, key = lax.cond(explore, _random_action, _forward_pass, key)
        return action, key
        

    @partial(jit, static_argnames=("self"))
    def update(self):
        pass