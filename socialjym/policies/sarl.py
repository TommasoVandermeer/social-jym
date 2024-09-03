import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from .base_policy import BasePolicy

MLP_1_PARAMS = {
    "output_sizes": [150, 100],
    "activation": nn.relu,
    "activate_final": True
}
MLP_2_PARAMS = {
    "output_sizes": [100, 50],
    "activation": nn.relu,
    "activate_final": False
}
MLP_3_PARAMS = {
    "output_sizes": [150, 100, 100, 1],
    "activation": nn.relu,
    "activate_final": False
}
ATTENTION_LAYER_PARAMS = {
    "output_sizes": [100, 100, 1],
    "activation": nn.relu,
    "activate_final": False
}

class ValueNetwork(hk.Module):
    def __init__(
            self,
            mlp1_params:dict=MLP_1_PARAMS,
            mlp2_params:dict=MLP_2_PARAMS,
            mlp3_params:dict=MLP_3_PARAMS,
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS
        ) -> None:
        self.mlp1 = hk.nets.MLP(**mlp1_params)
        self.mlp2 = hk.nets.MLP(**mlp2_params)
        self.mlp3 = hk.nets.MLP(**mlp3_params)
        self.attention = hk.nets.MLP(**attention_layer_params)

    def __call__(
            self, 
            x: jnp.ndarray
        ) -> jnp.ndarray:
        # Save self state variables
        self_state = x[:self.self_state_size]
        # Compute embeddings and global state
        mlp1_output = self.mlp1(x)
        global_state = jnp.mean(mlp1_output, axis=0)
        # Compute attention weights
        attention_input = jnp.concatenate([mlp1_output, global_state], axis=1)
        scores = self.attention(attention_input)
        attention_weights = nn.softmax(scores)
        # Compute hidden features
        features = self.mlp2(mlp1_output)
        # Compute weighted features (hidden features weighted by attention weights)
        weighted_features = jnp.sum(features * attention_weights, axis=0)
        # Compute state value
        mlp3_input = jnp.concatenate([self_state, weighted_features], axis=1)
        state_value = self.mlp3(mlp3_input)
        return state_value

@hk.transform
def value_network(x):
    vnet = ValueNetwork()
    return vnet(x)

class SARL(BasePolicy):
    def __init__(self, reward_function:FunctionType, v_max=1., gamma=0.9, dt=0.25, speed_samples=5, rotation_samples=16) -> None:
        # Configurable attributes
        super().__init__(gamma)
        self.reward_function = reward_function
        self.v_max = v_max
        self.dt = dt
        self.speed_samples = speed_samples
        self.rotation_samples = rotation_samples
        self.action_space = self._build_action_space()
        # Default attributes
        self.vnet_input_size = 13
        self.model = value_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _build_action_space(self) -> jnp.ndarray:
        speeds = lax.fori_loop(0,self.speed_samples,lambda i, speeds: speeds.at[i].set((jnp.exp((i + 1) / self.speed_samples) - 1) / (jnp.e - 1) * self.v_max), jnp.zeros((self.speed_samples,)))
        rotations = jnp.linspace(0, 2 * jnp.pi, self.rotation_samples, endpoint=False)
        action_space = jnp.empty((self.speed_samples * self.rotation_samples + 1,2))
        action_space = action_space.at[0].set(jnp.array([0, 0])) # First action is to stay still
        action_space = lax.fori_loop(0,
                                     len(action_space), 
                                     lambda i, acts: acts.at[i].set(jnp.array([speeds[i // self.rotation_samples] * jnp.cos(rotations[i % self.rotation_samples]),
                                                                               speeds[i // self.rotation_samples] * jnp.sin(rotations[i % self.rotation_samples])])),
                                     action_space)
        return action_space
    


    # Public methods