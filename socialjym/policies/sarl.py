import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn
from functools import partial
import haiku as hk
from types import FunctionType

from .cadrl import CADRL

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
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS,
            robot_state_size:int=6
        ) -> None:
        super().__init__()  
        self.mlp1 = hk.nets.MLP(**mlp1_params, name="mlp1")
        self.mlp2 = hk.nets.MLP(**mlp2_params, name="mlp2")
        self.mlp3 = hk.nets.MLP(**mlp3_params, name="mlp3")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size

    def __call__(
            self, 
            x: jnp.ndarray
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """

        # Dimensions examples (CrowdNav) with 5 humans and 13 features of reparametrized state
        # X Size : [1, 5, 13]
        # MLP1 input size:  [5, 13]
        # MLP1 output/MLP2 input size:  [5, 100]
        # Features size:  [1, 5, 50]
        # Global State size before expansion:  [1, 1, 100]
        # Global State size:  [5, 100]
        # Attention input size:  [5, 200]
        # Scores size:  [1, 5]
        # Weights size:  [1, 5, 1]
        # Weighted Feature size:  [1, 50]
        # Joint State/MLP3 input size:  [1, 56]

        # Dimensions examples (CrowdNav) with 5 humans and 13 features of reparametrized state
        # self_state size:  [6]
        # MLP1 output size: [5,100]
        # Features size: [5,50]
        # Global State size before expansion: [1,100]
        # Global State size: [5,100]
        # Attention input size: [5,200]
        # Scores size: [5,1]
        # Weights size: [5,1]
        # Weighted Feature size: [50]
        # Joint State/MLP3 input size: [56]

        # Save self state variables
        size = x.shape # (# of humans, length of reparametrized state)
        self_state = x[0,:self.robot_state_size] # The robot state is repeated in each row of axis 1, we take the first one
        # debug.print("self_state size:  {x}", x=self_state.shape)
        # Compute embeddings and global state
        mlp1_output = self.mlp1(x)
        # debug.print("MLP1 output size: {x}", x=mlp1_output.shape)
        # Compute hidden features
        features = self.mlp2(mlp1_output)
        # debug.print("Features size: {x}", x=features.shape)
        global_state = jnp.mean(mlp1_output, axis=0, keepdims=True)
        # debug.print("Global State size before expansion: {x}", x=global_state.shape)
        global_state = jnp.tile(global_state, (size[0], 1))
        # debug.print("Global State size: {x}", x=global_state.shape)
        # Compute attention weights (last step is softmax but setting attention_weight to zero for scores equal to zero)
        attention_input = jnp.concatenate([mlp1_output, global_state], axis=1)
        # debug.print("Attention input size: {x}", x=attention_input.shape)
        scores = self.attention(attention_input)
        # debug.print("Scores size: {x}", x=scores.shape)
        scores_exp = jnp.exp(scores) * jnp.array(scores != 0, dtype=jnp.float32)
        attention_weights = scores_exp / jnp.sum(scores_exp, axis=0)
        # debug.print("Weights size: {x}", x=attention_weights.shape)
        # Compute weighted features (hidden features weighted by attention weights)
        weighted_features = jnp.sum(jnp.multiply(attention_weights, features), axis=0)
        # debug.print("Weighted Feature size: {x}", x=weighted_features.shape)
        # Compute state value
        mlp3_input = jnp.concatenate([self_state, weighted_features], axis=0)
        # debug.print("Joint State/MLP3 input size: {x}", x=mlp3_input.shape)
        state_value = self.mlp3(mlp3_input)

        return state_value

@hk.transform
def value_network(x):
    vnet = ValueNetwork()
    return vnet(x)

class SARL(CADRL):
    def __init__(self, reward_function:FunctionType, v_max=1., gamma=0.9, dt=0.25, speed_samples=5, rotation_samples=16) -> None:
        # Configurable attributes
        super().__init__(
            reward_function=reward_function, 
            v_max=v_max, 
            gamma=gamma, 
            dt=dt, 
            speed_samples=speed_samples, 
            rotation_samples=rotation_samples
        )
        # Default attributes
        self.name = "SARL"
        self.model = value_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict) -> jnp.ndarray:
        n_humans = len(next_obs) - 1
        # Compute instantaneous reward
        current_obs = current_obs.at[n_humans,2:4].set(action)
        reward, _ = self.reward_function(current_obs, info, self.dt)
        # Apply robot action
        next_obs = next_obs.at[n_humans,2:4].set(action)
        next_obs = next_obs.at[n_humans].set(self._propagate_obs(next_obs[-1]))
        # Re-parametrize observation, for each human: [dg,v_pref,theta,radius,vx,vy,px1,py1,vx1,vy1,radius1,da,radius_sum]
        vnet_inputs = self.batch_compute_vnet_input(next_obs[n_humans], next_obs[0:n_humans], info)
        # Compute the output of the value network (value of the state)
        vnet_output = self.model.apply(vnet_params, None, vnet_inputs)
        # Compute the final value of the action
        value = reward + pow(self.gamma,self.dt * self.v_max) * vnet_output
        return value, vnet_inputs
    
    @partial(jit, static_argnames=("self"))
    def _batch_compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict) -> jnp.ndarray:
        return vmap(CADRL._compute_action_value, in_axes=(None,None,None,None,0,None))(self, next_obs, current_obs, info, action, vnet_params)

    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, vnet_params:dict, epsilon:float) -> jnp.ndarray:
        
        def _random_action(key):
            key, subkey = random.split(key)
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            return random.choice(subkey, self.action_space), key, vnet_inputs
        
        def _forward_pass(key):
            # Propagate humans state for dt time
            next_obs = jnp.vstack([self.batch_propagate_obs(obs[0:-1]),obs[-1]])
            # Compute action values
            action_values, vnet_inputs = self._batch_compute_action_value(next_obs, obs, info, self.action_space, vnet_params)
            action = self.action_space[jnp.argmax(action_values)]
            vnet_input = vnet_inputs[jnp.argmax(action_values)]
            # Return action with highest value
            return action, key, vnet_input
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        action, key, vnet_input = lax.cond(explore, _random_action, _forward_pass, key)
        return action, key, vnet_input