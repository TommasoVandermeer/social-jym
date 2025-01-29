import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from socialjym.envs.base_env import ROBOT_KINEMATICS
from .cadrl import CADRL
from .sarl import value_network as critic_network
from .sarl import MLP_1_PARAMS, MLP_2_PARAMS, ATTENTION_LAYER_PARAMS

MLP_4_PARAMS = {
    "output_sizes": [150, 100, 100, 4], # Output: [mu_Vleft, mu_Vright, sigma_Vleft, sigma_Vright]
    "activation": nn.relu,
    "activate_final": False,
    "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
    "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
}

class Actor(hk.Module):
    def __init__(
            self,
            mlp1_params:dict=MLP_1_PARAMS,
            mlp2_params:dict=MLP_2_PARAMS,
            mlp4_params:dict=MLP_4_PARAMS,
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS,
            robot_state_size:int=6,
            kinematics:str='unicycle',
            max_speed:float=1.,
            wheels_distance:float=0.7,
        ) -> None:
        super().__init__()  
        self.mlp1 = hk.nets.MLP(**mlp1_params, name="mlp1")
        self.mlp2 = hk.nets.MLP(**mlp2_params, name="mlp2")
        self.mlp4 = hk.nets.MLP(**mlp4_params, name="mlp4")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        self.max_speed = max_speed
        self.wheels_distance = wheels_distance

    def __call__(
            self, 
            x: jnp.ndarray,
            gaussian_key: random.PRNGKey,
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """

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
        mlp4_input = jnp.concatenate([self_state, weighted_features], axis=0)
        # debug.print("Joint State/MLP4 input size: {x}", x=mlp4_input.shape)
        if self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            mu_Vleft, mu_Vright, sigma_Vleft, sigma_Vright = self.mlp4(mlp4_input)
            sigma_Vleft = nn.softplus(sigma_Vleft) + 1e-5
            sigma_Vright = nn.softplus(sigma_Vright) + 1e-5
            key1, key2 = random.split(gaussian_key)
            vleft = mu_Vleft + sigma_Vleft * random.normal(key1)
            vright = mu_Vright + sigma_Vright * random.normal(key2)
            ## Bouind the wheels speed with HARD CLIPPING (gradients discontinuity and risk of vanishing gradients)
            # vleft = jnp.clip(vleft, -self.max_speed, self.max_speed)
            # vright = jnp.clip(vright, -self.max_speed, self.max_speed)
            ## Bound the wheels speed with SMOOTH CLIPPING (ensures gradient continuity)
            vleft = self.max_speed * jnp.tanh(vleft / self.max_speed)
            vright = self.max_speed * jnp.tanh(vright / self.max_speed)
            # WARNING: Robot can also go backwards
            action = jnp.array([(vleft + vright) / 2, (vright - vleft) / self.wheels_distance])
        elif self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            mu_Vx, mu_Vy, sigma_Vx, sigmaVy = self.mlp4(mlp4_input)
            sigma_Vx = nn.softplus(sigma_Vx) + 1e-5
            sigmaVy = nn.softplus(sigmaVy) + 1e-5
            key1, key2 = random.split(gaussian_key)
            vx = mu_Vx + sigma_Vx * random.normal(key1)
            vy = mu_Vy + sigmaVy * random.normal(key2)
            norm = jnp.linalg.norm(jnp.array([vx, vy]))
            ## Bound the norm of the velocity with HARD CLIPPING (gradients discontinuity and risk of vanishing gradients)
            # scaling_factor = jnp.clip(norm, 0., self.max_speed) / (norm + 1e-5)
            # vx = vx * scaling_factor
            # vy = vy * scaling_factor
            ## Bound the norm of the velocity with SMOOTH CLIPPING (ensures gradients continuity)
            scaling_factor = jnp.tanh(norm / self.max_speed) / (norm + 1e-5)
            vx = vx * scaling_factor
            vy = vy * scaling_factor
            action = jnp.array([vx, vy])
        return action

class SARLA2C(CADRL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            v_max=1., 
            gamma=0.9, 
            dt=0.25, 
            wheels_distance=0.7, 
            kinematics='unicycle',
            noise = False, # If True, noise is added to humams positions and velocities
            noise_sigma_percentage:float = 0., # Standard deviation of the noise as a percentage of the absolute value of the difference between the robot and the humans
        ) -> None:
        # Configurable attributes
        super().__init__(
            reward_function=reward_function, 
            v_max=v_max, 
            wheels_distance=wheels_distance,
            gamma=gamma, 
            dt=dt,
            kinematics=kinematics,
            noise=noise,
            noise_sigma_percentage=noise_sigma_percentage,
        )
        # Default attributes
        self.name = "SARL-A2C"
        self.critic = critic_network
        @hk.transform
        def actor_network(x:jnp.ndarray, gaussian_key:random.PRNGKey):
            actor = Actor(kinematics=kinematics, max_speed=v_max, wheels_distance=wheels_distance)
            return actor(x, gaussian_key)
        self.actor = actor_network

    # Private methods

    ####
    
    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, actor_params:dict, epsilon:float) -> jnp.ndarray:
        
        # Add noise to human observations
        if self.noise:
            key, subkey = random.split(key)
            obs = self._batch_add_noise_to_human_obs(obs, subkey)
        # Compute actor input
        actor_input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
        # Compute action 
        action = self.actor.apply(actor_params, None, actor_input, gaussian_key=key)

        return action, key, actor_input
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        vnet_params,
        epsilon):
        return vmap(SARLA2C.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos, 
            vnet_params, 
            epsilon)
    
    @partial(jit, static_argnames=("self","optimizer"))
    def update(
        self, 
        current_vnet_params:dict, 
        optimizer:optax.GradientTransformation, 
        optimizer_state: jnp.ndarray, 
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"vnet_inputs":jnp.ndarray, "targets":jnp.ndarray,}
    ) -> tuple:
        pass
        # Compute loss and gradients
        # Compute parameter updates
        # Apply updates
        # return updated_vnet_params, optimizer_state, loss