import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from jax.scipy.stats.norm import pdf
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from socialjym.envs.base_env import ROBOT_KINEMATICS
from .cadrl import CADRL
from .sarl import value_network as critic_network
from .sarl import MLP_1_PARAMS, MLP_2_PARAMS, ATTENTION_LAYER_PARAMS

@partial(jit, static_argnames=("kinematics"))
def _sample_action(
    kinematics:int, 
    sample:bool,
    key:random.PRNGKey,
    mu1:float,
    mu2:float,
    sigma1:float,
    sigma2:float,
    max_speed:float,
    wheels_distance:float=0.7
) -> jnp.ndarray:
    key1, key2 = random.split(key)
    if kinematics == ROBOT_KINEMATICS.index('unicycle'):
        vleft = mu1 + sigma1 * random.normal(key1) * jnp.squeeze(jnp.array([sample], dtype=int)) 
        vright = mu2 + sigma2 * random.normal(key2) * jnp.squeeze(jnp.array([sample], dtype=int))
        sampled_action = jnp.array([vleft, vright])
        ## Bouind the final action with HARD CLIPPING (gradients discontinuity)
        # vleft = jnp.clip(vleft, -max_speed, max_speed)
        # vright = jnp.clip(vright, -max_speed, max_speed)
        ## v = jnp.abs((vleft + vright) / 2) # Robot can only go forward
        # v = nn.relu((vleft + vright) / 2) # Robot can only go forward
        ## Bound the final action with SMOOTH CLIPPING (ensures gradient continuity)
        vleft = max_speed * jnp.tanh(vleft / max_speed)
        vright = max_speed * jnp.tanh(vright / max_speed)
        v = (vleft + vright) / 2
        ## v = v * jnp.tanh(v / 0.1) # Robot can only go forward. The smaller the denominator, the more the function is similar to abs (but less numerically stable)
        v = nn.leaky_relu(v) # Robot can only go forward. This is not soft clipping, but it is not constant below zero.
        ## Build final action
        action = jnp.array([v, (vright - vleft) / wheels_distance])
    elif kinematics == ROBOT_KINEMATICS.index('holonomic'):
        vx = mu1 + sigma1 * random.normal(key1) * jnp.squeeze(jnp.array([sample], dtype=int))
        vy = mu2 + sigma2 * random.normal(key2) * jnp.squeeze(jnp.array([sample], dtype=int))
        sampled_action = jnp.array([vx, vy])
        norm = jnp.linalg.norm(jnp.array([vx, vy]))
        ## Bound the norm of the velocity with HARD CLIPPING (gradients discontinuity)
        # scaling_factor = jnp.clip(norm, 0., max_speed) / (norm + EPSILON)
        # vx = vx * scaling_factor
        # vy = vy * scaling_factor
        ## Bound the norm of the velocity with SMOOTH CLIPPING (ensures gradients continuity)
        scaling_factor = jnp.tanh(norm / max_speed) / (norm + EPSILON)
        vx = vx * scaling_factor
        vy = vy * scaling_factor
        ## Build final action
        action = jnp.array([vx, vy])
    return action, sampled_action

@partial(jit, static_argnames=("kinematics"))
def batch_sample_action(
    kinematics:int,
    sample:bool,
    keys:random.PRNGKey,
    mu1:jnp.ndarray,
    mu2:jnp.ndarray,
    sigma1:jnp.ndarray,
    sigma2:jnp.ndarray,
    max_speed:float,
    wheels_distance:float=0.7
) -> jnp.ndarray:
    return vmap(_sample_action, in_axes=(None, None, 0, None, None, None, None, None, None))(kinematics, sample, keys, mu1, mu2, sigma1, sigma2, max_speed, wheels_distance)

MLP_4_PARAMS = {
    "output_sizes": [150, 100, 100, 4], # Output: [mu_Vleft, mu_Vright, sigma_Vleft, sigma_Vright]
    "activation": nn.relu,
    "activate_final": False,
    "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
    "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
}

EPSILON = 1e-5

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
            **kwargs:dict,
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """
        gaussian_key = kwargs.get("gaussian_key", random.PRNGKey(0))
        sample = kwargs.get("sample", True) # If false, the mean of the action distribution is returned
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
        # Compute normal distribution parameters
        mu1, mu2, sigma1, sigma2 = self.mlp4(mlp4_input)
        # debug.print("Joint State/MLP4 input size: {x}", x=mlp4_input.shape)
        ### TODO: Does it make sense to output a single sigma for both wheels? Ensuring they explore the same amount of space
        # Standard deviation must be strictly greater than zero
        sigma1 = nn.softplus(sigma1) + EPSILON
        sigma2 = nn.softplus(sigma2) + EPSILON
        # Sample action
        action, sampled_action = _sample_action(
            self.kinematics, 
            sample, 
            gaussian_key, 
            mu1, 
            mu2, 
            sigma1, 
            sigma2, 
            self.max_speed, 
            self.wheels_distance)
        # Save distributions data
        distributions = {
            "mu1": mu1,
            "mu2": mu2,
            "sigma1": sigma1,
            "sigma2": sigma2
        }
        return action, sampled_action, distributions

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
        def actor_network(x:jnp.ndarray, **kwargs) -> jnp.ndarray:
            actor = Actor(kinematics=kinematics, max_speed=v_max, wheels_distance=wheels_distance)
            return actor(x, **kwargs)
        self.actor = actor_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _compute_log_pdf_value(
        self, 
        mu1:jnp.ndarray, 
        sigma1:jnp.ndarray,
        mu2:jnp.ndarray,
        sigma2:jnp.ndarray, 
        action:jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.log(pdf(action[0], mu1, sigma1) * pdf(action[1], mu2, sigma2))

    @partial(jit, static_argnames=("self"))
    def _compute_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"inputs":jnp.ndarray, "critic_targets":jnp.ndarray, "sample_actions":jnp.ndarray},
    ) -> tuple:
        
        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray,  
            ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _loss_function(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:jnp.ndarray, 
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the loss
                return jnp.square(target - prediction)
            
            losses = _loss_function(
                current_critic_params,
                inputs,
                critic_targets)
            return jnp.mean(losses), losses
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
            advantages:jnp.ndarray,  
            ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0))
            def _loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
                advantage:jnp.ndarray, 
                ) -> jnp.ndarray:
                # Compute the prediction
                _, _, distrs = self.actor.apply(current_actor_params, None, input, random.PRNGKey(0))
                # Compute the log probability of the action
                log_pdf = self._compute_log_pdf_value(
                    distrs["mu1"],
                    distrs["sigma1"],
                    distrs["mu2"],
                    distrs["sigma2"],
                    sample_action)
                # Compute the loss
                return - log_pdf * advantage
            
            return jnp.mean(_loss_function(
                    current_actor_params,
                    inputs,
                    sample_actions,
                    advantages))

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        # Compute critic loss and gradients
        loss_and_advantages, critic_grads = value_and_grad(_batch_critic_loss_function, has_aux=True)(
            current_critic_params, 
            inputs,
            critic_targets)
        critic_loss, advantages = loss_and_advantages
        # Compute actor loss and gradients
        actor_loss, actor_grads = value_and_grad(_batch_actor_loss_function)(
            current_actor_params, 
            inputs,
            sample_actions, 
            advantages)
        return critic_loss, critic_grads, actor_loss, actor_grads
    
    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, actor_params:dict, sample=False) -> jnp.ndarray:

        # Add noise to human observations
        if self.noise:
            key, subkey = random.split(key)
            obs = self._batch_add_noise_to_human_obs(obs, subkey)
        # Compute actor input
        actor_input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
        # Compute action 
        key, subkey = random.split(key)
        action, sampled_action, distrs = self.actor.apply(actor_params, None, actor_input, gaussian_key=key, sample=sample)

        return action, key, actor_input, sampled_action, distrs
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        vnet_params,
        sample):
        return vmap(SARLA2C.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos, 
            vnet_params, 
            sample)
    
    @partial(jit, static_argnames=("self","actor_optimizer","critic_optimizer"))
    def update(
        self, 
        critic_params:dict, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        critic_optimizer:optax.GradientTransformation,
        critic_opt_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray], 
        # experiences: {"inputs":jnp.ndarray, "critic_targets":jnp.ndarray, "sample_actions":jnp.ndarray},
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads = self._compute_loss_and_gradients(
            critic_params, 
            actor_params,
            experiences,
        )
        ## CRITIC
        # Compute parameter updates
        critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
        # Apply updates
        updated_critic_params = optax.apply_updates(critic_params, critic_updates)
        ## ACTOR
        # Compute parameter updates
        actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
        # Apply updates
        updated_actor_params = optax.apply_updates(actor_params, actor_updates)
        
        return updated_critic_params, updated_actor_params, critic_opt_state, actor_opt_state, critic_loss, actor_loss