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

#### VERSION: TWO NETWORKS AND ACTOR OUTPUTS CONSTRAINED ACTION ####

EPSILON = 1e-5
MLP_4_PARAMS = {
    "output_sizes": [150, 100, 100, 2], # Output: [mu_Vleft, mu_Vright]
    "activation": nn.relu,
    "activate_final": False,
    "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
    "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
}

@partial(jit, static_argnames=("kinematics", "wheels_distance", "v_max"))
def _sample_action(
    kinematics:int,
    wheels_distance:float,
    v_max:float,
    mu1:float,
    mu2:float,
    sigma:float = 0.,
    key:random.PRNGKey = random.PRNGKey(0),
) -> jnp.ndarray:
    key1, key2 = random.split(key)
    if kinematics == ROBOT_KINEMATICS.index('unicycle'):
        vleft = mu1 + sigma * random.normal(key1)
        vright = mu2 + sigma * random.normal(key2)
        sampled_action = jnp.array([vleft, vright])
        ## Bouind the final action with HARD CLIPPING (gradients discontinuity)
        # vleft = jnp.clip(vleft, -v_max, v_max)
        # vright = jnp.clip(vright, -v_max, v_max)
        # v = nn.relu((vleft + vright) / 2) # Robot can only go forward
        ## Bound the final action with SMOOTH CLIPPING (ensures gradient continuity)
        vleft = v_max * jnp.tanh(vleft / v_max)
        vright = v_max * jnp.tanh(vright / v_max)
        v = (vleft + vright) / 2
        v = nn.leaky_relu(v) # Robot can only go forward. This is not soft clipping, but it is not constant below zero.
        ## Build final action
        constrained_action = jnp.array([v, (vright - vleft) / wheels_distance])
    elif kinematics == ROBOT_KINEMATICS.index('holonomic'):
        vx = mu1 + sigma * random.normal(key1)
        vy = mu2 + sigma * random.normal(key2)
        sampled_action = jnp.array([vx, vy])
        norm = jnp.linalg.norm(jnp.array([vx, vy]))
        ## Bound the norm of the velocity with HARD CLIPPING (gradients discontinuity)
        # scaling_factor = jnp.clip(norm, 0., v_max) / (norm + EPSILON)
        # vx = vx * scaling_factor
        # vy = vy * scaling_factor
        ## Bound the norm of the velocity with SMOOTH CLIPPING (ensures gradients continuity)
        scaling_factor = jnp.tanh(norm / v_max) / (norm + EPSILON)
        vx = vx * scaling_factor
        vy = vy * scaling_factor
        ## Build final action
        constrained_action = jnp.array([vx, vy])
    return constrained_action, sampled_action

@partial(jit, static_argnames=("kinematics", "wheels_distance", "v_max"))
def batch_sample_action(
    kinematics:int,
    wheels_distance:float,
    v_max:float,
    keys:random.PRNGKey,
    mu1:jnp.ndarray,
    mu2:jnp.ndarray,
    sigma:jnp.ndarray,
) -> jnp.ndarray:
    return vmap(_sample_action, in_axes=(None, None, None, 0, None, None, None))(
        kinematics,
        wheels_distance,
        v_max,
        keys, 
        mu1, 
        mu2, 
        sigma,
    )

class Actor(hk.Module):
    def __init__(
            self,
            kinematics:int,
            wheels_distance:float,
            v_max:float,
            mlp1_params:dict=MLP_1_PARAMS,
            mlp2_params:dict=MLP_2_PARAMS,
            mlp4_params:dict=MLP_4_PARAMS,
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS,
            robot_state_size:int=6,
        ) -> None:
        super().__init__()  
        self.mlp1 = hk.nets.MLP(**mlp1_params, name="mlp1")
        self.mlp2 = hk.nets.MLP(**mlp2_params, name="mlp2")
        self.mlp4 = hk.nets.MLP(**mlp4_params, name="mlp4")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size
        self.v_max = v_max
        self.kinematics = kinematics
        self.wheels_distance = wheels_distance

    def __call__(
            self, 
            x: jnp.ndarray,
            **kwargs:dict,
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """
        # Get keyword arguments
        gaussian_key = kwargs.get("gaussian_key", random.PRNGKey(0))
        sigma = kwargs.get("sigma", 0.)
        ## Save self state variables
        size = x.shape # (# of humans, length of reparametrized state)
        self_state = x[0,:self.robot_state_size] # The robot state is repeated in each row of axis 1, we take the first one
        # debug.print("self_state size:  {x}", x=self_state.shape)
        ## Compute embeddings and global state
        mlp1_output = self.mlp1(x)
        # debug.print("MLP1 output size: {x}", x=mlp1_output.shape)
        ## Compute hidden features
        features = self.mlp2(mlp1_output)
        # debug.print("Features size: {x}", x=features.shape)
        global_state = jnp.mean(mlp1_output, axis=0, keepdims=True)
        # debug.print("Global State size before expansion: {x}", x=global_state.shape)
        global_state = jnp.tile(global_state, (size[0], 1))
        # debug.print("Global State size: {x}", x=global_state.shape)
        ## Compute attention weights (last step is softmax but setting attention_weight to zero for scores equal to zero)
        attention_input = jnp.concatenate([mlp1_output, global_state], axis=1)
        # debug.print("Attention input size: {x}", x=attention_input.shape)
        scores = self.attention(attention_input)
        # debug.print("Scores size: {x}", x=scores.shape)
        scores_exp = jnp.exp(scores) * jnp.array(scores != 0, dtype=jnp.float32)
        attention_weights = scores_exp / jnp.sum(scores_exp, axis=0)
        # debug.print("Weights size: {x}", x=attention_weights.shape)
        ## Compute weighted features (hidden features weighted by attention weights)
        weighted_features = jnp.sum(jnp.multiply(attention_weights, features), axis=0)
        # debug.print("Weighted Feature size: {x}", x=weighted_features.shape)
        ## Compute state value
        mlp4_input = jnp.concatenate([self_state, weighted_features], axis=0)
        ## Compute normal distribution parameters
        mu1, mu2 = self.mlp4(mlp4_input)
        # debug.print("Joint State/MLP4 input size: {x}", x=mlp4_input.shape)
        ## Bound output (avoids problems of exploding gradients)
        mu1 = self.v_max * jnp.tanh(mu1)
        mu2 = self.v_max * jnp.tanh(mu2)
        ## Sample action
        constrained_action, sampled_action = _sample_action(
            self.kinematics,
            self.wheels_distance,
            self.v_max,
            mu1,
            mu2,
            sigma,
            key = gaussian_key,
        )
        return constrained_action, sampled_action, mu1, mu2

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
            actor = Actor(v_max = self.v_max, kinematics = self.kinematics, wheels_distance = self.wheels_distance)
            return actor(x, **kwargs)
        self.actor = actor_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _compute_log_pdf_value(
        self, 
        mu1:jnp.ndarray, 
        mu2:jnp.ndarray,
        sigma:jnp.ndarray,
        action:jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.log(pdf(action[0], mu1, sigma)) + jnp.log(pdf(action[1], mu2, sigma))

    @partial(jit, static_argnames=("self"))
    def _compute_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"inputs":jnp.ndarray, "critic_targets":jnp.ndarray, "sample_actions":jnp.ndarray},
        current_sigma:float,
        current_beta_entropy:float,
        imitation_learning:bool,
    ) -> tuple:
        
        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray,  
            ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _advantage(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:jnp.ndarray, 
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the loss
                return target - prediction
            
            advantage = _advantage(
                current_critic_params,
                inputs,
                critic_targets)
            return jnp.mean(jnp.square(advantage)), advantage
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
            advantages:jnp.ndarray,  
            sigma:float,
            beta_entropy:float = 0.0001,
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0, None))
            def _rl_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
                advantage:jnp.ndarray, 
                sigma:float,
            ) -> jnp.ndarray:
                # Compute the prediction
                _, _, mu1, mu2 = self.actor.apply(current_actor_params, None, input)
                # Compute the log probability of the action
                log_pdf = self._compute_log_pdf_value(
                    mu1,
                    mu2,
                    sigma,
                    sample_action)
                # Compute the entropy loss
                entropy_loss =  - (jnp.log(2*jnp.pi*sigma) + 1) / 2
                # Compute the loss
                return jnp.squeeze(- log_pdf * advantage), entropy_loss

            @partial(vmap, in_axes=(None, 0, 0, 0, None))
            def _il_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
                advantage:jnp.ndarray, 
                sigma:float,
            ) -> jnp.ndarray:
                # Compute the prediction (here we should input a key but for now we work only with mean actions)
                action, _, _, _ = self.actor.apply(current_actor_params, None, input, sigma=0.)
                # Compute the loss
                return 0.5 * jnp.sum(jnp.square(action - sample_action)), 0.
            
            actor_losses, entropy_losses = lax.cond(
                imitation_learning,
                lambda x: _il_loss_function(*x),
                lambda x: _rl_loss_function(*x),
                (current_actor_params, inputs, sample_actions, advantages, sigma),
            )
            actor_loss = jnp.mean(actor_losses)
            entropy_loss = beta_entropy * jnp.mean(entropy_losses)
            loss = actor_loss + entropy_loss
            return loss, {"actor_loss": actor_loss, "entropy_loss": entropy_loss}

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        # Compute critic loss and gradients
        loss_and_advantages, critic_grads = value_and_grad(_batch_critic_loss_function, has_aux=True)(
            current_critic_params, 
            inputs,
            critic_targets
        )
        critic_loss, advantages = loss_and_advantages
        # Compute actor loss and gradients
        actor_and_entropy_loss, actor_grads = value_and_grad(_batch_actor_loss_function, has_aux=True)(
            current_actor_params, 
            inputs,
            sample_actions, 
            advantages,
            current_sigma,
            current_beta_entropy
        )
        _, all_losses = actor_and_entropy_loss
        actor_loss = all_losses["actor_loss"]
        entropy_loss = all_losses["entropy_loss"]
        return critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss

    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict, 
        actor_params:dict, 
        sigma=0.
    ) -> jnp.ndarray:

        # Add noise to human observations
        if self.noise:
            key, subkey = random.split(key)
            obs = self._batch_add_noise_to_human_obs(obs, subkey)
        # Compute actor input
        actor_input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
        # Compute action 
        key, subkey = random.split(key)
        constrained_action, sampled_action, mu1, mu2 = self.actor.apply(actor_params, None, actor_input, sigma=sigma, key=subkey)
        return constrained_action, key, actor_input, sampled_action, {"mu1":mu1, "mu2":mu2}
    
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
        sigma:float,
        beta_entropy:float,
        imitation_learning:bool=False,
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss = self._compute_loss_and_gradients(
            critic_params, 
            actor_params,
            experiences,
            sigma,
            beta_entropy,
            imitation_learning,
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
        return (
            updated_critic_params, 
            updated_actor_params, 
            critic_opt_state, 
            actor_opt_state, 
            critic_loss, 
            actor_loss, 
            entropy_loss
        )