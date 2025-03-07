import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from jax.tree_util import tree_leaves
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from socialjym.envs.base_env import ROBOT_KINEMATICS
from .sarl import SARL
from .sarl import value_network as critic_network
from .sarl import MLP_1_PARAMS, MLP_2_PARAMS, ATTENTION_LAYER_PARAMS

EPSILON = 1e-5
MLP_1_PARAMS = {
    "output_sizes": [150, 100],
    "activation": nn.tanh, # nn.relu
    "activate_final": True,
    "w_init": hk.initializers.Orthogonal(scale=jnp.sqrt(2)),
    "b_init": hk.initializers.Constant(0.),
}
MLP_2_PARAMS = {
    "output_sizes": [100, 50],
    "activation": nn.tanh, # nn.relu
    "activate_final": False,
    "w_init": hk.initializers.Orthogonal(scale=jnp.sqrt(2)),
    "b_init": hk.initializers.Constant(0.),
}
MLP_4_PARAMS = {
    "output_sizes": [150, 100, 100],
    "activation": nn.tanh, # nn.relu
    "activate_final": False,
    "w_init": hk.initializers.Orthogonal(scale=jnp.sqrt(2)),
    "b_init": hk.initializers.Constant(0.),
}
ATTENTION_LAYER_PARAMS = {
    "output_sizes": [100, 100, 1],
    "activation": nn.tanh, # nn.relu
    "activate_final": False,
    "w_init": hk.initializers.Orthogonal(scale=jnp.sqrt(2)),
    "b_init": hk.initializers.Constant(0.),
}

class Actor(hk.Module):
    def __init__(
            self,
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
        self.output_layer = hk.Linear(2, w_init=hk.initializers.Orthogonal(scale=0.01), b_init=hk.initializers.Constant(0.), name="output_layer")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size

    def __call__(
            self, 
            x: jnp.ndarray,
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """
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
        ## Compute MLP4 output
        mlp4_input = jnp.concatenate([self_state, weighted_features], axis=0)
        mlp4_output = self.mlp4(mlp4_input)
        # debug.print("Joint State/MLP4 input size: {x}", x=mlp4_input.shape)
        ## Compute normal distribution parameters
        mu1, mu2 = self.output_layer(mlp4_output)
        logsigma = hk.get_parameter("logsigma", shape=[], init=hk.initializers.Constant(0.))
        return mu1, mu2, logsigma

class SARLPPO(SARL):
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
        self.name = "SARL-PPO"
        self.critic = critic_network
        @hk.transform
        def actor_network(x:jnp.ndarray) -> jnp.ndarray:
            actor = Actor()
            return actor(x)
        self.actor = actor_network

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _sample_action(
        self,
        mu1:float,
        mu2:float,
        sigma:float = 0.,
        key:random.PRNGKey = random.PRNGKey(0),
    ) -> jnp.ndarray:
        key1, key2 = random.split(key)
        if self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            v = mu1 + sigma * random.normal(key1)
            w = mu2 + sigma * random.normal(key2)
            sampled_action = jnp.array([v, w])
            ## Bound the final action with HARD CLIPPING
            v = jnp.clip(v, 0, self.v_max)
            w_max = (2 * (self.v_max - v)) / self.wheels_distance
            w = jnp.clip(w, -w_max, w_max)
            ## Build final action
            constrained_action = jnp.array([v, w])
        elif self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            vx = mu1 + sigma * random.normal(key1)
            vy = mu2 + sigma * random.normal(key2)
            sampled_action = jnp.array([vx, vy])
            norm = jnp.linalg.norm(jnp.array([vx, vy]))
            ## Bound the norm of the velocity with SMOOTH CLIPPING (ensures gradients continuity)
            scaling_factor = jnp.tanh(norm / self.v_max) / (norm + EPSILON)
            vx = vx * scaling_factor
            vy = vy * scaling_factor
            ## Build final action
            constrained_action = jnp.array([vx, vy])
        return constrained_action, sampled_action

    @partial(jit, static_argnames=("self"))
    def _compute_neg_log_pdf_value(
        self, 
        mu1:jnp.ndarray, 
        mu2:jnp.ndarray,
        logsigma:jnp.ndarray,
        action:jnp.ndarray
    ) -> jnp.ndarray:
        return .5 * jnp.sum(jnp.square((action - jnp.array([mu1, mu2])) / jnp.exp(logsigma))) + jnp.log(2 * jnp.pi) + logsigma

    @partial(jit, static_argnames=("self"))
    def _compute_rl_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
        current_beta_entropy:float,
        clip_range:float,
        debugging:bool=False,
    ) -> tuple:
        
        # Experiences: {
        #   "inputs":jnp.ndarray, 
        #   "critic_targets":jnp.ndarray, 
        #   "sample_actions":jnp.ndarray, 
        #   "old_values":jnp.ndarray, 
        #   "old_neglogpdfs":jnp.ndarray
        # },

        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray, 
            old_values:jnp.ndarray, 
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0))
            def _rl_loss_function(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:float, 
                old_value:float,
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the clipped prediction
                clipped_prediction = jnp.clip(prediction, old_value - clip_range, old_value + clip_range)
                # Compute the loss
                return jnp.maximum(jnp.square(target - prediction), jnp.square(target - clipped_prediction))
            
            critic_loss = _rl_loss_function(current_critic_params, inputs, critic_targets, old_values)
            return 0.5 * jnp.mean(critic_loss)
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
            advantages:jnp.ndarray,  
            old_neglogpdfs:jnp.ndarray,
            beta_entropy:float = 0.0001,
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0, 0, 0))
            def _rl_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
                advantage:jnp.ndarray, 
                old_neglogpdf:jnp.ndarray,
            ) -> jnp.ndarray:
                # Compute the prediction
                mu1, mu2, logsigma = self.actor.apply(current_actor_params, None, input)
                # Compute the log probability of the action
                neglogpdf = self._compute_neg_log_pdf_value(
                    mu1,
                    mu2,
                    logsigma,
                    sample_action
                )
                # Compute policy ratio
                ratio = jnp.exp(old_neglogpdf - neglogpdf)
                lax.cond(
                    debugging,
                    lambda _: debug.print(
                        "Ratio: {x} - Old pdf: {y} - New pdf: {z}", 
                        x=ratio,
                        y=jnp.exp(-old_neglogpdf),
                        z=jnp.exp(-neglogpdf),
                    ),
                    lambda _: None,
                    None,
                )
                # Compute actor loss
                actor_loss = jnp.maximum(- ratio * advantage, - jnp.clip(ratio, 1-clip_range, 1+clip_range) * advantage)
                # Compute the loss
                return actor_loss
            
            actor_losses = _rl_loss_function(current_actor_params, inputs, sample_actions, advantages, old_neglogpdfs)
            actor_loss = jnp.mean(actor_losses)
            entropy_loss = beta_entropy * (current_actor_params['actor']['logsigma'] + .5 * jnp.log(2 * jnp.pi * jnp.e)) * 2 # It is doubled because we have a sigma for each action (but it is the same, so double it)
            loss = actor_loss - entropy_loss
            return loss, {"actor_loss": actor_loss, "entropy_loss": entropy_loss}

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        old_values = experiences["old_values"]
        old_neglogpdfs = experiences["old_neglogpdfs"]
        # Compute and normalize advantages
        advantages = critic_targets - old_values
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + EPSILON)
        # Compute critic loss and gradients
        critic_loss, critic_grads = value_and_grad(_batch_critic_loss_function)(
            current_critic_params, 
            inputs,
            critic_targets,
            old_values
        )
        # Compute actor loss and gradients
        actor_and_entropy_loss, actor_grads = value_and_grad(_batch_actor_loss_function, has_aux=True)(
            current_actor_params, 
            inputs,
            sample_actions, 
            advantages,
            old_neglogpdfs,
            current_beta_entropy,
        )
        _, all_losses = actor_and_entropy_loss
        actor_loss = all_losses["actor_loss"]
        entropy_loss = all_losses["entropy_loss"]
        return critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss

    @partial(jit, static_argnames=("self"))
    def _compute_il_loss_and_gradients(
        self, 
        current_critic_params:dict, 
        current_actor_params:dict, 
        experiences:dict[str:jnp.ndarray],
    ) -> tuple:
        
        # Experiences: {
        #   "inputs":jnp.ndarray, 
        #   "critic_targets":jnp.ndarray, 
        #   "sample_actions":jnp.ndarray, 
        # },

        @jit
        def _batch_critic_loss_function(
            current_critic_params:dict,
            inputs:jnp.ndarray,
            critic_targets:jnp.ndarray, 
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _il_loss_function(
                current_critic_params:dict,
                input:jnp.ndarray,
                target:float, 
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.critic.apply(current_critic_params, None, input)
                # Compute the loss
                return jnp.square(target - prediction)
            
            critic_loss = _il_loss_function(
                current_critic_params,
                inputs,
                critic_targets)
            return jnp.mean(critic_loss)
        
        @jit
        def _batch_actor_loss_function(
            current_actor_params:dict,
            inputs:jnp.ndarray,
            sample_actions:jnp.ndarray,
        ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _il_loss_function(
                current_actor_params:dict,
                input:jnp.ndarray,
                sample_action:jnp.ndarray,
            ) -> jnp.ndarray:
                # Compute the prediction (here we should input a key but for now we work only with mean actions)
                mu1, mu2, logsigma = self.actor.apply(current_actor_params, None, input)
                # Get mean action
                constrained_action, _ = self._sample_action(
                    mu1, 
                    mu2, 
                    sigma = jnp.exp(logsigma),
                )
                # Compute the loss
                return 0.5 * jnp.sum(jnp.square(constrained_action - sample_action))
            
            actor_losses = _il_loss_function(current_actor_params, inputs, sample_actions)
            return jnp.mean(actor_losses)

        inputs = experiences["inputs"]
        critic_targets = experiences["critic_targets"]
        sample_actions = experiences["sample_actions"]
        # Compute critic loss and gradients
        critic_loss, critic_grads = value_and_grad(_batch_critic_loss_function)(
            current_critic_params, 
            inputs,
            critic_targets,
        )
        # Compute actor loss and gradients
        actor_loss, actor_grads = value_and_grad(_batch_actor_loss_function)(
            current_actor_params, 
            inputs,
            sample_actions,
        )
        return critic_loss, critic_grads, actor_loss, actor_grads, 0.

    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict, 
        actor_params:dict, 
        sample:bool = False,
    ) -> jnp.ndarray:

        # Add noise to human observations
        if self.noise:
            key, subkey = random.split(key)
            obs = self._batch_add_noise_to_human_obs(obs, subkey)
        # Compute actor input
        actor_input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
        # Compute action 
        key, subkey = random.split(key)
        mu1, mu2, logsigma = self.actor.apply(actor_params, None, actor_input)
        sigma = lax.cond(sample, lambda x: x, lambda _: 0., jnp.exp(logsigma))
        constrained_action, sampled_action = self._sample_action(
            mu1, 
            mu2, 
            sigma = sigma, 
            key = subkey
        )
        return constrained_action, key, actor_input, sampled_action, {"mu1":mu1, "mu2":mu2, "logsigma":logsigma}
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        actor_params,
        sample):
        return vmap(SARLPPO.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos, 
            actor_params, 
            sample,
        )
    
    @partial(jit, static_argnames=("self"))
    def batch_compute_neg_log_pdf_value(
        self, 
        mu1s:jnp.ndarray, 
        mu2s:jnp.ndarray,
        logsigmas:jnp.ndarray,
        actions:jnp.ndarray
    ) -> jnp.ndarray:
        return vmap(SARLPPO._compute_neg_log_pdf_value, in_axes=(None, 0, 0, 0, 0))(
            self,
            mu1s,
            mu2s,
            logsigmas,
            actions,
        )

    @partial(jit, static_argnames=("self"))
    def batch_sample_action(
        self,
        mu1:jnp.ndarray,
        mu2:jnp.ndarray,
        sigma:jnp.ndarray,
        keys:random.PRNGKey,
    ) -> jnp.ndarray:
        return vmap(SARLPPO._sample_action, in_axes=(None, None, None, None, 0))(
            self, 
            mu1, 
            mu2, 
            sigma,
            keys,
        )

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
        beta_entropy:float,
        clip_range:float,
        debugging:bool=False,
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss = self._compute_rl_loss_and_gradients(
                critic_params, 
                actor_params,
                experiences,
                beta_entropy,
                clip_range,
                debugging=False, #debugging,
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
        ## Debug
        lax.cond(
            debugging,
            lambda _: debug.print(
                "Sigma gradient: {x}\nSigma update: {y}", 
                x = actor_grads['actor']['logsigma'],
                y = actor_updates['actor']['logsigma'],
            ),
            lambda _: None,
            None,
        )
        return (
            updated_critic_params, 
            updated_actor_params, 
            critic_opt_state, 
            actor_opt_state, 
            critic_loss, 
            actor_loss, 
            entropy_loss
        )
    
    @partial(jit, static_argnames=("self","actor_optimizer","critic_optimizer"))
    def update_il(
        self, 
        critic_params:dict, 
        actor_params:dict,
        actor_optimizer:optax.GradientTransformation, 
        actor_opt_state: jnp.ndarray, 
        critic_optimizer:optax.GradientTransformation,
        critic_opt_state: jnp.ndarray,
        experiences:dict[str:jnp.ndarray], 
    ) -> tuple:
        # Compute loss and gradients for actor and critic
        critic_loss, critic_grads, actor_loss, actor_grads, entropy_loss = self._compute_il_loss_and_gradients(
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
        return (
            updated_critic_params, 
            updated_actor_params, 
            critic_opt_state, 
            actor_opt_state, 
            critic_loss, 
            actor_loss, 
            entropy_loss
        )