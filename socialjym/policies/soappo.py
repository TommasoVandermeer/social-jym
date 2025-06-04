import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from socialjym.envs.base_env import wrap_angle
from socialjym.utils.distributions.base_distribution import DISTRIBUTIONS
from socialjym.utils.distributions.dirichlet import Dirichlet
from .sarl import SARL
from .sarl import value_network as critic_network
from .sarl_ppo import EPSILON, NORM_EPSILON, MLP_1_PARAMS, MLP_2_PARAMS, MLP_3_PARAMS, ATTENTION_LAYER_PARAMS, Actor

class Actor(hk.Module):
    def __init__(
            self,
            v_max:float,
            wheels_distance:float,
            mlp1_params:dict=MLP_1_PARAMS,
            mlp2_params:dict=MLP_2_PARAMS,
            mlp4_params:dict=MLP_3_PARAMS,
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS,
            robot_state_size:int=6,
        ) -> None:
        super().__init__() 
        self.mlp1 = hk.nets.MLP(**mlp1_params, name="mlp1")
        self.mlp2 = hk.nets.MLP(**mlp2_params, name="mlp2")
        self.mlp4 = hk.nets.MLP(**mlp4_params, name="mlp4")
        self.vmax = v_max
        self.wheels_distance = wheels_distance
        self.distr = Dirichlet(v_max, wheels_distance, EPSILON)
        n_outputs = 3
        self.output_layer = hk.Linear(n_outputs, w_init=hk.initializers.Orthogonal(scale=0.01), b_init=hk.initializers.Constant(0.), name="output_layer")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size

    def __call__(
            self, 
            x: jnp.ndarray,
            **kwargs:dict,
        ) -> jnp.ndarray:
        """
        Computes the value of the state given the input x of shape (# of humans, length of reparametrized state)
        """
        ## Get kwargs
        random_key = kwargs.get("random_key", random.PRNGKey(0))
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
        alphas = self.output_layer(mlp4_output)
        ## Compute dirchlet distribution parameters
        alphas = nn.softplus(alphas) + 1
        distribution = {"alphas": alphas}
        ## Sample action
        sampled_action = self.distr.sample(distribution, random_key)
        return sampled_action, distribution

class SOAPPO(SARL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            normalize_and_clip_obs:bool=False,
            clip_obs_bound:float=10.,
            v_max:float=1., 
            gamma:float=0.9, 
            dt:float=0.25, 
            wheels_distance:float=0.7, 
            noise:bool=False, # If True, noise is added to humams positions and velocities
            noise_sigma_percentage:float=0., # Standard deviation of the noise as a percentage of the absolute value of the difference between the robot and the humans
        ) -> None:
        """
        SOAPPO (Social and Obstacle Aware Proximal Policy Optimization) is an RL policy that takes in input a local map of the static obstacles in the environment, the robot state and the human states, and
        outputs a constinuous action parameterized by a Dirichlet distribution whose support is guaranteed to avoid collisions with the static obstacles.
        """
        # Configurable attributes
        super().__init__(
            reward_function=reward_function, 
            v_max=v_max, 
            wheels_distance=wheels_distance,
            gamma=gamma, 
            dt=dt,
            kinematics='unicycle',
            noise=noise,
            noise_sigma_percentage=noise_sigma_percentage,
        )
        # Default attributes
        self.name = "SOAPPO"
        self.distr_id = DISTRIBUTIONS.index('dirichlet')
        self.normalize_and_clip_obs = normalize_and_clip_obs
        self.distr = Dirichlet(self.v_max, self.wheels_distance, EPSILON)
        if self.normalize_and_clip_obs:
            self.clip_obs_bound = clip_obs_bound
        self.critic = critic_network(robot_state_size=6+3) # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma)
        @hk.transform
        def actor_network(x:jnp.ndarray, **kwargs) -> jnp.ndarray:
            actor = Actor(v_max=self.v_max, wheels_distance=self.wheels_distance, robot_state_size=6+3) # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma)
            return actor(x, **kwargs)
        self.actor = actor_network

    # Private methods

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
                _, distr = self.actor.apply(current_actor_params, None, input)
                # Compute the log probability of the action
                neglogpdf = self.distr.neglogp(distr, sample_action)
                # Compute policy ratio
                ratio = jnp.exp(old_neglogpdf - neglogpdf)
                lax.cond(
                    debugging,
                    lambda _: debug.print(
                        "Ratio: {x} - Old neglogp: {y} - New neglogp: {z} - distr: {w} - action: {a} - advantage: {b}", 
                        x=ratio,
                        y=old_neglogpdf,
                        z=neglogpdf,
                        w=distr,
                        a=sample_action,
                        b=advantage,
                    ),
                    lambda _: None,
                    None,
                )
                # Compute actor loss
                actor_loss = jnp.maximum(- ratio * advantage, - jnp.clip(ratio, 1-clip_range, 1+clip_range) * advantage)
                # Compute the entropy loss
                entropy_loss = self.distr.entropy(distr)
                # Compute the loss
                return actor_loss, entropy_loss
            
            actor_losses, entropy_losses = _rl_loss_function(current_actor_params, inputs, sample_actions, advantages, old_neglogpdfs)
            actor_loss = jnp.mean(actor_losses)
            entropy_loss = - beta_entropy * jnp.mean(entropy_losses)
            loss = actor_loss + entropy_loss
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
                _, distr = self.actor.apply(current_actor_params, None, input)
                # Get mean action
                action = self.distr.mean(distr)
                # if self.distr_id == DISTRIBUTIONS.index('gaussian'):
                #     action = self.distr.bound_action(action, self.kinematics, self.v_max, self.wheels_distance)
                # Compute the loss
                return 0.5 * jnp.sum(jnp.square(action - sample_action))
            
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

    @partial(jit, static_argnames=("self"))
    def _compute_vnet_input(self, robot_obs:jnp.ndarray, human_obs:jnp.ndarray, action_space_params:jnp.ndarray, info:dict) -> jnp.ndarray:
        # Robot observation: [x,y,u1,u2,radius,theta]. Holonomic robot: [u1=vx,u2=vy]. Unicycle robot: [u1=v,u2=w].
        # Human observation: [x,y,vx,vy,radius]
        # Re-parametrized observation: [dg,v_pref,theta,radius,vx,vy,alpha,beta,gamma,px1,py1,vx1,vy1,radius1,da,radius_sum]
        rot = jnp.atan2(info["robot_goal"][1] - robot_obs[1],info["robot_goal"][0] - robot_obs[0])
        vnet_input = jnp.zeros((self.vnet_input_size,))
        # Robot state
        vnet_input = vnet_input.at[0].set(jnp.linalg.norm(info["robot_goal"] - robot_obs[0:2]))
        vnet_input = vnet_input.at[1].set(self.v_max)
        vnet_input = vnet_input.at[3].set(robot_obs[4])  
        vnet_input = vnet_input.at[2].set(wrap_angle(robot_obs[5] - rot))
        vnet_input = vnet_input.at[4].set(robot_obs[2] * jnp.cos(robot_obs[5]) * jnp.cos(rot) + robot_obs[2]  * jnp.sin(robot_obs[5]) * jnp.sin(rot))
        vnet_input = vnet_input.at[5].set(-robot_obs[2] * jnp.cos(robot_obs[5]) * jnp.sin(rot) + robot_obs[2]  * jnp.sin(robot_obs[5]) * jnp.cos(rot))
        vnet_input = vnet_input.at[6].set(action_space_params[0])  # alpha
        vnet_input = vnet_input.at[7].set(action_space_params[1])  # beta
        vnet_input = vnet_input.at[8].set(action_space_params[2])  # gamma
        # Humans state
        vnet_input = vnet_input.at[9].set((human_obs[0] - robot_obs[0]) * jnp.cos(rot) + (human_obs[1] - robot_obs[1]) * jnp.sin(rot))
        vnet_input = vnet_input.at[10].set(-(human_obs[0] - robot_obs[0]) * jnp.sin(rot) + (human_obs[1] - robot_obs[1]) * jnp.cos(rot))
        vnet_input = vnet_input.at[11].set(human_obs[2] * jnp.cos(rot) + human_obs[3] * jnp.sin(rot))
        vnet_input = vnet_input.at[12].set(-human_obs[2] * jnp.sin(rot) + human_obs[3] * jnp.cos(rot))
        vnet_input = vnet_input.at[13].set(human_obs[4])
        vnet_input = vnet_input.at[14].set(jnp.linalg.norm(human_obs[0:2] - robot_obs[0:2]))
        vnet_input = vnet_input.at[15].set(robot_obs[4] + human_obs[4])
        return vnet_input

    # Public methods

    @partial(jit, static_argnames=("self"))
    def batch_compute_vnet_input(self, robot_obs:jnp.ndarray, humans_obs:jnp.ndarray, action_space_params:jnp.ndarray, info:dict) -> jnp.ndarray:
        return vmap(SOAPPO._compute_vnet_input,in_axes=(None,None,0,None,None))(self, robot_obs, humans_obs, action_space_params, info)

    @partial(jit, static_argnames=("self"))
    def init_nns(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict,
    ) -> tuple:
        actor_params = self.actor.init(key, jnp.zeros((len(obs)-1, self.vnet_input_size)))
        critic_params = self.critic.init(key, jnp.zeros((len(obs)-1, self.vnet_input_size)))
        if self.normalize_and_clip_obs:
            input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
            norm_state = {
                "mean": jnp.zeros(input.shape[1:]),
                "var": jnp.ones(input.shape[1:]),
                "count": 0,
            }
            actor_params = {"norm_state": norm_state, "actor_params": actor_params}
        return actor_params, critic_params

    @partial(jit, static_argnames=("self"))
    def update_norm_state(
        self,
        inputs:jnp.ndarray,
        norm_state:dict,
    ) -> tuple:
        ## Update normalization state
        batch_mean = jnp.mean(inputs, axis=0)
        batch_var = jnp.var(inputs, axis=0)
        batch_count = inputs.shape[0]
        delta = batch_mean - norm_state["mean"]
        tot_count = norm_state["count"] + batch_count
        new_mean = norm_state["mean"] + delta * batch_count / tot_count
        m_a = norm_state["var"] * norm_state["count"]
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * norm_state["count"] * batch_count / tot_count
        new_var = m2 / tot_count
        norm_state = {
            "mean": new_mean,
            "var": new_var,
            "count": tot_count,
        }
        ## Normalize and clip inputs
        return norm_state

    @partial(jit, static_argnames=("self"))
    def normalize_and_clip_inputs(
        self,
        inputs:jnp.ndarray,
        norm_state:dict,
    ) -> jnp.ndarray:
        inputs = jnp.clip((inputs - norm_state["mean"]) / jnp.sqrt(norm_state["var"] + NORM_EPSILON), -self.clip_obs_bound, self.clip_obs_bound)
        return inputs

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
        input = self.batch_compute_vnet_input(obs[-1], obs[:-1], info)
        if self.normalize_and_clip_obs:
            input = self.normalize_and_clip_inputs(input, actor_params["norm_state"])
            actor_params = actor_params["actor_params"]
        # Compute action
        key, subkey = random.split(key)
        sampled_action, distr = self.actor.apply(
            actor_params, 
            None, 
            input, 
            random_key=subkey
        )
        action = lax.cond(sample, lambda _: sampled_action, lambda _: self.distr.mean(distr), None)
        return action, key, input, sampled_action, distr
    
    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        actor_params,
        sample,
    ):
        return vmap(SOAPPO.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos, 
            actor_params, 
            sample,
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
                debugging=debugging, #debugging,
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