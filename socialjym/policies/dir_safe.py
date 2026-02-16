import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from jax.tree_util import tree_map
from functools import partial
import haiku as hk
from types import FunctionType
import optax
import os
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

from socialjym.envs.base_env import EPSILON, HUMAN_POLICIES, wrap_angle
from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.distributions.base_distribution import DISTRIBUTIONS
from socialjym.utils.distributions.dirichlet import Dirichlet
from .sarl import SARL
from .sarl import ValueNetwork
from .sarl_ppo import EPSILON, MLP_1_PARAMS, MLP_2_PARAMS, MLP_3_PARAMS, ATTENTION_LAYER_PARAMS
from .jessi import JESSI
from jhsfm.hsfm import get_linear_velocity

class Actor(hk.Module):
    def __init__(
            self,
            v_max:float,
            wheels_distance:float,
            mlp1_params:dict=MLP_1_PARAMS,
            mlp2_params:dict=MLP_2_PARAMS,
            mlp4_params:dict=MLP_3_PARAMS,
            attention_layer_params:dict=ATTENTION_LAYER_PARAMS,
            robot_state_size:int=9,  # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma)
        ) -> None:
        super().__init__() 
        self.mlp1 = hk.nets.MLP(**mlp1_params, name="mlp1")
        self.mlp2 = hk.nets.MLP(**mlp2_params, name="mlp2")
        self.mlp4 = hk.nets.MLP(**mlp4_params, name="mlp4")
        self.vmax = v_max
        self.wheels_distance = wheels_distance
        self.distr = Dirichlet(EPSILON)
        n_outputs = 3
        self.output_layer = hk.Linear(n_outputs, w_init=hk.initializers.Orthogonal(scale=0.01), b_init=hk.initializers.Constant(0.), name="output_layer")
        self.attention = hk.nets.MLP(**attention_layer_params, name="attention")
        self.robot_state_size = robot_state_size

    def __call__(
            self, 
            x: jnp.ndarray,
            mask: jnp.ndarray = None,
            **kwargs:dict,
        ) -> jnp.ndarray:
        """
        Computes the action given the re-parametrized state x.
        """
        ## Get kwargs
        random_key = kwargs.get("random_key", random.PRNGKey(0))
        # WARNING: If later you want to normalize the input, the distribution dict must still contatain the unnormalized action space parameters
        action_space_params = x[0, self.robot_state_size-3:self.robot_state_size]  # The action space parameters are the last three elements of the robot state (whuch is repeated in each row of the state)
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
        if mask is not None: scores = scores * mask[:,None] # Set attention scores to zero for non existing humans
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
        alphas = nn.softplus(alphas) + 1
        ## Compute dirchlet distribution parameters
        vertices = jnp.array([
            [0, action_space_params[1] * (2 * self.vmax / self.wheels_distance)],
            [0, action_space_params[2] * (-2 * self.vmax / self.wheels_distance)],
            [action_space_params[0] * self.vmax, 0]
        ])
        distribution = {"alphas": alphas, "vertices": vertices}
        ## Sample action
        sampled_action = self.distr.sample(distribution, random_key)
        return sampled_action, distribution

class DIRSAFE(SARL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            v_max:float=1., 
            gamma:float=0.9, 
            dt:float=0.25, 
            wheels_distance:float=0.7, 
            noise:bool=False, # If True, noise is added to humams positions and velocities
            noise_sigma_percentage:float=0., # Standard deviation of the noise as a percentage of the absolute value of the difference between the robot and the humans
            # USED ONLY FOR LASERNAV:
            lidar_n_stack_to_use:int=1, # Number of past lidar observations to stack and use as input for the policy. If 1, only the current observation is used.
        ) -> None:
        """
        DIRSAFE (DIRichlet-based Socially Aware FEasible-action) is an RL policy that takes in input a local map of the static obstacles in the environment, the robot state and the human states, and
        outputs a continuous action parameterized by a Dirichlet distribution whose support is guaranteed to avoid collisions with the static obstacles.
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
        self.lidar_n_stack_to_use = lidar_n_stack_to_use
        # Default attributes
        self.name = "DIRSAFE"
        self.distr_id = DISTRIBUTIONS.index('dirichlet')
        self.distr = Dirichlet(EPSILON)
        self.normalize_and_clip_obs =  False
        self.vnet_input_size = 16 # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma), 7 as the human state (px, py, vx, vy, radius, dg, da)
        @hk.transform
        def value_network(x, mask=None, robot_state_size=9):  # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma)
            vnet = ValueNetwork(robot_state_size=robot_state_size)
            return vnet(x, mask=mask)
        self.critic = value_network
        @hk.transform
        def actor_network(x:jnp.ndarray, mask=None, **kwargs) -> jnp.ndarray:
            actor = Actor(v_max=self.v_max, wheels_distance=self.wheels_distance, robot_state_size=6+3) # 6 as the standard SARL implementation, 3 as the additional action space parameters (alpha, beta, gamma)
            return actor(x, mask=mask, **kwargs)
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
        # Robot observation: [x,y,v,w,radius,theta] (6)
        # Human observation: [x,y,vx,vy,radius] (5)
        # Aaction space parameters: [alpha, beta, gamma] (3)
        # Re-parametrized observation: [dg,v_pref,theta,radius,vx,vy,alpha,beta,gamma,px1,py1,vx1,vy1,radius1,da,radius_sum] (16)
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
    def batch_compute_vnet_input(self, robot_obs:jnp.ndarray, humans_obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        # Compute action space parameters (alpha, beta, gamma) - Done one time for all humans
        obstacles = info["static_obstacles"][-1] # Obstacles are repeated for each agent, we take the last one, corresponding to the robot agent
        obstacle_segments = obstacles.reshape((obstacles.shape[0] * obstacles.shape[1], 2, 2)) # Concatenate all segments regardless of the obstacle they belong to
        action_space_params = self.bound_action_space(
            obstacle_segments, 
            robot_obs[:2], # robot position
            robot_obs[5], # robot orientation
            robot_obs[4], # robot radius
        )
        return vmap(DIRSAFE._compute_vnet_input,in_axes=(None,None,0,None,None))(self, robot_obs, humans_obs, action_space_params, info)

    @partial(jit, static_argnames=("self"))
    def batch_compute_vnet_input_from_lidar(self, robot_obs:jnp.ndarray, humans_obs:jnp.ndarray, info:dict, point_cloud:jnp.ndarray) -> jnp.ndarray:
        # Compute action space parameters (alpha, beta, gamma) - Done one time for all humans
        action_space_params = self.bound_action_space_from_lidar(
            point_cloud, 
            robot_obs[4], # robot radius
        )
        return vmap(DIRSAFE._compute_vnet_input,in_axes=(None,None,0,None,None))(self, robot_obs, humans_obs, action_space_params, info)

    @partial(jit, static_argnames=("self"))
    def init_nns(
        self, 
        key:random.PRNGKey, 
        obs:jnp.ndarray, 
        info:dict,
    ) -> tuple:
        actor_params = self.actor.init(key, jnp.zeros((len(obs)-1, self.vnet_input_size)))
        critic_params = self.critic.init(key, jnp.zeros((len(obs)-1, self.vnet_input_size)))
        return actor_params, critic_params

    @partial(jit, static_argnames=("self"))
    def bound_action_space(self, obstacle_segments, robot_position, robot_orientation, robot_radius):
        """
        Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
        """
        # Convert obstacle segments to absolute coordinates
        # Translate segments to robot frame
        obstacle_segments = obstacle_segments.at[:, :, 0].set(obstacle_segments[:, :, 0] - robot_position[0])
        obstacle_segments = obstacle_segments.at[:, :, 1].set(obstacle_segments[:, :, 1] - robot_position[1])
        # Rotate segments by -robot_orientation
        c, s = jnp.cos(-robot_orientation), jnp.sin(-robot_orientation)
        rot = jnp.array([[c, -s], [s, c]])
        obstacle_segments = jnp.einsum('ij,klj->kli', rot, obstacle_segments)
        # Lower ALPHA
        _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
            obstacle_segments[:,0,0],
            obstacle_segments[:,0,1],
            obstacle_segments[:,1,0],
            obstacle_segments[:,1,1],
            # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
            0. + 1e-6, # xmin
            self.v_max * self.dt + robot_radius - 1e-6, # xmax
            -robot_radius + 1e-6, # ymin
            robot_radius - 1e-6, # ymax
        )
        intersection_points = jnp.vstack((intersection_points0, intersection_points1))
        min_x = jnp.nanmin(intersection_points[:,0])
        new_alpha = lax.cond(
            ~jnp.isnan(min_x),
            lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (self.v_max * self.dt),
            lambda _: 1.,
            None,
        )
        @jit
        def _lower_beta_and_gamma(tup:tuple):
            obstacle_segments, new_alpha, vmax, wheels_distance, dt, robot_radius = tup
            # Lower BETA
            _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
                obstacle_segments[:,0,0],
                obstacle_segments[:,0,1],
                obstacle_segments[:,1,0],
                obstacle_segments[:,1,1],
                # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
                -robot_radius + 1e-6, # xmin
                new_alpha * vmax * dt + robot_radius - 1e-6, # xmax
                robot_radius + 1e-6, # ymin
                robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance)) - 1e-6, # ymax
            )
            intersection_points = jnp.vstack((intersection_points0, intersection_points1))
            min_y = jnp.nanmin(intersection_points[:,1])
            new_beta = lax.cond(
                ~jnp.isnan(min_y),
                lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            # Lower GAMMA
            _, intersection_points0, intersection_points1 = self._batch_segment_rectangle_intersection(
                obstacle_segments[:,0,0],
                obstacle_segments[:,0,1],
                obstacle_segments[:,1,0],
                obstacle_segments[:,1,1],
                # Restricting the rectangle by 1e-6 avoids problems when obstacles are parallel or perpendicular to the robot's direction
                -robot_radius + 1e-6, # xmin
                new_alpha * vmax * dt + robot_radius - 1e-6, # xmax
                -robot_radius - (new_alpha*dt**2*vmax**2/(4*wheels_distance)) + 1e-6, # ymin
                -robot_radius - 1e-6, # ymax
            )
            intersection_points = jnp.vstack((intersection_points0, intersection_points1))
            max_y = jnp.nanmax(intersection_points[:,1])
            new_gamma = lax.cond(
                ~jnp.isnan(max_y),
                lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            return new_beta, new_gamma
        new_beta, new_gamma = lax.cond(
            new_alpha == 0.,
            lambda _: (1., 1.),
            _lower_beta_and_gamma,
            (obstacle_segments, new_alpha, self.v_max, self.wheels_distance, self.dt, robot_radius)
        )
        # Apply lower blound to new_alpha, new_beta, new_gamma
        new_alpha = jnp.max(jnp.array([EPSILON, new_alpha]))
        new_beta = jnp.max(jnp.array([EPSILON, new_beta]))
        new_gamma = jnp.max(jnp.array([EPSILON, new_gamma]))
        return jnp.array([new_alpha, new_beta, new_gamma])

    @partial(jit, static_argnames=("self"))
    def bound_action_space_from_lidar(self, lidar_point_cloud, robot_radius, eps=1e-6):
        """
        Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
        WARNING: Assumes LiDAR orientation is align with robot frame.
        """
        # Lower ALPHA
        is_inside_frontal_rect = (
            (lidar_point_cloud[:,0] >=  0 + eps) & # xmin
            (lidar_point_cloud[:,0] <= self.v_max * self.dt + robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= -robot_radius + eps) &  # ymin
            (lidar_point_cloud[:,1] <= robot_radius - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_frontal_rect[:, None],
            lidar_point_cloud,
            jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
        )
        min_x = jnp.nanmin(intersection_points[:,0])
        new_alpha = lax.cond(
            ~jnp.isnan(min_x),
            lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (self.v_max * self.dt),
            lambda _: 1.,
            None,
        )
        @jit
        def _lower_beta_and_gamma(tup:tuple):
            lidar_point_cloud, new_alpha, vmax, wheels_distance, dt = tup
            # Lower BETA
            is_inside_left_rect = (
                (lidar_point_cloud[:,0] >= -robot_radius + eps) & # xmin
                (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + robot_radius - eps) & # xmax
                (lidar_point_cloud[:,1] >= robot_radius + eps) &  # ymin
                (lidar_point_cloud[:,1] <= robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance)) - eps) # ymax
            )
            intersection_points = jnp.where(
                is_inside_left_rect[:, None],
                lidar_point_cloud,
                jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
            )
            min_y = jnp.nanmin(intersection_points[:,1])
            new_beta = lax.cond(
                ~jnp.isnan(min_y),
                lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            # Lower GAMMA
            is_inside_right_rect = (
                (lidar_point_cloud[:,0] >=  -robot_radius + eps) & # xmin
                (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + robot_radius - eps) & # xmax
                (lidar_point_cloud[:,1] >= -robot_radius - (new_alpha*dt**2*vmax**2/(4*wheels_distance)) + eps) & # ymin
                (lidar_point_cloud[:,1] <= -robot_radius - eps) # ymax
            )
            intersection_points = jnp.where(
                is_inside_right_rect[:, None],
                lidar_point_cloud,
                jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
            )
            max_y = jnp.nanmax(intersection_points[:,1])
            new_gamma = lax.cond(
                ~jnp.isnan(max_y),
                lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
                lambda _: 1.,
                None,
            )
            return new_beta, new_gamma
        new_beta, new_gamma = lax.cond(
            new_alpha == 0.,
            lambda _: (1., 1.),
            _lower_beta_and_gamma,
            (lidar_point_cloud, new_alpha, self.v_max, self.wheels_distance, self.dt)
        )
        # Apply lower bound to new_alpha, new_beta, new_gamma
        new_alpha = jnp.max(jnp.array([EPSILON, new_alpha]))
        new_beta = jnp.max(jnp.array([EPSILON, new_beta]))
        new_gamma = jnp.max(jnp.array([EPSILON, new_gamma]))
        return jnp.array([new_alpha, new_beta, new_gamma])

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
        return vmap(DIRSAFE.act, in_axes=(None, 0, 0, 0, None, None))(
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

    # LaserNav methods

    @partial(jit, static_argnames=("self","jessi"))
    def act_on_jessi_perception(
        self, 
        jessi:JESSI,
        perception_params:dict,
        key:random.PRNGKey, 
        lasernav_obs:jnp.ndarray, 
        info:dict, 
        actor_params:dict, 
        humans_radius:float,
        sample:bool = False,
    ) -> jnp.ndarray:
        ## Compute aligned point_cloud
        aligned_lidar = jessi.align_lidar(lasernav_obs[:self.lidar_n_stack_to_use])[1]
        point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        ## Identify visible humans with JESSI perception
        hcgs, _, _ = jessi.perception.apply(perception_params, None, jessi.compute_perception_input(lasernav_obs)[0])
        humans_mask = hcgs['weights'] > 0.5
        rc_humans_pos = hcgs['pos_distrs']['means']
        rc_humans_vel = hcgs['vel_distrs']['means']
        # Extract robot pose
        robot_position = lasernav_obs[0,:2]
        robot_theta = lasernav_obs[0,2]
        # Make humans positions and velocities in the world frame (later they will be transformed in the robot frame inside the vnet_input computation. This is inefficient but it's easier to reuse the same batch_compute_vnet_input function for both LaserNav and SocialNav observations)
        humans_pos = jnp.zeros_like(rc_humans_pos)
        humans_pos = humans_pos.at[:,0].set(rc_humans_pos[:,0] * jnp.cos(robot_theta) - rc_humans_pos[:,1] * jnp.sin(robot_theta) + robot_position[0])
        humans_pos = humans_pos.at[:,1].set(rc_humans_pos[:,0] * jnp.sin(robot_theta) + rc_humans_pos[:,1] * jnp.cos(robot_theta) + robot_position[1])
        humans_vel = jnp.zeros_like(rc_humans_vel)
        humans_vel = humans_vel.at[:,0].set(rc_humans_vel[:,0] * jnp.cos(robot_theta) - rc_humans_vel[:,1] * jnp.sin(robot_theta))
        humans_vel = humans_vel.at[:,1].set(rc_humans_vel[:,0] * jnp.sin(robot_theta) + rc_humans_vel[:,1] * jnp.cos(robot_theta))
        # SOCIALNAV OBSERVATION FORMAT:
        # - obs: observation of the current state. It is in the form:
        #         [[human1_px, human1_py, human1_vx, human1_vy, human1_radius, padding],
        #         [human2_px, human2_py, human2_vx, human2_vy, human2_radius, padding],
        #         ...
        #         [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius, padding],
        #         [robot_px, robot_py, robot_u1, robot_u2, robot_radius, robot_theta]].
        obs = jnp.zeros((len(humans_mask)+1, 6))
        obs = obs.at[:-1,0:2].set(humans_pos)
        obs = obs.at[:-1,2:4].set(humans_vel)
        obs = obs.at[:-1,4].set(humans_radius)
        obs = obs.at[-1,0:2].set(robot_position) # Current Robot position
        obs = obs.at[-1,2:4].set(lasernav_obs[0,4:6]) # Current Robot action (velocity)
        obs = obs.at[-1,4].set(lasernav_obs[0,3]) # Robot radius
        obs = obs.at[-1,5].set(robot_theta) # Robot theta

        # Compute actor input
        input = self.batch_compute_vnet_input_from_lidar(obs[-1], obs[:-1], info, point_cloud)

        @jit
        def _forward_pass_sample_action(val):
            vnet_input, key = val
            key, subkey = random.split(key)
            sampled_action, distr = self.actor.apply(
                actor_params, 
                None, 
                vnet_input, 
                humans_mask, # Mask to consider only visible humans in the actor forward pass
                random_key=subkey
            )
            return sampled_action, sampled_action, distr, key
        @jit
        def _forward_pass_mean_action(val):
            vnet_input, key = val
            sampled_action, distr = self.actor.apply(
                actor_params, 
                None, 
                vnet_input, 
                humans_mask, # Mask to consider only visible humans in the actor forward pass
            )
            return self.distr.mean(distr), sampled_action, distr, key
        
        # Compute action
        action, sampled_action, distr, key = lax.cond(
            sample, 
            _forward_pass_sample_action,
            _forward_pass_mean_action,
            (input, key)
        )
        return action, key, input, sampled_action, distr, hcgs