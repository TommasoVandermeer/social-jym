import jax.numpy as jnp
from jax import random, jit, vmap, lax, nn, value_and_grad, debug
from jax_tqdm import loop_tqdm
from jax.tree_util import tree_map
from functools import partial
import haiku as hk
from types import FunctionType
import optax
import os
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

from .base_policy import BasePolicy
from socialjym.envs.base_env import EPSILON, ROBOT_KINEMATICS, SCENARIOS, HUMAN_POLICIES, wrap_angle
from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import initialize_metrics_dict, compute_episode_metrics, print_average_metrics
from socialjym.policies.jessi import JESSI
from jhsfm.hsfm import get_linear_velocity

VN_PARAMS = {
    "output_sizes": [150, 100, 100, 1],
    "activation": nn.relu,
    "activate_final": False,
    "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
    "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
}

@hk.transform
def value_network(x):
    mlp = hk.nets.MLP(**VN_PARAMS)
    return mlp(x)

class CADRL(BasePolicy):
    def __init__(
            self, 
            reward_function:FunctionType, 
            v_max:float = 1., 
            gamma:float = 0.9, 
            dt:float = 0.25, 
            wheels_distance:float = 0.7, 
            kinematics:str = 'holonomic', 
            unicycle_box_action_space:bool = False, # If True, along with unicycle kinematics the action space is limited to a box (not triangle)
            noise:bool = False, # If True, noise is added to humams positions and velocities
            noise_sigma_percentage:float = 0., # Standard deviation of the noise as a percentage of the absolute value of the difference between the robot and the humans
            # position_noise_sigma_percentage_radius:float = 0., # Standard deviation of the noise as a percentage of the ditance between the robot and the humans
            # position_noise_sigma_angle:float = 0., # Standard deviation of the noise on the angle of humans' position in the robot frame
            # velocity_noise_sigma_percentage:float = 0., # Standard deviation of the noise as a percentage of the (vx,vy) coordinates of humans' velocity in the robot frame
        ) -> None:
        # Inputs validation
        assert gamma == reward_function.gamma, "gamma must be equal to the reward function gamma"
        assert v_max == reward_function.v_max, "v_max must be equal to the reward function v_max"
        assert v_max > 0, "v_max must be positive"
        assert dt > 0, "dt must be positive"
        assert wheels_distance > 0, "wheels_distance must be positive"
        assert kinematics in ROBOT_KINEMATICS, f"kinematics must be one of {ROBOT_KINEMATICS}"
        assert reward_function.kinematics == ROBOT_KINEMATICS.index(kinematics), "Reward function kinematics must match the robot policy kinematics"
        # Configurable attributes
        super().__init__(gamma)
        self.reward_function = reward_function
        self.v_max = v_max
        self.wheels_distance = wheels_distance # Distance between the wheels of the robot (used for unicycle kinematics). Value taken from real TurtleBot4 robot.
        self.dt = dt
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        self.noise = noise
        self.noise_sigma_percentage = noise_sigma_percentage
        self.unicycle_box_action_space = unicycle_box_action_space
        # self.position_noise_sigma_percentage_radius = position_noise_sigma_percentage_radius
        # self.position_noise_sigma_angle = position_noise_sigma_angle
        # self.velocity_noise_sigma_percentage = velocity_noise_sigma_percentage
        # Default attributes
        self.name = "CADRL"
        self.vnet_input_size = 13
        self.model = value_network
        self.action_space = self._build_action_space()

    # Private methods

    def _build_action_space(
        self,
        holonomic_speed_sample:int=5,
        holonomic_rotation_samples:int=16,
        unicycle_box_samples:int=9,
        unicycle_triangle_samples:int=9,
    ) -> jnp.ndarray:
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            # The number of actions will be equal to (holonomic_speed_sample * holonomic_rotation_samples + 1)
            speeds = lax.fori_loop(0,holonomic_speed_sample,lambda i, speeds: speeds.at[i].set((jnp.exp((i + 1) / holonomic_speed_sample) - 1) / (jnp.e - 1) * self.v_max), jnp.zeros((holonomic_speed_sample,)))
            rotations = jnp.linspace(0, 2 * jnp.pi, holonomic_rotation_samples, endpoint=False)
            action_space = jnp.empty((holonomic_speed_sample * holonomic_rotation_samples + 1,2))
            action_space = action_space.at[0].set(jnp.array([0., 0.])) # First action is to stay still
            action_space = lax.fori_loop(1,
                                        len(action_space), 
                                        lambda i, acts: acts.at[i].set(jnp.array([
                                            speeds[(i-1) % holonomic_speed_sample] * jnp.cos(rotations[(i-1) // holonomic_speed_sample]),
                                            speeds[(i-1) % holonomic_speed_sample] * jnp.sin(rotations[(i-1) // holonomic_speed_sample])])),
                                        action_space)
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            if self.unicycle_box_action_space:
                # We will subdivide the linear velocities in unicycle_box_samples intervals and the angular velocities in unicycle_box_samples intervals, for a total of unicycle_box_samples**2 actions
                linear_speeds = jnp.linspace(0, self.v_max, unicycle_box_samples)
                angular_speeds = jnp.linspace(-self.v_max/(self.wheels_distance/2), self.v_max/(self.wheels_distance/2), unicycle_box_samples)
                action_space = jnp.empty((len(linear_speeds)*len(angular_speeds),2))
                action_space = lax.fori_loop(
                    0,
                    len(angular_speeds),
                    lambda i, x: lax.fori_loop(
                        0,
                        len(linear_speeds),
                        lambda j, y: y.at[i*len(linear_speeds)+j].set(jnp.array([linear_speeds[j],angular_speeds[i]])),
                        x),
                    action_space)
            else:
                # The number of actions will be equal to (unicycle_triangle_samples**2) (which corresponds to the number of positive linear velocities)
                # The linear velocities are crossed with (unicycle_triangle_samples * 2 -1) angular speeds to create the feasible action space)
                angular_speeds = jnp.linspace(-self.v_max/(self.wheels_distance/2), self.v_max/(self.wheels_distance/2), 2*unicycle_triangle_samples-1)
                speeds = jnp.linspace(0, self.v_max, unicycle_triangle_samples)
                unconstrained_action_space = jnp.empty((len(angular_speeds)*len(speeds),2))
                unconstrained_action_space = lax.fori_loop(
                    0,
                    len(angular_speeds),
                    lambda i, x: lax.fori_loop(
                        0,
                        len(speeds),
                        lambda j, y: lax.cond(
                            jnp.all(jnp.array([i<len(angular_speeds)-j, i>=j])),
                            lambda z: z.at[i*len(speeds)+j].set(jnp.array([speeds[j],angular_speeds[i]])),
                            lambda z: z.at[i*len(speeds)+j].set(jnp.array([jnp.nan,jnp.nan])),
                            y),
                        x),
                    unconstrained_action_space)
                action_space = unconstrained_action_space[~jnp.isnan(unconstrained_action_space).any(axis=1)]
        return action_space

    @partial(jit, static_argnames=("self"))
    def _compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict, humans_mask:jnp.ndarray=None) -> jnp.ndarray:
        n_humans = len(next_obs) - 1
        # Compute instantaneous reward
        current_obs = current_obs.at[-1,2:4].set(action)
        reward, _ = self.reward_function(current_obs, info, self.dt)
        # Apply robot action
        next_obs = next_obs.at[-1,2:4].set(action)
        next_obs = next_obs.at[-1].set(self._propagate_robot_obs(next_obs[-1]))
        # Re-parametrize observation, for each human: [dg,v_pref,theta,radius,vx,vy,px1,py1,vx1,vy1,radius1,da,radius_sum]
        vnet_inputs = self.batch_compute_vnet_input(next_obs[n_humans], next_obs[0:n_humans], info)
        # Compute the output of the value network (value of the state)
        vnet_outputs = self.model.apply(vnet_params, None, vnet_inputs)
        if humans_mask is not None:
            vnet_outputs = jnp.where(jnp.expand_dims(humans_mask, axis=-1), vnet_outputs, jnp.full_like(vnet_outputs, jnp.inf)) # Mask the value of non visible humans (if any) so that they don't affect the final value of the action
        # Take the minimum among all outputs (representing the worst case scenario)
        min_vnet_output = jnp.min(vnet_outputs)
        # Compute the final value of the action
        value = reward + pow(self.gamma, self.dt * self.v_max) * min_vnet_output
        return value, vnet_inputs
        
    @partial(jit, static_argnames=("self"))
    def _batch_compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict, humans_mask:jnp.ndarray=None) -> jnp.ndarray:
        return vmap(CADRL._compute_action_value, in_axes=(None,None,None,None,0,None,None))(self, next_obs, current_obs, info, action, vnet_params, humans_mask)
    
    @partial(jit, static_argnames=("self"))
    def _add_noise_to_human_obs(self, human_obs:jnp.ndarray, robot_pos:jnp.ndarray, key:random.PRNGKey) -> jnp.ndarray:
        # ## Add noise to human position
        # key1, key2, key3 = random.split(key, 3)
        # r = jnp.linalg.norm(human_obs[0:2] - robot_pos) * (1. + random.normal(key1, (1,), dtype=jnp.float32)[0] * self.position_noise_sigma_percentage_radius)
        # theta = jnp.arctan2(human_obs[1] - robot_pos[1], human_obs[0] - robot_pos[0]) + (random.normal(key2, (1,), dtype=jnp.float32)[0] * self.position_noise_sigma_angle)
        # human_obs = human_obs.at[0].set(robot_pos[0] + r * jnp.cos(theta))
        # human_obs = human_obs.at[1].set(robot_pos[1] + r * jnp.sin(theta))
        # ## Add noise to human velocity
        # human_obs = human_obs.at[2:4].set(human_obs[2:4] + random.normal(key3, (2,), dtype=jnp.float32) * (self.velocity_noise_sigma_percentage * jnp.abs(human_obs[2:4])))
        
        ## Add noise
        human_obs = human_obs.at[0:2].set(human_obs[0:2] + random.normal(key, (2,), dtype=jnp.float32) * (self.noise_sigma_percentage * jnp.abs(human_obs[0:2] - robot_pos)))
        human_obs = human_obs.at[2:4].set(human_obs[2:4] + random.normal(key, (2,), dtype=jnp.float32) * (self.noise_sigma_percentage * jnp.abs(human_obs[2:4])))
        return human_obs

    @partial(jit, static_argnames=("self"))
    def _batch_add_noise_to_human_obs(self, obs:jnp.ndarray, key:random.PRNGKey) -> jnp.ndarray:
        keys = random.split(key, len(obs)-1)
        noisy_obs = obs.at[:-1].set(vmap(CADRL._add_noise_to_human_obs, in_axes=(None,0,None,0))(self, obs[:-1], obs[-1,0:2], keys))
        return noisy_obs

    @partial(jit, static_argnames=("self"))
    def _propagate_human_obs(self, obs:jnp.ndarray) -> jnp.ndarray:
        return obs.at[:2].set(obs[0:2] + obs[2:4] * self.dt)
    
    @partial(jit, static_argnames=("self"))
    def _propagate_robot_obs(self, obs:jnp.ndarray) -> jnp.ndarray:
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            obs = obs.at[0:2].set(obs[0:2] + obs[2:4] * self.dt)
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            obs = lax.cond(
                jnp.abs(obs[3]) > EPSILON,
                lambda x: x.at[:].set(jnp.array([
                    x[0] + (x[2]/x[3]) * (jnp.sin(x[5] + x[3] * self.dt) - jnp.sin(x[5])),
                    x[1] + (x[2]/x[3]) * (jnp.cos(x[5]) - jnp.cos(x[5] + x[3] * self.dt)),
                    *x[2:5],
                    wrap_angle(x[5] + x[3] * self.dt)
                ])),
                lambda x: x.at[:2].set(jnp.array([
                    x[0] + x[2] * self.dt * jnp.cos(x[5]),
                    x[1] + x[2] * self.dt * jnp.sin(x[5])
                ])),
                obs)
        return obs

    @partial(jit, static_argnames=("self"))
    def _compute_vnet_input(self, robot_obs:jnp.ndarray, human_obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        # Robot observation: [x,y,u1,u2,radius,theta]. Holonomic robot: [u1=vx,u2=vy]. Unicycle robot: [u1=v,u2=w].
        # Human observation: [x,y,vx,vy,radius]
        # Re-parametrized observation: [dg,v_pref,theta,radius,vx,vy,px1,py1,vx1,vy1,radius1,da,radius_sum]
        rot = jnp.atan2(info["robot_goal"][1] - robot_obs[1],info["robot_goal"][0] - robot_obs[0])
        vnet_input = jnp.zeros((self.vnet_input_size,))
        vnet_input = vnet_input.at[0].set(jnp.linalg.norm(info["robot_goal"] - robot_obs[0:2]))
        vnet_input = vnet_input.at[1].set(self.v_max)
        vnet_input = vnet_input.at[3].set(robot_obs[4])
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            vnet_input = vnet_input.at[2].set(0.)
            vnet_input = vnet_input.at[4].set(robot_obs[2] * jnp.cos(rot) + robot_obs[3] * jnp.sin(rot))
            vnet_input = vnet_input.at[5].set(-robot_obs[2] * jnp.sin(rot) + robot_obs[3] * jnp.cos(rot))
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):  
            vnet_input = vnet_input.at[2].set(wrap_angle(robot_obs[5] - rot))
            vnet_input = vnet_input.at[4].set(robot_obs[2] * jnp.cos(robot_obs[5]) * jnp.cos(rot) + robot_obs[2]  * jnp.sin(robot_obs[5]) * jnp.sin(rot))
            vnet_input = vnet_input.at[5].set(-robot_obs[2] * jnp.cos(robot_obs[5]) * jnp.sin(rot) + robot_obs[2]  * jnp.sin(robot_obs[5]) * jnp.cos(rot))
        vnet_input = vnet_input.at[6].set((human_obs[0] - robot_obs[0]) * jnp.cos(rot) + (human_obs[1] - robot_obs[1]) * jnp.sin(rot))
        vnet_input = vnet_input.at[7].set(-(human_obs[0] - robot_obs[0]) * jnp.sin(rot) + (human_obs[1] - robot_obs[1]) * jnp.cos(rot))
        vnet_input = vnet_input.at[8].set(human_obs[2] * jnp.cos(rot) + human_obs[3] * jnp.sin(rot))
        vnet_input = vnet_input.at[9].set(-human_obs[2] * jnp.sin(rot) + human_obs[3] * jnp.cos(rot))
        vnet_input = vnet_input.at[10].set(human_obs[4])
        vnet_input = vnet_input.at[11].set(jnp.linalg.norm(human_obs[0:2] - robot_obs[0:2]))
        vnet_input = vnet_input.at[12].set(robot_obs[4] + human_obs[4])
        return vnet_input

    # Public methods

    @partial(jit, static_argnames=("self"))
    def batch_compute_vnet_input(self, robot_obs:jnp.ndarray, humans_obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        return vmap(CADRL._compute_vnet_input,in_axes=(None,None,0,None))(self, robot_obs, humans_obs, info)

    @partial(jit, static_argnames=("self"))
    def batch_propagate_human_obs(self, obs:jnp.ndarray) -> jnp.ndarray:
        return vmap(CADRL._propagate_human_obs, in_axes=(None, 0))(self, obs)

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, vnet_params:dict, epsilon:float) -> jnp.ndarray:
        
        @jit
        def _random_action(val):
            obs, info, _, key = val
            key, subkey = random.split(key)
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            return random.choice(subkey, self.action_space), key, vnet_inputs, jnp.zeros((len(self.action_space)))
        
        @jit
        def _forward_pass(val):
            obs, info, vnet_params, key = val
            # Add noise to humans' observations
            if self.noise:
                key, subkey = random.split(key)
                obs = self._batch_add_noise_to_human_obs(obs, subkey)
            # Propagate humans state for dt time
            next_obs = jnp.vstack([self.batch_propagate_human_obs(obs[0:-1]),obs[-1]])
            # Compute action values
            action_values, vnet_inputs = self._batch_compute_action_value(next_obs, obs, info, self.action_space, vnet_params)
            action = self.action_space[jnp.argmax(action_values)]
            vnet_input = vnet_inputs[jnp.argmax(action_values)]
            # Return action with highest value
            return action, key, vnet_input, jnp.squeeze(action_values)
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        action, key, vnet_input, action_values = lax.cond(explore, _random_action, _forward_pass, (obs, info, vnet_params, key))
        return action, key, vnet_input, action_values

    @partial(jit, static_argnames=("self"))
    def batch_act(
        self,
        keys,
        obses,
        infos,
        vnet_params,
        epsilon):
        return vmap(CADRL.act, in_axes=(None,0,0,0,None,None))(
            self,
            keys,
            obses,
            infos,
            vnet_params,
            epsilon)

    @partial(jit, static_argnames=("self"))
    def _compute_loss_and_gradients(
        self, 
        current_vnet_params:dict,  
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"vnet_inputs":jnp.ndarray, "targets":jnp.ndarray,}
    ) -> tuple:
        @jit
        def _batch_loss_function(
            current_vnet_params:dict,
            vnet_inputs:jnp.ndarray,
            targets:jnp.ndarray,  
            ) -> jnp.ndarray:
            
            @partial(vmap, in_axes=(None, 0, 0))
            def _loss_function(
                current_vnet_params:dict,
                vnet_input:jnp.ndarray,
                target:jnp.ndarray, 
                ) -> jnp.ndarray:
                # Compute the prediction
                prediction = self.model.apply(current_vnet_params, None, vnet_input)
                # Compute the loss
                return jnp.square(target - prediction)
            
            return jnp.mean(_loss_function(
                    current_vnet_params,
                    vnet_inputs,
                    targets))

        vnet_inputs = experiences["vnet_inputs"]
        targets = experiences["targets"]
        # Compute the loss and gradients
        loss, grads = value_and_grad(_batch_loss_function)(
            current_vnet_params, 
            vnet_inputs,
            targets)
        return loss, grads
    
    @partial(jit, static_argnames=("self"))
    def batch_compute_loss_and_gradients(
        self, 
        current_vnet_params:dict, 
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"vnet_inputs":jnp.ndarray, "targets":jnp.ndarray,}
    ) -> tuple:
        return vmap(CADRL._compute_loss_and_gradients, in_axes=(None, None, 0))(self, current_vnet_params, experiences)
    
    @partial(jit, static_argnames=("self","optimizer"))
    def update(
        self, 
        current_vnet_params:dict, 
        optimizer:optax.GradientTransformation, 
        optimizer_state: jnp.ndarray, 
        experiences:dict[str:jnp.ndarray],
        # Experiences: {"vnet_inputs":jnp.ndarray, "targets":jnp.ndarray,}
    ) -> tuple:
        # Compute loss and gradients
        loss, grads = self._compute_loss_and_gradients(current_vnet_params, experiences)
        # Compute parameter updates
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        # Apply updates
        updated_vnet_params = optax.apply_updates(current_vnet_params, updates)
        return updated_vnet_params, optimizer_state, loss
    
    def evaluate(
        self,
        n_trials:int,
        random_seed:int,
        env:SocialNav,
        network_params:dict,
    ) -> dict:
        """
        Test the trained policy over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, SocialNav), "Environment must be an instance of SocialNav"
        assert env.kinematics == self.kinematics, "Policy kinematics must match environment kinematics"
        assert env.robot_dt == self.dt, f"Environment time step (dt={env.dt}) must be equal to policy time step (dt={self.dt}) for evaluation"
        time_limit = env.reward_function.time_limit
        @loop_tqdm(n_trials)
        @jit
        def _fori_body(i:int, for_val:tuple):   
            @jit
            def _while_body(while_val:tuple):
                # Retrieve data from the tuple
                state, obs, info, outcome, policy_key, steps, all_actions, all_states = while_val
                action, policy_key, _ = self.act(policy_key, obs, info, network_params, epsilon=0.)
                state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, policy_key, steps, all_actions, all_states

            ## Retrieve data from the tuple
            seed, metrics = for_val
            policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
            ## Reset the environment
            state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            initial_robot_position = state[-1,:2]
            ## Episode loop
            all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
            all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
            while_val_init = (state, obs, info, init_outcome, policy_key, 0, all_actions, all_states)
            _, _, end_info, outcome, policy_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
            ## Update metrics
            metrics = compute_episode_metrics(
                environment=env.environment,
                metrics=metrics,
                episode_idx=i, 
                initial_robot_position=initial_robot_position, 
                all_states=all_states, 
                all_actions=all_actions, 
                outcome=outcome, 
                episode_steps=episode_steps, 
                end_info=end_info, 
                max_steps=int(time_limit/env.robot_dt)+1, 
                personal_space=0.5,
                robot_dt=env.robot_dt,
                robot_radius=env.robot_radius,
                ccso_n_static_humans=env.ccso_n_static_humans,
                robot_specs={'kinematics': env.kinematics, 'v_max': self.v_max, 'wheels_distance': self.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
            )
            seed += 1
            return seed, metrics
        # Initialize metrics
        metrics = initialize_metrics_dict(n_trials)
        # Execute n_trials tests
        if env.scenario == SCENARIOS.index('circular_crossing_with_static_obstacles'):
            print(f"\nExecuting {n_trials} tests with {env.n_humans - env.ccso_n_static_humans} humans and {env.ccso_n_static_humans} obstacles...")
        else:
            print(f"\nExecuting {n_trials} tests with {env.n_humans} humans and {env.n_obstacles} obstacles...")
        _, metrics = lax.fori_loop(0, n_trials, _fori_body, (random_seed, metrics))
        # Print results
        print_average_metrics(n_trials, metrics)
        return metrics
    
    def animate_trajectory(
        self,
        robot_radius:float,
        robot_poses, # x, y, theta
        robot_actions,
        robot_goals,
        humans_poses, # x, y, theta
        humans_velocities, # vx, vy (in global frame)
        humans_radii,
        robot_actions_values=None, # (optional) values of the robot's actions (if not None, they will be visualized in the animation)
        action_distrs=None, # (optional) distribution of the robot's actions (if not None, it will be visualized in the animation)
        perception_distrs=None, # (optional) distribution of the robot's perception of humans' positions and velocities (if not None, it will be visualized in the animation)
        lidar_scans=None, # (optional) distance readings of the LiDAR sensor (if not None, it will be visualized in the animation)
        static_obstacles=None, # (optional) positions and radii of static obstacles (if not None, they will be visualized in the animation)
        occupancy_grids=None, # (optional) occupancy grids (if not None, they will be visualized in the animation)
        grid_cells=None, # (optional) grid cells (if not None, they will be visualized in the animation)
        grid_cells_size=None, # (optional) size of each grid cell (if not None, it will be visualized in the animation)
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
        p_visualization_threshold_dir:float=0.05,
    ):
        # Validate input args
        assert \
            len(robot_poses) == \
            len(robot_actions) == \
            len(robot_goals) == \
            len(humans_poses) == \
            len(humans_velocities) == \
            len(humans_radii), "All input lists must have the same length"
        # Set matplotlib fonts
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Compute informations for visualization
        n_steps = len(robot_poses)
        if action_distrs is not None:
            test_action_samples = self._build_action_space(unicycle_triangle_samples=35)
        # Animate trajectory
        fig = plt.figure(figsize=(21.43,13.57))
        fig.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        outer_gs = fig.add_gridspec(1, 2, width_ratios=[2, 0.4], wspace=0.09)
        axs = [
            fig.add_subplot(outer_gs[0]), # Simulation + LiDAR ranges (Top-Left)
            fig.add_subplot(outer_gs[1]),   # Action space (Right, tall)
        ]
        def animate(frame):
            for i, ax in enumerate(axs):
                ax.clear()
                if i == len(axs) - 1: continue
                ax.set(xlim=x_lims if x_lims is not None else [-10,10], ylim=y_lims if y_lims is not None else [-10,10])
                ax.set_xlabel('X', labelpad=-5)
                if i % 2 == 0:
                    ax.set_ylabel('Y', labelpad=-13)
                else:
                    ax.set_yticks([])
                ax.set_aspect('equal', adjustable='datalim')
                # Plot humans
                for h in range(len(humans_poses[frame])):
                    head = plt.Circle((humans_poses[frame][h,0] + jnp.cos(humans_poses[frame][h,2]) * humans_radii[frame][h], humans_poses[frame][h,1] + jnp.sin(humans_poses[frame][h,2]) * humans_radii[frame][h]), 0.1, color='black', alpha=0.6, zorder=1)
                    ax.add_patch(head)
                    circle = plt.Circle((humans_poses[frame][h,0], humans_poses[frame][h,1]), humans_radii[frame][h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                    ax.add_patch(circle)
                # Plot human velocities
                for h in range(len(humans_poses[frame])):
                    ax.arrow(
                        humans_poses[frame][h,0],
                        humans_poses[frame][h,1],
                        humans_velocities[frame][h,0],
                        humans_velocities[frame][h,1],
                        head_width=0.15,
                        head_length=0.15,
                        fc='blue',
                        ec='blue',
                        alpha=0.6,
                        zorder=30,
                    )
                # Plot robot
                robot_position = robot_poses[frame,:2]
                head = plt.Circle((robot_position[0] + robot_radius * jnp.cos(robot_poses[frame,2]), robot_position[1] + robot_radius * jnp.sin(robot_poses[frame,2])), 0.1, color='black', zorder=1)
                ax.add_patch(head)
                circle = plt.Circle((robot_position[0], robot_position[1]), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
                ax.add_patch(circle)
                # Plot robot goal
                ax.plot(
                    robot_goals[frame][0],
                    robot_goals[frame][1],
                    marker='*',
                    markersize=7,
                    color='red',
                    zorder=5,
                )
            ### FIRST ROW AXS: SIMULATION + INPUT VISUALIZATION
            # AX 0,0: Simulation with LiDAR ranges
            if static_obstacles is not None:
                if static_obstacles[frame].shape[1] > 1: # Polygon obstacles
                    for o in static_obstacles[frame]: axs[0].fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
                else: # One segment obstacles
                    for o in static_obstacles[frame]: axs[0].plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
            if (grid_cells is not None) and (grid_cells_size is not None):
                for i in range(len(grid_cells)):
                    for j in range(len(grid_cells[i])):
                        cell = grid_cells[i,j]
                        cell_occupancy = occupancy_grids[frame][i,j] if occupancy_grids is not None else 0
                        cell_color = 'red' if cell_occupancy else 'white'
                        rect = plt.Rectangle((cell[0]-grid_cells_size/2, cell[1]-grid_cells_size/2), grid_cells_size, grid_cells_size, facecolor=cell_color, edgecolor='black', alpha=0.5, zorder=0)
                        axs[0].add_patch(rect)
            if lidar_scans is not None:
                lidar_scan = lidar_scans[frame]
                for ray in range(len(lidar_scan)):
                    axs[0].plot(
                        [robot_poses[frame,0], robot_poses[frame,0] + lidar_scan[ray] * jnp.cos(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                        [robot_poses[frame,1], robot_poses[frame,1] + lidar_scan[ray] * jnp.sin(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                        color="black", 
                        linewidth=0.5, 
                        zorder=0
                    )
            if perception_distrs is not None:
                distr = tree_map(lambda x: x[frame], perception_distrs)
                for h in range(len(distr['pos_distrs']['means'])):
                    if distr['weights'][h] > 0.5:
                        pos_mean = distr['pos_distrs']['means'][h]
                        pos_world_frame = jnp.array([
                            pos_mean[0] * jnp.cos(robot_poses[frame,2]) - pos_mean[1] * jnp.sin(robot_poses[frame,2]) + robot_poses[frame,0],
                            pos_mean[0] * jnp.sin(robot_poses[frame,2]) + pos_mean[1] * jnp.cos(robot_poses[frame,2]) + robot_poses[frame,1],
                        ])
                        vel_mean = distr['vel_distrs']['means'][h]
                        vel_world_frame = jnp.array([
                            vel_mean[0] * jnp.cos(robot_poses[frame,2]) - vel_mean[1] * jnp.sin(robot_poses[frame,2]),
                            vel_mean[0] * jnp.sin(robot_poses[frame,2]) + vel_mean[1] * jnp.cos(robot_poses[frame,2]),
                        ])
                        axs[0].scatter(
                            pos_world_frame[0],
                            pos_world_frame[1],
                            color='red',
                            s=30,
                            alpha=1,
                            zorder=20,
                        )
                        axs[0].arrow(
                            pos_world_frame[0],
                            pos_world_frame[1],
                            vel_world_frame[0],
                            vel_world_frame[1],
                            head_width=0.3,
                            head_length=0.3,
                            fc='red',
                            ec='red',
                            alpha=1,
                            zorder=20,
                        )
            # AX :,2: Feasible and bounded action space + action space distribution and action taken
            axs[1].set_xlabel("$v$ (m/s)")
            axs[1].set_ylabel("$\omega$ (rad/s)", labelpad=-15)
            axs[1].set_xlim(-0.1, self.v_max + 0.1)
            axs[1].set_ylim(-2*self.v_max/self.wheels_distance - 0.3, 2*self.v_max/self.wheels_distance + 0.3)
            axs[1].set_xticks(jnp.arange(0, self.v_max+0.2, 0.2))
            axs[1].set_xticklabels([round(i,1) for i in jnp.arange(0, self.v_max, 0.2)] + [r"$\overline{v}$"])
            axs[1].set_yticks(jnp.arange(-2,3,1).tolist() + [2*self.v_max/self.wheels_distance,-2*self.v_max/self.wheels_distance])
            axs[1].set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
            axs[1].grid()
            axs[1].add_patch(
                plt.Polygon(
                    [   
                        [0,2*self.v_max/self.wheels_distance],
                        [0,-2*self.v_max/self.wheels_distance],
                        [self.v_max,0],
                    ],
                    closed=True,
                    fill=True,
                    edgecolor='black',
                    facecolor='white',
                    linewidth=2,
                    zorder=2,
                ),
            )
            if robot_actions_values is not None:
                feasible_actions_idx = jnp.where(self.action_space > -jnp.inf)[0]
                feasible_actions = self.action_space[feasible_actions_idx]
                feasible_actions_values = robot_actions_values[frame][feasible_actions_idx] if robot_actions_values is not None else None
                axs[1].scatter(
                    feasible_actions[:,0],
                    feasible_actions[:,1],
                    c=feasible_actions_values if feasible_actions_values is not None else 'black',
                    cmap='Reds'if feasible_actions_values is not None else None,
                    s=20,
                    alpha=0.7,
                    label='Feasible actions',
                    zorder=3,
                )
            if action_distrs is not None:
                bounded_action_space_vertices = action_distrs["vertices"][frame]
                axs[1].add_patch(
                    plt.Polygon(
                        [   
                            bounded_action_space_vertices[0],
                            bounded_action_space_vertices[1],
                            bounded_action_space_vertices[2],
                        ],
                        closed=True,
                        fill=True,
                        edgecolor='green',
                        facecolor='lightgreen',
                        linewidth=2,
                        zorder=3,
                    ),
                )
                actor_distr = tree_map(lambda x: x[frame], action_distrs)
                samples = test_action_samples[self.distr.batch_is_in_support(actor_distr, test_action_samples)]
                test_action_p = self.distr.batch_p(actor_distr, samples)
                points_high_p = samples[test_action_p > p_visualization_threshold_dir]
                corresponding_colors = test_action_p[test_action_p > p_visualization_threshold_dir]
                axs[1].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
            axs[1].plot(robot_actions[frame,0], robot_actions[frame,1], marker='^',markersize=9,color='blue',zorder=51) # Action taken
        anim = FuncAnimation(fig, animate, interval=self.dt*1000, frames=n_steps)
        if save_video:
            save_path = os.path.join(os.path.dirname(__file__), f'jessi_trajectory.mp4')
            writer_video = FFMpegWriter(fps=int(1/self.dt), bitrate=1800)
            anim.save(save_path, writer=writer_video, dpi=300)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()

    def animate_socialnav_trajectory(
        self,
        states,
        actions,
        goals,
        humans_radii,
        socialnav_env:SocialNav,
        action_values:jnp.ndarray=None,
        action_distrs:jnp.ndarray=None,
        perception_distrs:dict=None,
        static_obstacles=None,
        occupancy_grids=None,
        grid_cells=None,
        grid_cells_size=None,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        robot_positions = states[:,-1,:2]
        robot_orientations = states[:,-1,4]
        robot_poses = jnp.hstack((robot_positions, robot_orientations.reshape(-1,1)))
        humans_positions = states[:,:-1,:2]
        humans_orientations = states[:,:-1,4]
        humans_poses = jnp.dstack((humans_positions, humans_orientations))
        humans_body_velocities = states[:,:-1,2:4]
        humans_velocities = lax.cond(
            socialnav_env.humans_policy == HUMAN_POLICIES.index('hsfm'),
            lambda: vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(
                    humans_orientations,
                    humans_body_velocities,
                ),
            lambda: humans_body_velocities,
        )
        self.animate_trajectory(
            socialnav_env.robot_radius,
            robot_poses,
            actions,
            goals,
            humans_poses,
            humans_velocities,
            humans_radii,
            robot_actions_values=action_values,
            perception_distrs=perception_distrs,
            action_distrs=action_distrs,
            static_obstacles=static_obstacles,
            occupancy_grids=occupancy_grids,
            grid_cells=grid_cells,
            grid_cells_size=grid_cells_size,
            x_lims=x_lims,
            y_lims=y_lims,
            save_video=save_video,
        )

    # LaserNav methods

    @partial(jit, static_argnames=("self","jessi"))
    def act_on_jessi_perception(
        self, 
        jessi:JESSI,
        perception_params:dict,
        key:random.PRNGKey,
        lasernav_obs:jnp.ndarray, # LaserNav observations
        info:dict,
        vnet_params:dict, 
        epsilon:float,
        humans_radius:float,
    ) -> jnp.ndarray:
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

        @jit
        def _random_action(val):
            obs, _, info, _, key = val
            key, subkey = random.split(key)
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            return random.choice(subkey, self.action_space), key, vnet_inputs, jnp.zeros(len(self.action_space))
        @jit
        def _forward_pass(val):
            obs, humans_mask, info, vnet_params, key = val
            # Propagate humans state for dt time
            next_obs = jnp.vstack([self.batch_propagate_human_obs(obs[0:-1]),obs[-1]])
            # Compute action values
            action_values, vnet_inputs = self._batch_compute_action_value(next_obs, obs, info, self.action_space, vnet_params, humans_mask=humans_mask)
            action = self.action_space[jnp.argmax(action_values)]
            vnet_input = vnet_inputs[jnp.argmax(action_values)]
            # Return action with highest value
            return action, key, vnet_input, jnp.squeeze(action_values)
        @jit
        def _towards_goal_action(val):
            obs, _, info, _, key = val
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            # Compute the action that goes straight towards the goal with maximum speed allowed for the unicycle robot
            diff = info["robot_goal"] - obs[-1,0:2]
            distance = jnp.linalg.norm(diff)
            direction = jnp.arctan2(diff[1], diff[0])
            direction_error = wrap_angle(direction - obs[-1,5])
            w = jnp.clip(direction_error/self.dt, -self.v_max/(self.wheels_distance/2), self.v_max/(self.wheels_distance/2))
            v = self.v_max - (self.v_max * jnp.abs(w) / (self.v_max/(self.wheels_distance/2)))
            v = jnp.clip(v, 0, jnp.min(jnp.array([v, distance/self.dt])))
            optimal_action = jnp.array([v,w])
            # Find discrete action closest to the optimal action
            action_idx = jnp.argmin(jnp.linalg.norm(self.action_space - optimal_action, axis=1))
            action = self.action_space[action_idx]
            return action, key, vnet_inputs, jnp.zeros(len(self.action_space))
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        case = jnp.argmax(jnp.array([
            explore, 
            (~explore) & jnp.any(humans_mask), 
            (~explore) & ~jnp.any(humans_mask)
        ]))
        action, key, vnet_input, action_values = lax.switch(
            case, 
            [
                _random_action, 
                _forward_pass, 
                _towards_goal_action, 
            ],
            (obs, humans_mask, info, vnet_params, key)
        )
        return action, key, vnet_input, action_values, hcgs
    
    def evaluate_on_jessi_perception(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
        jessi:JESSI,
        perception_params:dict,
        network_params:dict,
        humans_radius_hypothesis:jnp.ndarray
    ) -> dict:
        """
        Test the trained policy over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == self.kinematics, "Policy kinematics must match environment kinematics"
        assert env.robot_dt == self.dt, f"Environment time step (dt={env.dt}) must be equal to policy time step (dt={self.dt}) for evaluation"
        time_limit = env.reward_function.time_limit
        @loop_tqdm(n_trials)
        @jit
        def _fori_body(i:int, for_val:tuple):   
            @jit
            def _while_body(while_val:tuple):
                # Retrieve data from the tuple
                state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states = while_val
                if self.name == "SARL*":
                    action, _, _, _, _, _, _= self.act_on_jessi_perception(jessi, perception_params, policy_key, obs, info, network_params, 0., humans_radius_hypothesis)
                elif self.name == "DIRSAFE":
                    action, _, _, _, _, _ = self.act_on_jessi_perception(jessi, perception_params, policy_key, obs, info, network_params, humans_radius_hypothesis, sample=False)
                else:
                    action, _, _, _, _ = self.act_on_jessi_perception(jessi, perception_params, policy_key, obs, info, network_params, 0., humans_radius_hypothesis)
                state, obs, info, _, outcome, (_, env_key)  = env.step(state,info,action,test=True)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states

            ## Retrieve data from the tuple
            seed, metrics = for_val
            policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
            ## Reset the environment
            state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            initial_robot_position = state[-1,:2]
            ## Episode loop
            all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
            all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
            while_val_init = (state, obs, info, init_outcome, policy_key, env_key, 0, all_actions, all_states)
            _, _, end_info, outcome, policy_key, env_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
            ## Update metrics
            metrics = compute_episode_metrics(
                environment=env.environment,
                metrics=metrics,
                episode_idx=i, 
                initial_robot_position=initial_robot_position, 
                all_states=all_states, 
                all_actions=all_actions, 
                outcome=outcome, 
                episode_steps=episode_steps, 
                end_info=end_info, 
                max_steps=int(time_limit/env.robot_dt)+1, 
                personal_space=0.5,
                robot_dt=env.robot_dt,
                robot_radius=env.robot_radius,
                ccso_n_static_humans=env.ccso_n_static_humans,
                robot_specs={'kinematics': env.kinematics, 'v_max': self.v_max, 'wheels_distance': self.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
            )
            seed += 1
            return seed, metrics
        # Initialize metrics
        metrics = initialize_metrics_dict(n_trials)
        # Execute n_trials tests
        if env.scenario == SCENARIOS.index('circular_crossing_with_static_obstacles'):
            print(f"\nExecuting {n_trials} tests with {env.n_humans - env.ccso_n_static_humans} humans and {env.ccso_n_static_humans} obstacles...")
        else:
            print(f"\nExecuting {n_trials} tests with {env.n_humans} humans and {env.n_obstacles} obstacles...")
        _, metrics = lax.fori_loop(0, n_trials, _fori_body, (random_seed, metrics))
        # Print results
        print_average_metrics(n_trials, metrics)
        return metrics