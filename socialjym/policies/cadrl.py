import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad, tree_map
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from .base_policy import BasePolicy
from socialjym.envs.base_env import EPSILON, ROBOT_KINEMATICS, wrap_angle

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
    def _compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict) -> jnp.ndarray:
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
        # Take the minimum among all outputs (representing the worst case scenario)
        min_vnet_output = jnp.min(vnet_outputs)
        # Compute the final value of the action
        value = reward + pow(self.gamma, self.dt * self.v_max) * min_vnet_output
        return value, vnet_inputs
        
    @partial(jit, static_argnames=("self"))
    def _batch_compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict) -> jnp.ndarray:
        return vmap(CADRL._compute_action_value, in_axes=(None,None,None,None,0,None))(self, next_obs, current_obs, info, action, vnet_params)
    
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
            return random.choice(subkey, self.action_space), key, vnet_inputs
        
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
            return action, key, vnet_input
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        action, key, vnet_input = lax.cond(explore, _random_action, _forward_pass, (obs, info, vnet_params, key))
        return action, key, vnet_input

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