import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn, value_and_grad
from functools import partial
import haiku as hk
from types import FunctionType
import optax

from .base_policy import BasePolicy
from socialjym.envs.base_env import ROBOT_KINEMATICS

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
    def __init__(self, reward_function:FunctionType, v_max=1., gamma=0.9, dt=0.25, wheels_distance=0.7, kinematics='holonomic') -> None:
        # Configurable attributes
        super().__init__(gamma)
        self.reward_function = reward_function
        self.v_max = v_max
        self.wheels_distance = wheels_distance # Distance between the wheels of the robot (used for unicycle kinematics). Value taken from real TurtleBot4 robot.
        self.dt = dt
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        # Default attributes
        self.name = "CADRL"
        self.vnet_input_size = 13
        self.model = value_network
        self.action_space = self._build_action_space()

    # Private methods

    # @partial(jit, static_argnames=("self"))
    def _build_action_space(self) -> jnp.ndarray:
        # TODO: Adjust this mess, it's not very readable
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            speed_samples = 5
            rotation_samples = 16
            speeds = lax.fori_loop(0,speed_samples,lambda i, speeds: speeds.at[i].set((jnp.exp((i + 1) / speed_samples) - 1) / (jnp.e - 1) * self.v_max), jnp.zeros((speed_samples,)))
            rotations = jnp.linspace(0, 2 * jnp.pi, rotation_samples, endpoint=False)
            action_space = jnp.empty((speed_samples * rotation_samples + 1,2))
            action_space = action_space.at[0].set(jnp.array([0., 0.])) # First action is to stay still
            action_space = lax.fori_loop(1,
                                        len(action_space), 
                                        lambda i, acts: acts.at[i].set(jnp.array([
                                            speeds[(i-1) % speed_samples] * jnp.cos(rotations[(i-1) // speed_samples]),
                                            speeds[(i-1) % speed_samples] * jnp.sin(rotations[(i-1) // speed_samples])])),
                                        action_space)
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'): # In this case speed samples are doubles and rotation samples are halved because of the kinematic constraints
            ## METHOD 1
            # speed_samples = 4
            # angular_speed_samples = 19
            # speeds = jnp.linspace(-self.v_max, self.v_max, speed_samples)
            # angular_speeds = jnp.linspace(-self.v_max/(self.wheels_distance/2), self.v_max/(self.wheels_distance/2), angular_speed_samples)
            # action_space = jnp.empty((speed_samples * angular_speed_samples + 5,2))
            # action_space = action_space.at[0].set(jnp.array([0., 0.])) # First action is to stay still
            # action_space = action_space.at[1].set(jnp.array([self.v_max, 0.])) # Second action is to move forward at max speed
            # action_space = action_space.at[2].set(jnp.array([-self.v_max, 0.])) # Second action is to move backwards at max speed
            # action_space = action_space.at[3].set(jnp.array([0., self.v_max/(self.wheels_distance/2)])) # Second action is to rotate counter-clockwise at max speed
            # action_space = action_space.at[4].set(jnp.array([0., -self.v_max/(self.wheels_distance/2)])) # Second action is to rotate clockwise at max speed
            # action_space = lax.fori_loop(5,
            #                             len(action_space), 
            #                             lambda i, acts: acts.at[i].set(jnp.array([
            #                                 (speeds[(i-5) // angular_speed_samples] / self.v_max) * (self.v_max - jnp.abs(angular_speeds[(i-5) % angular_speed_samples]) * (self.wheels_distance/2)),
            #                                 angular_speeds[(i-5) % angular_speed_samples]])),
            #                             action_space)
            ## METHOD 2
            action_space = jnp.empty((81,2))
            action_space = action_space.at[0].set(jnp.array([0., 0.])) # First action is to stay still
            action_space = action_space.at[1].set(jnp.array([self.v_max, 0.])) # Second action is to move forward at max speed
            action_space = action_space.at[2].set(jnp.array([-self.v_max, 0.])) # Second action is to move backwards at max speed
            action_space = action_space.at[3].set(jnp.array([0., self.v_max/(self.wheels_distance/2)])) # Second action is to rotate counter-clockwise at max speed
            action_space = action_space.at[4].set(jnp.array([0., -self.v_max/(self.wheels_distance/2)])) # Second action is to rotate clockwise at max speed
            action_space = action_space.at[5:7].set(jnp.c_[jnp.linspace(-self.v_max, 0, 4)[1:-1], jnp.zeros((2,))]) # Actions to move bacwards at different speeds
            action_space = action_space.at[7:9].set(jnp.c_[jnp.linspace(0, self.v_max, 4)[1:-1], jnp.zeros((2,))]) # Actions to move forwards at different speeds
            # half_w_samples = 9
            # v_for_each_w = [1, 1, 2, 2, 2, 2, 2, 3, 3]
            half_w_samples = 6
            v_for_each_w = [2,2,3,3,4,4]
            first_quadrant_points = jnp.empty((18,2))
            for i in range(half_w_samples):
                angular_speed = ((half_w_samples-i) / (half_w_samples + 2)) * (self.v_max/(self.wheels_distance/2))
                for j in range(v_for_each_w[i]):
                    first_quadrant_points = first_quadrant_points.at[sum(v_for_each_w[0:i])+j].set(jnp.array([
                        ((j+1) / (v_for_each_w[i] + 1)) / self.v_max * (self.v_max - angular_speed * (self.wheels_distance/2)),
                        angular_speed]))
            second_quadrant_points = jnp.array([-first_quadrant_points[:,0], first_quadrant_points[:,1]]).T
            third_quadrant_points = jnp.array([-first_quadrant_points[:,0], -first_quadrant_points[:,1]]).T
            fourth_quadrant_points = jnp.array([first_quadrant_points[:,0], -first_quadrant_points[:,1]]).T
            action_space = action_space.at[9:27].set(first_quadrant_points)
            action_space = action_space.at[27:45].set(second_quadrant_points)
            action_space = action_space.at[45:63].set(third_quadrant_points)
            action_space = action_space.at[63:81].set(fourth_quadrant_points)
        return action_space

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
        vnet_outputs = self.model.apply(vnet_params, None, vnet_inputs)
        # Take the minimum among all outputs (representing the worst case scenario)
        min_vnet_output = jnp.min(vnet_outputs)
        # Compute the final value of the action
        value = reward + pow(self.gamma,self.dt * self.v_max) * min_vnet_output
        return value, vnet_inputs
        
    @partial(jit, static_argnames=("self"))
    def _batch_compute_action_value(self, next_obs:jnp.ndarray, current_obs:jnp.ndarray, info:dict, action:jnp.ndarray, vnet_params:dict) -> jnp.ndarray:
        return vmap(CADRL._compute_action_value, in_axes=(None,None,None,None,0,None))(self, next_obs, current_obs, info, action, vnet_params)
    
    @partial(jit, static_argnames=("self"))
    def _propagate_obs(self, obs:jnp.ndarray) -> jnp.ndarray:
        obs = obs.at[0:2].set(obs[0:2] + obs[2:4] * self.dt)
        return obs

    @partial(jit, static_argnames=("self"))
    def _compute_vnet_input(self, robot_obs:jnp.ndarray, human_obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        # Robot observation: [x,y,ux,uy,radius]
        # Human observation: [x,y,vx,vy,radius]
        # Re-parametrized observation: [dg,v_pref,theta,radius,vx,vy,px1,py1,vx1,vy1,radius1,da,radius_sum]
        rot = jnp.atan2(info["robot_goal"][1] - robot_obs[1],info["robot_goal"][0] - robot_obs[0])
        vnet_input = jnp.zeros((self.vnet_input_size,))
        vnet_input = vnet_input.at[0].set(jnp.linalg.norm(info["robot_goal"] - robot_obs[0:2]))
        vnet_input = vnet_input.at[1].set(self.v_max)
        vnet_input = vnet_input.at[2].set(robot_obs[5])
        vnet_input = vnet_input.at[3].set(robot_obs[4])
        vnet_input = vnet_input.at[4].set(robot_obs[2] * jnp.cos(rot) + robot_obs[3] * jnp.sin(rot))
        vnet_input = vnet_input.at[5].set(-robot_obs[2] * jnp.sin(rot) + robot_obs[3] * jnp.cos(rot))
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
    def batch_propagate_obs(self, obs:jnp.ndarray) -> jnp.ndarray:
        return vmap(CADRL._propagate_obs, in_axes=(None, 0))(self, obs)

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

    @partial(jit, static_argnames=("self","optimizer"))
    def update(self, 
               current_vnet_params:dict, 
               optimizer:optax.GradientTransformation, 
               optimizer_state: jnp.ndarray, 
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
        # Compute parameter updates
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        # Apply updates
        updated_vnet_params = optax.apply_updates(current_vnet_params, updates)
        return updated_vnet_params, optimizer_state, loss