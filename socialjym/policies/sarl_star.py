import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn
from functools import partial
import haiku as hk
from types import FunctionType

from .sarl import SARL
from socialjym.utils.global_planners.base_global_planner import GLOBAL_PLANNERS
from socialjym.utils.global_planners.a_star import AStarPlanner
from socialjym.utils.global_planners.dijkstra import DijkstraPlanner

class SARLStar(SARL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            planner="A*", # "A*" or "Dijkstra"
            v_max=1., 
            gamma=0.9, 
            dt=0.25, 
            wheels_distance=0.7, 
            kinematics='holonomic',
            unicycle_box_action_space=False,
            noise = False, # If True, noise is added to humams positions and velocities
            noise_sigma_percentage:float = 0., # Standard deviation of the noise as a percentage of the absolute value of the difference between the robot and the humans
            # position_noise_sigma_percentage_radius = 0., # Standard deviation of the noise as a percentage of the ditance between the robot and the humans
            # position_noise_sigma_angle = 0., # Standard deviation of the noise on the angle of humans' position in the robot frame
            # velocity_noise_sigma_percentage = 0., # Standard deviation of the noise as a percentage of the (vx,vy) coordinates of humans' velocity in the robot frame
        ) -> None:
        assert planner in GLOBAL_PLANNERS, f"Planner {planner} not recognized. Available planners: {GLOBAL_PLANNERS}"
        # Configurable attributes
        super().__init__(
            reward_function=reward_function, 
            v_max=v_max, 
            gamma=gamma, 
            dt=dt, 
            wheels_distance=wheels_distance, 
            kinematics=kinematics,
            unicycle_box_action_space=unicycle_box_action_space,
            noise=noise,
            noise_sigma_percentage=noise_sigma_percentage,
            # position_noise_sigma_percentage_radius = position_noise_sigma_percentage_radius,
            # position_noise_sigma_angle = position_noise_sigma_angle,
            # velocity_noise_sigma_percentage = velocity_noise_sigma_percentage, # Standard deviation of the noise as a percentage of the (vx,vy) coordinates of humans' velocity in the robot frame
        )
        if planner == "A*":
            self.planner = AStarPlanner
        elif planner == "Dijkstra":
            self.planner = DijkstraPlanner
        # Default attributes
        self.name = "SARL*"

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _compute_safe_action_space(self, obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        """
        For each action in the action space, simulate the robot's movement and check if it collides with any obstacle or human.
        """
        pass

    # Public methods

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
        return vmap(SARL.act, in_axes=(None, 0, 0, 0, None, None))(
            self,
            keys, 
            obses, 
            infos, 
            vnet_params, 
            epsilon)