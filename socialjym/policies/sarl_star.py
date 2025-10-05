import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, nn
from functools import partial
import haiku as hk
from types import FunctionType

from .sarl import SARL
from jhsfm.hsfm import vectorized_compute_obstacle_closest_point
from socialjym.utils.global_planners.base_global_planner import GLOBAL_PLANNERS
from socialjym.utils.global_planners.a_star import AStarPlanner
from socialjym.utils.global_planners.dijkstra import DijkstraPlanner

class SARLStar(SARL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            grid_size:jnp.ndarray,
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
            self.planner = AStarPlanner(grid_size)
        elif planner == "Dijkstra":
            self.planner = DijkstraPlanner(grid_size)
        # Default attributes
        self.name = "SARL*"

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _is_action_safe(self, obs:jnp.ndarray, info:dict, action:jnp.ndarray) -> bool:
        """
        For a given action, simulate the robot's movement and check if it collides with any static obstacle.
        """
        obs = obs.at[-1,2:4].set(action)
        next_pos = self._propagate_robot_obs(obs[-1])[:2]
        # Check collision with obstacles
        closest_points = vectorized_compute_obstacle_closest_point(
            next_pos,
            info['static_obstacles'][-1] # Robot viewed obstacles
        )
        distances = jnp.linalg.norm(closest_points - next_pos, axis=1)
        min_distance = jnp.nanmin(distances)
        return lax.cond(
            jnp.isnan(min_distance), # All dummy obstacles
            lambda _: True,
            lambda x: x > obs[-1,4],
            min_distance,
        )

    @partial(jit, static_argnames=("self"))
    def _compute_safe_action_space(self, obs:jnp.ndarray, info:dict) -> jnp.ndarray:
        """
        For each action in the action space, simulate the robot's movement and check if it collides with any static obstacle.
        """
        return vmap(SARLStar._is_action_safe, in_axes=(None, None, None, 0))(self, obs, info, self.action_space)

    @partial(jit, static_argnames=("self"))
    def _compute_safe_action_value(self, next_obs, obs, info, action, vnet_params, is_action_safe) -> jnp.ndarray:
        """
        Compute the value of a given action only if it is safe, otherwise return -inf.
        """
        return lax.cond(
            is_action_safe,
            lambda: self._compute_action_value(next_obs, obs, info, action, vnet_params),
            lambda: (jnp.array([-jnp.inf]), self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)), # Return -inf and dummy vnet_input
        )

    @partial(jit, static_argnames=("self"))
    def _batch_compute_safe_action_value(self, next_obs, obs, info, action, vnet_params, is_action_safe) -> jnp.ndarray:
        """
        Compute the value of a batch of actions only if they are safe, otherwise return -inf.
        """
        return vmap(SARLStar._compute_safe_action_value, in_axes=(None,None,None,None,0,None,0))(
            self,
            next_obs, 
            obs, 
            info, 
            action, 
            vnet_params, 
            is_action_safe
        )

    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(self, key:random.PRNGKey, obs:jnp.ndarray, info:dict, vnet_params:dict, epsilon:float) -> jnp.ndarray:
        
        @jit
        def _random_action(val):
            obs, info, _, safe_actions, key = val
            key, subkey = random.split(key)
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            # Set to zero the probabilities to sample unsafe actions
            probabilities = jnp.where(safe_actions, 1/jnp.sum(safe_actions), 0.)
            return random.choice(subkey, self.action_space, p=probabilities), key, vnet_inputs

        @jit
        def _forward_pass(val):
            obs, info, vnet_params, safe_actions, key = val
            # Add noise to humans' observations
            if self.noise:
                key, subkey = random.split(key)
                obs = self._batch_add_noise_to_human_obs(obs, subkey)
            # Propagate humans state for dt time
            next_obs = jnp.vstack([self.batch_propagate_human_obs(obs[0:-1]),obs[-1]])
            # Compute action values
            action_values, vnet_inputs = self._batch_compute_safe_action_value(
                next_obs, 
                obs, 
                info, 
                self.action_space, 
                vnet_params, 
                safe_actions
            )
            action = self.action_space[jnp.argmax(action_values)]
            vnet_input = vnet_inputs[jnp.argmax(action_values)]
            # Return action with highest value
            return action, key, vnet_input
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        # Run global planner to find next subgoal
        path, path_length = self.planner.find_path(
            obs[-1,:2], 
            info['robot_goal'], 
            info['grid_cells'], 
            info['occupancy_grid']
        )
        info['robot_goal'] = lax.cond(
            path_length > 1,
            lambda: path[1], # Next waypoint in the path
            lambda: info['robot_goal'], # Already at goal cell
        )
        # Compute safe actions
        safe_actions = self._compute_safe_action_space(obs, info)
        # Compute best action
        action, key, vnet_input = lax.cond(explore, _random_action, _forward_pass, (obs, info, vnet_params, safe_actions, key))
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