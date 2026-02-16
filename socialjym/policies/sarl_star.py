import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug
from jax.tree_util import tree_map
from functools import partial
from types import FunctionType
import os

from .sarl import SARL
from .jessi import JESSI
from socialjym.envs.base_env import wrap_angle
from jhsfm.hsfm import vectorized_compute_obstacle_closest_point
from socialjym.utils.global_planners.base_global_planner import GLOBAL_PLANNERS
from socialjym.utils.global_planners.a_star import AStarPlanner
from socialjym.utils.global_planners.dijkstra import DijkstraPlanner

# IMPLEMENTATION OF SARL*
# @inproceedings{li2019sarl,
#   title={SARL: Deep reinforcement learning based human-aware navigation for mobile robot in indoor environments},
#   author={Li, Keyu and Xu, Yangxin and Wang, Jiankun and Meng, Max Q-H},
#   booktitle={2019 IEEE International Conference on Robotics and Biomimetics (ROBIO)},
#   pages={688--694},
#   year={2019},
#   organization={IEEE}
# }

class SARLStar(SARL):
    def __init__(
            self, 
            reward_function:FunctionType, 
            grid_size:jnp.ndarray,
            use_planner=True,
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
            # USED ONLY FOR LASERNAV:
            lidar_n_stack_to_use = 1, # Number of past lidar observations to stack and use as input for the policy. If 1, only the current lidar observation is used.
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
        self.use_planner = use_planner
        if planner == "A*":
            self.planner = AStarPlanner(grid_size)
        elif planner == "Dijkstra":
            self.planner = DijkstraPlanner(grid_size)
        self.lidar_n_stack_to_use = lidar_n_stack_to_use
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
    def _is_action_safe_from_lidar(self, point_cloud:jnp.ndarray, obs:jnp.ndarray, action:jnp.ndarray) -> bool:
        """
        For a given action, simulate the robot's movement and check if it collides with any obstacle using the lidar point cloud.
        """
        obs = obs.at[-1,2:4].set(action)
        next_pos = self._propagate_robot_obs(obs[-1])[:2]
        # Check collision with obstacles
        distances = jnp.linalg.norm(point_cloud - next_pos, axis=1)
        min_distance = jnp.min(distances)
        return min_distance > obs[-1,4]
    
    @partial(jit, static_argnames=("self"))
    def _compute_safe_action_space_from_lidar(self, point_cloud:jnp.ndarray, obs:jnp.ndarray) -> jnp.ndarray:
        """
        For each action in the action space, simulate the robot's movement and check if it collides with any obstacle using the lidar point cloud.
        """
        return vmap(SARLStar._is_action_safe_from_lidar, in_axes=(None, None, None, 0))(self, point_cloud, obs, self.action_space)

    @partial(jit, static_argnames=("self"))
    def _compute_safe_action_value(self, next_obs, obs, info, action, vnet_params, is_action_safe, humans_mask=None) -> jnp.ndarray:
        """
        Compute the value of a given action only if it is safe, otherwise return -inf.
        """
        return lax.cond(
            is_action_safe,
            lambda: self._compute_action_value(next_obs, obs, info, action, vnet_params, humans_mask=humans_mask),
            lambda: (jnp.array([-jnp.inf]), self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)), # Return -inf and dummy vnet_input
        )

    @partial(jit, static_argnames=("self"))
    def _batch_compute_safe_action_value(self, next_obs, obs, info, action, vnet_params, is_action_safe, humans_mask=None) -> jnp.ndarray:
        """
        Compute the value of a batch of actions only if they are safe, otherwise return -inf.
        """
        return vmap(SARLStar._compute_safe_action_value, in_axes=(None,None,None,None,0,None,0, None))(
            self,
            next_obs, 
            obs, 
            info, 
            action, 
            vnet_params, 
            is_action_safe,
            humans_mask,
        )

    @partial(jit, static_argnames=("self"))
    def _is_cell_occupied_from_lidar(self, point_cloud:jnp.ndarray, cell_center:jnp.ndarray, cell_size:float, epsilon=1e-5) -> bool:
        """
        Check if a cell is free by checking if any point in the point cloud is within the cell. 
        The cell is considered occupied if there is at least one point in the point cloud that is within the cell.
        """
        # Check if any point in the point cloud is within the cell
        half_cell_size = cell_size / 2
        in_cell = \
            (point_cloud[:,0] >= cell_center[0] - half_cell_size - epsilon) & \
            (point_cloud[:,0] <= cell_center[0] + half_cell_size + epsilon) & \
            (point_cloud[:,1] >= cell_center[1] - half_cell_size - epsilon) & \
            (point_cloud[:,1] <= cell_center[1] + half_cell_size + epsilon)
        # If there are no points in the cell, it is free
        return jnp.any(in_cell)

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
            return random.choice(subkey, self.action_space, p=probabilities), key, vnet_inputs, jnp.zeros((len(self.action_space)))

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
            return action, key, vnet_input, jnp.squeeze(action_values)
        
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        ## Run global planner to find next subgoal
        if self.use_planner:
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
            # debug.print("New subgoal: {x}, path length: {y}", x=info['robot_goal'], y=path_length)
        ## Compute safe actions
        safe_actions = self._compute_safe_action_space(obs, info)
        ## Compute best action
        action, key, vnet_input, action_values = lax.cond(explore, _random_action, _forward_pass, (obs, info, vnet_params, safe_actions, key))
        return action, key, vnet_input, action_values, info['robot_goal'], info['occupancy_grid']
    
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
        ## Compute aligned point_cloud
        aligned_lidar = jessi.align_lidar(lasernav_obs[:self.lidar_n_stack_to_use])[1]
        point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        # Compute occupancy grid for the planner based on the point_cloud
        occupancy_grid = vmap(vmap(self._is_cell_occupied_from_lidar, in_axes=(None, 0, None)), in_axes=(None, 0, None))(
            point_cloud, 
            info['grid_cells'], 
            info['grid_cells_size']
        )
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
            obs, _, info, _, key, safe_actions = val
            key, subkey = random.split(key)
            vnet_inputs = self.batch_compute_vnet_input(obs[-1], obs[0:-1], info)
            probabilities = jnp.where(safe_actions, 1/jnp.sum(safe_actions), 0.)
            # Fictitious action values
            action_values = jnp.zeros(len(self.action_space))
            action_values = jnp.where(safe_actions, action_values, -jnp.inf) # Set to -inf the value of unsafe actions
            return random.choice(subkey, self.action_space, p=probabilities), key, vnet_inputs, action_values
        @jit
        def _forward_pass(val):
            obs, humans_mask, info, vnet_params, key, safe_actions = val
            # Propagate humans state for dt time
            next_obs = jnp.vstack([self.batch_propagate_human_obs(obs[0:-1]),obs[-1]])
            # Compute action values
            action_values, vnet_inputs = self._batch_compute_safe_action_value(
                next_obs, 
                obs, 
                info, 
                self.action_space, 
                vnet_params, 
                safe_actions, 
                humans_mask=humans_mask
            )
            action = self.action_space[jnp.argmax(action_values)]
            vnet_input = vnet_inputs[jnp.argmax(action_values)]
            # Return action with highest value
            return action, key, vnet_input, jnp.squeeze(action_values)
        ## Compute path
        if self.use_planner:
            # Find path
            path, path_length = self.planner.find_path(
                obs[-1,:2], 
                info['robot_goal'], 
                info['grid_cells'], 
                occupancy_grid
            )
            info['robot_goal'] = lax.cond(
                path_length > 1,
                lambda: path[1], # Next waypoint in the path
                lambda: info['robot_goal'], # Already at goal cell
            )
            # debug.print("New subgoal: {x}, path length: {y}", x=info['robot_goal'], y=path_length)
        ## Compute safe actions
        safe_actions = self._compute_safe_action_space_from_lidar(point_cloud, obs)
        key, subkey = random.split(key)
        explore = random.uniform(subkey) < epsilon
        action, key, vnet_input, action_values = lax.cond(
            explore, 
            lambda x: _random_action(x), 
            lambda x: _forward_pass(x), 
            (obs, humans_mask, info, vnet_params, key, safe_actions)
        )
        return action, key, vnet_input, action_values, hcgs, info['robot_goal'], occupancy_grid