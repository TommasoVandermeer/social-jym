import jax.numpy as jnp
from jax import jit
from functools import partial
from types import FunctionType

from .base_env import BaseEnv, SCENARIOS

class LaserNav(BaseEnv):
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL using a 2D LiDAR as sensor.
    """
    def __init__(
            self, 
            robot_radius:float, 
            robot_dt:float, 
            humans_dt:float, 
            scenario:str, 
            n_humans:int, 
            reward_function:FunctionType,
            humans_policy='hsfm', 
            robot_visible=False, 
            circle_radius=7, 
            traffic_height=3, # traffic_height=3 # Conform with Social-Navigation-PyEnvs
            traffic_length=14,
            crowding_square_side=14,
            hybrid_scenario_subset=jnp.arange(0, len(SCENARIOS)-1, dtype=jnp.int32),
            lidar_angular_range=jnp.pi,
            lidar_max_dist=10.,
            lidar_num_rays=60,
            kinematics='holonomic',
            max_cc_delay = 5.,
            ccso_n_static_humans:int = 3,
        ) -> None:
        ## SocialNav env initialization
        super().__init__(
            robot_radius=robot_radius,
            robot_dt=robot_dt,
            humans_dt=humans_dt,
            scenario=scenario,
            n_humans=n_humans,
            reward_function=reward_function,
            humans_policy=humans_policy,
            robot_visible=robot_visible,
            circle_radius=circle_radius,
            traffic_height=traffic_height,
            traffic_length=traffic_length,
            crowding_square_side=crowding_square_side,
            hybrid_scenario_subset=hybrid_scenario_subset,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            kinematics=kinematics,
            max_cc_delay=max_cc_delay,
            ccso_n_static_humans=ccso_n_static_humans,
            )

    # --- Private methods ---

    @partial(jit, static_argnames=("self"))
    def _get_obs(self, state:jnp.ndarray, info:dict, action:jnp.ndarray) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment, and the robot's action,
        this function computes the observation of the current state.

        args:
        - state: current state of the environment.
        - info: dictionary containing additional information about the environment.
        - action: action to be taken by the robot (vx,vy) or (v,w).

        output:
        - obs: [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements]
        """
        measurements, _ = self.get_lidar_measurements(
            state[-1, :2], # Lidar position (robot position)
            state[-1,4], # Lidar yaw angle (robot orientation)
            state[1:-1, :2], # Human positions
            info['humans_parameters'][:,0], # Human radii
        )
        # Compute the observation
        obs = jnp.array([
            *state[-1,:2], # Robot position
            state[-1,4], # Robot orientation
            self.robot_radius, # Robot radius
            *action, # Robot action (either (vx,vy) or (v,w))
            *measurements, # LiDAR measurements
        ])
        return obs
        