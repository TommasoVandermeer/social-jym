from abc import ABC, abstractmethod
from functools import partial
from jax import jit, vmap, lax
import jax.numpy as jnp

class BaseEnv(ABC):
    def __init__(
        self,
        lidar_angular_range=jnp.pi,
        lidar_max_dist=10.,
        lidar_num_rays=60
    ) -> None:
        self.lidar_angular_range = lidar_angular_range
        self.lidar_max_dist = lidar_max_dist
        self.lidar_num_rays = lidar_num_rays

    # --- Private methods ---

    @abstractmethod
    def _get_obs(self, state):
        pass

    @abstractmethod
    def _reset(self, key):
        pass

    @partial(jit, static_argnames=("self"))
    def _human_ray_intersect(self, direction:jnp.ndarray, human_position:jnp.ndarray, robot_position:jnp.ndarray, human_radius:float) -> float:
        s = robot_position - human_position
        b = jnp.dot(s, direction)
        c = jnp.dot(s, s) - human_radius**2
        h = b * b - c
        distance = lax.cond(
            h < 0,
            lambda x: x,
            lambda x: lax.cond(
                - b - jnp.sqrt(h) < 0,
                lambda y: y,
                lambda _: - b - jnp.sqrt(h),
                x),
            self.lidar_max_dist)
        return distance        
    
    @partial(jit, static_argnames=("self"))
    def _batch_human_ray_intersect(self, direction:jnp.ndarray, human_positions:jnp.ndarray, robot_position:jnp.ndarray, human_radiuses:float) -> jnp.ndarray:
        return jnp.min(vmap(BaseEnv._human_ray_intersect, in_axes=(None,None,0,None,0))(self, direction, human_positions, robot_position, human_radiuses))

    @partial(jit, static_argnames=("self"))
    def _ray_cast(self, angle:float, robot_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray) -> float:
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        measurement = self._batch_human_ray_intersect(direction, human_positions, robot_position, human_radiuses)
        return measurement

    @partial(jit, static_argnames=("self"))
    def _batch_ray_cast(self, angles:float, robot_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray) -> jnp.ndarray:
        return vmap(BaseEnv._ray_cast, in_axes=(None,0,None,None,None))(self, angles, robot_position, human_positions, human_radiuses)

    # --- Public methods ---

    @abstractmethod
    def reset(self, key):
        pass

    @abstractmethod
    def step(self, env_state, action):
        pass

    @partial(jit, static_argnames=("self"))
    def get_lidar_measurements(
        self, 
        robot_position:jnp.ndarray, 
        robot_yaw:float,  
        human_positions:jnp.ndarray, 
        human_radiuses:jnp.ndarray
    ) -> jnp.ndarray:
        """
        Given the current state of the environment, the robot orientation and the additional information about the environment,
        this function computes the lidar measurements of the robot.

        args:
        - robot_position (2,): jnp.ndarray containing the x and y coordinates of the robot.
        - robot_yaw (1,): float containing the orientation of the robot.
        - human_positions (self.n_humans,2): jnp.ndarray containing the x and y coordinates of the humans.
        - human_radiuses (self.n_humans,): jnp.ndarray containing the radius of the humans.

        output:
        - lidar_output (self.lidar_num_rays,2): jnp.ndarray containing the lidar measurements of the robot and the angle for each ray.
        """
        angles = jnp.linspace(robot_yaw - self.lidar_angular_range/2, robot_yaw + self.lidar_angular_range/2, self.lidar_num_rays)
        measurements = self._batch_ray_cast(angles, robot_position, human_positions, human_radiuses)
        lidar_output = jnp.stack((measurements, angles), axis=-1)
        return lidar_output