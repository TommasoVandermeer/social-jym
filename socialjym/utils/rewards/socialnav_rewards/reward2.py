from jax import jit, lax, vmap, debug
import jax.numpy as jnp
from functools import partial

from socialjym.utils.aux_functions import batch_point_to_line_distance, binary_to_decimal
from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, wrap_angle

@jit
def batch_wrap_angle(angles:jnp.ndarray) -> jnp.ndarray:
    return vmap(wrap_angle, in_axes=0)(angles)

@jit
def _set_nan_if_false(x:float, y:bool) -> float:
    return lax.cond(
        y,
        lambda x: x,
        lambda _: jnp.nan,
        x)

@jit
def _batch_set_nan_if_false(x:jnp.ndarray, y:jnp.ndarray) -> jnp.ndarray:
    return vmap(_set_nan_if_false, in_axes=(0,0))(x,y)

class Reward2(BaseReward):
    def __init__(
        self, 
        target_reached_reward: bool=True,
        collision_penalty_reward: bool=True,
        discomfort_penalty_reward: bool=True,
        progress_to_goal_reward: bool=True,
        time_penalty_reward: bool=True,
        high_rotation_penalty_reward: bool=True,
        heading_deviation_from_goal_penalty_reward: bool=True,
        time_limit: float=50.,
        goal_reward: float=1., 
        collision_penalty: float=-0.25, 
        discomfort_distance: float=0.2, 
        progress_to_goal_weight: float=0.1,
        time_penalty: float=0.01,
        angular_speed_bound: float=0.7,
        angular_speed_penalty_weight: float=0.1,
        n_heading_samples: int=30,
        max_heading_deviation: float=jnp.pi/6,
        heading_deviation_weight: float=0.6
    ) -> None:
        # Check input parameters
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_penalty < 0, "collision_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"
        assert progress_to_goal_weight > 0, "progress_to_goal_weight must be positive"
        assert time_penalty > 0, "time_penalty must be positive"
        assert angular_speed_bound > 0, "angular_speed_bound must be positive"
        assert angular_speed_penalty_weight > 0, "angular_speed_penalty_weight must be positive"
        # Define reward type
        self.target_reached_reward = target_reached_reward
        self.collision_penalty_reward = collision_penalty_reward
        self.discomfort_distance_penalty_reward = discomfort_penalty_reward
        self.progress_to_goal_reward = progress_to_goal_reward
        self.time_penalty_reward = time_penalty_reward
        self.high_rotation_penalty_reward = high_rotation_penalty_reward
        self.heading_deviation_from_goal_penalty_reward = heading_deviation_from_goal_penalty_reward
        self.binary_reward = jnp.array([
            heading_deviation_from_goal_penalty_reward,
            high_rotation_penalty_reward,
            time_penalty_reward,
            progress_to_goal_reward,
            discomfort_penalty_reward,
            collision_penalty_reward,
            target_reached_reward], dtype=int)
        self.decimal_reward = binary_to_decimal(self.binary_reward)
        self.type = f"socialnav_reward2_{self.decimal_reward}"
        # Initialize reward parameters
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.discomfort_distance = discomfort_distance
        self.time_limit = time_limit
        self.progress_to_goal_weight = progress_to_goal_weight
        self.time_penalty = time_penalty
        self.angular_speed_bound = angular_speed_bound
        self.angular_speed_penalty_weight = angular_speed_penalty_weight
        self.n_heading_samples = n_heading_samples
        self.max_heading_deviation = max_heading_deviation
        self.heading_deviation_weight = heading_deviation_weight
        # Default parameters
        self.kinematics = ROBOT_KINEMATICS.index('unicycle')   

    @partial(jit, static_argnames=("self"))
    def _is_collision_free_for_single_human(
        self,
        heading:float, 
        robot_position:jnp.ndarray,
        robot_lin_velocity:jnp.ndarray, 
        robot_radius:float,
        human_position:jnp.ndarray, 
        human_velocity:jnp.ndarray,
        human_radius:float
    ) -> bool:
        theta_vab = jnp.atan2(
            robot_lin_velocity[0] * jnp.sin(heading) - human_velocity[1], 
            robot_lin_velocity[0] * jnp.cos(heading) - human_velocity[0])
        theta = jnp.arctan2(human_position[1] - robot_position[1], human_position[0] - robot_position[0])
        beta = jnp.arcsin((robot_radius + human_radius) / jnp.linalg.norm(human_position - robot_position))
        return jnp.all(jnp.array([theta_vab < theta - beta, theta_vab > theta + beta]))
    
    @partial(jit, static_argnames=("self"))
    def _is_collision_free_for_all_humans(
        self,
        heading:float, 
        robot_position:jnp.ndarray,
        robot_lin_velocity:jnp.ndarray, 
        robot_radius:float,
        humans_positions:jnp.ndarray, 
        humans_velocities:jnp.ndarray,
        humans_radiuses:jnp.ndarray
    ) -> bool:
        return jnp.all(vmap(
            Reward2._is_collision_free_for_single_human, 
            in_axes=(None, None, None, None, None, 0, 0, 0))(
                self,
                heading, 
                robot_position, 
                robot_lin_velocity, 
                robot_radius, 
                humans_positions,
                humans_velocities,
                humans_radiuses))

    @partial(jit, static_argnames=("self"))
    def _batch_is_heading_collision_free(
        self,
        headings:float, 
        robot_position:jnp.ndarray,
        robot_lin_velocity:jnp.ndarray, 
        robot_radius:float,
        humans_positions:jnp.ndarray, 
        humans_velocities:jnp.ndarray,
        humans_radiuses:jnp.ndarray
    ) -> bool:
        return vmap(
            Reward2._is_collision_free_for_all_humans, 
            in_axes=(None, 0, None, None, None, None, None, None))(
                self,
                headings,
                robot_position,
                robot_lin_velocity,
                robot_radius,
                humans_positions,
                humans_velocities,
                humans_radiuses)

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        obs:jnp.ndarray, 
        info:dict, 
        dt:float
    ) -> tuple[float, dict]:
        """
        Given a state and a dictionary containing additional information about the environment,
        this function computes the reward of the current state and wether the episode is finished or not.
        This function is public so that it can be called by the agent policy to compute the best action.

        This reward contains several contributions which can be activated or deactivated by setting the corresponding parameters to True or False.

        args:
        - obs: observation of the current state of the environment (IMPORTANT: action is embedded in here and its (v,w) (unicycle kinematics)
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation

        output:
        - reward: reward obtained in the current state.
        - outcome: dictionary indicating if the episode is finished or not and why.
        """
        robot_pos = obs[-1,0:2]
        humans_pos = obs[0:len(obs)-1,0:2]
        robot_goal = info["robot_goal"]
        humans_radiuses = obs[0:len(obs)-1,4]
        robot_radius = obs[-1,4]
        action = obs[-1,2:4]
        time = info["time"]
        ### COMPUTE NECESSARY DATA ###
        # Compute next robot position and theta
        next_robot_pos = lax.cond(
            action[1] != 0,
            lambda x: x.at[:].set(jnp.array([
                x[0] + (action[0]/action[1]) * (jnp.sin(obs[-1,5] + action[1] * dt) - jnp.sin(obs[-1,5])),
                x[1] + (action[0]/action[1]) * (jnp.cos(obs[-1,5]) - jnp.cos(obs[-1,5] + action[1] * dt))
            ])),
            lambda x: x.at[:].set(jnp.array([
                x[0] + action[0] * dt * jnp.cos(obs[-1,5]),
                x[1] + action[0] * dt * jnp.sin(obs[-1,5])
            ])),
            robot_pos)
        # Compute next humans positions
        next_humans_pos = humans_pos + obs[0:-1,2:4] * dt
        # Progress to goal
        progress_to_goal = jnp.linalg.norm(robot_pos - robot_goal) - jnp.linalg.norm(next_robot_pos - robot_goal)
        # Collision and discomfort detection with humans (within a duration of dt)
        distances = batch_point_to_line_distance(jnp.zeros((len(obs)-1,2)), humans_pos - robot_pos, next_humans_pos - next_robot_pos) - (humans_radiuses + robot_radius)
        collision = jnp.any(distances < 0)
        min_distance = jnp.max(jnp.array([jnp.min(distances),0]))
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision), min_distance < self.discomfort_distance]))
        # Check if the robot reached its goal
        reached_goal = jnp.linalg.norm(next_robot_pos - robot_goal) < robot_radius
        # Timeout
        timeout = jnp.all(jnp.array([time >= self.time_limit, jnp.logical_not(collision), jnp.logical_not(reached_goal)]))
        # Desired direction angle search
        @jit
        def _compute_desired_heading(
            robot_pos:jnp.ndarray,
            robot_heading:float,
            robot_action:jnp.ndarray,
            robot_radius:float,
            robot_goal:jnp.ndarray,
            humans_pos:jnp.ndarray,
            humans_velocities:jnp.ndarray,
            humans_radiuses:jnp.ndarray
        ) -> float:
            headings = jnp.linspace(-jnp.pi, jnp.pi, self.n_heading_samples) 
            feasible_headings = self._batch_is_heading_collision_free(
                headings, 
                robot_pos, 
                robot_action[0] * jnp.array([jnp.cos(robot_heading), jnp.sin(robot_heading)]),
                robot_radius, 
                humans_pos, 
                humans_velocities, 
                humans_radiuses)
            # debug.print("feasible_headings: {x}", x=feasible_headings)
            goal_heading = jnp.arctan2(robot_goal[1] - robot_pos[1], robot_goal[0] - robot_pos[0])
            headings_diff = jnp.abs(batch_wrap_angle(headings - goal_heading))
            headings_diff = _batch_set_nan_if_false(headings_diff, feasible_headings)
            desired_heading = lax.cond(
                jnp.all(jnp.logical_not(feasible_headings)), # If no feasible heading is found, desired heading is nan (the reward contribution will be 0)
                lambda _: jnp.nan,
                lambda _: headings[jnp.nanargmin(headings_diff)],
                None)
            return desired_heading    
        desired_heading = lax.cond(
            jnp.all(jnp.array([self.heading_deviation_from_goal_penalty_reward, jnp.logical_not(reached_goal)])),
            lambda _: _compute_desired_heading(
                robot_pos, 
                obs[-1,5], 
                action, 
                robot_radius, 
                robot_goal, 
                humans_pos, 
                obs[0:-1,2:4], 
                humans_radiuses),
            lambda _: jnp.nan,
            None)
        ### COMPUTE REWARD ###
        reward = 0.
        # Reward for reaching the goal
        reward = lax.cond(
            jnp.all(jnp.array([self.target_reached_reward, reached_goal])), 
            lambda r: r + self.goal_reward, 
            lambda r: r, 
            reward)
        # Penalty for collision
        reward = lax.cond(
            jnp.all(jnp.array([self.collision_penalty_reward, collision])), 
            lambda r: r + self.collision_penalty, 
            lambda r: r, 
            reward) 
        # Penalty for getting too close to humans
        reward = lax.cond(
            jnp.all(jnp.array([self.discomfort_distance_penalty_reward, discomfort])), 
            lambda r: r - 0.5 * dt * (self.discomfort_distance - min_distance), 
            lambda r: r, 
            reward)
        # Progress to goal reward
        reward = lax.cond(
            jnp.all(jnp.array([self.progress_to_goal_reward, jnp.logical_not(reached_goal)])), 
            lambda r: r + self.progress_to_goal_weight * progress_to_goal, 
            lambda r: r, 
            reward)
        # Time penalty
        reward = lax.cond(
            jnp.all(jnp.array([self.time_penalty_reward, jnp.logical_not(reached_goal)])), 
            lambda r: r - self.time_penalty, 
            lambda r: r, 
            reward)
        # High rotation penalty
        reward = lax.cond(
            jnp.all(jnp.array([self.high_rotation_penalty_reward, jnp.abs(action[1]) > self.angular_speed_bound])), 
            lambda r: r - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
            lambda r: r, 
            reward)
        # Active heading direction
        reward = lax.cond(
            jnp.all(jnp.array([self.heading_deviation_from_goal_penalty_reward, jnp.logical_not(reached_goal), jnp.logical_not(jnp.isnan(desired_heading))])), 
            lambda r: r + self.heading_deviation_weight * wrap_angle(self.max_heading_deviation - jnp.abs(desired_heading)),
            lambda r: r, 
            reward)
        ### COMPUTE OUTCOME ###
        outcome = {
            "nothing": jnp.logical_not(jnp.any(jnp.array([collision,reached_goal,timeout]))),
            "success": reached_goal,
            "failure": collision,
            "timeout": timeout
        }
        # # DEBUG
        # debug.print("\n")
        # debug.print("collision: {x}", x=collision)
        # debug.print("reached_goal: {x}", x=reached_goal)
        # debug.print("timeout: {x}", x=timeout)
        return reward, outcome