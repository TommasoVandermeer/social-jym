from jax import jit, lax, vmap, debug
import jax.numpy as jnp
from functools import partial

from socialjym.utils.aux_functions import batch_point_to_line_distance, binary_to_decimal
from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, wrap_angle

@jit
def batch_wrap_angle(angles:jnp.ndarray) -> jnp.ndarray:
    return vmap(wrap_angle, in_axes=0)(angles)

class Reward2(BaseReward):
    def __init__(
        self, 
        gamma:float=0.9, # Discount factor
        v_max:float=1., # Maximum speed of the robot
        target_reached_reward: bool=True,
        collision_penalty_reward: bool=True,
        discomfort_penalty_reward: bool=True,
        progress_to_goal_reward: bool=False,
        time_penalty_reward: bool=False,
        high_rotation_penalty_reward: bool=False,
        time_limit: float=50.,
        goal_reward: float=1., 
        collision_penalty: float=-0.25, 
        discomfort_distance: float=0.2, 
        progress_to_goal_weight: float=0.15,
        time_penalty: float=0.01,
        angular_speed_bound: float=2.,
        angular_speed_penalty_weight: float=0.1,
    ) -> None:
        super().__init__(gamma)
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
        self.binary_reward = jnp.array([
            high_rotation_penalty_reward,
            time_penalty_reward,
            progress_to_goal_reward,
            discomfort_penalty_reward,
            collision_penalty_reward,
            target_reached_reward], dtype=int)
        self.decimal_reward = binary_to_decimal(self.binary_reward)
        self.type = f"socialnav_reward2_{self.decimal_reward}"
        # Initialize reward parameters
        self.v_max = v_max
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.discomfort_distance = discomfort_distance
        self.time_limit = time_limit
        self.progress_to_goal_weight = progress_to_goal_weight
        self.time_penalty = time_penalty
        self.angular_speed_bound = angular_speed_bound
        self.angular_speed_penalty_weight = angular_speed_penalty_weight
        # Default parameters
        self.kinematics = ROBOT_KINEMATICS.index('unicycle')   

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
        # Collision with humans (within a duration of dt)
        distances = batch_point_to_line_distance(jnp.zeros((len(obs)-1,2)), humans_pos - robot_pos, next_humans_pos - next_robot_pos) - (humans_radiuses + robot_radius)
        collision = jnp.any(distances < 0)
        # Check if the robot reached its goal
        reached_goal = jnp.linalg.norm(next_robot_pos - robot_goal) < robot_radius
        # Timeout
        timeout = (time >= self.time_limit) & (~(collision)) & (~(reached_goal))
        ### COMPUTE REWARD ###
        reward = 0.
        # Reward for reaching the goal
        if self.target_reached_reward:
            reward = lax.cond(
                ~(collision) & (reached_goal), 
                lambda r: r + self.goal_reward, 
                lambda r: r, 
                reward
            )
        # Penalty for collision
        if self.collision_penalty_reward:
            reward = lax.cond(
                collision, 
                lambda r: r + self.collision_penalty, 
                lambda r: r, 
                reward
            ) 
        # Penalty for getting too close to humans
        if self.discomfort_distance_penalty_reward:
            min_distance = jnp.max(jnp.array([jnp.min(distances),0]))
            discomfort = (~(collision)) & (min_distance < self.discomfort_distance)
            reward = lax.cond(
                discomfort, 
                lambda r: r - 0.5 * dt * (self.discomfort_distance - min_distance), 
                lambda r: r, 
                reward
            )
        # Progress to goal reward
        if self.progress_to_goal_reward:
            progress_to_goal = jnp.linalg.norm(robot_pos - robot_goal) - jnp.linalg.norm(next_robot_pos - robot_goal)
            reward = lax.cond(
                ~(reached_goal), 
                lambda r: r + self.progress_to_goal_weight * progress_to_goal, 
                lambda r: r, 
                reward
            )
        # Time penalty
        if self.time_penalty_reward:
            reward = lax.cond(
                ~(reached_goal), 
                lambda r: r - self.time_penalty, 
                lambda r: r, 
                reward
            )
        # High rotation penalty
        if self.high_rotation_penalty_reward:
            reward = lax.cond(
                jnp.abs(action[1]) > self.angular_speed_bound, 
                lambda r: r - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
                lambda r: r, 
                reward
            )
        ### COMPUTE OUTCOME ###
        outcome = {
            "nothing": ~((collision) | (reached_goal) | (timeout)),
            "success": ~(collision) & (reached_goal),
            "failure": collision,
            "timeout": timeout
        }
        # # DEBUG
        # debug.print("\n")
        # debug.print("collision: {x}", x=collision)
        # debug.print("reached_goal: {x}", x=reached_goal)
        # debug.print("timeout: {x}", x=timeout)
        return reward, outcome