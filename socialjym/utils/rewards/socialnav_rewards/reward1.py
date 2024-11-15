from jax import jit, lax
import jax.numpy as jnp
from functools import partial

from socialjym.utils.aux_functions import batch_point_to_line_distance
from socialjym.utils.rewards.base_reward import BaseReward

class Reward1(BaseReward):
    def __init__(
        self, 
        goal_reward: float=1., 
        collision_penalty: float=-0.25, 
        discomfort_distance: float=0.2, 
        time_limit: float=50.,
    ) -> None:
        # Check input parameters
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_penalty < 0, "collision_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"
        # Initialize reward parameters
        self.type = "socialnav_reward1"
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.discomfort_distance = discomfort_distance
        self.time_limit = time_limit

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

        This is the classical sparse reward with personal space invasion penalization used in the Social Navigation literature.

        args:
        - obs: observation of the current state of the environment (IMPORTANT: action is embedded in here)
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
        time = info["time"]
        next_robot_pos = robot_pos + obs[-1,2:4] * dt
        next_humans_pos = humans_pos + obs[0:len(obs)-1,2:4] * dt
        # Collision and discomfort detection with humans (within a duration of dt)
        distances = batch_point_to_line_distance(jnp.zeros((len(obs)-1,2)), humans_pos - robot_pos, next_humans_pos - next_robot_pos) - (humans_radiuses + robot_radius)
        collision = jnp.any(distances < 0)
        min_distance = jnp.max(jnp.array([jnp.min(distances),0]))
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision), min_distance < self.discomfort_distance]))
        # Check if the robot reached its goal
        reached_goal = jnp.linalg.norm(next_robot_pos - robot_goal) < robot_radius
        # Timeout
        timeout = jnp.all(jnp.array([time >= self.time_limit, jnp.logical_not(collision), jnp.logical_not(reached_goal)]))
        # Compute reward
        # reward = - (jnp.linalg.norm(next_robot_pos - robot_goal)/100) # Debug reward (agent should always move towards the goal)
        reward = 0.
        reward = lax.cond(reached_goal, lambda r: r + self.goal_reward, lambda r: r, reward) # Reward for reaching the goal
        reward = lax.cond(collision, lambda r: r + self.collision_penalty, lambda r: r, reward) # Penalty for collision
        reward = lax.cond(discomfort, lambda r: r - 0.5 * dt * (self.discomfort_distance - min_distance), lambda r: r, reward) # Penalty for getting too close to humans
        # Compute outcome 
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