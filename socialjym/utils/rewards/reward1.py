from types import FunctionType
from jax import jit, lax, debug
import jax.numpy as jnp

from socialjym.utils.aux_functions import batch_point_to_line_distance

def generate_reward_done_function(
        goal_reward: float, 
        collision_penalty : float, 
        discomfort_distance: float, 
        time_limit: float
    ) -> FunctionType:
    
    @jit
    def get_reward_done(obs:jnp.ndarray, info:dict, dt:float) -> tuple[float, bool]:
        """
        Given a state and a dictionary containing additional information about the environment,
        this function computes the reward of the current state and wether the episode is finished or not.
        This function is public so that it can be called by the agent policy to compute the best action.

        This is the classical sparse reward with personal space invasion penalization used in the Social Navigation literature.

        args:
        - obs: observation of the current state of the environment
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation

        output:
        - reward: reward obtained in the current state.
        - done: boolean indicating whether the episode is done.
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
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision), min_distance < discomfort_distance]))
        # Check if the robot reached its goal
        reached_goal = jnp.linalg.norm(next_robot_pos - robot_goal) < robot_radius
        # Timeout
        timeout = jnp.any(time >= time_limit)
        # Compute reward
        # reward = - (jnp.linalg.norm(next_robot_pos - robot_goal)/100) # Debug reward (agent should always move towards the goal)
        reward = 0.
        reward = lax.cond(reached_goal, lambda r: r + goal_reward, lambda r: r, reward) # Reward for reaching the goal
        reward = lax.cond(collision, lambda r: r - collision_penalty, lambda r: r, reward) # Penalty for collision
        reward = lax.cond(discomfort, lambda r: r - 0.5 * dt * (discomfort_distance - min_distance), lambda r: r, reward) # Penalty for getting too close to humans
        # Compute done 
        done = jnp.any(jnp.array([collision,reached_goal,timeout]))
        # # DEBUG
        # debug.print("\n")
        # debug.print("collision: {x}", x=collision)
        # debug.print("reached_goal: {x}", x=reached_goal)
        # debug.print("timeout: {x}", x=timeout)
        return reward, done
    
    return get_reward_done