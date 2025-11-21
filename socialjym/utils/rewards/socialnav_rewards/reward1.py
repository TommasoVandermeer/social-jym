from jax import jit, lax
import jax.numpy as jnp
from functools import partial

from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout

class Reward1(BaseReward):
    def __init__(
        self, 
        gamma:float=0.9, # Discount factor
        v_max:float=1., # Maximum speed of the robot
        goal_reward: float=1., 
        collision_penalty: float=-0.25, 
        discomfort_distance: float=0.2, 
        time_limit: float=50.,
        kinematics: str='holonomic'
    ) -> None:
        super().__init__(gamma)
        # Check input parameters
        assert v_max > 0, "v_max must be positive"
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_penalty < 0, "collision_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"
        # Initialize reward parameters
        self.type = "socialnav_reward1"
        self.v_max = v_max
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.discomfort_distance = discomfort_distance
        self.time_limit = time_limit
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        # Define terminations
        self.interval_collision_termination = IntervalRobotHumanCollision()
        self.instant_collision_termination = InstantRobotHumanCollision()
        self.goal_reached_termination = RobotReachedGoal()
        self.timeout = Timeout(time_limit)

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
        - obs: observation of the current state of the environment (IMPORTANT: action is embedded in here and its (Vx,Vy) in case of holonomic kinematics and (v,w) in case of unicycle kinematics)
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
        # Compute next positions
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            next_robot_pos = robot_pos + action * dt
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
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
        next_humans_pos = humans_pos + obs[0:-1,2:4] * dt
        # Collision and discomfort detection with humans (within a duration of dt)
        collision, collision_info = self.interval_collision_termination(
            robot_pos, 
            next_robot_pos,
            robot_radius,
            humans_pos,
            next_humans_pos,
            humans_radiuses
        )
        min_distance = collision_info['min_distance']
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision), min_distance < self.discomfort_distance]))
        # Check if the robot reached its goal
        reached_goal, _ = self.goal_reached_termination(
            next_robot_pos,
            robot_radius,
            robot_goal,
        )
        # Timeout
        timeout, _ =  self.timeout(time) 
        # Compute reward
        reward = 0.
        reward = lax.cond(~(collision) & (reached_goal), lambda r: r + self.goal_reward, lambda r: r, reward) # Reward for reaching the goal
        reward = lax.cond(collision, lambda r: r + self.collision_penalty, lambda r: r, reward) # Penalty for collision
        reward = lax.cond(discomfort, lambda r: r - 0.5 * dt * (self.discomfort_distance - min_distance), lambda r: r, reward) # Penalty for getting too close to humans
        # Compute outcome 
        outcome = {
            "nothing": ~((collision) | (reached_goal) | (timeout)),
            "success": (~(collision)) & (reached_goal),
            "failure": collision,
            "timeout": timeout & (~(collision)) & (~(reached_goal))
        }
        # # DEBUG
        # debug.print("\n")
        # debug.print("collision: {x}", x=collision)
        # debug.print("reached_goal: {x}", x=reached_goal)
        # debug.print("timeout: {x}", x=timeout)
        return reward, outcome