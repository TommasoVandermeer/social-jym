from jax import jit, lax
import jax.numpy as jnp
from functools import partial

from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout

class DummyReward(BaseReward):
    def __init__(
        self, 
        v_max: float=1.0,
        time_limit: float=50.,
        kinematics: str='holonomic'
    ) -> None:
        super().__init__(0.9)
        # Initialize reward parameters
        self.v_max = v_max
        self.type = "socialnav_dummyreward"
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
        - reward: 0.0 (dummy reward)
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
        collision, _ = self.interval_collision_termination(
            robot_pos, 
            next_robot_pos,
            robot_radius,
            humans_pos,
            next_humans_pos,
            humans_radiuses
        )
        # Check if the robot reached its goal
        reached_goal, _ = self.goal_reached_termination(
            next_robot_pos,
            robot_radius,
            robot_goal,
        )
        # Timeout
        timeout, _ =  self.timeout(time) # Compute outcome 
        outcome = {
            "nothing": ~((collision) | (reached_goal) | (timeout)),
            "success": (~(collision)) & (reached_goal),
            "failure": collision,
            "timeout": timeout & (~(collision)) & (~(reached_goal))
        }
        return 0., outcome