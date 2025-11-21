from jax import jit, lax, vmap
import jax.numpy as jnp
from functools import partial

from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, HUMAN_POLICIES
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_obstacle_collision import InstantRobotObstacleCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout
from jhsfm.hsfm import get_linear_velocity

class DummyReward(BaseReward):
    def __init__(
        self, 
        robot_radius: float,
        v_max: float=1.0,
        time_limit: float=50.,
    ) -> None:
        super().__init__(0.9)
        # Initialize reward parameters
        self.v_max = v_max
        self.type = "lasernav_dummyreward"
        self.time_limit = time_limit
        self.kinematics = ROBOT_KINEMATICS.index('unicycle')
        self.robot_radius = robot_radius
        self.humans_policy = HUMAN_POLICIES.index('hsfm')
        # Define terminations
        self.interval_human_collision_termination = IntervalRobotHumanCollision()
        self.instant_human_collision_termination = InstantRobotHumanCollision()
        self.instant_obstacle_collision_termination = InstantRobotObstacleCollision()
        self.goal_reached_termination = RobotReachedGoal()
        self.timeout = Timeout(time_limit)

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        state:jnp.ndarray, 
        action:jnp.ndarray,
        info:dict, 
        dt:float
    ) -> tuple[float, dict]:
        """
        Given a state and a dictionary containing additional information about the environment,
        this function computes the reward of the current state and wether the episode is finished or not.
        This function is public so that it can be called by the agent policy to compute the best action.

        This is the classical sparse reward with personal space invasion penalization used in the Social Navigation literature.

        args:
        - state: current state of the environment
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation

        output:
        - reward: 0.0 (dummy reward)
        - outcome: dictionary indicating if the episode is finished or not and why.
        """
        robot_pos = state[-1,:2]
        robot_yaw = state[-1,4]
        humans_pos = state[:-1,:2]
        robot_goal = info["robot_goal"]
        humans_radiuses = info["humans_parameters"][:,0]
        robot_radius = self.robot_radius
        time = info["time"]
        # Compute next positions
        next_robot_pos = lax.cond(
            action[1] != 0,
            lambda x: x.at[:].set(jnp.array([
                x[0] + (action[0]/action[1]) * (jnp.sin(robot_yaw + action[1] * dt) - jnp.sin(robot_yaw)),
                x[1] + (action[0]/action[1]) * (jnp.cos(robot_yaw) - jnp.cos(robot_yaw + action[1] * dt))
            ])),
            lambda x: x.at[:].set(jnp.array([
                x[0] + action[0] * dt * jnp.cos(robot_yaw),
                x[1] + action[0] * dt * jnp.sin(robot_yaw)
            ])),
            robot_pos)
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            humans_orientations = state[:-1,4]
            humans_velocities = vmap(get_linear_velocity)(humans_orientations, state[:-1,2:4])
            next_humans_pos = humans_pos + humans_velocities * dt
        else:
            next_humans_pos = humans_pos + state[:-1,2:4] * dt
        # Collision detection with humans (within a duration of dt)
        collision_with_human, _ = self.interval_human_collision_termination(
            robot_pos, 
            next_robot_pos,
            robot_radius,
            humans_pos,
            next_humans_pos,
            humans_radiuses
        )
        # Collision detection with obstacles
        collision_with_obstacle, _ = self.instant_obstacle_collision_termination(
            next_robot_pos,
            robot_radius,
            info['static_obstacles'][-1],
        )
        # Check if the robot reached its goal
        reached_goal, _ = self.goal_reached_termination(
            next_robot_pos,
            robot_radius,
            robot_goal,
        )
        # Timeout
        timeout, _ =  self.timeout(time) # Compute outcome 
        collision = collision_with_human | collision_with_obstacle
        outcome = {
            "nothing": ~((collision) | (reached_goal) | (timeout)),
            "success": (~(collision)) & (reached_goal),
            "collision_with_human": collision_with_human,
            "collision_with_obstacle": collision_with_obstacle,
            "timeout": timeout & (~(collision)) & (~(reached_goal))
        }
        return 0., outcome