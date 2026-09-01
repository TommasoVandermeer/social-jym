from jax import jit, lax, vmap
import jax.numpy as jnp
from functools import partial
from typing import Union

from socialjym.utils.aux_functions import binary_to_decimal
from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, HUMAN_POLICIES
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_obstacle_collision import InstantRobotObstacleCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout
from jhsfm.hsfm import get_linear_velocity

class Reward1(BaseReward):
    def __init__(
        self, 
        robot_radius: float,
        gamma:Union[float, list, tuple, jnp.ndarray] = 0.9, # Discount factor
        v_max: float=1.0,
        time_limit: float=50.,
        target_reached_reward: bool=True,
        collision_with_humans_penalty_reward: bool=True,
        collision_with_obstacles_penalty_reward: bool=True,
        discomfort_penalty_reward: bool=True,
        progress_to_goal_reward: bool=True,
        high_rotation_penalty_reward: bool=True,
        goal_reward: float=1., 
        collision_with_humans_penalty: float=-0.25, 
        collision_with_obstacles_penalty: float=-0.05,
        discomfort_distance: float=0.2, 
        progress_to_goal_weight: float=0.03,
        angular_speed_bound: float=1.,
        angular_speed_penalty_weight: float=0.0075,
    ) -> None:
        super().__init__(gamma)
        # Check input parameters
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_with_humans_penalty < 0, "collision_with_humans_penalty must be negative"
        assert collision_with_obstacles_penalty < 0, "collision_with_obstacles_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"
        assert progress_to_goal_weight > 0, "progress_to_goal_weight must be positive"
        assert angular_speed_bound > 0, "angular_speed_bound must be positive"
        assert angular_speed_penalty_weight > 0, "angular_speed_penalty_weight must be positive"
        # Define reward type
        self.target_reached_reward = target_reached_reward
        self.collision_with_humans_penalty_reward = collision_with_humans_penalty_reward
        self.collision_with_obstacles_penalty_reward = collision_with_obstacles_penalty_reward
        self.discomfort_distance_penalty_reward = discomfort_penalty_reward
        self.progress_to_goal_reward = progress_to_goal_reward
        self.high_rotation_penalty_reward = high_rotation_penalty_reward
        self.binary_reward = jnp.array(
            [
                high_rotation_penalty_reward,
                progress_to_goal_reward,
                discomfort_penalty_reward,
                collision_with_humans_penalty_reward,
                collision_with_obstacles_penalty_reward,
                target_reached_reward
            ], 
            dtype=int
        )
        if isinstance(gamma, (list, tuple, jnp.ndarray)):
            print(
                "REWARD - Multi-discount mode active. Gammas will be assigned in this order:",
                "\n- Rotational penalty" if self.high_rotation_penalty_reward else "",
                "\n- Progress to goal" if self.progress_to_goal_reward else "",
                "\n- Discomfort" if self.discomfort_distance_penalty_reward else "",
                "\n- Collision w/ humans" if self.collision_with_humans_penalty_reward else "",
                "\n- Collision w/ obstacles" if self.collision_with_obstacles_penalty_reward else "",
                "\n- Target reached" if self.target_reached_reward else "",
            )
            self.multi_gamma = True
            gamma_list = [float(g) for g in gamma]
            assert len(gamma_list) == jnp.sum(self.binary_reward), "Number of gammas must be the same as active reward terms."
            idx = 0
            self.g_rot = gamma_list[idx] if self.high_rotation_penalty_reward else None
            idx += 1 if self.high_rotation_penalty_reward else 0
            self.g_prog = gamma_list[idx] if self.progress_to_goal_reward else None
            idx += 1 if self.progress_to_goal_reward else 0
            self.g_disc = gamma_list[idx] if self.collision_with_obstacles_penalty_reward else None
            idx += 1 if self.collision_with_obstacles_penalty_reward else 0
            self.g_coll_hum = gamma_list[idx] if self.collision_with_humans_penalty_reward else None
            idx += 1 if self.collision_with_humans_penalty_reward else 0
            self.g_coll_obs = gamma_list[idx] if self.collision_with_obstacles_penalty_reward else None
            idx += 1 if self.collision_with_obstacles_penalty_reward else 0
            self.g_goal = gamma_list[idx] if self.target_reached_reward else None
            idx += 1 if self.target_reached_reward else 0
            self.unique_gammas = tuple(set(gamma_list))
        else:
            self.multi_gamma = False
            self.unique_gammas = (float(gamma),)
        self.decimal_reward = binary_to_decimal(self.binary_reward)
        self.type = f"lasernav_reward2_{self.decimal_reward}"
        # Initialize reward parameters
        self.v_max = v_max
        self.time_limit = time_limit
        self.goal_reward = goal_reward
        self.collision_with_humans_penalty = collision_with_humans_penalty
        self.collision_with_obstacles_penalty = collision_with_obstacles_penalty
        self.discomfort_distance = discomfort_distance
        self.progress_to_goal_weight = progress_to_goal_weight
        self.angular_speed_bound = angular_speed_bound
        self.angular_speed_penalty_weight = angular_speed_penalty_weight
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
        - reward_terms: dictionary of all the reward terms with different discounts
        """
        robot_pos = state[-1,:2]
        robot_yaw = state[-1,4]
        humans_pos = state[:-1,:2]
        robot_goal = info["robot_goal"]
        humans_radiuses = info["humans_parameters"][:,0]
        robot_radius = info["_robot_params"]["radius"] if "_robot_params" in info else self.robot_radius
        time = info["time"]
        # Compute next positions
        next_robot_pos = lax.cond(
            jnp.abs(action[1]) > 1e-3,
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
        collision_with_human, collision_with_human_info = self.interval_human_collision_termination(
            robot_pos, 
            next_robot_pos,
            robot_radius,
            humans_pos,
            next_humans_pos,
            humans_radiuses
        )
        min_distance = collision_with_human_info['min_distance']
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision_with_human), min_distance < self.discomfort_distance]))
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
        ### COMPUTE OUTCOME ###
        failure = collision_with_human | collision_with_obstacle
        outcome = {
            "nothing": ~((failure) | (reached_goal) | (timeout)),
            "success": (~(failure)) & (reached_goal),
            "collision_with_human": collision_with_human,
            "collision_with_obstacle": collision_with_obstacle,
            "timeout": timeout & (~(failure)) & (~(reached_goal))
        }
        ### COMPUTE REWARD ###
        # Reward for reaching the goal
        if self.target_reached_reward:
            goal_reward = lax.cond(
                ~(failure) & (reached_goal), 
                lambda: self.goal_reward, 
                lambda: 0., 
            )
        else:
            goal_reward = 0.
        # Penalty for collision with humans
        if self.collision_with_humans_penalty_reward:
            collision_human_reward = lax.cond(
                collision_with_human, 
                lambda: self.collision_with_humans_penalty, 
                lambda: 0., 
            ) 
        else:
            collision_human_reward = 0.
        # Penalty for collision with obstacles
        if self.collision_with_obstacles_penalty_reward:
            collision_obstacle_reward = lax.cond(
                collision_with_obstacle, 
                lambda: self.collision_with_obstacles_penalty, 
                lambda: 0., 
            )
        else:
            collision_obstacle_reward = 0.
        # Penalty for getting too close to humans
        if self.discomfort_distance_penalty_reward:
            discomfort = (~(failure)) & (min_distance < self.discomfort_distance)
            discomfort_reward = lax.cond(
                discomfort, 
                lambda: - 0.5 * dt * (self.discomfort_distance - min_distance), 
                lambda: 0., 
            )
        else:
            discomfort_reward = 0.
        # Progress to goal reward
        if self.progress_to_goal_reward:
            progress_to_goal = jnp.linalg.norm(robot_pos - robot_goal) - jnp.linalg.norm(next_robot_pos - robot_goal)
            progress_reward = lax.cond(
                ~(reached_goal), 
                lambda: + self.progress_to_goal_weight * progress_to_goal, 
                lambda: 0., 
            )
        else:
            progress_reward = 0.
        # High rotation penalty
        if self.high_rotation_penalty_reward:
            rotation_reward = lax.cond(
                jnp.abs(action[1]) > self.angular_speed_bound, 
                lambda: - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
                lambda: 0., 
            )
        else:
            rotation_reward = 0.
        reward = goal_reward + collision_human_reward + collision_obstacle_reward + discomfort_reward + progress_reward + rotation_reward
        if self.multi_gamma:
            reward_terms = {g: 0.0 for g in self.unique_gammas}
            if self.target_reached_reward:
                reward_terms[self.g_goal] += goal_reward
            if self.collision_with_humans_penalty_reward:
                reward_terms[self.g_coll_hum] += collision_human_reward
            if self.collision_with_obstacles_penalty_reward:
                reward_terms[self.g_coll_obs] += collision_obstacle_reward
            if self.discomfort_distance_penalty_reward:
                reward_terms[self.g_disc] += discomfort_reward
            if self.progress_to_goal_reward:
                reward_terms[self.g_prog] += progress_reward
            if self.high_rotation_penalty_reward:
                reward_terms[self.g_rot] += rotation_reward
        else:
            reward_terms = {self.gamma: reward}
        return reward, outcome, reward_terms
