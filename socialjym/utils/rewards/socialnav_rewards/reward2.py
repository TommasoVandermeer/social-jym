from jax import jit, lax, vmap, debug
import jax.numpy as jnp
from functools import partial
from typing import Union

from socialjym.utils.aux_functions import binary_to_decimal
from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, wrap_angle
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout

@jit
def batch_wrap_angle(angles:jnp.ndarray) -> jnp.ndarray:
    return vmap(wrap_angle, in_axes=0)(angles)

class Reward2(BaseReward):
    def __init__(
        self, 
        gamma:Union[float, list, tuple, jnp.ndarray] = 0.9, # Discount factor
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
        if isinstance(gamma, (list, tuple, jnp.ndarray)):
            print(
                "REWARD - Multi-discount mode active. Gammas will be assigned in this order:" \
                "\n- Rotational penalty" if self.high_rotation_penalty_reward else "" \
                "\n- Time penalty" if self.time_penalty_reward else "" \
                "\n- Progress to goal" if self.progress_to_goal_reward else "" \
                "\n- Discomfort" if self.discomfort_distance_penalty_reward else "" \
                "\n- Collision" if self.collision_penalty_reward else "" \
                "\n- Target reached" if self.target_reached_reward else "" \
            )
            self.multi_gamma = True
            gamma_list = [float(g) for g in gamma]
            assert len(gamma_list) == jnp.sum(self.binary_reward), "Number of gammas must be the same as active reward terms."
            idx = 0
            self.g_rot = gamma_list[idx] if self.high_rotation_penalty_reward else None
            idx += 1 if self.high_rotation_penalty_reward else 0
            self.g_time = gamma_list[idx] if self.time_penalty_reward else None
            idx += 1 if self.time_penalty_reward else 0
            self.g_prog = gamma_list[idx] if self.progress_to_goal_reward else None
            idx += 1 if self.progress_to_goal_reward else 0
            self.g_disc = gamma_list[idx] if self.discomfort_distance_penalty_reward else None
            idx += 1 if self.discomfort_distance_penalty_reward else 0
            self.g_coll = gamma_list[idx] if self.collision_penalty_reward else None
            idx += 1 if self.collision_penalty_reward else 0
            self.g_goal = gamma_list[idx] if self.target_reached_reward else None
            idx += 1 if self.target_reached_reward else 0
            self.unique_gammas = jnp.array(list(set(gamma_list)))
        else:
            self.multi_gamma = False
            self.unique_gammas = jnp.array([gamma])
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
    ) -> tuple[float, dict, dict]:
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
        - reward_terms: dictionary of all the reward terms with different discounts
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
        ### COMPUTE REWARD ###
        # Reward for reaching the goal
        if self.target_reached_reward:
            goal_reward = lax.cond(
                ~(collision) & (reached_goal), 
                lambda: self.goal_reward, 
                lambda: 0., 
            )
        else:
            goal_reward = 0.
        # Penalty for collision
        if self.collision_penalty_reward:
            collision_reward = lax.cond(
                collision, 
                lambda: self.collision_penalty, 
                lambda: 0., 
            ) 
        else:
            collision_reward = 0.
        # Penalty for getting too close to humans
        if self.discomfort_distance_penalty_reward:
            discomfort = (~(collision)) & (min_distance < self.discomfort_distance)
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
                lambda: self.progress_to_goal_weight * progress_to_goal, 
                lambda: 0., 
            )
        else:
            progress_reward = 0.
        # Time penalty
        if self.time_penalty_reward:
            time_reward = lax.cond(
                ~(reached_goal), 
                lambda: - self.time_penalty, 
                lambda: 0., 
            )
        else:
            time_reward = 0.
        # High rotation penalty
        if self.high_rotation_penalty_reward:
            rotation_reward = lax.cond(
                jnp.abs(action[1]) > self.angular_speed_bound, 
                lambda: - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
                lambda: 0., 
            )
        else:
            rotation_reward = 0.
        reward = goal_reward + collision_reward + discomfort_reward + progress_reward + time_reward + rotation_reward
        if self.multi_gamma:
            reward_terms = {g: 0.0 for g in self.unique_gammas}
            if self.target_reached_reward:
                reward_terms[self.g_goal] += goal_reward
            if self.collision_penalty_reward:
                reward_terms[self.g_coll] += collision_reward
            if self.discomfort_distance_penalty_reward:
                reward_terms[self.g_disc] += discomfort_reward
            if self.progress_to_goal_reward:
                reward_terms[self.g_prog] += progress_reward
            if self.time_penalty_reward:
                reward_terms[self.g_time] += time_reward
            if self.high_rotation_penalty_reward:
                reward_terms[self.g_rot] += rotation_reward
        else:
            reward_terms = {self.gamma: reward}
        ### COMPUTE OUTCOME ###
        outcome = {
            "nothing": ~((collision) | (reached_goal) | (timeout)),
            "success": ~(collision) & (reached_goal),
            "failure": collision,
            "timeout": timeout & (~(collision)) & (~(reached_goal))
        }
        # # DEBUG
        # debug.print("\n")
        # debug.print("collision: {x}", x=collision)
        # debug.print("reached_goal: {x}", x=reached_goal)
        # debug.print("timeout: {x}", x=timeout)
        return reward, outcome, reward_terms