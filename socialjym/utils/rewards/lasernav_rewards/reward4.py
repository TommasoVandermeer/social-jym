from jax import jit, lax, vmap
import jax
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
from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward

class Reward4(BaseReward):
    def __init__(
        self, 
        robot_radius: float,
        gamma:Union[float, list, tuple, jnp.ndarray] = 0.9, # Discount factor
        v_max: float=1.0,
        wheels_distance: float=0.7,
        dt: float=0.25,
        time_limit: float=50.,
        target_reached_reward: bool=True,
        collision_with_humans_penalty_reward: bool=True,
        collision_with_obstacles_penalty_reward: bool=True,
        discomfort_penalty_reward: bool=True,
        progress_to_goal_reward: bool=True,
        high_rotation_penalty_reward: bool=True,
        risk_reward: bool=True,
        escape_reward: bool=True,
        goal_reward: float=1., 
        collision_with_humans_penalty: float=-0.5, 
        collision_with_obstacles_penalty: float=-0.05,
        discomfort_distance: float=0.2, 
        progress_to_goal_weight: float=0.03,
        angular_speed_bound: float=1.,
        angular_speed_penalty_weight: float=0.0075,
        risk_avoidance_horizon: float=3.0,
        risk_avoidance_distance: float=0.5,
        risk_avoidance_top_k: int=3,
        risk_max_weight: float=0.1,
        risk_mean_weight: float=0.05,
        escape_alpha_threshold: float=0.5,
        escape_rotation_reward_weight: float=0.01,
        escape_rotation_stall_penalty_weight: float=0.01,
        escape_rotation_switch_penalty_weight: float=0.02,
        escape_rotation_deadband: float=0.05,
    ) -> None:
        super().__init__(gamma)
        # DIR-SAFE for action space bounding utils
        self.dir_safe = DIRSAFE(
            reward_function = DummyReward(v_max=v_max, time_limit=time_limit, kinematics='unicycle'),
            v_max = v_max,
            wheels_distance = wheels_distance,
            dt = dt,
        )
        # Check input parameters
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_with_humans_penalty < 0, "collision_with_humans_penalty must be negative"
        assert collision_with_obstacles_penalty < 0, "collision_with_obstacles_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"
        assert progress_to_goal_weight > 0, "progress_to_goal_weight must be positive"
        assert angular_speed_bound > 0, "angular_speed_bound must be positive"
        assert angular_speed_penalty_weight > 0, "angular_speed_penalty_weight must be positive"
        assert risk_avoidance_horizon > 0, "risk_avoidance_horizon must be positive"
        assert risk_avoidance_distance > 0, "risk_avoidance_distance must be positive"
        assert (risk_avoidance_top_k > 0) and (type(risk_avoidance_top_k) == int), "risk_avoidance_top_k must be a positive integer"
        assert (risk_max_weight >= 0) and (risk_mean_weight >= 0), "risk_max_weight and risk_mean_weight must be non-negative"
        assert (escape_alpha_threshold >= 0) and (escape_alpha_threshold <= 1), "escape_alpha_threshold must be in [0,1]"
        # Define reward type
        self.target_reached_reward = target_reached_reward
        self.collision_with_humans_penalty_reward = collision_with_humans_penalty_reward
        self.collision_with_obstacles_penalty_reward = collision_with_obstacles_penalty_reward
        self.discomfort_distance_penalty_reward = discomfort_penalty_reward
        self.progress_to_goal_reward = progress_to_goal_reward
        self.high_rotation_penalty_reward = high_rotation_penalty_reward
        self.risk_reward = risk_reward
        self.escape_reward = escape_reward
        self.binary_reward = jnp.array(
            [
                escape_reward,
                risk_reward,
                high_rotation_penalty_reward,
                progress_to_goal_reward,
                discomfort_penalty_reward,
                collision_with_humans_penalty_reward,
                collision_with_obstacles_penalty_reward,
                target_reached_reward,
            ], 
            dtype=int
        )
        if isinstance(gamma, (list, tuple, jnp.ndarray)):
            print(
                "REWARD - Multi-discount mode active. Gammas will be assigned in this order:",
                "\n- Escape" if self.escape_reward else "",
                "\n- Risk" if self.risk_reward else "",
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
            self.g_escape = gamma_list[idx] if self.escape_reward else None
            idx += 1 if self.escape_reward else 0
            self.g_risk = gamma_list[idx] if self.risk_reward else None
            idx += 1 if self.risk_reward else 0
            self.g_rot = gamma_list[idx] if self.high_rotation_penalty_reward else None
            idx += 1 if self.high_rotation_penalty_reward else 0
            self.g_prog = gamma_list[idx] if self.progress_to_goal_reward else None
            idx += 1 if self.progress_to_goal_reward else 0
            self.g_disc = gamma_list[idx] if self.discomfort_distance_penalty_reward else None
            idx += 1 if self.discomfort_distance_penalty_reward else 0
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
        self.type = f"lasernav_reward4_{self.decimal_reward}"
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
        self.risk_avoidance_horizon = risk_avoidance_horizon
        self.risk_avoidance_distance = risk_avoidance_distance
        self.risk_avoidance_top_k = risk_avoidance_top_k
        self.w_max_risk = risk_max_weight
        self.w_mean_risk = risk_mean_weight
        self.escape_alpha_threshold = escape_alpha_threshold
        self.escape_angular_speed = 2 * v_max / wheels_distance
        self.escape_rotation_reward_weight = escape_rotation_reward_weight
        self.escape_rotation_stall_penalty_weight = escape_rotation_stall_penalty_weight
        self.escape_rotation_switch_penalty_weight = escape_rotation_switch_penalty_weight
        self.escape_rotation_deadband = escape_rotation_deadband
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
        new_states:jnp.ndarray,
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
        - new_states: next states of the environment evaluated at the humans_dt integration step
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation

        output:
        - reward: 0.0 (dummy reward)
        - outcome: dictionary indicating if the episode is finished or not and why.
        - reward_terms: dictionary of all the reward terms with different discounts
        """
        robot_pos = state[-1,:2]
        robot_orientation = state[-1,4]
        robot_goal = info["robot_goal"]
        humans_radiuses = info["humans_parameters"][:,0]
        robot_radius = self.robot_radius
        time = info["time"]
        next_robot_pos = new_states[-1,-1,:2]
        # Collision detection with humans in all the next states
        collision_with_humans, collision_with_human_infos = vmap(self.instant_human_collision_termination, in_axes=(0, None, 0, None))(
            new_states[:,-1,:2], 
            robot_radius,
            new_states[:,:-1,:2],
            humans_radiuses
        )
        collision_with_human = jnp.any(collision_with_humans)
        min_distance = jnp.min(collision_with_human_infos['min_distance'])
        discomfort = jnp.all(jnp.array([jnp.logical_not(collision_with_human), min_distance < self.discomfort_distance]))
        # Collision detection with obstacles in all the next states
        collision_with_obstacles, collision_with_obstacle_infos = vmap(self.instant_obstacle_collision_termination, in_axes=(0, None, None))(
            new_states[:,-1,:2],
            robot_radius,
            info['static_obstacles'][-1],
        )
        collision_with_obstacle = jnp.any(collision_with_obstacles)
        min_clearance = jnp.min(collision_with_obstacle_infos['min_distance'])
        # Escape detection (based on DIR-SAFE action space bounding)
        obstacles = info["static_obstacles"][-1] # Obstacles are repeated for each agent, we take the last one, corresponding to the robot agent
        obstacle_segments = obstacles.reshape((obstacles.shape[0] * obstacles.shape[1], 2, 2)) # Concatenate all segments regardless of the obstacle they belong to
        alpha = self.dir_safe.bound_action_space(
            obstacle_segments,
            robot_pos,
            robot_orientation,
            robot_radius,
        )[0]
        escape = alpha < self.escape_alpha_threshold
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
                (jnp.abs(action[1]) > self.angular_speed_bound) & (~(escape)), 
                lambda: - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
                lambda: 0., 
            )
        else:
            rotation_reward = 0.
        # Risk reward
        if self.risk_reward:
            current_max_risk, current_mean_risk = self._risk(state, info, dt)
            next_max_risk, next_mean_risk = self._risk(new_states[-1], info, dt)
            risk_reward = lax.cond(
                ~(failure),
                lambda: self.w_max_risk * (current_max_risk - next_max_risk) + self.w_mean_risk * (current_mean_risk - next_mean_risk),
                lambda: 0.0,
            )
        else:
            risk_reward = 0.
        # Escape reward
        if self.escape_reward:
            angular_speed = jnp.abs(action[1])
            turn_fraction = jnp.clip(
                angular_speed / self.escape_angular_speed,
                0.0,
                1.0,
            )
            escape_turn_reward = lax.cond(
                escape,
                lambda: self.escape_rotation_reward_weight * dt * turn_fraction,
                lambda: 0.0,
            )
            escape_stall_penalty = lax.cond(
                escape,
                lambda: -self.escape_rotation_stall_penalty_weight * dt * (1.0 - turn_fraction),
                lambda: 0.0,
            )
            previous_angular_speed = state[-1, 3]
            direction_switch = (
                escape
                & (jnp.abs(previous_angular_speed) > self.escape_rotation_deadband)
                & (angular_speed > self.escape_rotation_deadband)
                & (previous_angular_speed * action[1] < 0.0)
            )
            escape_switch_penalty = lax.cond(
                direction_switch,
                lambda: -self.escape_rotation_switch_penalty_weight * dt,
                lambda: 0.0,
            )
            escape_reward = escape_turn_reward + escape_stall_penalty + escape_switch_penalty
        else:
            escape_reward = 0.
        ### TOTAL REWARD ###
        reward = goal_reward + collision_human_reward + collision_obstacle_reward + discomfort_reward + progress_reward + rotation_reward + risk_reward + escape_reward
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
            if self.risk_reward:
                reward_terms[self.g_risk] += risk_reward
            if self.escape_reward:
                reward_terms[self.g_escape] += escape_reward
        else:
            reward_terms = {self.gamma: reward}
        return reward, outcome, reward_terms

    @partial(jit, static_argnames=("self"))
    def _risk(self, state, info, dt):
        """
        Given a state and a dictionary containing additional information about the environment,
        this function computes the risk of the current state with respect to nearby humans.

        args:
        - state: current state of the environment
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation

        output:
        - max_risk: scalar between 0 and 1 indicating the risk of the current state with respect to the riskiest interacting human.
        - mean_risk: scalar between 0 and 1 indicating the average risk of the current state with respect to the k riskiest interacting humans.
        """
        # TODO: Query HSFM to propagate humans' states over the risk_avoidance_horizon in the future and compute risk based on that.
        # Robot
        robot_pos = state[-1,:2]
        robot_yaw = state[-1,4]
        robot_velocity_unicycle = state[-1,2:4]
        robot_radius = self.robot_radius
        next_robot_pos = lax.cond(
            jnp.abs(robot_velocity_unicycle[1]) > 1e-3,
            lambda x: x.at[:].set(jnp.array([
                x[0] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.sin(robot_yaw + robot_velocity_unicycle[1] * dt) - jnp.sin(robot_yaw)),
                x[1] + (robot_velocity_unicycle[0]/robot_velocity_unicycle[1]) * (jnp.cos(robot_yaw) - jnp.cos(robot_yaw + robot_velocity_unicycle[1] * dt))
            ])),
            lambda x: x.at[:].set(jnp.array([
                x[0] + robot_velocity_unicycle[0] * dt * jnp.cos(robot_yaw),
                x[1] + robot_velocity_unicycle[0] * dt * jnp.sin(robot_yaw)
            ])),
            robot_pos)
        robot_velocity = (next_robot_pos - robot_pos) / dt
        # Humans
        humans_pos = state[:-1,:2]
        humans_radiuses = info["humans_parameters"][:,0]
        humans_orientations = state[:-1,4]
        humans_velocities = vmap(get_linear_velocity)(humans_orientations, state[:-1,2:4])
        # Relative
        humans_rel_pos = humans_pos - robot_pos
        humans_rel_vel = humans_velocities - robot_velocity
        # Risk
        humans_rel_vel_squared = jnp.sum(humans_rel_vel**2, axis=1)
        closest_times = -jnp.sum(humans_rel_pos * humans_rel_vel, axis=-1) / jnp.maximum(humans_rel_vel_squared, 1e-8)
        closest_times = jnp.clip(closest_times, 0., self.risk_avoidance_horizon)
        closest_offsets = humans_rel_pos + closest_times[:,None] * humans_rel_vel
        clearance_distances = jnp.linalg.norm(closest_offsets, axis=-1) - (humans_radiuses + robot_radius)
        risks = jnp.clip((self.risk_avoidance_distance - clearance_distances) / self.risk_avoidance_distance, 0., 1.)
        # Top k risks
        k_eff = min(self.risk_avoidance_top_k, risks.shape[0])
        top_k_risks = jax.lax.top_k(risks, k_eff)[0]
        return jnp.max(top_k_risks), jnp.mean(top_k_risks)