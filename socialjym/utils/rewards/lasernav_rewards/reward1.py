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
from socialjym.utils.terminations.base_termination import point_to_line_distance
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
        timeout_penalty_reward: bool=False,
        goal_reward: float=1., 
        collision_with_humans_penalty: float=-0.25, 
        collision_with_obstacles_penalty: float=-0.05,
        discomfort_distance: float=0.2, 
        progress_to_goal_weight: float=0.03,
        angular_speed_bound: float=1.,
        angular_speed_penalty_weight: float=0.0075,
        timeout_penalty: float=-0.25,
        use_leg_collisions: bool=True,
        effective_foot_radius: float=0.20,
        anticipatory_avoidance_reward: bool=False,
        avoidance_distance: float=1.0,
        avoidance_horizon: float=1.5,
        avoidance_penalty_weight: float=0.15,
        avoidance_improvement_weight: float=0.20,
        head_on_risk_multiplier: float=1.0,
        local_minimum_escape_reward: bool=False,
        escape_clearance_distance: float=2.0,
        escape_clearance_weight: float=0.05,
        stalled_rotation_penalty_weight: float=0.01,
        escape_forward_bonus_weight: float=0.02,
        turn_in_place_linear_threshold: float=0.05,
        turn_in_place_angular_threshold: float=0.20,
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
        assert timeout_penalty < 0, "timeout_penalty must be negative"
        assert effective_foot_radius > 0, "effective_foot_radius must be positive"
        assert avoidance_distance > 0, "avoidance_distance must be positive"
        assert avoidance_horizon > 0, "avoidance_horizon must be positive"
        assert avoidance_penalty_weight >= 0, "avoidance_penalty_weight must be non-negative"
        assert avoidance_improvement_weight >= 0, "avoidance_improvement_weight must be non-negative"
        assert head_on_risk_multiplier >= 0, "head_on_risk_multiplier must be non-negative"
        assert escape_clearance_distance > 0, "escape_clearance_distance must be positive"
        assert escape_clearance_weight >= 0, "escape_clearance_weight must be non-negative"
        assert stalled_rotation_penalty_weight >= 0, "stalled_rotation_penalty_weight must be non-negative"
        assert escape_forward_bonus_weight >= 0, "escape_forward_bonus_weight must be non-negative"
        # Define reward type
        self.target_reached_reward = target_reached_reward
        self.collision_with_humans_penalty_reward = collision_with_humans_penalty_reward
        self.collision_with_obstacles_penalty_reward = collision_with_obstacles_penalty_reward
        self.discomfort_distance_penalty_reward = discomfort_penalty_reward
        self.progress_to_goal_reward = progress_to_goal_reward
        self.high_rotation_penalty_reward = high_rotation_penalty_reward
        self.timeout_penalty_reward = timeout_penalty_reward
        self.binary_reward = jnp.array(
            [
                high_rotation_penalty_reward,
                progress_to_goal_reward,
                discomfort_penalty_reward,
                collision_with_humans_penalty_reward,
                collision_with_obstacles_penalty_reward,
                timeout_penalty_reward,
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
                "\n- Timeout" if self.timeout_penalty_reward else "",
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
            self.g_disc = gamma_list[idx] if self.discomfort_distance_penalty_reward else None
            idx += 1 if self.discomfort_distance_penalty_reward else 0
            self.g_coll_hum = gamma_list[idx] if self.collision_with_humans_penalty_reward else None
            idx += 1 if self.collision_with_humans_penalty_reward else 0
            self.g_coll_obs = gamma_list[idx] if self.collision_with_obstacles_penalty_reward else None
            idx += 1 if self.collision_with_obstacles_penalty_reward else 0
            self.g_timeout = gamma_list[idx] if self.timeout_penalty_reward else None
            idx += 1 if self.timeout_penalty_reward else 0
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
        self.timeout_penalty = timeout_penalty
        self.use_leg_collisions = use_leg_collisions
        self.effective_foot_radius = effective_foot_radius
        self.anticipatory_avoidance_reward = anticipatory_avoidance_reward
        self.avoidance_distance = avoidance_distance
        self.avoidance_horizon = avoidance_horizon
        self.avoidance_penalty_weight = avoidance_penalty_weight
        self.avoidance_improvement_weight = avoidance_improvement_weight
        self.head_on_risk_multiplier = head_on_risk_multiplier
        self.local_minimum_escape_reward = local_minimum_escape_reward
        self.escape_clearance_distance = escape_clearance_distance
        self.escape_clearance_weight = escape_clearance_weight
        self.stalled_rotation_penalty_weight = stalled_rotation_penalty_weight
        self.escape_forward_bonus_weight = escape_forward_bonus_weight
        self.turn_in_place_linear_threshold = turn_in_place_linear_threshold
        self.turn_in_place_angular_threshold = turn_in_place_angular_threshold
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
        robot_radius = self.robot_radius
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
        return self._compute_reward(
            robot_pos,
            next_robot_pos,
            robot_goal,
            min_distance,
            collision_with_human,
            collision_with_obstacle,
            reached_goal,
            timeout,
            action,
            dt,
        )

    @partial(jit, static_argnames=("self"))
    def _compute_reward(
        self,
        robot_pos,
        next_robot_pos,
        robot_goal,
        min_distance,
        collision_with_human,
        collision_with_obstacle,
        reached_goal,
        timeout,
        action,
        dt,
        avoidance_reward=0.0,
        escape_reward=0.0,
    ):
        ### COMPUTE OUTCOME ###
        failure = collision_with_human | collision_with_obstacle
        success = (~failure) & reached_goal
        timeout = timeout & (~failure) & (~reached_goal)
        outcome = {
            "nothing": ~(failure | success | timeout),
            "success": success,
            "collision_with_human": collision_with_human,
            "collision_with_obstacle": collision_with_obstacle,
            "timeout": timeout,
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
        if self.anticipatory_avoidance_reward:
            discomfort_reward += avoidance_reward
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
        if self.local_minimum_escape_reward:
            progress_reward += escape_reward
        # High rotation penalty
        if self.high_rotation_penalty_reward:
            rotation_reward = lax.cond(
                jnp.abs(action[1]) > self.angular_speed_bound, 
                lambda: - self.angular_speed_penalty_weight * jnp.abs(action[1]), 
                lambda: 0., 
            )
        else:
            rotation_reward = 0.
        if self.timeout_penalty_reward:
            timeout_reward = lax.cond(timeout, lambda: self.timeout_penalty, lambda: 0.)
        else:
            timeout_reward = 0.
        reward = goal_reward + collision_human_reward + collision_obstacle_reward + discomfort_reward + progress_reward + rotation_reward + timeout_reward
        if self.multi_gamma:
            reward_terms = {g: 0.0 for g in self.unique_gammas}
            if self.target_reached_reward:
                reward_terms[self.g_goal] += goal_reward
            if self.collision_with_humans_penalty_reward:
                reward_terms[self.g_coll_hum] += collision_human_reward
            if self.collision_with_obstacles_penalty_reward:
                reward_terms[self.g_coll_obs] += collision_obstacle_reward
            if self.timeout_penalty_reward:
                reward_terms[self.g_timeout] += timeout_reward
            if self.discomfort_distance_penalty_reward:
                reward_terms[self.g_disc] += discomfort_reward
            if self.progress_to_goal_reward:
                reward_terms[self.g_prog] += progress_reward
            if self.high_rotation_penalty_reward:
                reward_terms[self.g_rot] += rotation_reward
        else:
            reward_terms = {self.gamma: reward}
        return reward, outcome, reward_terms

    def _collision_segments(self, robot_starts, robot_ends, entity_starts, entity_ends, radii):
        collisions, collision_info = vmap(
            self.interval_human_collision_termination,
            in_axes=(0, 0, None, 0, 0, None),
        )(
            robot_starts,
            robot_ends,
            self.robot_radius,
            entity_starts,
            entity_ends,
            radii,
        )
        return jnp.any(collisions), jnp.min(collision_info["min_distance"])

    def _human_risk(self, state, human_radii, valid_mask=None):
        """Bounded closest-approach risk, with extra weight for head-on motion."""
        robot_pos = state[-1, :2]
        robot_heading = state[-1, 4]
        robot_velocity = state[-1, 2] * jnp.array(
            [jnp.cos(robot_heading), jnp.sin(robot_heading)]
        )
        human_velocity = vmap(get_linear_velocity)(state[:-1, 4], state[:-1, 2:4])
        relative_position = state[:-1, :2] - robot_pos
        relative_velocity = human_velocity - robot_velocity
        velocity_squared = jnp.sum(relative_velocity ** 2, axis=-1)
        closest_time = jnp.clip(
            -jnp.sum(relative_position * relative_velocity, axis=-1)
            / jnp.maximum(velocity_squared, 1e-8),
            0.0,
            self.avoidance_horizon,
        )
        closest_offset = relative_position + closest_time[:, None] * relative_velocity
        clearance = jnp.linalg.norm(closest_offset, axis=-1) - (
            self.robot_radius + human_radii
        )
        proximity = jnp.clip(
            (self.avoidance_distance - clearance) / self.avoidance_distance,
            0.0,
            1.0,
        )
        robot_speed = jnp.linalg.norm(robot_velocity)
        human_speed = jnp.linalg.norm(human_velocity, axis=-1)
        opposing = jnp.clip(
            -jnp.sum(human_velocity * robot_velocity, axis=-1)
            / jnp.maximum(human_speed * robot_speed, 1e-8),
            0.0,
            1.0,
        )
        risks = proximity * (1.0 + self.head_on_risk_multiplier * opposing)
        if valid_mask is not None:
            risks = jnp.where(valid_mask, risks, 0.0)
        return jnp.max(risks)

    def _forward_clearance(self, state, human_positions, human_radii, obstacles):
        """Approximate body-width clearance in front of the robot."""
        robot_pos = state[-1, :2]
        heading = state[-1, 4]
        forward = jnp.array([jnp.cos(heading), jnp.sin(heading)])
        lateral = jnp.array([-forward[1], forward[0]])

        obstacle_segments = obstacles.reshape((-1, 2, 2))
        samples = jnp.linspace(0.0, 1.0, 9)
        obstacle_points = (
            obstacle_segments[:, None, 0]
            + samples[None, :, None]
            * (obstacle_segments[:, None, 1] - obstacle_segments[:, None, 0])
        ).reshape((-1, 2))
        valid_obstacles = jnp.all(jnp.isfinite(obstacle_points), axis=-1)
        obstacle_offset = obstacle_points - robot_pos
        obstacle_forward = obstacle_offset @ forward
        obstacle_lateral = jnp.abs(obstacle_offset @ lateral)
        obstacle_clearance = jnp.where(
            valid_obstacles
            & (obstacle_forward > 0.0)
            & (obstacle_lateral < self.robot_radius + 0.15),
            obstacle_forward - self.robot_radius,
            jnp.inf,
        )

        human_offset = human_positions - robot_pos
        human_forward = human_offset @ forward
        human_lateral = jnp.abs(human_offset @ lateral)
        human_clearance = jnp.where(
            (human_forward > 0.0)
            & (human_lateral < self.robot_radius + human_radii),
            human_forward - self.robot_radius - human_radii,
            jnp.inf,
        )
        clearance = jnp.minimum(jnp.min(obstacle_clearance), jnp.min(human_clearance))
        return jnp.clip(clearance, 0.0, self.escape_clearance_distance)

    @partial(jit, static_argnames=("self", "leg_dynamics"))
    def transition(
        self,
        old_state,
        new_state,
        intermediate_states,
        action,
        info,
        dt,
        intermediate_leg_states=None,
        intermediate_human_end_positions=None,
        intermediate_leg_end_states=None,
        intermediate_human_respawns=None,
        leg_dynamics=False,
    ):
        """Reward the trajectory that was actually executed by LaserNav."""
        trajectory = jnp.concatenate((old_state[None, ...], intermediate_states), axis=0)
        starts = trajectory[:-1]
        ends = trajectory[1:]

        human_radii = info["humans_parameters"][:, 0]
        human_starts = trajectory[:-1, :-1, :2]
        human_ends = (
            intermediate_states[:, :-1, :2]
            if intermediate_human_end_positions is None
            else intermediate_human_end_positions
        )
        collision_with_human, min_human_distance = self._collision_segments(
            starts[:, -1, :2], ends[:, -1, :2], human_starts, human_ends, human_radii
        )
        if self.use_leg_collisions and leg_dynamics and intermediate_leg_states is not None:
            old_feet = jnp.stack(
                (info["humans_leg_state"][:, 0:2], info["humans_leg_state"][:, 3:5]),
                axis=1,
            ).reshape((-1, 2))
            feet_history = jnp.stack(
                (intermediate_leg_states[:, :, 0:2], intermediate_leg_states[:, :, 3:5]),
                axis=2,
            ).reshape((intermediate_leg_states.shape[0], -1, 2))
            foot_starts = jnp.concatenate((old_feet[None, ...], feet_history[:-1]), axis=0)
            if intermediate_leg_end_states is None:
                foot_ends = feet_history
            else:
                foot_ends = jnp.stack(
                    (intermediate_leg_end_states[:, :, 0:2], intermediate_leg_end_states[:, :, 3:5]),
                    axis=2,
                ).reshape((intermediate_leg_end_states.shape[0], -1, 2))
            collision_with_human, min_human_distance = self._collision_segments(
                starts[:, -1, :2],
                ends[:, -1, :2],
                foot_starts,
                foot_ends,
                jnp.full((foot_starts.shape[1],), self.effective_foot_radius),
            )

        obstacle_segments = info["static_obstacles"][-1].reshape((-1, 2, 2))
        valid_obstacles = jnp.all(jnp.isfinite(obstacle_segments), axis=(1, 2))

        def segment_distance(robot_start, robot_end, obstacle):
            obstacle_start, obstacle_end = obstacle[0], obstacle[1]
            robot_delta = robot_end - robot_start
            obstacle_delta = obstacle_end - obstacle_start
            offset = obstacle_start - robot_start
            denominator = robot_delta[0] * obstacle_delta[1] - robot_delta[1] * obstacle_delta[0]
            safe_denominator = jnp.where(jnp.abs(denominator) > 1e-9, denominator, 1.0)
            t = (offset[0] * obstacle_delta[1] - offset[1] * obstacle_delta[0]) / safe_denominator
            u = (offset[0] * robot_delta[1] - offset[1] * robot_delta[0]) / safe_denominator
            intersects = (
                (jnp.abs(denominator) > 1e-9)
                & (t >= 0.0) & (t <= 1.0)
                & (u >= 0.0) & (u <= 1.0)
            )
            distances = jnp.array([
                point_to_line_distance(robot_start, obstacle_start, obstacle_end),
                point_to_line_distance(robot_end, obstacle_start, obstacle_end),
                point_to_line_distance(obstacle_start, robot_start, robot_end),
                point_to_line_distance(obstacle_end, robot_start, robot_end),
            ])
            return jnp.where(intersects, 0.0, jnp.min(distances))

        distances_to_obstacles = vmap(
            lambda robot_start, robot_end: vmap(segment_distance, in_axes=(None, None, 0))(
                robot_start, robot_end, obstacle_segments
            )
        )(starts[:, -1, :2], ends[:, -1, :2])
        distances_to_obstacles = jnp.where(valid_obstacles[None, :], distances_to_obstacles, jnp.inf)
        collision_with_obstacle = jnp.any(distances_to_obstacles < self.robot_radius)

        goal_distances = jnp.linalg.norm(ends[:, -1, :2] - info["robot_goal"], axis=-1)
        reached_goal = jnp.any(goal_distances < self.robot_radius)
        timeout, _ = self.timeout(info["time"] + dt)

        avoidance_reward = 0.0
        non_respawned = (
            jnp.ones((old_state.shape[0] - 1,), dtype=jnp.bool_)
            if intermediate_human_respawns is None
            else ~jnp.any(intermediate_human_respawns, axis=0)
        )
        if self.anticipatory_avoidance_reward:
            old_risk = self._human_risk(old_state, human_radii, non_respawned)
            new_risk = self._human_risk(new_state, human_radii, non_respawned)
            avoidance_reward = (
                self.avoidance_improvement_weight * (old_risk - new_risk)
                - self.avoidance_penalty_weight * dt * new_risk
            )

        escape_reward = 0.0
        if self.local_minimum_escape_reward:
            old_human_positions = old_state[:-1, :2]
            new_human_positions = new_state[:-1, :2]
            clearance_radii = human_radii
            if self.use_leg_collisions and leg_dynamics and intermediate_leg_states is not None:
                old_human_positions = old_feet
                new_human_positions = feet_history[-1]
                clearance_radii = jnp.full(
                    (old_human_positions.shape[0],), self.effective_foot_radius
                )
                clearance_mask = jnp.repeat(non_respawned, 2)
            else:
                clearance_mask = non_respawned
            old_human_positions = jnp.where(
                clearance_mask[:, None], old_human_positions, jnp.inf
            )
            new_human_positions = jnp.where(
                clearance_mask[:, None], new_human_positions, jnp.inf
            )
            old_clearance = self._forward_clearance(
                old_state, old_human_positions, clearance_radii, info["static_obstacles"][-1]
            )
            new_clearance = self._forward_clearance(
                new_state, new_human_positions, clearance_radii, info["static_obstacles"][-1]
            )
            translation = jnp.linalg.norm(new_state[-1, :2] - old_state[-1, :2])
            heading_delta = jnp.abs(jnp.arctan2(
                jnp.sin(new_state[-1, 4] - old_state[-1, 4]),
                jnp.cos(new_state[-1, 4] - old_state[-1, 4]),
            ))
            turning_in_place = (
                (translation < self.turn_in_place_linear_threshold * dt)
                & (heading_delta > self.turn_in_place_angular_threshold * dt)
            )
            previous_actions = info["action_history"]
            repeatedly_turning = jnp.all(
                (jnp.abs(previous_actions[:, 0]) < self.turn_in_place_linear_threshold)
                & (jnp.abs(previous_actions[:, 1]) > self.turn_in_place_angular_threshold)
            )
            clearance_gain = new_clearance - old_clearance
            escape_reward = jnp.where(
                turning_in_place,
                self.escape_clearance_weight * jnp.clip(clearance_gain, -0.5, 0.5),
                0.0,
            )
            escape_reward -= jnp.where(
                turning_in_place & repeatedly_turning & (clearance_gain <= 1e-3),
                self.stalled_rotation_penalty_weight * dt,
                0.0,
            )
            resumed_forward = repeatedly_turning & (
                translation > self.turn_in_place_linear_threshold * dt
            )
            escape_reward += jnp.where(
                resumed_forward,
                self.escape_forward_bonus_weight
                * jnp.clip(translation / jnp.maximum(self.v_max * dt, 1e-8), 0.0, 1.0),
                0.0,
            )

        return self._compute_reward(
            old_state[-1, :2],
            new_state[-1, :2],
            info["robot_goal"],
            min_human_distance,
            collision_with_human,
            collision_with_obstacle,
            reached_goal,
            timeout,
            action,
            dt,
            avoidance_reward,
            escape_reward,
        )
