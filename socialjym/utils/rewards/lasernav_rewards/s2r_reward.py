from functools import partial
from typing import Union

import jax.numpy as jnp
from jax import jit, nn, vmap

from jhsfm.hsfm import get_linear_velocity
from socialjym.envs.base_env import HUMAN_POLICIES, ROBOT_KINEMATICS
from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.utils.terminations.robot_human_collision import (
    InstantRobotHumanCollision,
)
from socialjym.utils.terminations.robot_obstacle_collision import (
    InstantRobotObstacleCollision,
)
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout


class S2RReward(BaseReward):
    """Transition-accurate social-navigation reward for LaserNav.

    Collision and clearance are evaluated at the simulator integration rate,
    rather than reconstructing one straight segment between control frames.
    This is important for traffic scenarios whose humans can be respawned at a
    boundary during a control interval.
    """

    def __init__(
        self,
        robot_radius: float,
        gamma: Union[float, list, tuple, jnp.ndarray] = 0.9,
        v_max: float = 1.0,
        time_limit: float = 50.0,
        goal_reward: float = 5.0,
        human_collision_penalty: float = -5.0,
        obstacle_collision_penalty: float = -1.0,
        timeout_penalty: float = -1.0,
        progress_weight: float = 0.3,
        clearance_distance: float = 0.5,
        clearance_weight: float = 0.5,
        ttc_horizon: float = 2.5,
        ttc_decay: float = 1.0,
        ttc_safety_margin: float = 0.4,
        ttc_weight: float = 0.25,
        yielding_distance: float = 2.5,
        yielding_half_angle: float = jnp.pi / 3.0,
        yielding_weight: float = 0.5,
        idle_weight: float = 0.01,
        angular_speed_bound: float = 1.0,
        angular_speed_penalty_weight: float = 0.0075,
    ):
        super().__init__(gamma)
        if isinstance(gamma, (list, tuple, jnp.ndarray)):
            raise ValueError("S2RReward currently requires one scalar discount factor.")
        self.v_max = v_max
        self.time_limit = time_limit
        self.robot_radius = robot_radius
        self.goal_reward = goal_reward
        self.human_collision_penalty = human_collision_penalty
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.timeout_penalty = timeout_penalty
        self.progress_weight = progress_weight
        self.clearance_distance = clearance_distance
        self.clearance_weight = clearance_weight
        self.ttc_horizon = ttc_horizon
        self.ttc_decay = ttc_decay
        self.ttc_safety_margin = ttc_safety_margin
        self.ttc_weight = ttc_weight
        self.yielding_distance = yielding_distance
        self.yielding_half_angle = yielding_half_angle
        self.yielding_weight = yielding_weight
        self.idle_weight = idle_weight
        self.angular_speed_bound = angular_speed_bound
        self.angular_speed_penalty_weight = angular_speed_penalty_weight
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.humans_policy = HUMAN_POLICIES.index("hsfm")
        self.instant_human_collision_termination = InstantRobotHumanCollision()
        self.instant_obstacle_collision_termination = InstantRobotObstacleCollision()
        self.goal_reached_termination = RobotReachedGoal()
        self.timeout = Timeout(time_limit)

    @partial(jit, static_argnames=("self",))
    def __call__(self, state, action, info, dt):
        """Fallback for callers without internal transition history."""
        next_state = state.at[-1, :2].set(
            state[-1, :2]
            + dt
            * action[0]
            * jnp.array([jnp.cos(state[-1, 4]), jnp.sin(state[-1, 4])])
        )
        return self.evaluate_transition(
            state, action, info, dt, state_history=next_state[None, ...]
        )

    @partial(jit, static_argnames=("self",))
    def evaluate_transition(self, state, action, info, dt, state_history=None):
        del action
        states = jnp.concatenate((state[None, ...], state_history), axis=0)
        robot_radius = (
            info["_robot_params"]["radius"]
            if "_robot_params" in info
            else self.robot_radius
        )
        v_max = (
            info["_robot_params"]["v_max"]
            if "_robot_params" in info
            else self.v_max
        )
        human_radii = info["humans_parameters"][:, 0]
        robot_positions = states[:, -1, :2]
        human_positions = states[:, :-1, :2]
        centre_distances = jnp.linalg.norm(
            human_positions - robot_positions[:, None, :], axis=-1
        )
        surface_clearances = centre_distances - (
            robot_radius + human_radii[None, :]
        )
        human_collision = jnp.any(surface_clearances < 0.0)

        obstacles = info["static_obstacles"][-1]
        obstacle_collisions = vmap(
            lambda position: self.instant_obstacle_collision_termination(
                position, robot_radius, obstacles
            )[0]
        )(robot_positions)
        obstacle_collision = jnp.any(obstacle_collisions)
        failure = human_collision | obstacle_collision

        final_state = states[-1]
        reached_goal, _ = self.goal_reached_termination(
            final_state[-1, :2], robot_radius, info["robot_goal"]
        )
        timeout, _ = self.timeout(info["time"] + dt)
        success = reached_goal & (~failure)
        timeout = timeout & (~failure) & (~success)
        outcome = {
            "nothing": ~(failure | success | timeout),
            "success": success,
            "collision_with_human": human_collision & (~success),
            "collision_with_obstacle": obstacle_collision & (~success),
            "timeout": timeout,
        }

        # Per-integration-step risk.  Positions are never connected by a swept
        # segment, so scenario respawns cannot create fictitious collisions.
        # Score the realised post-integration states.  Using ``states[:-1]``
        # would make the first (and for a one-substep transition, the only)
        # yielding term depend on the velocity from the previous command.
        sampled_states = states[1:]
        sampled_clearances = surface_clearances[1:]
        human_theta = sampled_states[:, :-1, 4]
        human_velocities = vmap(vmap(get_linear_velocity))(
            human_theta, sampled_states[:, :-1, 2:4]
        )
        robot_yaw = sampled_states[:, -1, 4]
        robot_speed = sampled_states[:, -1, 2]
        robot_velocities = robot_speed[:, None] * jnp.stack(
            (jnp.cos(robot_yaw), jnp.sin(robot_yaw)), axis=-1
        )
        relative_position = (
            sampled_states[:, :-1, :2] - sampled_states[:, -1:, :2]
        )
        relative_velocity = human_velocities - robot_velocities[:, None, :]
        relative_speed_sq = jnp.sum(relative_velocity**2, axis=-1) + 1e-6
        raw_ttc = -jnp.sum(relative_position * relative_velocity, axis=-1) / relative_speed_sq
        clipped_ttc = jnp.clip(raw_ttc, 0.0, self.ttc_horizon)
        closest_position = (
            relative_position + clipped_ttc[..., None] * relative_velocity
        )
        predicted_clearance = jnp.linalg.norm(closest_position, axis=-1) - (
            robot_radius + human_radii[None, :]
        )
        approaching = (raw_ttc > 0.0) & (raw_ttc < self.ttc_horizon)
        collision_course = nn.sigmoid(
            (self.ttc_safety_margin - predicted_clearance) / 0.1
        )
        ttc_risk = (
            approaching
            * collision_course
            * jnp.exp(-clipped_ttc / self.ttc_decay)
        )
        max_ttc_risk = jnp.max(ttc_risk, axis=-1)

        cos_yaw = jnp.cos(robot_yaw)[:, None]
        sin_yaw = jnp.sin(robot_yaw)[:, None]
        rel_x = cos_yaw * relative_position[..., 0] + sin_yaw * relative_position[..., 1]
        rel_y = -sin_yaw * relative_position[..., 0] + cos_yaw * relative_position[..., 1]
        rel_angle = jnp.abs(jnp.arctan2(rel_y, rel_x))
        front_mask = (
            (rel_x > 0.0)
            & (rel_angle <= self.yielding_half_angle)
            & (jnp.linalg.norm(relative_position, axis=-1) <= self.yielding_distance)
        )
        yielding_risk = jnp.max(ttc_risk * front_mask, axis=-1)

        integration_dt = dt / jnp.maximum(state_history.shape[0], 1)
        min_clearance_per_sample = jnp.min(sampled_clearances, axis=-1)
        clearance_violation = jnp.clip(
            (self.clearance_distance - min_clearance_per_sample)
            / self.clearance_distance,
            0.0,
            1.0,
        )
        clearance_penalty = -self.clearance_weight * integration_dt * jnp.sum(
            clearance_violation
        )
        ttc_penalty = -self.ttc_weight * integration_dt * jnp.sum(max_ttc_risk)
        normalized_forward_speed = jnp.clip(robot_speed / jnp.maximum(v_max, 1e-6), 0.0, 1.0)
        yielding_penalty = -self.yielding_weight * integration_dt * jnp.sum(
            yielding_risk * normalized_forward_speed
        )
        low_risk = jnp.max(max_ttc_risk) < 0.1
        idle_penalty = jnp.where(
            low_risk,
            -self.idle_weight * dt * (1.0 - jnp.mean(normalized_forward_speed)),
            0.0,
        )
        angular_speed = sampled_states[:, -1, 3]
        rotation_penalty = -self.angular_speed_penalty_weight * integration_dt * jnp.sum(
            jnp.where(
                jnp.abs(angular_speed) > self.angular_speed_bound,
                jnp.abs(angular_speed),
                0.0,
            )
        )
        progress = (
            jnp.linalg.norm(state[-1, :2] - info["robot_goal"])
            - jnp.linalg.norm(final_state[-1, :2] - info["robot_goal"])
        )
        progress_reward = self.progress_weight * progress
        terminal_reward = jnp.where(success, self.goal_reward, 0.0)
        terminal_reward += jnp.where(
            outcome["collision_with_human"], self.human_collision_penalty, 0.0
        )
        terminal_reward += jnp.where(
            outcome["collision_with_obstacle"], self.obstacle_collision_penalty, 0.0
        )
        terminal_reward += jnp.where(timeout, self.timeout_penalty, 0.0)
        reward = (
            terminal_reward
            + progress_reward
            + clearance_penalty
            + ttc_penalty
            + yielding_penalty
            + idle_penalty
            + rotation_penalty
        )
        return reward, outcome, {self.gamma: reward}
