from jax import jit, lax, vmap, debug
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

class Reward4(BaseReward):
    def __init__(
        self, 
        robot_radius: float,
        gamma:Union[float, list, tuple, jnp.ndarray] = 0.9,
        v_max: float=1.0,
        time_limit: float=50.,
        
        # Reward Flags
        target_reached_reward: bool=True,
        collision_with_humans_penalty_reward: bool=True,
        collision_with_obstacles_penalty_reward: bool=True,
        timeout_penalty_reward: bool=True,
        predictive_ttc_penalty_reward: bool=True,
        social_intrusion_penalty_reward: bool=True,
        progress_to_goal_reward: bool=True,
        angular_jerk_penalty_reward: bool=True,
        linear_jerk_penalty_reward: bool=True,
        
        # Reward Weights & Parameters (Ideal Set)
        goal_weight: float = 1.0,
        collision_weight: float = -2.0,
        timeout_weight: float = -0.005,
        progress_weight: float = 0.05,
        ttc_weight: float = -0.1,
        social_intrusion_weight: float = -0.1,
        angular_jerk_weight: float = -0.05,
        linear_jerk_weight: float = -0.02,
        
        # Thresholds
        ttc_horizon: float = 3.0,
        ttc_threshold: float = 1.0,
        social_comfort_dist: float = 0.7,
        goal_threshold: float = 0.3
    ) -> None:
        super().__init__(gamma)
        
        print("WARNING: this reward is still not stable for training. Use at your own risk.")

        # Assign flags
        self.target_reached_reward = target_reached_reward
        self.collision_with_humans_penalty_reward = collision_with_humans_penalty_reward
        self.collision_with_obstacles_penalty_reward = collision_with_obstacles_penalty_reward
        self.timeout_penalty_reward = timeout_penalty_reward
        self.predictive_ttc_penalty_reward = predictive_ttc_penalty_reward
        self.social_intrusion_penalty_reward = social_intrusion_penalty_reward
        self.progress_to_goal_reward = progress_to_goal_reward
        self.angular_jerk_penalty_reward = angular_jerk_penalty_reward
        self.linear_jerk_penalty_reward = linear_jerk_penalty_reward
        
        self.binary_reward = jnp.array(
            [
                linear_jerk_penalty_reward,
                angular_jerk_penalty_reward,
                progress_to_goal_reward,
                social_intrusion_penalty_reward,
                predictive_ttc_penalty_reward,
                timeout_penalty_reward,
                collision_with_humans_penalty_reward,
                collision_with_obstacles_penalty_reward,
                target_reached_reward
            ], 
            dtype=int
        )
        
        if isinstance(gamma, (list, tuple, jnp.ndarray)):
            self.multi_gamma = True
            gamma_list = [float(g) for g in gamma]
            assert len(gamma_list) == jnp.sum(self.binary_reward), "Number of gammas must be the same as active reward terms."
            idx = 0
            self.g_ljerk = gamma_list[idx] if self.linear_jerk_penalty_reward else None
            idx += 1 if self.linear_jerk_penalty_reward else 0
            self.g_ajerk = gamma_list[idx] if self.angular_jerk_penalty_reward else None
            idx += 1 if self.angular_jerk_penalty_reward else 0
            self.g_prog = gamma_list[idx] if self.progress_to_goal_reward else None
            idx += 1 if self.progress_to_goal_reward else 0
            self.g_soc = gamma_list[idx] if self.social_intrusion_penalty_reward else None
            idx += 1 if self.social_intrusion_penalty_reward else 0
            self.g_ttc = gamma_list[idx] if self.predictive_ttc_penalty_reward else None
            idx += 1 if self.predictive_ttc_penalty_reward else 0
            self.g_timeout = gamma_list[idx] if self.timeout_penalty_reward else None
            idx += 1 if self.timeout_penalty_reward else 0
            self.g_coll_hum = gamma_list[idx] if self.collision_with_humans_penalty_reward else None
            idx += 1 if self.collision_with_humans_penalty_reward else 0
            self.g_coll_obs = gamma_list[idx] if self.collision_with_obstacles_penalty_reward else None
            idx += 1 if self.collision_with_obstacles_penalty_reward else 0
            self.g_goal = gamma_list[idx] if self.target_reached_reward else None
            
            self.unique_gammas = tuple(set(gamma_list))
        else:
            self.multi_gamma = False
            self.unique_gammas = (float(gamma),)
            
        self.decimal_reward = binary_to_decimal(self.binary_reward)
        self.type = f"lasernav_reward4_{self.decimal_reward}"
        
        # Initialize parameters
        self.v_max = v_max
        self.time_limit = time_limit
        self.goal_weight = goal_weight
        self.collision_weight = collision_weight
        self.timeout_weight = timeout_weight
        self.progress_weight = progress_weight
        self.ttc_weight = ttc_weight
        self.social_intrusion_weight = social_intrusion_weight
        self.angular_jerk_weight = angular_jerk_weight
        self.linear_jerk_weight = linear_jerk_weight
        
        self.ttc_horizon = ttc_horizon
        self.ttc_threshold = ttc_threshold
        self.social_comfort_dist = social_comfort_dist
        self.goal_threshold = goal_threshold
        
        self.robot_radius = robot_radius
        self.humans_policy = HUMAN_POLICIES.index('hsfm')
        self.kinematics = ROBOT_KINEMATICS.index('unicycle')
        
        # Terminations
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
        
        robot_pos = state[-1,:2]
        robot_yaw = state[-1,4]
        
        humans_pos = state[:-1,:2]
        robot_goal = info["robot_goal"]
        humans_radiuses = info["humans_parameters"][:,0]
        time = info["time"]
        
        # Calculate next states
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
            
        # Reconstruct velocity vector
        robot_vel_vec = (next_robot_pos - robot_pos) / dt

        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            humans_orientations = state[:-1,4]
            humans_velocities = vmap(get_linear_velocity)(humans_orientations, state[:-1,2:4])
        else:
            humans_velocities = state[:-1,2:4]
            
        next_humans_pos = humans_pos + humans_velocities * dt

        # Terminations
        collision_with_human, _ = self.interval_human_collision_termination(
            robot_pos, next_robot_pos, self.robot_radius, humans_pos, next_humans_pos, humans_radiuses
        )
        collision_with_obstacle, _ = self.instant_obstacle_collision_termination(
            next_robot_pos, self.robot_radius, info['static_obstacles'][-1]
        )
        reached_goal, _ = self.goal_reached_termination(
            next_robot_pos, self.robot_radius, robot_goal
        )
        timeout, _ = self.timeout(time)
        
        failure = collision_with_human | collision_with_obstacle
        outcome = {
            "nothing": ~((failure) | (reached_goal) | (timeout)),
            "success": (~(failure)) & (reached_goal),
            "collision_with_human": collision_with_human,
            "collision_with_obstacle": collision_with_obstacle,
            "timeout": timeout & (~(failure)) & (~(reached_goal))
        }

        # --- 1. TASK REWARDS ---
        goal_reward = lax.cond(
            self.target_reached_reward & ~(failure) & reached_goal, 
            lambda: self.goal_weight, lambda: 0.
        )
        
        timeout_penalty = lax.cond(
            self.timeout_penalty_reward & ~(reached_goal),
            lambda: self.timeout_weight, lambda: 0.
        )

        progress_reward = 0.
        if self.progress_to_goal_reward:
            # Directional progress based on velocity vector
            vec_to_goal = robot_goal - robot_pos
            dist_to_goal = jnp.sqrt(jnp.sum(vec_to_goal**2) + 1e-8)
            
            progress_reward = lax.cond(
                ~(reached_goal) & (dist_to_goal > 1e-5),
                lambda: self.progress_weight * jnp.dot(robot_vel_vec, vec_to_goal / dist_to_goal) * dt,
                lambda: 0.
            )

        # --- 2. SOCIAL SAFETY REWARDS ---
        collision_human_penalty = lax.cond(
            self.collision_with_humans_penalty_reward & collision_with_human, 
            lambda: self.collision_weight, lambda: 0.
        )
        collision_obs_penalty = lax.cond(
            self.collision_with_obstacles_penalty_reward & collision_with_obstacle, 
            lambda: self.collision_weight * 0.2, lambda: 0.
        )

        predictive_ttc_penalty = 0.
        if self.predictive_ttc_penalty_reward:
            rel_pos = robot_pos - humans_pos
            rel_vel = robot_vel_vec - humans_velocities
            
            # TTC calculation: t_min = -(rel_pos dot rel_vel) / (rel_vel dot rel_vel)
            rel_vel_sq = jnp.sum(rel_vel**2, axis=1) + 1e-8
            t_min = -jnp.sum(rel_pos * rel_vel, axis=1) / rel_vel_sq
            safe_t_min = jnp.clip(t_min, 0.0, self.ttc_horizon + 1.0)
            
            # Predict minimum distance at t_min
            min_dist_predicted = jnp.linalg.norm(rel_pos + jnp.expand_dims(safe_t_min, 1) * rel_vel, axis=1)
            collision_radius = self.robot_radius + humans_radiuses
            
            # Conditions for TTC penalty: moving towards, within horizon, predicted distance < safe threshold
            valid_ttc = (t_min > 0) & (t_min < self.ttc_horizon) & (min_dist_predicted < collision_radius * 1.5)
            
            penalties = jnp.where(valid_ttc, jnp.exp(-safe_t_min / self.ttc_threshold), 0.0)
            predictive_ttc_penalty = lax.cond(
                failure, lambda: 0., lambda: self.ttc_weight * jnp.sum(penalties) * dt
            )

        social_intrusion_penalty = 0.
        if self.social_intrusion_penalty_reward:
            dists = jnp.sqrt(jnp.sum((next_robot_pos - next_humans_pos)**2, axis=1) + 1e-8)
            d_min = self.robot_radius + humans_radiuses
            
            # Exponential penalty when entering comfort zone
            violations = (dists < self.social_comfort_dist) & (dists > d_min)
            safe_social_denom = jnp.maximum(self.social_comfort_dist - d_min, 1e-3)
            safe_dists_diff = jnp.maximum(dists - d_min, 0.0)
            penalties = jnp.where(
                violations, 
                jnp.exp(-safe_dists_diff / safe_social_denom), 
                0.0
            )
            social_intrusion_penalty = lax.cond(
                failure, lambda: 0., lambda: self.social_intrusion_weight * jnp.sum(penalties) * dt
            )

        # --- 3. COMFORT REWARDS (Kinematics) ---
        prev_action = info["action_history"][0]
        prev_prev_action = info["action_history"][1]
        
        angular_jerk_penalty = 0.
        if self.angular_jerk_penalty_reward:
            # RL-friendly smoothness (L1 norm of discrete 2nd derivative)
            delta_w = action[1] - prev_action[1]
            prev_delta_w = prev_action[1] - prev_prev_action[1]
            angular_jerk_penalty = self.angular_jerk_weight * jnp.abs(delta_w - prev_delta_w)

        linear_jerk_penalty = 0.
        if self.linear_jerk_penalty_reward:
            # RL-friendly smoothness (L1 norm of discrete 2nd derivative)
            delta_v = action[0] - prev_action[0]
            prev_delta_v = prev_action[0] - prev_prev_action[0]
            linear_jerk_penalty = self.linear_jerk_weight * jnp.abs(delta_v - prev_delta_v)

        # --- AGGREGATION ---
        reward = (goal_reward + timeout_penalty + progress_reward + 
                 collision_human_penalty + collision_obs_penalty + 
                 predictive_ttc_penalty + social_intrusion_penalty + 
                 angular_jerk_penalty + linear_jerk_penalty)
        
        ## DEBUG
        # _ = lax.cond(jnp.isnan(reward) | jnp.isinf(reward) | (reward > 1e5), lambda: debug.print("Reward is {x}",x=reward), lambda: None)
        debug.callback(
            lambda x: print(f"WARNING: Found invalid reward value: {x[1]}") if x[0] else None, 
            (jnp.isnan(reward) | jnp.isinf(reward) | (reward > 1e5), {
                "goal_reward": goal_reward,
                "timeout_penalty": timeout_penalty,
                "progress_reward": progress_reward,
                "collision_human_penalty": collision_human_penalty,
                "collision_obs_penalty": collision_obs_penalty,
                "predictive_ttc_penalty": predictive_ttc_penalty,
                "social_intrusion_penalty": social_intrusion_penalty,
                "angular_jerk_penalty": angular_jerk_penalty,
                "linear_jerk_penalty": linear_jerk_penalty,
                "reward": reward,
                "robot_pos": robot_pos,
                "next_robot_pos": next_robot_pos,
                "robot_vel_vec": robot_vel_vec,
                "humans_pos": humans_pos,
                "next_humans_pos": next_humans_pos,
                "humans_velocities": humans_velocities,
                "collision_with_human": collision_with_human,
                "collision_with_obstacle": collision_with_obstacle,
            })
        )

        if self.multi_gamma:
            reward_terms = {g: 0.0 for g in self.unique_gammas}
            if self.target_reached_reward: reward_terms[self.g_goal] += goal_reward
            if self.collision_with_humans_penalty_reward: reward_terms[self.g_coll_hum] += collision_human_penalty
            if self.collision_with_obstacles_penalty_reward: reward_terms[self.g_coll_obs] += collision_obs_penalty
            if self.timeout_penalty_reward: reward_terms[self.g_timeout] += timeout_penalty
            if self.progress_to_goal_reward: reward_terms[self.g_prog] += progress_reward
            if self.predictive_ttc_penalty_reward: reward_terms[self.g_ttc] += predictive_ttc_penalty
            if self.social_intrusion_penalty_reward: reward_terms[self.g_soc] += social_intrusion_penalty
            if self.angular_jerk_penalty_reward: reward_terms[self.g_ajerk] += angular_jerk_penalty
            if self.linear_jerk_penalty_reward: reward_terms[self.g_ljerk] += linear_jerk_penalty
        else:
            reward_terms = {self.gamma: reward}
            
        return reward, outcome, reward_terms