from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import os

from socialjym.envs.base_env import ROBOT_KINEMATICS
from socialjym.policies.mppi import MPPI
from socialjym.utils.distributions.gaussian import BivariateGaussian

class DRAMPPI(MPPI):
    def __init__(
            self, 
            # DRA-MPPI hyperparameters
            num_samples=400, 
            horizon=20, 
            temperature=0.1, 
            noise_sigma=jnp.array([0.4, 1.2]), # Sigma for [v, w]
            monte_carlo_risk_estimation_samples=20_000, # Number of samples to estimate the risk of a trajectory using Monte Carlo sampling
            humans_radius_hypothesis=0.3, # Radius to consider for the humans when estimating the risk of a trajectory (used in the risk critic)
            humans_motion_variance=0.3**2, # Variance to consider for the humans motion when estimating the risk of a trajectory (used in the risk critic)
            risk_threshold=0.05, # Threshold for the risk to consider a trajectory as too risky (used in the risk critic)
            # MPPI critics weights
            velocity_cost_weight = 0.5,
            goal_distance_cost_weight = 3.0,
            obstacle_cost_weight = 3.0,
            risk_soft_cost_weight = 5.0,
            risk_overall_weight = 1.0,
            control_cost_weight = 0.1,
            # Base hyperparameters 
            robot_radius:float=0.3,
            v_max:float=1., 
            gamma:float=0.9, 
            dt:float=0.25, 
            wheels_distance:float=0.7, 
            n_stack:int=5,
            lidar_angular_range=2*jnp.pi,
            lidar_max_dist=10.,
            lidar_num_rays=100,
            lidar_angles_robot_frame=None, # If not specified, rays are evenly distributed in the angular range
            lidar_n_stack_to_use=1, # Number of lidar scans to use to compute the action. If 1, only the most recent scan is used. If >1, the most recent n_stack_to_use scans are used and stacked together (e.g. if n_stack_to_use=3 and lidar_num_rays=100, the input point cloud will have 300 points).
        ):
        """
        Dynamic Risk-Aware Model Predictive Path Integral (DRA-MPPI) control policy for navigation.
        Implementation inspired by "Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations" by Trevisan et al. (IROS 2025).
        """
        # Intialize parent class
        super().__init__(
            robot_radius=robot_radius,
            v_max=v_max, 
            gamma=gamma, 
            dt=dt, 
            wheels_distance=wheels_distance, 
            n_stack=n_stack,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            lidar_angles_robot_frame=lidar_angles_robot_frame, 
            lidar_n_stack_to_use=lidar_n_stack_to_use, 
            use_box_action_space=False, # MPPI uses continuous action space clamped in the triangle
        )
        # Save DRA-MPPI hyperparameters
        self.num_samples = num_samples # K
        self.horizon = horizon         # T
        self.temperature = temperature         # Temperature
        self.noise_sigma = noise_sigma # Sigma for the noise added to the controls (shape: (2,) for [v, w])
        self.risk_threshold = risk_threshold # Maximum acceptable risk for a trajectory
        self.monte_carlo_risk_estimation_samples = monte_carlo_risk_estimation_samples # Number of samples to estimate the risk of a trajectory using Monte Carlo sampling
        self.risk_soft_cost_weight = risk_soft_cost_weight # Weight of the soft risk cost (which increases as the estimated risk approaches the maximum acceptable risk)
        self.humans_radius_hypothesis = humans_radius_hypothesis # Radius to consider for the humans when estimating the risk of a trajectory (used in the risk critic)
        self.radius = robot_radius + humans_radius_hypothesis # Radius to consider for the collision between the robot and the humans when estimating the risk of a trajectory (used in the risk critic)
        self.humans_motion_variance = humans_motion_variance # Variance to consider for the humans motion when estimating the risk of a trajectory (used in the risk critic)
        # Default parameters
        self.name = "DRA-MPPI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.u_mean_shape = (self.horizon, 2) # Shape: (Horizon, 2) -> (20 steps, 2 controls)
        self.biv_gaussian = BivariateGaussian() # Used for risk estimation
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, humans_state, monte_carlo_keys, rect)
            'velocity':  lambda p, a, aidx, g, cp: self._velocity_critic(a), # Heredited from DWA
            'goal_distance':   lambda p, a, aidx, g, cp: self._goal_distance_critic(p, g),
            'control_cost': lambda p, a, aidx, g, cp: self._control_critic(a),
            'risk': lambda p, a, aidx, g, cp: self._risk_critic(cp),
        }
        self.weights = {
            'velocity':  velocity_cost_weight,
            'goal_distance':   goal_distance_cost_weight,
            'control_cost': control_cost_weight,
            'risk': risk_overall_weight,
        }

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _risk_critic(self, collision_probability):
        return self.risk_soft_cost_weight * collision_probability + 1_000_000. * (collision_probability > self.risk_threshold).astype(jnp.float32)

    @partial(jit, static_argnames=("self"))
    def _control_critic(self, action):
        return jnp.abs(action[1]) # Prefer smaller angular velocities to produce smoother trajectories

    @partial(jit, static_argnames=("self"))
    def cost(self, robot_pose, action, action_idx, robot_goal, collision_probability):
        total_cost = 0.0
        for name, critic_fn in self.critics.items():
            weight = self.weights.get(name, 0.0)
            cost_val = critic_fn(robot_pose, action, action_idx, robot_goal, collision_probability)
            total_cost = total_cost + (weight * cost_val) 
        return total_cost

    @partial(jit, static_argnames=("self"))
    def _rollouts_and_costs(self, start_pose, controls_sequences, goal, humans_state, monte_carlo_keys):
        """
        Simulates all the trajectories and computes the cumulative cost.
        """
        init_poses = jnp.tile(start_pose, (self.num_samples, 1))
        inputs = {
            'controls_sequences': controls_sequences, # Shape: (horizon, num_samples, 2)
            'humans_state': humans_state, # Shape: (n_humans, 4)
            'monte_carlo_keys': monte_carlo_keys, # Shape: (horizon, num_samples, 2)
        }
        def step_fn(carry, inputs):
            poses, seq_idx, current_costs = carry
            actions = inputs['controls_sequences'][seq_idx]
            humans_state = inputs['humans_state']
            mc_key = inputs['monte_carlo_keys'][seq_idx]
            # Compute next poses
            next_poses = vmap(self.motion, in_axes=(0, 0, None))(poses, actions, self.dt)
            # Approximate joint collision probability on rect of all next_poses with Monte Carlo sampling
            x_min = jnp.min(next_poses[:, 0]) - self.radius
            x_max = jnp.max(next_poses[:, 0]) + self.radius
            y_min = jnp.min(next_poses[:, 1]) - self.radius
            y_max = jnp.max(next_poses[:, 1]) + self.radius
            mc_points = random.uniform(
                mc_key, 
                shape=(self.monte_carlo_risk_estimation_samples, 2), 
                minval=jnp.array([x_min, y_min]), 
                maxval=jnp.array([x_max, y_max])
            ) # Shape: (monte_carlo_risk_estimation_samples, 2)
            t_elapsed = (seq_idx + 1) * self.dt
            humans_position_distributions = {
                'means': humans_state[:, :2] + t_elapsed * humans_state[:, 2:], # Shape: (n_humans, 2)
                'correlation': jnp.zeros((len(humans_state),)), # Shape: (n_humans,)
                'logsigmas': jnp.log(jnp.sqrt(jnp.ones((len(humans_state), 2)) * self.humans_motion_variance) * t_elapsed), # Shape: (n_humans, 2)
            }
            pdfs_mc_points = vmap(vmap(self.biv_gaussian.p, in_axes=(None, 0)), in_axes=(0, None))(humans_position_distributions, mc_points) # Shape: (n_humans, mc_samples)
            pdfs_mc_points = jnp.clip(pdfs_mc_points, max=1)
            p_joint_mc = 1.0 - jnp.prod(1.0 - pdfs_mc_points, axis=0) # Shape: (mc_samples,)
            rob_mc_diff = next_poses[:, None, :] - mc_points[None, :, :] # (K, M, 2)
            rob_mc_dist_sq = jnp.sum(rob_mc_diff**2, axis=-1) # (K, M)
            in_robot_mask = (rob_mc_dist_sq <= self.radius**2).astype(jnp.float32) # (K, M)
            count_in = jnp.sum(in_robot_mask, axis=1) + 1e-6
            mean_density = jnp.dot(in_robot_mask, p_joint_mc) / count_in # (K, M) @ (M,) -> (K,)
            collision_area = jnp.pi * (self.radius**2) # (K,) * scalar -> (K,)
            collision_probabilities = mean_density * collision_area  # (K,) * scalar -> (K,)
            # Compute cost of the current step for all trajectories
            step_costs = vmap(self.cost, in_axes=(0, 0, None, None, 0))(next_poses, actions, seq_idx, goal, collision_probabilities)
            return (next_poses, seq_idx + 1, current_costs + step_costs), (next_poses, humans_position_distributions)
        (_, _, total_costs), (trajectories, humans_distributions) = lax.scan(step_fn, (init_poses, 0, jnp.zeros(self.num_samples)), inputs)
        trajectories = jnp.concatenate((start_pose[None, :], trajectories), axis=0) # Shape: (Horizon+1, 3)
        # TERMINAL COST
        # total_costs += 5 * vmap(self._goal_distance_critic, in_axes=(0, None))(final_poses, goal) # Add terminal cost based on distance to goal
        return total_costs, trajectories, humans_distributions

    # Public methods

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        obs:jnp.ndarray, # SocialNav obs
        info:dict, 
        u_mean:jnp.ndarray, 
        key
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Split key
        key, key_noise, key_mc = random.split(key)
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_position = obs[-1, :2] # Shape: (2,)
        robot_orientation = obs[-1, 5]
        robot_pose = jnp.concatenate((robot_position, robot_orientation[None])) # Shape: (3,)
        # Extract humans states
        humans_state = obs[:-1, :4] # Shape: (n_humans, 4)
        # Sample control sequences and compute their costs
        noise = random.normal(key_noise, (self.num_samples, self.horizon, 2)) * self.noise_sigma
        V = u_mean[None, :, :] + noise
        V_clamped = vmap(vmap(self._clamp_action))(V).transpose((1, 0, 2)) # Shape: (horizon, num_samples, 2)
        monte_carlo_keys = random.split(key_mc, self.horizon)
        costs, trajectories, humans_distributions = self._rollouts_and_costs(
            robot_pose, V_clamped, robot_goal, humans_state, monte_carlo_keys
        )
        # Compute weights and update u_mean
        beta = jnp.min(costs)
        weights = jnp.exp(-(costs - beta) / self.temperature)
        weights = weights / (jnp.sum(weights) + 1e-5) # Normalize to sum 1
        perturbations_weighted = jnp.sum(weights[:, None, None] * noise, axis=0)
        u_mean = u_mean + perturbations_weighted
        u_mean_clamped = vmap(self._clamp_action)(u_mean)
        action = u_mean_clamped[0]
        u_mean_clamped = jnp.roll(u_mean_clamped, -1, axis=0)
        u_mean_clamped = u_mean_clamped.at[-1].set(jnp.zeros(2)) 
        return action, u_mean_clamped, trajectories, costs, humans_distributions, key