from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial
import scipy.interpolate as interpolate
from jax.scipy.special import ndtri

from socialjym.envs.base_env import ROBOT_KINEMATICS, HUMAN_POLICIES, SCENARIOS
from socialjym.policies.mppi import MPPI
from socialjym.utils.distributions.gaussian import BivariateGaussian
from socialjym.envs.lasernav import LaserNav
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics
from .jessi import JESSI
from jhsfm.hsfm import get_linear_velocity, vectorized_compute_obstacle_closest_point

class DRAMPPI(MPPI):
    def __init__(
            self, 
            # DRA-MPPI hyperparameters
            use_halton_spline=True,
            num_knots=5,
            num_samples=400, 
            horizon=20, 
            temperature=0.1, 
            noise_sigma=jnp.array([0.2, 0.6]), # Sigma for [v, w]
            monte_carlo_risk_estimation_samples=10_000, # Number of samples to estimate the risk of a trajectory using Monte Carlo sampling
            humans_radius_hypothesis=0.3, # Radius to consider for the humans when estimating the risk of a trajectory (used in the risk critic)
            humans_motion_variance=0.1**2, # Variance to consider for the humans motion when estimating the risk of a trajectory (used in the risk critic)
            risk_threshold=0.05, # Threshold for the risk to consider a trajectory as too risky (used in the risk critic)
            eta_min = 5,
            eta_max = 10,
            # MPPI critics weights
            velocity_cost_weight = 0.5,
            goal_distance_cost_weight = 3.0,
            obstacle_cost_weight = 3.0,
            risk_soft_cost_weight = 5.0,
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
        self.eta_min = eta_min # Minimum value for eta (used to normalize the weights of the trajectories based on their costs)
        self.eta_max = eta_max # Maximum value for eta (used to normalize the weights of the trajectories based on their costs)
        self.use_halton_spline = use_halton_spline
        self.num_knots = num_knots # Number of knots for the spline smoothing of the noise (should be < horizon)
        if use_halton_spline:
            self.spline_matrix = self._precompute_spline_matrix(
                n_knots=self.num_knots, 
                horizon=self.horizon, 
                degree=3
            ) # Shape: (horizon, num_knots)
            total_dims = self.num_knots * 2 
            self.halton_base = self._generate_halton_sequence(
                n_samples=self.num_samples, 
                n_dims=total_dims
            ) # Shape: (num_samples, num_knots, 2)
            self.spline_matrix = jnp.array(self.spline_matrix)
        # Default parameters
        self.name = "DRA-MPPI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.u_mean_shape = (self.horizon, 2) # Shape: (Horizon, 2) -> (20 steps, 2 controls)
        self.biv_gaussian = BivariateGaussian() # Used for risk estimation
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, static_obstacles, collision_probability)
            'velocity':  lambda p, a, aidx, g, so, cp: self._velocity_critic(a), # Heredited from DWA
            'goal_distance':   lambda p, a, aidx, g, so, cp: self._goal_distance_critic(p, g),
            'control_cost': lambda p, a, aidx, g, so, cp: self._control_critic(a),
            'risk': lambda p, a, aidx, g, so, cp: self._risk_critic(cp),
            'obstacle': lambda p, a, aidx, g, so, cp: self._obstacle_critic(p, so),
        }
        self.weights = {
            'velocity':  velocity_cost_weight,
            'goal_distance':   goal_distance_cost_weight,
            'control_cost': control_cost_weight,
            'risk': 1.0, # The risk cost is already weighted by the risk_soft_cost_weight, so we set it to 1.0 here to avoid double weighting
            'obstacle': obstacle_cost_weight,
        }
        # Initialize critics and weights for the cost function from lidar
        self.critics_lidar = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, point_cloud, collision_probability)
            'velocity':  lambda p, a, aidx, g, pc, cp: self._velocity_critic(a), # Heredited from DWA
            'goal_distance':   lambda p, a, aidx, g, pc, cp: self._goal_distance_critic(p, g),
            'control_cost': lambda p, a, aidx, g, pc, cp: self._control_critic(a),
            'risk': lambda p, a, aidx, g, pc, cp: self._risk_critic(cp),
            'obstacle': lambda p, a, aidx, g, pc, cp: MPPI._obstacle_critic(self, p, pc),
        }
        self.weights_lidar = {
            'velocity':  velocity_cost_weight,
            'goal_distance':   goal_distance_cost_weight,
            'control_cost': control_cost_weight,
            'risk': 1.0, # The risk cost is already weighted by the risk_soft_cost_weight, so we set it to 1.0 here to avoid double weighting
            'obstacle': obstacle_cost_weight,
        }

    # Private methods

    def _precompute_spline_matrix(self, n_knots, horizon, degree=3):
        t_eval = jnp.linspace(0, n_knots - 1, horizon)
        knots_identity = jnp.eye(n_knots)
        x_knots = jnp.arange(n_knots)  
        basis_matrix = []
        for i in range(n_knots):
            y_dummy = knots_identity[i]
            spl = interpolate.make_interp_spline(x_knots, y_dummy, k=degree)
            y_eval = spl(t_eval)
            basis_matrix.append(y_eval)  
        M = jnp.stack(basis_matrix).T
        return M

    def _generate_halton_sequence(self, n_samples, n_dims):
        def primes_from_2_to(n):
            primes = [2]
            attempt = 3
            while len(primes) < n:
                if all(attempt % p != 0 for p in primes):
                    primes.append(attempt)
                attempt += 2
            return primes
        primes = primes_from_2_to(n_dims)
        seq = []
        for i in range(n_dims):
            b = primes[i]
            n, d = 0, 1
            pts = []
            for _ in range(n_samples):
                x = d - n
                if x == 1:
                    n += d
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    n = (b + 1) * y - x
                pts.append(n / d)
            seq.append(pts)
        halton_flat = jnp.array(seq).T # (n_samples, total_dims)
        return halton_flat.reshape(n_samples, -1, 2)

    @partial(jit, static_argnames=("self"))
    def _sample_halton_spline_noise(self, key):
        # 1. Randomize Halton (Cranley-Patterson Rotation)
        shift = random.uniform(key, shape=(1, self.num_knots, 2))
        randomized_halton = (self.halton_base + shift) % 1.0
        # 2. Transform to Gaussian (Box-Muller o Inverse CDF)
        knots_noise = ndtri(randomized_halton) 
        # 3. Apply Spline Smoothing via Matrix Multiplication
        # M: (T, K_knots)
        # Knots: (N_samples, K_knots, Actions)
        # Result: (N_samples, T, Actions)
        smooth_noise = jnp.einsum('tk, nka -> nta', self.spline_matrix, knots_noise)
        return smooth_noise

    @partial(jit, static_argnames=("self"))
    def _obstacle_critic(self, robot_pose, static_obstacles):
        closest_points = vectorized_compute_obstacle_closest_point(
            robot_pose[:2],
            static_obstacles
        )
        distances = jnp.linalg.norm(closest_points - robot_pose[:2], axis=1)
        min_distance = jnp.nanmin(distances)
        clearance_cost = lax.cond(
            min_distance - self.robot_radius <= 0,
            lambda: 1_000_000.,  # Collision, assign infinite cost
            lambda: 1/min_distance,  # Prefer larger clearance (i.e. smaller cost)
        )
        return clearance_cost  # Prefer larger clearance (i.e. smaller cost)

    @partial(jit, static_argnames=("self"))
    def _risk_critic(self, collision_probability):
        return self.risk_soft_cost_weight * collision_probability + 1_000_000. * (collision_probability > self.risk_threshold).astype(jnp.float32)

    @partial(jit, static_argnames=("self"))
    def _control_critic(self, action):
        return jnp.abs(action[1]) # Prefer smaller angular velocities to produce smoother trajectories

    @partial(jit, static_argnames=("self"))
    def _rollouts_and_costs(self, start_pose, controls_sequences, goal, humans_state, monte_carlo_keys, static_obstacles):
        """
        Simulates all the trajectories and computes the cumulative cost.
        """
        init_poses = jnp.tile(start_pose, (self.num_samples, 1)) # Shape: (num_samples, 3)
        inputs = {
            'controls_sequences': controls_sequences, # Shape: (horizon, num_samples, 2)
            'monte_carlo_keys': monte_carlo_keys, # Shape: (horizon, 2)
        }
        def step_fn(carry, inputs):
            poses, seq_idx, current_costs = carry
            actions = inputs['controls_sequences']
            mc_key = inputs['monte_carlo_keys']
            # Compute next poses
            next_poses = vmap(self.motion, in_axes=(0, 0, None))(poses, actions, self.dt) # Shape: (num_samples, 3)
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
            rob_mc_diff = next_poses[:, None, :2] - mc_points[None, :, :] # (K, M, 2)
            rob_mc_dist_sq = jnp.sum(rob_mc_diff**2, axis=-1) # (K, M)
            in_robot_mask = (rob_mc_dist_sq <= self.radius**2).astype(jnp.float32) # (K, M)
            count_in = jnp.sum(in_robot_mask, axis=1) + 1e-6
            mean_density = jnp.dot(in_robot_mask, p_joint_mc) / count_in # (K, M) @ (M,) -> (K,)
            collision_area = jnp.pi * (self.radius**2) # (K,) * scalar -> (K,)
            collision_probabilities = mean_density * collision_area  # (K,) * scalar -> (K,)
            # debug.print("Step {s}: Avg. Collision probability: {cp}", s=seq_idx, cp=jnp.mean(collision_probabilities))
            # Compute cost of the current step for all trajectories
            step_costs = vmap(self.cost, in_axes=(0, 0, None, None, None, 0))(next_poses, actions, seq_idx, goal, static_obstacles, collision_probabilities)
            discounted_step_costs = step_costs * (self.gamma ** (seq_idx * self.dt * self.v_max))
            return (next_poses, seq_idx + 1, current_costs + discounted_step_costs), (next_poses, humans_position_distributions)
        (_, _, total_costs), (trajectories, humans_distributions) = lax.scan(step_fn, (init_poses, 0, jnp.zeros(self.num_samples)), inputs)
        trajectories = jnp.concatenate((init_poses[None, :], trajectories), axis=0) # Shape: (Horizon+1, num_samples, 3)
        # TERMINAL COST (Not used in DRA-MPPI)
        return total_costs, trajectories, humans_distributions

    @partial(jit, static_argnames=("self"))
    def _rollouts_and_costs_from_lidar(self, start_pose, controls_sequences, goal, humans_state, monte_carlo_keys, point_cloud):
        """
        Simulates all the trajectories and computes the cumulative cost.
        """
        init_poses = jnp.tile(start_pose, (self.num_samples, 1)) # Shape: (num_samples, 3)
        inputs = {
            'controls_sequences': controls_sequences, # Shape: (horizon, num_samples, 2)
            'monte_carlo_keys': monte_carlo_keys, # Shape: (horizon, 2)
        }
        humans_mask = humans_state['mask']
        humans_pos_means = humans_state['distrs']['pos_distrs']['means']
        humans_pos_covs = vmap(self.biv_gaussian.covariance)(humans_state['distrs']['pos_distrs'])
        humans_vel_means = humans_state['distrs']['vel_distrs']['means']
        humans_vel_covs = vmap(self.biv_gaussian.covariance)(humans_state['distrs']['vel_distrs'])
        def step_fn(carry, inputs):
            poses, seq_idx, current_costs = carry
            actions = inputs['controls_sequences']
            mc_key = inputs['monte_carlo_keys']
            # Compute next poses
            next_poses = vmap(self.motion, in_axes=(0, 0, None))(poses, actions, self.dt) # Shape: (num_samples, 3)
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
            next_humans_pos_means = humans_pos_means + t_elapsed * humans_vel_means
            next_humans_pos_covs = humans_pos_covs + (t_elapsed**2) * humans_vel_covs 
            humans_position_distributions = vmap(self.biv_gaussian.covariance_to_parameters)(next_humans_pos_covs)
            humans_position_distributions['means'] = next_humans_pos_means
            pdfs_mc_points = vmap(vmap(self.biv_gaussian.p, in_axes=(None, 0)), in_axes=(0, None))(humans_position_distributions, mc_points) # Shape: (n_humans, mc_samples)
            # Mask all the points extracted from not real humans (i.e. those with low score in the perception)
            pdfs_mc_points = pdfs_mc_points * humans_mask[:, None] # Shape: (n_humans, mc_samples)
            pdfs_mc_points = jnp.clip(pdfs_mc_points, max=1)
            # Mask distributions extracted from not real humans (i.e. those with low score in the perception)
            humans_position_distributions['means'] = jnp.where(humans_mask[:, None], humans_position_distributions['means'], jnp.full_like(humans_position_distributions['means'], jnp.nan))
            humans_position_distributions['logsigmas'] = jnp.where(humans_mask[:, None], humans_position_distributions['logsigmas'], jnp.full_like(humans_position_distributions['logsigmas'], jnp.nan))
            humans_position_distributions['correlation'] = jnp.where(humans_mask, humans_position_distributions['correlation'], jnp.full_like(humans_position_distributions['correlation'], jnp.nan))
            p_joint_mc = 1.0 - jnp.prod(1.0 - pdfs_mc_points, axis=0) # Shape: (mc_samples,)
            rob_mc_diff = next_poses[:, None, :2] - mc_points[None, :, :] # (K, M, 2)
            rob_mc_dist_sq = jnp.sum(rob_mc_diff**2, axis=-1) # (K, M)
            in_robot_mask = (rob_mc_dist_sq <= self.radius**2).astype(jnp.float32) # (K, M)
            count_in = jnp.sum(in_robot_mask, axis=1) + 1e-6
            mean_density = jnp.dot(in_robot_mask, p_joint_mc) / count_in # (K, M) @ (M,) -> (K,)
            collision_area = jnp.pi * (self.radius**2) # (K,) * scalar -> (K,)
            collision_probabilities = mean_density * collision_area  # (K,) * scalar -> (K,)
            # debug.print("Step {s}: Avg. Collision probability: {cp}", s=seq_idx, cp=jnp.mean(collision_probabilities))
            # Compute cost of the current step for all trajectories
            step_costs = vmap(self.cost_from_lidar, in_axes=(0, 0, None, None, None, 0))(next_poses, actions, seq_idx, goal, point_cloud, collision_probabilities)
            discounted_step_costs = step_costs * (self.gamma ** (seq_idx * self.dt * self.v_max))
            return (next_poses, seq_idx + 1, current_costs + discounted_step_costs), (next_poses, humans_position_distributions)
        (_, _, total_costs), (trajectories, humans_distributions) = lax.scan(step_fn, (init_poses, 0, jnp.zeros(self.num_samples)), inputs)
        trajectories = jnp.concatenate((init_poses[None, :], trajectories), axis=0) # Shape: (Horizon+1, num_samples, 3)
        # TERMINAL COST (Not used in DRA-MPPI)
        return total_costs, trajectories, humans_distributions

    # Public methods

    @partial(jit, static_argnames=("self"))
    def init_u_mean_and_beta(self):
        return self.init_u_mean(), self.temperature

    @partial(jit, static_argnames=("self"))
    def cost(self, robot_pose, action, action_idx, robot_goal, static_obstacles, collision_probability):
        total_cost = 0.0
        for name, critic_fn in self.critics.items():
            weight = self.weights.get(name, 0.0)
            cost_val = critic_fn(robot_pose, action, action_idx, robot_goal, static_obstacles, collision_probability)
            total_cost = total_cost + (weight * cost_val) 
        return total_cost

    @partial(jit, static_argnames=("self"))
    def cost_from_lidar(self, robot_pose, action, action_idx, robot_goal, point_cloud, collision_probability):
        total_cost = 0.0
        for name, critic_fn in self.critics_lidar.items():
            weight = self.weights_lidar.get(name, 0.0)
            cost_val = critic_fn(robot_pose, action, action_idx, robot_goal, point_cloud, collision_probability)
            total_cost = total_cost + (weight * cost_val) 
        return total_cost

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        obs:jnp.ndarray, # SocialNav obs
        info:dict, 
        u_mean:jnp.ndarray, 
        beta:float,
        key
    ) -> tuple:
        # Split key
        key, key_noise, key_mc = random.split(key, 3)
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_position = obs[-1, :2] # Shape: (2,)
        robot_orientation = obs[-1, 5]
        robot_pose = jnp.concatenate((robot_position, robot_orientation[None])) # Shape: (3,)
        # Extract humans states
        humans_state = obs[:-1, :4] # Shape: (n_humans, 4)
        # Generate control sequences
        if self.use_halton_spline:
            base_noise = self._sample_halton_spline_noise(key_noise)
        else:
            base_noise = random.normal(key_noise, (self.num_samples, self.horizon, 2))
        noise = base_noise * self.noise_sigma
        V = u_mean[None, :, :] + noise # Shape: (num_samples, horizon, 2)
        V_clamped = vmap(vmap(self._clamp_action))(V).transpose((1, 0, 2)) # Shape: (horizon, num_samples, 2)
        V_clamped = V_clamped.at[:,0,:].set(jnp.zeros((self.horizon, 2))) # Ensure one trajectory is full braking
        monte_carlo_keys = random.split(key_mc, self.horizon)
        costs, trajectories, humans_distributions = self._rollouts_and_costs(
            robot_pose, V_clamped, robot_goal, humans_state, monte_carlo_keys, info['static_obstacles'][-1],
        )
        # Compute weights and update u_mean
        rho = jnp.min(costs)
        weights = jnp.exp(-(costs - rho) / beta)
        eta = jnp.sum(weights) + 1e-5
        case = jnp.argmax(jnp.array([
            eta > self.eta_max, 
            eta < self.eta_min, 
            (eta <= self.eta_max) & (eta >= self.eta_min),
        ]))
        new_beta = lax.switch(
            case,
            [
                lambda: beta * 0.9, # If eta is too high, decrease beta
                lambda: beta * 1.2, # If eta is too low, increase beta
                lambda: beta,       # If eta is good, keep beta the same
            ]
        )
        weights = weights / eta # Normalize to sum 1
        perturbations_weighted = jnp.sum(weights[:, None, None] * noise, axis=0)
        u_mean = u_mean + perturbations_weighted
        u_mean_clamped = vmap(self._clamp_action)(u_mean)
        action = u_mean_clamped[0]
        u_mean_clamped = jnp.roll(u_mean_clamped, -1, axis=0)
        u_mean_clamped = u_mean_clamped.at[-1].set(jnp.zeros(2)) 
        return action, u_mean_clamped, new_beta, trajectories.transpose((1, 0, 2)), costs, humans_distributions, key

    def animate_socialnav_trajectory(
        self,
        states,
        actions,
        u_means,
        trajectories,
        trajectories_costs,
        goals,
        static_obstacles,
        humans_radii,
        humans_distributions,
        socialnav_env:SocialNav,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        robot_positions = states[:,-1,:2]
        robot_orientations = states[:,-1,4]
        robot_poses = jnp.hstack((robot_positions, robot_orientations.reshape(-1,1)))
        humans_positions = states[:,:-1,:2]
        humans_orientations = states[:,:-1,4]
        humans_poses = jnp.dstack((humans_positions, humans_orientations))
        humans_body_velocities = states[:,:-1,2:4]
        humans_velocities = lax.cond(
            socialnav_env.humans_policy == HUMAN_POLICIES.index('hsfm'),
            lambda: vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(
                    humans_orientations,
                    humans_body_velocities,
                ),
            lambda: humans_body_velocities,
        )
        # Compute full trajectories with U_mean
        @jit
        def compute_chosen_trajectory(robot_pose, robot_u_mean):
            trajectory = jnp.empty((self.horizon+1, 3))
            trajectory = trajectory.at[0].set(robot_pose)
            for t in range(1,self.horizon+1):
                next_pose = self.motion(trajectory[t-1], robot_u_mean[t-1], self.dt)
                trajectory = trajectory.at[t].set(next_pose)
            return trajectory
        chosen_trajectories = vmap(compute_chosen_trajectory)(robot_poses, u_means)
        self.animate_trajectory(
            robot_poses,
            actions,
            chosen_trajectories,
            trajectories,
            trajectories_costs,
            goals,
            humans_poses,
            humans_velocities,
            humans_radii,
            static_obstacles,
            humans_distributions=humans_distributions,
            x_lims=x_lims,
            y_lims=y_lims,
            save_video=save_video,
        )
   
    # LaserNav methods

    @partial(jit, static_argnames=("self","jessi"))
    def act_on_jessi_perception(
        self, 
        jessi:JESSI,
        perception_params:dict,
        key:random.PRNGKey,
        lasernav_obs:jnp.ndarray, 
        info:dict, 
        u_mean:jnp.ndarray, 
        beta:float,
    ) -> tuple:
        # Split key
        key, key_noise, key_mc = random.split(key, 3)
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_pose = lasernav_obs[0, :3] # Shape: (3,)
        ## Compute aligned point_cloud
        aligned_lidar = self.align_lidar(lasernav_obs[:self.lidar_n_stack_to_use])[1]
        point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        ## Identify visible humans with JESSI perception
        hcgs, _, _ = jessi.perception.apply(perception_params, None, jessi.compute_perception_input(lasernav_obs)[0])
        humans_mask = hcgs['weights'] > 0.5
        ## Roto-translate hcgs to be in the global frame
        hcgs['pos_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
            hcgs['pos_distrs'], 
            robot_pose
        )
        hcgs['vel_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
            hcgs['vel_distrs'], 
            jnp.array([0, 0, robot_pose[2]]) # Velocities are not affected by translation, only rotation
        )
        humans_state = {
            'distrs': hcgs, # Shape: (n_visible_humans, 2)
            'mask': humans_mask, # Shape: (n_humans,)
        }
        # Generate control sequences
        if self.use_halton_spline:
            base_noise = self._sample_halton_spline_noise(key_noise)
        else:
            base_noise = random.normal(key_noise, (self.num_samples, self.horizon, 2))
        noise = base_noise * self.noise_sigma
        V = u_mean[None, :, :] + noise # Shape: (num_samples, horizon, 2)
        V_clamped = vmap(vmap(self._clamp_action))(V).transpose((1, 0, 2)) # Shape: (horizon, num_samples, 2)
        monte_carlo_keys = random.split(key_mc, self.horizon)
        costs, trajectories, humans_distributions = self._rollouts_and_costs_from_lidar(
            robot_pose, V_clamped, robot_goal, humans_state, monte_carlo_keys, point_cloud
        )
        # Compute weights and update u_mean
        rho = jnp.min(costs)
        weights = jnp.exp(-(costs - rho) / beta)
        eta = jnp.sum(weights) + 1e-5
        case = jnp.argmax(jnp.array([
            eta > self.eta_max, 
            eta < self.eta_min, 
            (eta <= self.eta_max) & (eta >= self.eta_min),
        ]))
        new_beta = lax.switch(
            case,
            [
                lambda: beta * 0.9, # If eta is too high, decrease beta
                lambda: beta * 1.2, # If eta is too low, increase beta
                lambda: beta,       # If eta is good, keep beta the same
            ]
        )
        weights = weights / eta # Normalize to sum 1
        perturbations_weighted = jnp.sum(weights[:, None, None] * noise, axis=0)
        u_mean = u_mean + perturbations_weighted
        u_mean_clamped = vmap(self._clamp_action)(u_mean)
        action = u_mean_clamped[0]
        u_mean_clamped = jnp.roll(u_mean_clamped, -1, axis=0)
        u_mean_clamped = u_mean_clamped.at[-1].set(jnp.zeros(2)) 
        return action, u_mean_clamped, new_beta, trajectories.transpose((1, 0, 2)), costs, humans_distributions, hcgs, key
    
    def animate_lasernav_trajectory(
        self,
        states,
        observations,
        actions,
        u_means,
        trajectories,
        trajectories_costs,
        goals,
        static_obstacles,
        humans_radii,
        humans_distributions,
        perception_distributions,
        lasernav_env:LaserNav,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        robot_positions = states[:,-1,:2]
        robot_orientations = states[:,-1,4]
        robot_poses = jnp.hstack((robot_positions, robot_orientations.reshape(-1,1)))
        humans_positions = states[:,:-1,:2]
        humans_orientations = states[:,:-1,4]
        humans_poses = jnp.dstack((humans_positions, humans_orientations))
        humans_body_velocities = states[:,:-1,2:4]
        humans_velocities = lax.cond(
            lasernav_env.humans_policy == HUMAN_POLICIES.index('hsfm'),
            lambda: vmap(vmap(get_linear_velocity, in_axes=(0,0)), in_axes=(0,0))(
                    humans_orientations,
                    humans_body_velocities,
                ),
            lambda: humans_body_velocities,
        )
        # Compute full trajectories with U_mean
        @jit
        def compute_chosen_trajectory(robot_pose, robot_u_mean):
            trajectory = jnp.empty((self.horizon+1, 3))
            trajectory = trajectory.at[0].set(robot_pose)
            for t in range(1,self.horizon+1):
                next_pose = self.motion(trajectory[t-1], robot_u_mean[t-1], self.dt)
                trajectory = trajectory.at[t].set(next_pose)
            return trajectory
        chosen_trajectories = vmap(compute_chosen_trajectory)(robot_poses, u_means)
        # # Compute point clouds from LaserNav observations
        # @jit
        # def compute_point_cloud(observation):
        #     return self.align_lidar(observation)[1][:self.lidar_n_stack_to_use].reshape(-1, 2)
        # point_clouds = vmap(compute_point_cloud)(observations)
        self.animate_trajectory(
            robot_poses,
            actions,
            chosen_trajectories,
            trajectories,
            trajectories_costs,
            goals,
            humans_poses,
            humans_velocities,
            humans_radii,
            static_obstacles,
            lidar_scans=observations[:,0,6:],
            # point_clouds=point_clouds,
            humans_distributions=humans_distributions,
            perception_distributions=perception_distributions,
            x_lims=x_lims,
            y_lims=y_lims,
            save_video=save_video,
        )

    def evaluate_on_jessi_perception(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
        jessi:JESSI,
        perception_params:dict,
    ) -> dict:
        """
        Test the trained policy over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == self.kinematics, "Policy kinematics must match environment kinematics"
        assert env.robot_dt == self.dt, f"Environment time step (dt={env.dt}) must be equal to policy time step (dt={self.dt}) for evaluation"
        time_limit = env.reward_function.time_limit
        @loop_tqdm(n_trials)
        @jit
        def _fori_body(i:int, for_val:tuple):   
            @jit
            def _while_body(while_val:tuple):
                # Retrieve data from the tuple
                state, obs, info, outcome, u_mean, beta, policy_key, env_key, steps, all_actions, all_states = while_val
                action, u_mean, beta, _, _, _, _, policy_key = self.act_on_jessi_perception(jessi, perception_params, policy_key, obs, info, u_mean, beta)
                state, obs, info, _, outcome, (_, env_key)  = env.step(state,info,action,test=True)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, u_mean, beta, policy_key, env_key, steps, all_actions, all_states

            ## Retrieve data from the tuple
            seed, metrics = for_val
            policy_key, reset_key, env_key = vmap(random.PRNGKey)(jnp.zeros(3, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
            ## Reset the environment
            state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            initial_robot_position = state[-1,:2]
            ## Episode loop
            all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
            all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
            u_mean_init, beta_init = self.init_u_mean_and_beta()
            while_val_init = (state, obs, info, init_outcome, u_mean_init, beta_init, policy_key, env_key, 0, all_actions, all_states)
            _, _, end_info, outcome, _ , _, policy_key, env_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
            ## Update metrics
            metrics = compute_episode_metrics(
                environment=env.environment,
                metrics=metrics,
                episode_idx=i, 
                initial_robot_position=initial_robot_position, 
                all_states=all_states, 
                all_actions=all_actions, 
                outcome=outcome, 
                episode_steps=episode_steps, 
                end_info=end_info, 
                max_steps=int(time_limit/env.robot_dt)+1, 
                personal_space=0.5,
                robot_dt=env.robot_dt,
                robot_radius=env.robot_radius,
                ccso_n_static_humans=env.ccso_n_static_humans,
                robot_specs={'kinematics': env.kinematics, 'v_max': self.v_max, 'wheels_distance': self.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
            )
            seed += 1
            return seed, metrics
        # Initialize metrics
        metrics = initialize_metrics_dict(n_trials)
        # Execute n_trials tests
        if env.scenario == SCENARIOS.index('circular_crossing_with_static_obstacles'):
            print(f"\nExecuting {n_trials} tests with {env.n_humans - env.ccso_n_static_humans} humans and {env.ccso_n_static_humans} obstacles...")
        else:
            print(f"\nExecuting {n_trials} tests with {env.n_humans} humans and {env.n_obstacles} obstacles...")
        _, metrics = lax.fori_loop(0, n_trials, _fori_body, (random_seed, metrics))
        # Print results
        print_average_metrics(n_trials, metrics)
        return metrics