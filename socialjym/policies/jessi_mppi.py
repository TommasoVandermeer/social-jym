import jax.numpy as jnp
from jax import random, jit, vmap, lax, debug, ops
from jax.tree_util import tree_map
from functools import partial

from socialjym.policies.jessi import JESSI
from socialjym.envs.base_env import wrap_angle
from socialjym.utils.distributions.gaussian import BivariateGaussian

class JESSI_MPPI(JESSI):
    def __init__(
        self, 
        # MPPI parameters
        num_samples=50, 
        horizon=20, 
        temperature=0.15, 
        velocity_cost_weight = 1.0,
        goal_distance_cost_weight = 2.0,
        obstacle_cost_weight = 0.1,
        control_cost_weight = 0.1,
        humans_cost_weight = 5.0,
        safety_from_humans = 0.3,
        lidar_n_stack_to_use=1, # Number of lidar scans to use to compute the action.
        humans_detection_threshold=0.5, # Minimum weight for a human to be considered in the cost function
        # JESSI parameters
        robot_radius:float=0.3,
        v_max:float=1., 
        dt:float=0.25, 
        wheels_distance:float=0.7, 
        n_stack:int=5,
        lidar_angular_range=2*jnp.pi,
        lidar_max_dist=10.,
        lidar_num_rays=100,
        lidar_angles_robot_frame=None, # If not specified, rays are evenly distributed in the angular range
        lidar_position_robot_frame:tuple[float, float]=(0.0, 0.0),
        n_detectable_humans:int=10,
        max_humans_velocity:float=1.5,
        max_beam_range:float=10.0, # This is only used to normalize the LiDAR readings before feeding them to the encoder
        embedding_dim:int=32,
        n_sectors:int=60,
        angular_sectors_width_deg:float=18.0, # This is the width of the attention sectors in degrees. It determines how many rays are attended to by each sector.
        n_stack_for_action_space_bounding:int=1,
        beam_dropout_rate:float=0.0,
        legit:bool=False, 
    ) -> None:
        ## Input validation
        assert isinstance(num_samples, int) and num_samples > 0, "num_samples must be a positive integer."
        assert isinstance(horizon, int) and horizon > 0, "horizon must be a positive integer."
        assert isinstance(temperature, (int, float)) and temperature > 0, "temperature must be a positive number."
        assert isinstance(velocity_cost_weight, (int, float)) and velocity_cost_weight >= 0, "velocity_cost_weight must be a non-negative number."
        assert isinstance(goal_distance_cost_weight, (int, float)) and goal_distance_cost_weight >= 0, "goal_distance_cost_weight must be a non-negative number."
        assert isinstance(obstacle_cost_weight, (int, float)) and obstacle_cost_weight >= 0, "obstacle_cost_weight must be a non-negative number."
        assert isinstance(control_cost_weight, (int, float)) and control_cost_weight >= 0, "control_cost_weight must be a non-negative number."
        assert isinstance(humans_cost_weight, (int, float)) and humans_cost_weight >= 0, "humans_cost_weight must be a non-negative number."
        assert isinstance(safety_from_humans, (int, float)) and safety_from_humans >= 0, "safety_from_humans must be a non-negative number."
        assert isinstance(lidar_n_stack_to_use, int) and lidar_n_stack_to_use > 0, "lidar_n_stack_to_use must be a positive integer."
        ## JESSI initialization
        super().__init__(
            robot_radius=robot_radius,
            v_max=v_max, 
            dt=dt, 
            wheels_distance=wheels_distance, 
            n_stack=n_stack,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            lidar_angles_robot_frame=lidar_angles_robot_frame,
            lidar_position_robot_frame=lidar_position_robot_frame,
            n_detectable_humans=n_detectable_humans,
            max_humans_velocity=max_humans_velocity,
            max_beam_range=max_beam_range,
            embedding_dim=embedding_dim,
            n_sectors=n_sectors,
            angular_sectors_width_deg=angular_sectors_width_deg,
            n_stack_for_action_space_bounding=n_stack_for_action_space_bounding,
            beam_dropout_rate=beam_dropout_rate,
            ablation_mode=None,
            legit=legit
        )
        ## MPPI initialization
        self.num_samples = num_samples
        self.horizon = horizon
        self.temperature = temperature
        self.velocity_cost_weight = velocity_cost_weight
        self.goal_distance_cost_weight = goal_distance_cost_weight
        self.obstacle_cost_weight = obstacle_cost_weight
        self.control_cost_weight = control_cost_weight
        self.humans_cost_weight = humans_cost_weight
        self.safety_from_humans = safety_from_humans
        self.lidar_n_stack_to_use = lidar_n_stack_to_use
        self.humans_detection_threshold = humans_detection_threshold
        self.biv_gaussian = BivariateGaussian()
        self.w_max = 2 * v_max / wheels_distance  
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, robot_goal, point_cloud, humans_positions, humans_mask)
            'velocity':  lambda p, a, g, pc, h, m: self._velocity_critic(a),
            'goal_distance':   lambda p, a, g, pc, h, m: self._goal_distance_critic(p, g),
            'obstacle_cost': lambda p, a, g, pc, h, m: self._obstacle_critic(p, pc),
            'control_cost': lambda p, a, g, pc, h, m: self._control_critic(a),
            'humans_cost': lambda p, a, g, pc, h, m: self._humans_critic(p, h, m),
        }
        self.weights = {
            'velocity':  velocity_cost_weight,
            'goal_distance':   goal_distance_cost_weight,
            'obstacle_cost': obstacle_cost_weight,
            'control_cost': control_cost_weight,
            'humans_cost': humans_cost_weight,
        }

    ### Private methods

    @partial(jit, static_argnames=("self"))
    def _velocity_critic(self, action):
        vmax = self.v_max - (self.v_max * jnp.abs(action[1]) / self.w_max)  # Max linear velocity for the given angular velocity, to be in the triangle defined by the kinematic constraints
        return lax.cond(
            vmax > 0,
            lambda: (vmax - action[0]) / vmax,  # Prefer higher speeds (given the maximum speed for the given angular velocity)
            lambda: 0.0 # Complete turning in place is not penalized
        )

    @partial(jit, static_argnames=("self"))
    def _obstacle_critic(self, robot_pose, point_cloud):
        distances = jnp.linalg.norm(robot_pose[None, :2] - point_cloud, axis=1)
        min_distance = jnp.min(distances)
        clearance_cost = lax.cond(
            min_distance - self.robot_radius <= 0,
            lambda: 1_000_000. / self.obstacle_cost_weight,  # Collision, assign infinite cost
            lambda: 1/min_distance,  # Prefer larger clearance (i.e. smaller cost)
        )
        return clearance_cost  # Prefer larger clearance (i.e. smaller cost)
    
    @partial(jit, static_argnames=("self"))
    def _goal_distance_critic(self, robot_pose, robot_goal):
        distance_to_goal = jnp.linalg.norm(robot_pose[:2] - robot_goal)
        return distance_to_goal
    
    @partial(jit, static_argnames=("self"))
    def _control_critic(self, action):
        return jnp.linalg.norm(action) # Prefer smaller actions (i.e. less energy consumption and smoother trajectories)

    @partial(jit, static_argnames=("self"))
    def _humans_critic(self, robot_pose, humans_positions, humans_mask):
        distances = jnp.linalg.norm(robot_pose[None, :2] - humans_positions[:, :2], axis=1)
        distances = jnp.where(humans_mask, distances, jnp.inf)  # Set distance to infinity for undetectable humans
        min_distance = jnp.min(distances) - (self.robot_radius + self.safety_from_humans)  # Subtract robot radius and safety margin
        cost = lax.cond(
            min_distance <= 0,
            lambda: 1_000_000. / self.humans_cost_weight,  # Collision or too close, assign infinite cost
            lambda: 1/min_distance,  # Prefer larger clearance (i.e. smaller cost)
        )
        return cost

    @partial(jit, static_argnames=("self"))
    def reproject_scan(self, old_scan, old_pose, new_pose):
        num_rays = old_scan.shape[0]
        fov = self.lidar_angular_range
        angles = jnp.linspace(-fov/2, fov/2, num_rays)
        x_local = old_scan * jnp.cos(angles)
        y_local = old_scan * jnp.sin(angles)
        local_pc = jnp.stack([x_local, y_local], axis=-1)
        c_old, s_old = jnp.cos(old_pose[2]), jnp.sin(old_pose[2])
        R_old = jnp.array([[c_old, -s_old], [s_old, c_old]])
        global_pc = local_pc @ R_old.T + old_pose[:2]
        c_new, s_new = jnp.cos(new_pose[2]), jnp.sin(new_pose[2])
        R_new_inv = jnp.array([[c_new, s_new], [-s_new, c_new]])
        new_local_pc = (global_pc - new_pose[:2]) @ R_new_inv.T
        dists = jnp.linalg.norm(new_local_pc, axis=-1)
        new_angles = jnp.arctan2(new_local_pc[:, 1], new_local_pc[:, 0])
        min_angle = -fov / 2.0
        angle_res = fov / num_rays
        bin_idx = jnp.floor((new_angles - min_angle) / angle_res).astype(jnp.int32)
        valid_mask = (bin_idx >= 0) & (bin_idx < num_rays)
        safe_bin_idx = jnp.where(valid_mask, bin_idx, num_rays)
        safe_dists = jnp.where(valid_mask, dists, jnp.inf)
        min_dists = ops.segment_min(safe_dists, safe_bin_idx, num_segments=num_rays + 1)
        scan = min_dists[:num_rays]
        return jnp.where(jnp.isinf(scan), self.lidar_max_dist, scan)

    @partial(jit, static_argnames=("self"))
    def _rollout_and_cost(self, key, fix_scan_embeddings, perception_params, actor_critic_params, observation, start_pose, goal, point_cloud, humans_state_seq):
        """
        Simulates a trajectory and computes the cumulative cost.
        """
        ref_poses = observation[:, :3]  # Shape: (n_stack, 3)
        humans_mask = humans_state_seq['weights'][0] > self.humans_detection_threshold  # Mask for detectable humans
        def step_fn(carry, perception_output):
            pose, prec_poses, current_cost, time_idx, key = carry
            key1, key2 = random.split(key)
            # Extract human positions from the state
            humans_positions = perception_output['pos_distrs']['means']  # Shape: (n_detectable_humans, 2)
            # Compute perception output in the robot frame
            theta_inv = -pose[2]
            c_inv, s_inv = jnp.cos(theta_inv), jnp.sin(theta_inv)
            rx_inv = -(c_inv * pose[0] - s_inv * pose[1])
            ry_inv = -(s_inv * pose[0] + c_inv * pose[1])
            robot_pose_inv = jnp.array([rx_inv, ry_inv, theta_inv])
            perception_output['pos_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
                perception_output['pos_distrs'], 
                robot_pose_inv
            )
            perception_output['vel_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
                perception_output['vel_distrs'], 
                jnp.array([0, 0, robot_pose_inv[2]]) # Velocities are not affected by translation, only rotation
            )
            # Compute action space parameters
            point_cloud_robot_frame = (point_cloud - pose[None, :2]) @ jnp.array([[c_inv, s_inv], [-s_inv, c_inv]])
            action_space_params = self.bound_action_space(point_cloud_robot_frame)
            # Compute actor input 
            goal_robot_frame = jnp.array([[c_inv, -s_inv], [s_inv, c_inv]]) @ (goal - pose[:2])
            actor_input = self.compute_actor_input(None, perception_output, action_space_params, goal_robot_frame)
            # Compute new scan embeddings
            ## Very expensive computationally
            # new_lidar_measurements = vmap(self.reproject_scan, in_axes=(0, 0, 0))(
            #     observation[:, 11:],
            #     ref_poses, 
            #     prec_poses
            # )
            # new_observation = observation.at[:, :3].set(prec_poses)
            # new_observation = new_observation.at[:, 11:].set(new_lidar_measurements)
            # perception_input, _ = self.compute_perception_input(new_observation)
            # _, scan_embeddings, _, _ = self.perception.apply(perception_params, None, perception_input)
            ## Faster method but not really accurate
            delta_theta = wrap_angle(pose[2] - ref_poses[0, 2])
            shift = jnp.round(delta_theta / jnp.deg2rad(self.angular_sectors_width_deg)).astype(int)
            scan_embeddings = jnp.roll(fix_scan_embeddings, shift, axis=0)
            # Compute robot action
            action, _, _, _, _ = self.actor_critic.apply(
                actor_critic_params, 
                None, 
                actor_input, 
                scan_embeddings,
                random_key=key2,
            ) 
            # Apply robot action
            next_pose = self.motion(pose, action, self.dt)
            # Update poses
            prec_poses = jnp.roll(prec_poses, shift=1, axis=0)
            prec_poses = prec_poses.at[0].set(next_pose)
            # Compute step cost
            step_cost = self.cost(pose, action, goal, point_cloud, humans_positions, humans_mask) 
            return (next_pose, prec_poses, current_cost + step_cost, time_idx + 1, key1), (next_pose, action)
        (final_pose, _,total_cost, _, _), (trajectory, actions) = lax.scan(step_fn, (start_pose, observation[:, :3], 0.0, 0, key), humans_state_seq)
        trajectory = jnp.concatenate((start_pose[None, :], trajectory), axis=0) # Shape: (Horizon+1, 3)
        # TERMINAL COST
        total_cost += 5 * self._goal_distance_critic(final_pose, goal) # Add terminal cost based on distance to goal
        return total_cost, actions, trajectory

    @partial(jit, static_argnames=("self"))
    def _propagate_humans_states(self, humans_pos_means, humans_pos_covs, humans_vel_means, humans_vel_covs, humans_scores):
        """
        Propagates the humans' states over the horizon using a simple constant velocity model.
        """
        vel_distrs = vmap(self.biv_gaussian.covariance_to_parameters)(humans_vel_covs)
        vel_distrs["means"] = humans_vel_means
        def step_fn(carry, _):
            pos_means, pos_covs = carry
            # Update positions based on velocities
            new_pos_means = pos_means + humans_vel_means * self.dt
            new_pos_covs = pos_covs + humans_vel_covs * (self.dt ** 2)  # Simple propagation of uncertainty
            new_pos_distr = vmap(self.biv_gaussian.covariance_to_parameters)(new_pos_covs)
            new_pos_distr["means"] = new_pos_means
            new_distr = {"pos_distrs": new_pos_distr, "vel_distrs": vel_distrs, "weights": humans_scores}
            return (new_pos_means, new_pos_covs), new_distr
        _, distr_seq = lax.scan(step_fn, (humans_pos_means, humans_pos_covs), None, length=self.horizon)
        return distr_seq 

    ### Public methods

    @partial(jit, static_argnames=("self"))
    def bound_action(self, action, action_space_vertices):
        """
        Bounds the action to be within the convex hull defined by the action_space_vertices.
        """
        v0, v1, v2 = action_space_vertices[0], action_space_vertices[1], action_space_vertices[2]
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        denom = jnp.where(denom == 0, 1e-8, denom) # Protezione NaN
        w1 = ((v1[1] - v2[1]) * (action[0] - v2[0]) + (v2[0] - v1[0]) * (action[1] - v2[1])) / denom
        w2 = ((v2[1] - v0[1]) * (action[0] - v2[0]) + (v0[0] - v2[0]) * (action[1] - v2[1])) / denom
        w3 = 1.0 - w1 - w2
        is_inside = (w1 >= -1e-5) & (w2 >= -1e-5) & (w3 >= -1e-5)
        def project_edge(a, b):
            ab = b - a
            ap = action - a
            t = jnp.dot(ap, ab) / jnp.maximum(jnp.dot(ab, ab), 1e-8)
            t = jnp.clip(t, 0.0, 1.0)
            return a + t * ab
        proj0 = project_edge(v0, v1)
        proj1 = project_edge(v1, v2)
        proj2 = project_edge(v2, v0)
        dist0 = jnp.sum((action - proj0)**2)
        dist1 = jnp.sum((action - proj1)**2)
        dist2 = jnp.sum((action - proj2)**2)
        min_dist = jnp.minimum(dist0, jnp.minimum(dist1, dist2))
        best_proj = jnp.where(min_dist == dist0, proj0,
                    jnp.where(min_dist == dist1, proj1, proj2))
        return jnp.where(is_inside, action, best_proj)

    @partial(jit, static_argnames=("self"))
    def cost(self, robot_pose, action, robot_goal, point_cloud, humans_positions, humans_mask):
        total_cost = 0.0
        for name, critic_fn in self.critics.items():
            weight = self.weights.get(name, 0.0)
            cost_val = critic_fn(robot_pose, action, robot_goal, point_cloud, humans_positions, humans_mask)
            total_cost = total_cost + (weight * cost_val) 
        return total_cost

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        key:random.PRNGKey,
        obs:jnp.ndarray, 
        info:dict, 
        perception_params:dict,
        actor_critic_params:dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Split key
        key, subkey = random.split(key)
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_pose = obs[0, :3]
        # Compute aligned point cloud from lidar observations - IN THE WORLD FRAME
        aligned_lidar_rf, aligned_lidar_wf = self.align_lidar(obs[:self.lidar_n_stack_to_use])
        point_cloud = jnp.reshape(aligned_lidar_wf, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        # Compute probabilistic humans states 
        perception_input, _ = self.compute_perception_input(obs)
        perception_output, scan_embeddings, spatial_attn, temporal_attn = self.perception.apply(perception_params, None, perception_input)
        perception_output_rf = perception_output.copy() # Keep a copy of the perception output in the robot frame for debugging
        # Compute actor_distribution for visualization purposes
        action_space_params = self.bound_action_space(jnp.reshape(aligned_lidar_rf, (-1, 2)))
        robot_goal_rf = jnp.array([[jnp.cos(-robot_pose[2]), -jnp.sin(-robot_pose[2])], [jnp.sin(-robot_pose[2]), jnp.cos(-robot_pose[2])]]) @(robot_goal - robot_pose[:2])
        actor_input = self.compute_actor_input(None, perception_output, action_space_params, robot_goal_rf)
        _, actor_distr, _, state_value, human_attn = self.actor_critic.apply(actor_critic_params, None, actor_input, scan_embeddings)
        # Probabilistic humans states in the world frame
        perception_output['pos_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
            perception_output['pos_distrs'], 
            robot_pose
        )
        perception_output['vel_distrs'] = vmap(self.biv_gaussian.roto_translate, in_axes=(0, None))(
            perception_output['vel_distrs'], 
            jnp.array([0, 0, robot_pose[2]]) # Velocities are not affected by translation, only rotation
        )
        # Propagate (world frame) humans states over the horizon
        humans_pos_means = perception_output['pos_distrs']['means']
        humans_pos_covs = vmap(self.biv_gaussian.covariance)(perception_output['pos_distrs'])
        humans_vel_means = perception_output['vel_distrs']['means']
        humans_vel_covs = vmap(self.biv_gaussian.covariance)(perception_output['vel_distrs'])
        humans_scores = perception_output["weights"]
        humans_distr_seq = self._propagate_humans_states(humans_pos_means, humans_pos_covs, humans_vel_means, humans_vel_covs, humans_scores)
        # humans_distr_seq = tree_map(lambda x, y: jnp.concatenate((x[None, :], y), axis=0), perception_output, humans_distr_seq)
        # Sample trajectories and compute costs
        keys = random.split(subkey, self.num_samples)
        costs, actions, trajectories = vmap(self._rollout_and_cost, in_axes=(0, None, None, None, None, None, None, None, None))(
            keys, scan_embeddings, perception_params, actor_critic_params, obs, robot_pose, robot_goal, point_cloud, humans_distr_seq
        )
        # Compute weights and update u_mean
        beta = jnp.min(costs)
        weights = jnp.exp(-(costs - beta) / self.temperature)
        weights = weights / (jnp.sum(weights) + 1e-5) # Normalize to sum 1
        control_sequence = jnp.sum(weights[:, None, None] * actions, axis=0)
        # Bound first action to be within the bounded action space
        action_space_vertices = actor_distr['vertices']  # Shape: (3, 2)
        control_sequence = control_sequence.at[0].set(self.bound_action(control_sequence[0], action_space_vertices))
        action = control_sequence[0]
        return (
            action, 
            control_sequence, 
            trajectories, 
            costs, 
            perception_output_rf, 
            {"pos_distrs": humans_distr_seq["pos_distrs"], "weights": humans_distr_seq["weights"]}, 
            actor_distr, 
            state_value, 
            spatial_attn, 
            temporal_attn, 
            human_attn, 
            key
        )    