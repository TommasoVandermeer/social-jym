from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import os

from socialjym.envs.base_env import SCENARIOS, ROBOT_KINEMATICS, HUMAN_POLICIES
from socialjym.policies.dwa import DWA
from socialjym.envs.base_env import wrap_angle
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics
from jhsfm.hsfm import get_linear_velocity

class MPPI(DWA):
    def __init__(
            self, 
            # MPPI hyperparameters
            num_samples=500, 
            horizon=20, 
            temperature=0.3, 
            noise_sigma=jnp.array([0.3, 0.9]), # Sigma for [v, w]
            # MPPI critics weights
            velocity_cost_weight = 0.5,
            goal_distance_cost_weight = 3.0,
            obstacle_cost_weight = 3.0,
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
        Model Predictive Path Integral (MPPI) control policy for navigation.
        Implementation inspired by "Model Predictive Path Integral Control: From Theory to Parallel Computation and Practice" by Williams et al. (2017).
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
        # Save MPPI hyperparameters
        self.num_samples = num_samples # K
        self.horizon = horizon         # T
        self.temperature = temperature         # Temperature
        self.noise_sigma = noise_sigma # Sigma for the noise added to the controls (shape: (2,) for [v, w])
        # Default parameters
        self.name = "MPPI"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        self.u_mean_shape = (self.horizon, 2) # Shape: (Horizon, 2) -> (20 steps, 2 controls)
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, point_cloud)
            'velocity':  lambda p, a, aidx, g, pc: self._velocity_critic(a), # Heredited from DWA
            'goal_distance':   lambda p, a, aidx, g, pc: self._goal_distance_critic(p, g),
            'obstacle_cost': lambda p, a, aidx, g, pc: self._obstacle_critic(p, aidx, pc),
            'control_cost': lambda p, a, aidx, g, pc: self._control_critic(a),
        }
        self.weights = {
            'velocity':  velocity_cost_weight,
            'goal_distance':   goal_distance_cost_weight,
            'obstacle_cost': obstacle_cost_weight,
            'control_cost': control_cost_weight,
        }

    # Private methods

    @partial(jit, static_argnames=("self"))
    def _velocity_critic(self, action):
        vmax = self.v_max - (self.v_max * jnp.abs(action[1]) / self.w_max)  # Max linear velocity for the given angular velocity, to be in the triangle defined by the kinematic constraints
        return lax.cond(
            vmax > 0,
            lambda: (vmax - action[0]) / vmax,  # Prefer higher speeds (given the maximum speed for the given angular velocity)
            lambda: 0.0 # Complete turning in place is not penalized
        )

    @partial(jit, static_argnames=("self"))
    def _obstacle_critic(self, robot_pose, action_idx, point_cloud):
        distances = jnp.linalg.norm(robot_pose[None, :2] - point_cloud, axis=1)
        min_distance = jnp.min(distances)
        clearance_cost = lax.cond(
            min_distance - self.robot_radius <= 0,
            lambda: 1_000_000.,  # Collision, assign infinite cost
            lambda: 1/min_distance,  # Prefer larger clearance (i.e. smaller cost)
        )
        return clearance_cost  # Prefer larger clearance (i.e. smaller cost)
    
    @partial(jit, static_argnames=("self"))
    def _goal_distance_critic(self, robot_pose, robot_goal):
        distance_to_goal = jnp.linalg.norm(robot_pose[:2] - robot_goal)
        return distance_to_goal
    
    @partial(jit, static_argnames=("self"))
    def _control_critic(self, action):
        return jnp.linalg.norm(action) # Prefer smaller actions (i.e. smoother trajectories)

    @partial(jit, static_argnames=("self"))
    def _rollout_and_cost(self, start_pose, controls_seq, goal, point_cloud):
        """
        Simulates a trajectory and computes the cumulative cost.
        """
        def step_fn(carry, action):
            pose, seq_idx, current_cost = carry
            next_pose = self.motion(pose, action, self.dt)
            step_cost = self.cost(pose, action, seq_idx, goal, point_cloud) 
            return (next_pose, seq_idx + 1, current_cost + step_cost), next_pose
        (final_pose, _, total_cost), trajectory = lax.scan(step_fn, (start_pose, 0, 0.0), controls_seq)
        trajectory = jnp.concatenate((start_pose[None, :], trajectory), axis=0) # Shape: (Horizon+1, 3)
        # TERMINAL COST
        total_cost += 5 * self._goal_distance_critic(final_pose, goal) # Add terminal cost based on distance to goal
        return total_cost, trajectory

    @partial(jit, static_argnames=("self"))
    def _clamp_action(self, action):
        # Action is [v, w] is clamped to be in the triangle defined by the kinematic constraints (i.e. v >= 0, w_min <= w <= w_max, and v <= v_max - (v_max/w_max) * |w|)
        # The clamping is done by connecting the origin with the original action and finding the intersection with the triangle. 
        # If the original action is inside the triangle, it is unchanged. If it is outside, it is projected onto the triangle.
        v = jnp.maximum(action[0], 0.0)
        w = action[1]
        constraint_val = (v / self.v_max) + (jnp.abs(w) / self.w_max)
        scale_factor = 1.0 / (constraint_val + 1e-5)
        final_scale = jnp.minimum(1.0, scale_factor)
        v_clamped = v * final_scale
        w_clamped = w * final_scale
        return jnp.array([v_clamped, w_clamped])

    # Public methods

    @partial(jit, static_argnames=("self"))
    def init_u_mean(self):
        return jnp.zeros(self.u_mean_shape)

    @partial(jit, static_argnames=("self"))
    def act(
        self, 
        obs:jnp.ndarray, 
        info:dict, 
        u_mean:jnp.ndarray, 
        key
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Split key
        key, subkey = random.split(key)
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_pose = obs[0, :3]
        # Compute aligned point cloud from lidar observations
        aligned_lidar = self.align_lidar(obs[:self.lidar_n_stack_to_use])[1]
        point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        # Sample control sequences and compute their costs
        noise = random.normal(subkey, (self.num_samples, self.horizon, 2)) * self.noise_sigma
        V = u_mean[None, :, :] + noise
        V_clamped = vmap(vmap(self._clamp_action))(V) # Shape: (num_samples, horizon, 2)
        costs, trajectories = vmap(self._rollout_and_cost, in_axes=(None, 0, None, None))(
            robot_pose, V_clamped, robot_goal, point_cloud
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
        return action, u_mean_clamped, trajectories, costs, key

    def evaluate(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
    ) -> dict:
        """
        Test MPPI over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "MPPI policy can only be evaluated on unicycle kinematics"
        assert env.robot_dt == self.dt, f"Environment time step (dt={env.dt}) must be equal to policy time step (dt={self.dt}) for evaluation"
        assert env.lidar_angular_range == self.lidar_angular_range, f"Environment LiDAR angular range (lidar_angular_range={env.lidar_angular_range}) must be equal to policy LiDAR angular range (lidar_angular_range={self.lidar_angular_range}) for evaluation"
        assert env.lidar_max_dist == self.lidar_max_dist, f"Environment LiDAR max distance (lidar_max_dist={env.lidar_max_dist}) must be equal to policy LiDAR max distance (lidar_max_dist={self.lidar_max_dist}) for evaluation"
        assert env.lidar_num_rays == self.lidar_num_rays, f"Environment LiDAR number of rays (lidar_num_rays={env.lidar_num_rays}) must be equal to policy LiDAR number of rays (lidar_num_rays={self.lidar_num_rays}) for evaluation"
        time_limit = env.reward_function.time_limit
        @loop_tqdm(n_trials)
        @jit
        def _fori_body(i:int, for_val:tuple):   
            @jit
            def _while_body(while_val:tuple):
                # Retrieve data from the tuple
                state, obs, info, outcome, u_mean, policy_key, env_key, steps, all_actions, all_states = while_val
                action, u_mean, _, _, policy_key = self.act(obs, info, u_mean, policy_key)
                state, obs, info, _, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, u_mean, policy_key, env_key, steps, all_actions, all_states

            ## Retrieve data from the tuple
            seed, metrics = for_val
            policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
            env_key = random.PRNGKey(seed + 1_000_000)
            ## Reset the environment
            state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
            initial_robot_position = state[-1,:2]
            ## Episode loop
            all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
            all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
            while_val_init = (state, obs, info, init_outcome, self.init_u_mean(), policy_key, env_key, 0, all_actions, all_states)
            _, _, end_info, outcome, _, _, _, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
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

    def animate_trajectory(
        self,
        robot_poses, # x, y, theta
        robot_actions,
        robot_u_means,
        robot_trajectories, # Sampled trajectories (num_steps, num_samples, horizon+1, 3)
        robot_trajectories_costs,
        robot_goals,
        observations,
        humans_poses, # x, y, theta
        humans_velocities, # vx, vy (in global frame)
        humans_radii,
        static_obstacles,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        save_video:bool=False,
    ):
        # Validate input args
        assert \
            len(robot_poses) == \
            len(robot_actions) == \
            len(robot_u_means) == \
            len(robot_trajectories) == \
            len(robot_trajectories_costs) == \
            len(robot_goals) == \
            len(observations) == \
            len(humans_poses) == \
            len(humans_velocities) == \
            len(humans_radii) == \
            len(static_obstacles), "All inputs must have the same length"
        n_steps = len(robot_poses)
        # Set matplotlib fonts
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Compute full trajectories with U_mean
        @jit
        def compute_chosen_trajectory(robot_pose, robot_u_mean, robot_goal, obs):
            aligned_lidar = self.align_lidar(obs[:self.lidar_n_stack_to_use])[1]
            point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
            _, chosen_trajectory = self._rollout_and_cost(robot_pose, robot_u_mean, robot_goal, point_cloud)
            return chosen_trajectory
        chosen_trajectories = vmap(compute_chosen_trajectory)(robot_poses, robot_u_means, robot_goals, observations)
        # Compute first sampled actions at each frame
        @jit
        def inverse_kinematics(robot_pose, next_robot_pose):
            dtheta = wrap_angle(next_robot_pose[2] - robot_pose[2])
            w = dtheta / self.dt
            dx = next_robot_pose[0] - robot_pose[0]
            dy = next_robot_pose[1] - robot_pose[1]
            theta_mid = robot_pose[2] + dtheta / 2.0
            chord_len = dx * jnp.cos(theta_mid) + dy * jnp.sin(theta_mid)
            v = lax.cond(
                jnp.abs(w) > 1e-5,
                lambda: chord_len * w / (2.0 * jnp.sin(dtheta / 2.0)),
                lambda: (dx * jnp.cos(robot_pose[2]) + dy * jnp.sin(robot_pose[2])) / self.dt,
            )
            return jnp.array([v, w])
        first_sampled_actions = vmap(vmap(inverse_kinematics, in_axes=(None, 0)))(robot_poses, robot_trajectories[:,:,1]) # Shape: (n_steps-1, 2)
        # Animate trajectory
        fig = plt.figure(figsize=(21.43,13.57))
        fig.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        outer_gs = fig.add_gridspec(1, 2, width_ratios=[2, 0.4], wspace=0.09)
        gs_left = outer_gs[0].subgridspec(1, 2, wspace=0.0, hspace=0.0)
        axs = [
            fig.add_subplot(gs_left[0]), # Simulation + LiDAR ranges (Top-Left)
            fig.add_subplot(gs_left[1]), # Simulation + Point cloud (Top-Right)
            fig.add_subplot(outer_gs[1]),   # Action space (Right, tall)
        ]
        def animate(frame):
            for i, ax in enumerate(axs):
                ax.clear()
                if i == len(axs) - 1: continue
                ax.set(xlim=x_lims if x_lims is not None else [-10,10], ylim=y_lims if y_lims is not None else [-10,10])
                ax.set_xlabel('X', labelpad=-5)
                if i % 2 == 0:
                    ax.set_ylabel('Y', labelpad=-13)
                else:
                    ax.set_yticks([])
                ax.set_aspect('equal', adjustable='datalim')
                # Plot humans
                for h in range(len(humans_poses[frame])):
                    head = plt.Circle((humans_poses[frame][h,0] + jnp.cos(humans_poses[frame][h,2]) * humans_radii[frame][h], humans_poses[frame][h,1] + jnp.sin(humans_poses[frame][h,2]) * humans_radii[frame][h]), 0.1, color='black', alpha=0.6, zorder=1)
                    ax.add_patch(head)
                    circle = plt.Circle((humans_poses[frame][h,0], humans_poses[frame][h,1]), humans_radii[frame][h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                    ax.add_patch(circle)
                # Plot human velocities
                for h in range(len(humans_poses[frame])):
                    ax.arrow(
                        humans_poses[frame][h,0],
                        humans_poses[frame][h,1],
                        humans_velocities[frame][h,0],
                        humans_velocities[frame][h,1],
                        head_width=0.15,
                        head_length=0.15,
                        fc='blue',
                        ec='blue',
                        alpha=0.6,
                        zorder=30,
                    )
                # Plot robot
                robot_position = robot_poses[frame,:2]
                head = plt.Circle((robot_position[0] + self.robot_radius * jnp.cos(robot_poses[frame,2]), robot_position[1] + self.robot_radius * jnp.sin(robot_poses[frame,2])), 0.1, color='black', zorder=1)
                ax.add_patch(head)
                circle = plt.Circle((robot_position[0], robot_position[1]), self.robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
                ax.add_patch(circle)
                # Plot robot goal
                ax.plot(
                    robot_goals[frame][0],
                    robot_goals[frame][1],
                    marker='*',
                    markersize=7,
                    color='red',
                    zorder=5,
                )
                # Plot static obstacles
                if static_obstacles[frame].shape[1] > 1: # Polygon obstacles
                    for o in static_obstacles[frame]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
                else: # One segment obstacles
                    for o in static_obstacles[frame]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
            ### FIRST ROW AXS: SIMULATION + INPUT VISUALIZATION
            # AX 0,0: Simulation with LiDAR ranges
            lidar_scan = observations[frame,0,6:]
            for ray in range(len(lidar_scan)):
                axs[0].plot(
                    [robot_poses[frame,0], robot_poses[frame,0] + lidar_scan[ray] * jnp.cos(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    [robot_poses[frame,1], robot_poses[frame,1] + lidar_scan[ray] * jnp.sin(robot_poses[frame,2] + self.lidar_angles_robot_frame[ray])],
                    color="black", 
                    linewidth=0.5, 
                    zorder=0
                )
            axs[0].set_title("Trajectory")
            # AX 0,1: Simulation with LiDAR point cloud stack and DWA possible paths
            point_cloud = self.align_lidar(observations[frame])[1][:self.lidar_n_stack_to_use].reshape(-1, 2)
            if len(point_cloud.shape) == 2:
                point_cloud = point_cloud[None, ...]
            for j, cloud in enumerate(point_cloud):
                # color/alpha fade with j (smaller j -> less faded)
                t = (1 - j / (self.n_stack - 1))  # in [0,1]
                axs[1].scatter(
                    cloud[:,0],
                    cloud[:,1],
                    c=0.3 + 0.7 * jnp.ones((self.lidar_num_rays,)) * t,
                    cmap='Reds',
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.3 + 0.7 * t,
                    zorder=20 + self.n_stack - j,
                )
            axs[1].set_title("Pointcloud & Paths")
            for i, trajectory in enumerate(robot_trajectories[frame]):
                cost = robot_trajectories_costs[frame][i]
                if cost < 1_000_000:
                    axs[1].plot(trajectory[:,0], trajectory[:,1], color='green', alpha=0.5, linewidth=1, zorder=10)
            axs[1].plot(chosen_trajectories[frame][:,0], chosen_trajectories[frame][:,1], color='blue', alpha=1, linewidth=3, zorder=30)
            # AX :,2: Feasible and bounded action space + action space distribution and action taken
            axs[2].set_xlabel("$v$ (m/s)")
            axs[2].set_ylabel("$\omega$ (rad/s)", labelpad=-15)
            axs[2].set_xlim(-0.1, self.v_max + 0.1)
            axs[2].set_ylim(-2*self.v_max/self.wheels_distance - 0.3, 2*self.v_max/self.wheels_distance + 0.3)
            axs[2].set_xticks(jnp.arange(0, self.v_max+0.2, 0.2))
            axs[2].set_xticklabels([round(i,1) for i in jnp.arange(0, self.v_max, 0.2)] + [r"$\overline{v}$"])
            axs[2].set_yticks(jnp.arange(-2,3,1).tolist() + [2*self.v_max/self.wheels_distance,-2*self.v_max/self.wheels_distance])
            axs[2].set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
            axs[2].grid()
            axs[2].add_patch(
                plt.Polygon(
                    [   
                        [0,2*self.v_max/self.wheels_distance],
                        [0,-2*self.v_max/self.wheels_distance],
                        [self.v_max,0],
                    ],
                    closed=True,
                    fill=True,
                    edgecolor='green',
                    facecolor='lightgreen',
                    linewidth=2,
                    zorder=2,
                ),
            )
            axs[2].scatter(
                first_sampled_actions[frame,:,0],
                first_sampled_actions[frame,:,1],
                color='orange',
                alpha=0.5,
                zorder=10,
                s=5,
            )
            axs[2].plot(robot_actions[frame,0], robot_actions[frame,1], marker='^',markersize=9,color='blue',zorder=51) # Action taken
        anim = FuncAnimation(fig, animate, interval=self.dt*1000, frames=n_steps)
        if save_video:
            save_path = os.path.join(os.path.dirname(__file__), f'jessi_trajectory.mp4')
            writer_video = FFMpegWriter(fps=int(1/self.dt), bitrate=1800)
            anim.save(save_path, writer=writer_video, dpi=300)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()

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
        self.animate_trajectory(
            robot_poses,
            actions,
            u_means,
            trajectories,
            trajectories_costs,
            goals,
            observations,
            humans_poses,
            humans_velocities,
            humans_radii,
            static_obstacles,
            x_lims,
            y_lims,
            save_video,
        )
    