from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import os

from socialjym.envs.base_env import SCENARIOS, ROBOT_KINEMATICS, HUMAN_POLICIES
from socialjym.policies.base_policy import BasePolicy
from socialjym.envs.base_env import wrap_angle
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics
from jhsfm.hsfm import get_linear_velocity

class DWA(BasePolicy):
    def __init__(
        self,
        actions_discretization = 9,
        predict_time_horizon = .5,
        heading_cost_coeff = 0.126,
        clearance_cost_coeff = 0.2,
        velocity_cost_coeff = 0.2,
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
        use_box_action_space=True, # If True, the action space for which the DWA cost is computed is the box action space (i.e. all combinations of linear and angular speeds up to the maximum) but then the action is bounded in the triangle, otherwise it is the discretized action space (i.e. a subset of the box action space). Using the box action space is more computationally expensive, but allows to find better actions.
    ):
        """
        Dynamic Window Approach (DWA) policy for navigation.
        Implementation inspired by https://github.com/goktug97/DynamicWindowApproach
        """
        # Input validation
        assert robot_radius > 0, "Robot radius must be positive"
        assert v_max > 0, "Maximum robot velocity must be positive"
        assert gamma >= 0 and gamma <= 1, "Discount factor must be in [0, 1]"
        assert dt > 0, "Time step must be positive"
        assert wheels_distance > 0, "Wheels distance must be positive"
        assert n_stack >= 2, "Number of stacked observations must be at least 2, to observe motion"
        assert lidar_angular_range > 0 and lidar_angular_range <= 2*jnp.pi, "LiDAR angular range must be in (0, 2pi]"
        assert lidar_max_dist > 1, "LiDAR maximum distance must be greater than 1 meter"
        assert lidar_num_rays >= 10, "LiDAR number of rays must be at least 10"
        assert lidar_n_stack_to_use >= 1, "Number of lidar scans to use must be at least 1"
        assert lidar_n_stack_to_use <= n_stack, "Number of lidar scans to use cannot be greater than n_stack"
        assert predict_time_horizon > 0, "Predict time horizon must be positive"
        assert predict_time_horizon % dt == 0, "Predict time horizon must be a multiple of dt"
        assert actions_discretization >= 3, "Actions discretization must be at least 2"
        # Initialize policy parameters
        super().__init__(discount=gamma)
        self.predict_time_horizon = predict_time_horizon
        self.heading_cost_coeff = heading_cost_coeff
        self.clearance_cost_coeff = clearance_cost_coeff
        self.velocity_cost_coeff = velocity_cost_coeff
        self.robot_radius = robot_radius
        self.v_max = v_max
        self.dt = dt
        self.n_steps = int(predict_time_horizon // dt)
        self.wheels_distance = wheels_distance
        self.w_max = 2 * v_max / wheels_distance
        self.w_min = - self.w_max
        self.n_stack = n_stack
        self.lidar_angular_range = lidar_angular_range
        self.lidar_max_dist = lidar_max_dist
        self.lidar_num_rays = lidar_num_rays
        self.use_box_action_space = use_box_action_space
        if lidar_angles_robot_frame is None:
            self.lidar_angles_robot_frame = jnp.linspace(-lidar_angular_range/2, lidar_angular_range/2, lidar_num_rays)
        else:
            assert len(lidar_angles_robot_frame) == lidar_num_rays, "Length of lidar_angles_robot_frame must be equal to lidar_num_rays"
            self.lidar_angles_robot_frame = lidar_angles_robot_frame
        self.lidar_n_stack_to_use = lidar_n_stack_to_use
        # Default attributes
        self.name = "DWA"
        self.kinematics = ROBOT_KINEMATICS.index("unicycle")
        # Compute action space
        self.actions_discretization = actions_discretization
        angular_speeds = jnp.linspace(-self.v_max/(self.wheels_distance/2), self.v_max/(self.wheels_distance/2), 2*actions_discretization-1)
        speeds = jnp.linspace(0, self.v_max, actions_discretization)
        unconstrained_action_space = jnp.empty((len(angular_speeds)*len(speeds),2))
        unconstrained_action_space = lax.fori_loop(
            0,
            len(angular_speeds),
            lambda i, x: lax.fori_loop(
                0,
                len(speeds),
                lambda j, y: lax.cond(
                    jnp.all(jnp.array([i<len(angular_speeds)-j, i>=j])),
                    lambda z: z.at[i*len(speeds)+j].set(jnp.array([speeds[j],angular_speeds[i]])),
                    lambda z: z.at[i*len(speeds)+j].set(jnp.array([jnp.nan,jnp.nan])),
                    y),
                x),
            unconstrained_action_space)
        self.action_space = unconstrained_action_space[~jnp.isnan(unconstrained_action_space).any(axis=1)]
        self.box_action_space = lax.fori_loop(
            0,
            len(angular_speeds),
            lambda i, x: lax.fori_loop(
                0,
                len(speeds),
                lambda j, z: z.at[i*len(speeds)+j].set(jnp.array([speeds[j],angular_speeds[i]])),
                x
            ),
            unconstrained_action_space
        )
        # Compute delta trajectories for each action in the action space, to speed up computations in the critics
        def compute_trajectory(action):
            return lax.fori_loop(
                1,
                self.n_steps+1,
                lambda i, x: x.at[i].set(self.motion(x[i-1], action, self.dt)),
                jnp.zeros((self.n_steps+1, 3)),
            )
        if self.use_box_action_space:
            self.delta_trajectories = vmap(compute_trajectory)(self.box_action_space)
            self.action_idxs = jnp.arange(len(self.box_action_space))
        else:
            self.delta_trajectories = vmap(compute_trajectory)(self.action_space)
            self.action_idxs = jnp.arange(len(self.action_space))
        # Initialize critics and weights for the cost function
        self.critics = {
            # Inputs are (current_robot_pose, action, action_idx, robot_goal, point_cloud)
            'velocity':  lambda p, a, aidx, g, pc: self._velocity_critic(a),
            'goal_heading':   lambda p, a, aidx, g, pc: self._goal_heading_critic(p, a, g),
            'clearance': lambda p, a, aidx, g, pc: self._clearance_critic(p, aidx, pc),
        }
        self.weights = {
            'velocity': velocity_cost_coeff,
            'goal_heading': heading_cost_coeff,
            'clearance': clearance_cost_coeff,
        }
           
    # Private methods

    @partial(jit, static_argnames=("self"))
    def _align_lidar_stack(self, obs_stack, ref_position, ref_orientation):
        """
        args:
        - obs_stack (lidar_num_rays + 6):  [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].

        outputs:
        - pointcloud_and_action (lidar_num_rays, 2): LiDAR points in robot reference frame
        - pointcloud_world_frame (lidar_num_rays, 2): LiDAR points in world frame
        """
        ## Split obs stack
        robot_position = obs_stack[:2]  # Shape: (2,)
        robot_orientation = obs_stack[2]  # Shape: ()
        #robot_radius = obs_stack[3]  # Shape: ()
        #robot_action = obs_stack[4:6]  # Shape: (2,)
        lidar_measurements = obs_stack[6:]  # Shape: (lidar_num_rays)
        ## Align scan to reference frame
        # Compute LiDAR angles in world frame
        lidar_angles = self.lidar_angles_robot_frame + robot_orientation  # Shape: (lidar_num_rays)
        # Compute cartesian coordinates of LiDAR points in world frame
        xs = lidar_measurements * jnp.cos(lidar_angles) + robot_position[0]
        ys = lidar_measurements * jnp.sin(lidar_angles) + robot_position[1]
        points_world = jnp.stack((xs, ys), axis=-1)  # Shape: (lidar_num_rays, 2)
        # Roto-translate points to robot frame
        c, s = jnp.cos(ref_orientation), jnp.sin(ref_orientation)
        R = jnp.array([
            [c, -s],
            [s,  c]
        ])
        points_robot = jnp.dot(points_world - ref_position, R)
        return points_robot, points_world

    @partial(jit, static_argnames=("self"))
    def _velocity_critic(self, action):
        if self.use_box_action_space:
            return (self.v_max - action[0]) / self.v_max  # Prefer higher speeds
        else:
            vmax = self.v_max - (self.v_max * jnp.abs(action[1]) / self.w_max)  # Max linear velocity for the given angular velocity, to be in the triangle defined by the kinematic constraints
            return lax.cond(
                vmax > 0,
                lambda: (vmax - action[0]) / vmax,  # Prefer higher speeds (given the maximum speed for the given angular velocity)
                lambda: 0.0 # Complete turning in place is not penalized
            )
    
    @partial(jit, static_argnames=("self"))
    def _goal_heading_critic(self, robot_pose, action, robot_goal):
        next_robot_pose = self.motion(robot_pose, action, self.predict_time_horizon)
        goal_direction = jnp.atan2(robot_goal[1] - next_robot_pose[1], robot_goal[0] - next_robot_pose[0])
        heading_error = wrap_angle(goal_direction - next_robot_pose[2])
        return jnp.abs(heading_error) / jnp.pi  # Prefer smaller heading error
    
    @partial(jit, static_argnames=("self"))
    def _clearance_critic(self, robot_pose, action_idx, point_cloud):
        # Predict robot trajectory for the given action
        c, s = jnp.cos(robot_pose[2]), jnp.sin(robot_pose[2])
        rot = jnp.array([[c, -s], [s, c]])
        robot_poses =  robot_pose[:2] + jnp.dot(self.delta_trajectories[action_idx,:,:2], rot.T)
        # Compute distance from predicted trajectory to each point in the point cloud
        def min_distance_to_trajectory(point):
            distances = jnp.linalg.norm(robot_poses[1:, :2] - point[None, :], axis=1)
            return jnp.min(distances)
        distances = vmap(min_distance_to_trajectory)(point_cloud)
        min_distance = jnp.min(distances)
        clearance_cost = lax.cond(
            min_distance - self.robot_radius <= 0,
            lambda: jnp.inf,  # Collision, assign infinite cost
            lambda: 1/min_distance,  # Prefer larger clearance (i.e. smaller cost)
        )
        return clearance_cost  # Prefer larger clearance (i.e. smaller cost)

    # Public methods

    @partial(jit, static_argnames=("self"))
    def cost(self, robot_pose, action, action_idx, robot_goal, point_cloud):
        total_cost = 0.0
        for name, critic_fn in self.critics.items():
            weight = self.weights.get(name, 0.0)
            cost_val = critic_fn(robot_pose, action, action_idx, robot_goal, point_cloud)
            total_cost = total_cost + (weight * cost_val) 
        return total_cost

    @partial(jit, static_argnames=("self"))
    def align_lidar(
        self,
        obs:jnp.ndarray,
    ):
        """
        Align lidar scans in the observation stacks to the robot frame of the most recent observation.
        Prepare the input for the encoder network.

        args:
        - obs (n_stack, lidar_num_rays + 6): Each stack [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        The first stack is the most recent one.

        output:
        - processed_obs (n_stack, lidar_num_rays * 2): aligned LiDAR stack. First information corresponds to the most recent observation.
        """
        ref_position = obs[0,:2]
        ref_orientation = obs[0,2]
        return vmap(DWA._align_lidar_stack, in_axes=(None, 0, None, None))(self, obs, ref_position, ref_orientation)

    @partial(jit, static_argnames=("self"))
    def motion(
        self,
        pose:jnp.ndarray,
        action:jnp.ndarray,
        dt:float,
    ) -> jnp.ndarray:
        new_pose = lax.cond(
            jnp.abs(action[1]) > 1e-5,
            lambda _: jnp.array([
                pose[0]+(action[0]/action[1])*(jnp.sin(pose[2]+action[1]*dt)-jnp.sin(pose[2])),
                pose[1]+(action[0]/action[1])*(jnp.cos(pose[2])-jnp.cos(pose[2]+action[1]*dt)),
                wrap_angle(pose[2]+action[1]*dt),
            ]),
            lambda _: jnp.array([
                pose[0]+action[0]*dt*jnp.cos(pose[2]),
                pose[1]+action[0]*dt*jnp.sin(pose[2]),
                pose[2],
            ]),
            None)
        return new_pose

    @partial(jit, static_argnames=("self"))
    def act(
        self,
        obs:jnp.ndarray, 
        info:dict,
    ) -> tuple[jnp.ndarray, float]:
        """
        Compute the action to take given the current observation and info (which contains the robot goal).

        Args:
            obs: (n_stack, lidar_num_rays + 6) The current observation, which includes the robot state and lidar scans.
            info: (dict) Additional information, including the robot goal.

        Returns:
            action: (2,) The action to take, consisting of [linear_velocity, angular_velocity].
            action_cost: The cost of the chosen action, according to the DWA cost function.
        """
        # Extract robot goal from info
        robot_goal = info['robot_goal']
        # Extract current robot state from observation
        robot_pose = obs[0, :3]
        # Compute point cloud 
        aligned_lidar = self.align_lidar(obs[:self.lidar_n_stack_to_use])[1]
        point_cloud = jnp.reshape(aligned_lidar, (-1, 2))  # Shape: (lidar_n_stack_to_use * lidar_num_rays, 2)
        # Compute action using DWA
        if self.use_box_action_space:          
            actions_costs = vmap(self.cost, in_axes=(None, 0, 0, None, None))(robot_pose, self.box_action_space, self.action_idxs, robot_goal, point_cloud)
            best_action_box_idx = jnp.nanargmin(actions_costs)
            best_action_box = self.box_action_space[best_action_box_idx]
            best_action = self.action_space[jnp.nanargmin(jnp.linalg.norm(self.action_space - best_action_box, axis=1))]
        else:
            actions_costs = vmap(self.cost, in_axes=(None, 0, 0, None, None))(robot_pose, self.action_space, self.action_idxs, robot_goal, point_cloud)
            best_action_idx = jnp.nanargmin(actions_costs)
            best_action = self.action_space[best_action_idx]
        return best_action, actions_costs
    
    def evaluate(
        self,
        n_trials:int,
        random_seed:int,
        env:LaserNav,
    ) -> dict:
        """
        Test DWA over n_trials episodes and compute relative metrics.
        """
        assert isinstance(env, LaserNav), "Environment must be an instance of LaserNav"
        assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "DWA policy can only be evaluated on unicycle kinematics"
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
                state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states = while_val
                action, _ = self.act(obs, info)
                state, obs, info, _, outcome, (_, env_key) = env.step(state,info,action,test=True,env_key=env_key)    
                # Save data
                all_actions = all_actions.at[steps].set(action)
                all_states = all_states.at[steps].set(state)
                # Update step counter
                steps += 1
                return state, obs, info, outcome, policy_key, env_key, steps, all_actions, all_states

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
            while_val_init = (state, obs, info, init_outcome, policy_key, env_key, 0, all_actions, all_states)
            _, _, end_info, outcome, policy_key, env_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
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
        robot_actions_costs,
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
            len(robot_goals) == \
            len(observations) == \
            len(humans_poses) == \
            len(humans_velocities) == \
            len(humans_radii) == \
            len(static_obstacles), "All inputs must have the same length"
        # Set matplotlib fonts
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Compute informations for visualization
        n_steps = len(robot_poses)
        def compute_trajectory(action):
            return lax.fori_loop(
                1,
                self.n_steps+1,
                lambda i, x: x.at[i].set(self.motion(x[i-1], action, self.dt)),
                jnp.zeros((self.n_steps+1, 3)),
            )
        delta_trajectories = vmap(compute_trajectory)(self.action_space)
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
            actions_cost = robot_actions_costs[frame]
            feasible_actions_idxs = ~jnp.isinf(actions_cost) 
            if self.use_box_action_space:
                feasible_actions = self.box_action_space[feasible_actions_idxs]
            else:
                feasible_actions = self.action_space[feasible_actions_idxs]
            feasible_actions_cost = actions_cost[feasible_actions_idxs]
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
            c, s = jnp.cos(robot_poses[frame,2]), jnp.sin(robot_poses[frame,2])
            rot = jnp.array([[c, -s], [s, c]])
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
            for action in feasible_actions:
                if (action == self.action_space).all(axis=1).any():
                    idx = jnp.argmax((action == self.action_space).all(axis=1))
                    trajectory =  robot_poses[frame,:2] + jnp.dot(delta_trajectories[idx,:,:2], rot.T)
                    is_best_action = jnp.array_equal(action, robot_actions[frame])
                    color = 'blue' if is_best_action else 'green'
                    s = 3 if is_best_action else 1
                    alpha = 1 if is_best_action else 0.5
                    axs[1].plot(trajectory[:,0], trajectory[:,1], color=color, alpha=alpha, linewidth=s, zorder=10)
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
                feasible_actions[:,0],
                feasible_actions[:,1],
                c=1/feasible_actions_cost,
                cmap='Reds',
                s=20,
                alpha=0.7,
                label='Feasible actions',
                zorder=3,
            )
            axs[2].plot(robot_actions[frame,0], robot_actions[frame,1], marker='^',markersize=9,color='blue',zorder=51) # Action taken
            if self.use_box_action_space:
                min_cost_box_action = feasible_actions[jnp.nanargmin(feasible_actions_cost)]
                axs[2].plot(min_cost_box_action[0], min_cost_box_action[1], marker='^',markersize=9,color='darkorange',zorder=50) # Best action in the box action space
                axs[2].plot([robot_actions[frame,0], min_cost_box_action[0]], [robot_actions[frame,1], min_cost_box_action[1]], color='darkorange', linestyle='--', zorder=49) # Line between best box action and action taken
        anim = FuncAnimation(fig, animate, interval=self.dt*1000, frames=n_steps)
        if save_video:
            save_path = os.path.join(os.path.dirname(__file__), f'{self.name}_trajectory.mp4')
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
        actions_costs,
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
            actions_costs,
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