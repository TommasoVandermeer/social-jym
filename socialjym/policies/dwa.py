from jax import jit, lax, vmap, debug, random
from jax_tqdm import loop_tqdm
import jax.numpy as jnp
from functools import partial

from socialjym.envs.base_env import SCENARIOS, ROBOT_KINEMATICS
from socialjym.policies.base_policy import BasePolicy
from socialjym.envs.base_env import wrap_angle
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.aux_functions import compute_episode_metrics, initialize_metrics_dict, print_average_metrics

class DWA(BasePolicy):
    def __init__(
        self,
        actions_discretization = 9,
        predict_time_horizon = .75,
        heading_cost_coeff = 0.2,
        clearance_cost_coeff = 0.2,
        velocity_cost_coeff = 0.2,
        distance_cost_coeff = 0.1,
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
        self.distance_cost_coeff = distance_cost_coeff
        self.robot_radius = robot_radius
        self.v_max = v_max
        self.dt = dt
        self.n_steps = int(predict_time_horizon // dt)
        self.wheels_distance = wheels_distance
        self.n_stack = n_stack
        self.lidar_angular_range = lidar_angular_range
        self.lidar_max_dist = lidar_max_dist
        self.lidar_num_rays = lidar_num_rays
        if lidar_angles_robot_frame is None:
            self.lidar_angles_robot_frame = jnp.linspace(-lidar_angular_range/2, lidar_angular_range/2, lidar_num_rays)
        else:
            assert len(lidar_angles_robot_frame) == lidar_num_rays, "Length of lidar_angles_robot_frame must be equal to lidar_num_rays"
            self.lidar_angles_robot_frame = lidar_angles_robot_frame
        self.lidar_n_stack_to_use = lidar_n_stack_to_use
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
    def _velocity_cost(self, action):
        return self.v_max - action[0]  # Prefer higher speeds
    
    @partial(jit, static_argnames=("self"))
    def _heading_cost(self, robot_pose, action, robot_goal):
        next_robot_pose = self.motion(robot_pose, action, self.predict_time_horizon)
        goal_direction = jnp.atan2(robot_goal[1] - next_robot_pose[1], robot_goal[0] - next_robot_pose[0])
        heading_error = wrap_angle(goal_direction - next_robot_pose[2])
        return jnp.abs(heading_error) / jnp.pi  # Prefer smaller heading error

    @partial(jit, static_argnames=("self"))
    def _distance_cost(self, robot_pose, action, robot_goal):
        next_robot_pose = self.motion(robot_pose, action, self.predict_time_horizon)
        distance_to_goal = jnp.linalg.norm(robot_goal - next_robot_pose[:2])
        return lax.cond(
            distance_to_goal <= 1.5,
            lambda: distance_to_goal/1.5,  # Prefer smaller distance to goal when close to it
            lambda: 0.,  # Ignore distance to goal when far from it, to prioritize obstacle avoidance
        )
    
    @partial(jit, static_argnames=("self"))
    def _clearance_cost(self, pose, action, point_cloud):
        # Predict robot trajectory for the given action
        robot_poses = jnp.empty((self.n_steps+1, 3))
        robot_poses = robot_poses.at[0].set(pose)
        robot_poses = lax.fori_loop(
            1,
            self.n_steps+1,
            lambda i, x: x.at[i].set(self.motion(x[i-1], action, self.dt)),
            robot_poses)
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

    @partial(jit, static_argnames=("self"))
    def _dwa_cost(self, robot_pose, action, robot_goal, point_cloud):
        velocity_cost = self._velocity_cost(action)
        heading_cost = self._heading_cost(robot_pose, action, robot_goal)
        clearance_cost = self._clearance_cost(robot_pose, action, point_cloud)
        distance_cost = self._distance_cost(robot_pose, action, robot_goal)
        total_cost = (
            self.velocity_cost_coeff * velocity_cost +
            self.heading_cost_coeff * heading_cost +
            self.clearance_cost_coeff * clearance_cost + 
            self.distance_cost_coeff * distance_cost
        )
        return total_cost

    # Public methods

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
        actions_costs = vmap(self._dwa_cost, in_axes=(None, 0, None, None))(robot_pose, self.action_space, robot_goal, point_cloud)
        best_action_idx = jnp.nanargmin(actions_costs)
        best_action = self.action_space[best_action_idx]
        return best_action, actions_costs[best_action_idx]
    
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