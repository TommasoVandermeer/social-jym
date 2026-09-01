import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from .base_env import BaseEnv, SCENARIOS, ENVIRONMENTS, is_multiple

class LaserNav(BaseEnv):
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL. 
    The robot senses the environment through a 2D LiDAR. 
    Humans move according the Headed Social Force Model (HSFM).
    Humans legs dynamics are also simulated.
    LiDAR rays collide with humans legs and static obstacles.

    Suitable for policies: JESSI
    """
    def __init__(
            self, 
            robot_radius:float, 
            robot_dt:float, 
            humans_dt:float, 
            scenario:str, 
            n_humans:int, 
            n_obstacles:int,
            reward_function:FunctionType,
            robot_visible=False, 
            circle_radius=7, 
            traffic_height=3, 
            traffic_length=14,
            crowding_square_side=14,
            hybrid_scenario_subset=jnp.arange(0, len(SCENARIOS)-1, dtype=jnp.int32),
            n_stack=5,
            lidar_angular_range=2*jnp.pi,
            lidar_max_dist=10.,
            lidar_num_rays=100,
            lidar_noise=False,
            lidar_noise_fixed_std=0.01,  # 1cm base noise
            lidar_noise_proportional_std=0.01, # 1% of the distance noise
            lidar_salt_and_pepper_prob=0.03, # 3% of the rays are affected by salt and pepper noise
            lidar_dt=None, # If None, LiDAR is updated at every environment step. Otherwise, it is updated according to the specified frequency (in Hz).
            odometry_dt=None, # If None, Odometry is updated at every environment step. Otherwise, it is updated according to the specified frequency (in Hz).
            velocity_dynamics="coupled_slew_rate",
            tau_linear_velocity=0.0, # To model linear velocity dynamics as a first order system
            tau_angular_velocity=0.0, # To model angular velocity dynamics as a second order system
            wheels_max_linear_acceleration=jnp.inf,
            wheels_distance=0.,
            control_delay_mean=0.0, # Mean of the control delay distribution (assumed to be Gaussian)
            control_delay_sigma=0.0, # Standard deviation of the control delay distribution (assumed to be Gaussian)
            kinematics='unicycle',
            max_cc_delay = 5.,
            ccso_n_static_humans:int = 3,
            ccso_static_humans_radius_mean:float = 1.,
            ccso_static_humans_radius_std:float = 0.2,
            thick_default_obstacle:bool = True,
            obstacles_noise:float = 0.,
            noisy_walls:bool = False,
            grid_map_computation:bool = False,
            grid_cell_size:float = 0.9, # Such parameter is suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT)
            grid_min_size:float = 18., # Such parameter is the minimum suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT) in order to always include all static obstacles, the robot and its goal.
            leg_dynamics:bool = False,
        ) -> None:
        ## BaseEnv initialization
        super().__init__(
            robot_radius=robot_radius,
            robot_dt=robot_dt,
            humans_dt=humans_dt,
            n_humans=n_humans,
            n_obstacles=n_obstacles,
            scenario=scenario,
            humans_policy='hsfm',
            robot_visible=robot_visible,
            circle_radius=circle_radius,
            traffic_height=traffic_height,
            traffic_length=traffic_length,
            crowding_square_side=crowding_square_side,
            hybrid_scenario_subset=hybrid_scenario_subset,
            lidar_angular_range=lidar_angular_range,
            lidar_max_dist=lidar_max_dist,
            lidar_num_rays=lidar_num_rays,
            lidar_noise=lidar_noise,
            lidar_noise_fixed_std=lidar_noise_fixed_std,
            lidar_noise_proportional_std=lidar_noise_proportional_std,
            lidar_salt_and_pepper_prob=lidar_salt_and_pepper_prob,
            velocity_dynamics=velocity_dynamics,
            tau_action_0=tau_linear_velocity,
            tau_action_1=tau_angular_velocity,
            wheels_max_linear_acceleration=wheels_max_linear_acceleration,
            wheels_distance=wheels_distance,
            control_delay_mean=control_delay_mean,
            control_delay_sigma=control_delay_sigma,
            kinematics=kinematics,
            max_cc_delay=max_cc_delay,
            ccso_n_static_humans=ccso_n_static_humans,
            ccso_static_humans_radius_mean=ccso_static_humans_radius_mean,
            ccso_static_humans_radius_std=ccso_static_humans_radius_std,
            grid_map_computation=grid_map_computation,
            grid_cell_size=grid_cell_size,
            grid_min_size=grid_min_size,
            thick_default_obstacle=thick_default_obstacle,
            obstacles_noise=obstacles_noise,
            noisy_walls=noisy_walls,
            leg_dynamics=leg_dynamics,
        )
        ## Args validation
        assert reward_function.kinematics == self.kinematics, "The reward function's kinematics must be the same as the environment's kinematics."
        assert n_stack >=1, "The number of stacked observations must be at least 1."
        if lidar_dt is None:
            self.lidar_dt = self.robot_dt
            self.lidar_misalignment = False
        else:
            self.lidar_dt = lidar_dt
            self.lidar_misalignment = True
        assert is_multiple(self.lidar_dt, humans_dt), "The LiDAR update frequency must be a multiple of simulation frequency."
        assert self.lidar_dt <= self.robot_dt, "The LiDAR update frequency must be higher than or equal to the robot control frequency."
        if odometry_dt is None:
            self.odometry_dt = self.lidar_dt
            self.odometry_misalignment = False
        else:
            self.odometry_dt = odometry_dt
            self.odometry_misalignment = True
        assert is_multiple(self.odometry_dt, humans_dt), "The Odometry update frequency must be a multiple of simulation frequency."
        assert self.odometry_dt <= self.lidar_dt, "The Odometry update frequency must be higher than or equal to the LiDAR update frequency."
        ## Env initialization
        self.n_stack = n_stack
        self.reward_function = reward_function
        self.environment = ENVIRONMENTS.index('lasernav')
        self.lidar_substeps = int(self.lidar_dt / self.humans_dt) # Number of simulation steps between two LiDAR updates
        self.odometry_substeps = int(self.odometry_dt / self.humans_dt) # Number of simulation steps between two Odometry updates
        self.control_substeps = int(self.robot_dt / self.humans_dt) # Number of simulation steps between two robot action updates

    # --- Private methods --- #

    def __repr__(self) -> str:
        return str(self.__dict__)

    @partial(jit, static_argnames=("self"))
    def _init_info(
        self,
        initial_state:jnp.ndarray,
        humans_goal:jnp.ndarray,
        robot_goal:jnp.ndarray,
        robot_goal_list:jnp.ndarray,
        humans_parameters:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        current_scenario:int,
        humans_delay:jnp.ndarray,
        is_x_flipped:bool,
        is_y_flipped:bool,
        noise_key:random.PRNGKey,
        visibility_chance:float,
    ) -> dict:
        """
        OVERRIDES BaseEnv._init_info method.

        Initializes the info dictionary with the given parameters.

        args:
        - initial_state: initial state of the environment.
        - humans_goal: array of humans' goals.
        - robot_goal: array of robot's goal.
        - humans_parameters: array of humans' parameters.
        - static_obstacles: array of static obstacles.
        - current_scenario: current scenario index.
        - humans_delay: array of humans' delays.

        output:
        - info: dictionary containing the initialized values.
        """
        info = super()._init_info(
            initial_state,
            humans_goal,
            robot_goal,
            robot_goal_list,
            humans_parameters,
            static_obstacles,
            current_scenario,
            humans_delay,
            is_x_flipped,
            is_y_flipped,
            noise_key,
            visibility_chance=visibility_chance,
        )
        noise_key1, noise_key2, noise_key3 = random.split(noise_key, 3)
        # Time from last LiDAR scan initialization
        info["substeps_from_last_scan"] = 0
        if self.lidar_misalignment:
            info["substeps_from_last_scan"] += random.randint(noise_key2, (), 0, self.lidar_substeps)
        # Time from last Odometry update initialization
        info["substeps_from_last_odom_ref_scan"] = info["substeps_from_last_scan"]
        if self.odometry_misalignment:
            info["substeps_from_last_odom_ref_scan"] += random.randint(noise_key3, (), 0, self.odometry_substeps)
        # Previous observation initialization
        info["previous_obs"], humans_visibility_mask, obstacles_visibility_mask = vmap(self._get_current_obs, in_axes=(None,None,None,None,None,None,None,None,None,None,0))(
            initial_state,
            info["humans_leg_state"],
            initial_state,
            jnp.zeros((2,)),
            humans_parameters[:,0],
            info["humans_leg_parameters"][:,-1],
            static_obstacles[-1],
            info["time"] - info["substeps_from_last_scan"]*self.humans_dt,
            info["time"] - info["substeps_from_last_odom_ref_scan"]*self.humans_dt,
            info["time"],
            random.split(noise_key1, self.n_stack),
        )
        info["humans_visibility_mask"] = humans_visibility_mask[0]
        info["obstacles_visibility_mask"] = obstacles_visibility_mask[0]
        return info

    @partial(jit, static_argnames=("self"))
    def _get_current_obs(
        self,
        lidar_state:jnp.ndarray, 
        legs_lidar_state:jnp.ndarray, 
        odom_state:jnp.ndarray, 
        robot_action:jnp.ndarray,
        humans_radii:jnp.ndarray, 
        legs_radii:jnp.ndarray, 
        static_obstacles:jnp.ndarray, 
        lidar_timestamp:float,
        odom_timestamp:float,
        control_timestamp:float,
        noise_key:random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment,
        this function computes the current observation of the state.

        args:
        - lidar_state: current state at the LiDAR update step.
        - legs_lidar_state: current state of humans' legs at the LiDAR update step.
        - odom_state: current state at the Odometry update step.
        - robot_action: last reference velocity commanded to the robot
        - humans_radii: radii of the humans.
        - legs_radii: radii of the humans' legs.
        - static_obstacles: static obstacles in the environment.
        - lidar_timestamp: timestamp of the current LiDAR state.
        - odom_timestamp: timestamp of the current Odometry state.
        - control_timestamp: timestamp of the current control step (environment step).
        - noise_key: random.PRNGKey for noise generation.

        output:
        - current_obs: [rx,ry,r_theta,r_radius,r_v,r_w,r_a1,r_a2,lidar_timestamp,odom_timestamp,control_timestamp,lidar_measurements]
        """
        measurements, humans_visibility_mask, obstacles_visibility_mask = self.get_lidar_measurements(
            lidar_state[-1, :2], # Lidar position (robot position)
            lidar_state[-1,4], # Lidar yaw angle (robot orientation)
            lidar_state[:-1, :2], # Human positions
            legs_lidar_state[:,[0,1,3,4]], # Humans legs positions (lx, ly, rx, ry)
            humans_radii,
            legs_radii,
            static_obstacles, 
            noise_key=noise_key
        )
        robot_velocity = odom_state[-1,2:4] # Robot action (either (vx,vy) or (v,w))
        robot_position = odom_state[-1,:2]
        robot_orientation = odom_state[-1,4]
        # Compute the current observation
        current_obs = jnp.array([
            *robot_position, # Robot position
            robot_orientation, # Robot orientation
            self.robot_radius, # Robot radius
            *robot_velocity, # Robot velocity (either (vx,vy) or (v,w))
            *robot_action, # Last robot reference velocity (either (vx,vy) or (v,w))
            lidar_timestamp,
            odom_timestamp,
            control_timestamp,
            *measurements[:,0], # LiDAR measurements
        ])
        return current_obs, humans_visibility_mask, obstacles_visibility_mask

    @partial(jit, static_argnames=("self"))
    def _get_obs(self, state:jnp.ndarray, info:dict, action:jnp.ndarray, noise_key:random.PRNGKey) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment,
        this function computes the observation of the current state (which is a stack of the last n_stack observations).

        args:
        - state: current state of the environment. (UNUSED HERE, STATE IS GATHERED FROM INTERMEDIATE_STATES IN INFO BASED ON SENSORS FREQUENCIES)
        - info: dictionary containing additional information about the environment.
        - action: last robot action (t-1).
        - noise_key: random.PRNGKey for noise generation.

        output:
        - obs (n_stack, lidar_num_rays + 10): Each stack [rx,ry,r_theta,r_radius,r_v,r_w,r_a1,r_a2,lidar_timestamp,odom_timestamp,control_timestamp,lidar_measurements].
        The first stack is the most recent one.
        """
        lidar_state = info['intermediate_states'][-(1+info["substeps_from_last_scan"])]
        legs_lidar_state = info['intermediate_leg_states'][-(1+info["substeps_from_last_scan"])]
        odom_state = info['intermediate_states'][-(1+info["substeps_from_last_odom_ref_scan"])]
        current_obs, humans_visibility_mask, obstacles_visibility_mask = self._get_current_obs(
            lidar_state, 
            legs_lidar_state, 
            odom_state, 
            action,
            info["humans_parameters"][:,0], 
            info["humans_leg_parameters"][:,-1], 
            info["static_obstacles"][-1], 
            info["time"] - info["substeps_from_last_scan"]*self.humans_dt,
            info["time"] - info["substeps_from_last_odom_ref_scan"]*self.humans_dt,
            info["time"],
            noise_key
        )
        # Stack the current observation with the previous ones
        obs = jnp.vstack((current_obs,info["previous_obs"][:-1]))
        return obs, humans_visibility_mask, obstacles_visibility_mask
        
    # --- Public methods --- #

    @partial(jit, static_argnames=("self"))
    def step(
        self, 
        state:jnp.ndarray, 
        info:dict, 
        action:jnp.ndarray, 
        test:bool=False,
        reset_if_done:bool=False,
        reset_key:random.PRNGKey=random.PRNGKey(0),
        env_key:random.PRNGKey=random.PRNGKey(0),
        scenarios_prob:jnp.ndarray=None,
        visibility_chance:float=0.,
    )-> tuple[jnp.ndarray, jnp.ndarray, dict, float, bool]:
        """
        Given an environment state, a dictionary containing additional information about the environment, and an action,
        this function computes the next state, the observation, the reward, and whether the episode is done.

        args:
        - state: jnp.ndarray containing the state of the environment.
        - info: dictionary containing additional information about the environment.
        - action: action to be taken by the robot.
        - test: boolean indicating whether the function is being used for testing purposes.
        - reset_if_done: boolean indicating whether the environment should be reset if the episode is done.

        output:
        - new_state: jnp.ndarray containing the updated state of the environment.
        - obs: observation of the new state.
        - info: dictionary containing additional information about the environment.
        - reward: reward obtained in the transition.
        - outcome: dictionary indicating whether the episode is in a terminal state or not.
        - (reset_key, env_key): tuple of random.PRNGKey used to reset the environment (only if reset_if_done is True) and to advance the environment key.
        """
        ### Advance Environment noise key
        new_env_key, delay_key,_ = random.split(env_key, 3) 
        ### Robot goal update (next waypoint, if present)
        if self.scenario != -1: # Custom scenario, no automatic goal update
            info["robot_goal"], info["robot_goal_index"] = lax.cond(
                (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= self.robot_radius*3) & # Waypoint reached threshold is set to be higher
                (info['robot_goal_index'] < len(info['robot_goal_list'])-1) & # Check if current goal is not the last one
                (~(jnp.any(jnp.isnan(info['robot_goal_list'][info['robot_goal_index']+1])))), # Check if next goal is not NaN
                lambda _: (info['robot_goal_list'][info['robot_goal_index']+1], info['robot_goal_index']+1),
                lambda x: x,
                (info["robot_goal"], info["robot_goal_index"])
            )
        ### Compute reward and outcome
        reward, outcome, reward_terms = self.reward_function(state, action, info, self.robot_dt)
        ### Compute robot delay
        info["robot_delay"] = jnp.clip(random.normal(delay_key) * self.control_delay_sigma + self.control_delay_mean, 0., self.actions_history_length * self.robot_dt) # Delay must be positive and lower than maximum history length * robot_dt
        ### Update state and info
        new_state, new_info, (state_history, humans_leg_state_history) = self._step(state, info, action) 
        ### Test outcome computation (during tests we check for actual collision or reaching goal)
        @jit
        def _test_outcome(val:tuple):
            state, info, outcome = val
            success, _ = self.reward_function.goal_reached_termination(
                state[-1,:2],
                self.robot_radius,
                info["robot_goal"],
            )
            collision_with_human, _ = self.reward_function.instant_human_collision_termination(
                state[-1,:2],
                self.robot_radius,
                state[:-1,:2],
                info["humans_parameters"][:,0]
            )
            collision_with_obstacle, _ = self.reward_function.instant_obstacle_collision_termination(
                state[-1,:2],
                self.robot_radius,
                info['static_obstacles'][-1],
            )
            failure = collision_with_human | collision_with_obstacle
            outcome["success"] = success
            outcome["collision_with_human"] = (collision_with_human) & (~success)
            outcome["collision_with_obstacle"] = (collision_with_obstacle) & (~success)
            outcome["timeout"] = jnp.all(jnp.array([outcome["timeout"], ~failure, jnp.logical_not(outcome["success"])]))
            outcome["nothing"] = jnp.logical_not(jnp.any(jnp.array([outcome["success"], failure, outcome["timeout"]])))
            return outcome
        outcome = lax.cond(test, lambda x: _test_outcome(x), lambda x: x[2], (new_state, info, outcome))
        ### Update time, step, return, previous observation
        new_info["time"] += self.robot_dt
        new_info["step"] += 1
        new_info["action_history"] = jnp.concatenate((action[None,:], new_info["action_history"][:-1]), axis=0)
        new_info["intermediate_states"] = state_history
        new_info["intermediate_leg_states"] = humans_leg_state_history
        new_info["substeps_from_last_scan"] = (new_info["substeps_from_last_scan"] + self.control_substeps) % self.lidar_substeps
        new_info["substeps_from_last_odom_ref_scan"] = ((new_info["substeps_from_last_odom_ref_scan"] + self.control_substeps - new_info["substeps_from_last_scan"]) % self.odometry_substeps) + new_info["substeps_from_last_scan"]
        gammas = jnp.array(tuple(reward_terms.keys()))
        rewards = jnp.array(tuple(reward_terms.values()))
        exponent = info["step"] * self.robot_dt * self.reward_function.v_max
        new_info["return"] += jnp.sum(jnp.power(gammas, exponent) * rewards)
        ### If done and reset_if_done, automatically reset the environment (available only if using standard scenarios)
        if self.scenario != -1: # Custom scenario, no automatic reset
            new_state, reset_key, new_info = lax.cond(
                (reset_if_done) & (~(outcome["nothing"])),
                lambda x: self._reset(x[1], scenarios_prob=scenarios_prob, visibility_chance=visibility_chance),
                lambda x: x,
                (new_state, reset_key, new_info)
            )
        # TODO: Filter obstacles based on the robot position and grid cell decomposition of static obstacles
        new_obs, new_info["humans_visibility_mask"], new_info["obstacles_visibility_mask"] = self._get_obs(new_state, new_info, action, new_env_key)
        new_info["previous_obs"] = new_obs
        return new_state, new_obs, new_info, (reward, reward_terms), outcome, (reset_key, new_env_key)

    @partial(jit, static_argnames=("self"))
    def batch_step(
        self, 
        states:jnp.ndarray, 
        infos:dict, 
        actions:jnp.ndarray, 
        reset_keys:jnp.ndarray, # This is moved upwards because a default value cannot be given.
        env_keys:jnp.ndarray,
        test:bool=False,
        reset_if_done:bool=False,
        scenarios_prob:jnp.ndarray=None,
        visibility_chance:float=0.,
    ):
        return vmap(LaserNav.step, in_axes=(None, 0, 0, 0, None, None, 0, 0, None, None))(
            self, 
            states, 
            infos, 
            actions, 
            test, 
            reset_if_done,
            reset_keys,
            env_keys,
            scenarios_prob,
            visibility_chance,
        )
    
    @partial(jit, static_argnames=("self"))
    def reset(self, key:random.PRNGKey, scenarios_prob:jnp.ndarray=None, visibility_chance:float=0.) -> tuple:
        initial_state, key, info = self._reset(key, scenarios_prob=scenarios_prob, visibility_chance=visibility_chance)
        return \
            initial_state, \
            key, \
            info["previous_obs"], \
            info, \
            {"success": False, "collision_with_human": False, "collision_with_obstacle": False, "timeout": False, "nothing": True}
    
    @partial(jit, static_argnames=("self"))
    def batch_reset(self, keys):
        return vmap(LaserNav.reset, in_axes=(None,0))(self, keys)

    @partial(jit, static_argnames=("self"))
    def reset_custom_episode(self, key:random.PRNGKey, custom_episode:dict) -> tuple:
        """
        Resets the environment to a user-specified episode (custom scenario, scenario index -1).

        args:
        - key: PRNG key (also used to seed sensor/leg noise).
        - custom_episode: dictionary with keys:
            full_state (n_humans+1, 6): initial full state. WARNING: humans' velocities
                must be given in the GLOBAL frame; they are converted to the body frame
                here since LaserNav humans are driven by HSFM.
            humans_goal (n_humans, 2): humans' goal positions.
            robot_goal (2,): robot's goal position.
            static_obstacles (n_humans+1, n_obstacles, 1, 2, 2): static obstacles.
            scenario (int): scenario index (use -1 for custom scenario).
            humans_radius (n_humans,): humans' radii.
            humans_speed (n_humans,): humans' desired speeds.

        output:
        - initial_state, key, obs (previous_obs stack), info, outcome
          (same format as LaserNav.reset).
        """
        full_state = jnp.array(custom_episode["full_state"])
        # LaserNav humans are always HSFM: convert global-frame velocities to the body frame
        full_state = lax.fori_loop(
            0,
            self.n_humans,
            lambda i, x: x.at[i].set(jnp.array(
                [x[i,0],
                 x[i,1],
                 *jnp.matmul(jnp.array([[jnp.cos(x[i,4]), -jnp.sin(x[i,4])], [jnp.sin(x[i,4]), jnp.cos(x[i,4])]]), x[i,2:4]),
                 x[i,4],
                 x[i,5]])),
            full_state)
        humans_goal = jnp.array(custom_episode["humans_goal"])
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        humans_parameters = humans_parameters.at[:,0].set(jnp.array(custom_episode["humans_radius"]))
        humans_parameters = humans_parameters.at[:,2].set(jnp.array(custom_episode["humans_speed"]))
        robot_goal = jnp.array(custom_episode["robot_goal"])
        robot_goal_list = jnp.array([robot_goal, jnp.full((2,), jnp.nan)]) # Dummy waypoint list (unused for custom scenario)
        if self.n_obstacles == 0:
            static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan)
        else:
            static_obstacles = jnp.array(custom_episode["static_obstacles"])
        key, noise_key = random.split(key)
        info = self._init_info(
            full_state,
            humans_goal,
            robot_goal,
            robot_goal_list,
            humans_parameters,
            static_obstacles,
            custom_episode["scenario"],
            jnp.zeros((self.n_humans,)),
            False,
            False,
            noise_key,
            visibility_chance=0.,
        )
        return \
            full_state, \
            key, \
            info["previous_obs"], \
            info, \
            {"success": False, "collision_with_human": False, "collision_with_obstacle": False, "timeout": False, "nothing": True}