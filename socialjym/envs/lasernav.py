import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from .base_env import BaseEnv, SCENARIOS, ENVIRONMENTS, is_multiple, wrap_angle
from .parameter_context import (
    bounds_from_nominal,
    sample_context,
    validate_env_params,
    validate_robot_params,
)

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

    @partial(jit, static_argnames=("self"))
    def _attach_parameter_context(self, info, robot_params, env_params):
        """Attach v2 contexts to runtime state without affecting legacy resets."""
        contextual_info = info.copy()
        contextual_info["_robot_params"] = robot_params
        contextual_info["_env_params"] = env_params
        contextual_info["previous_obs"] = contextual_info["previous_obs"].at[:, 3].set(
            robot_params["radius"]
        )
        return contextual_info

    @partial(jit, static_argnames=("self"))
    def _runtime_sensor_substeps(self, info):
        """Return discrete sensor periods supported by the simulator history."""
        if "_env_params" not in info:
            return self.lidar_substeps, self.odometry_substeps
        lidar_substeps = jnp.clip(
            jnp.rint(info["_env_params"]["lidar_period"] / self.humans_dt).astype(jnp.int32),
            1,
            self.control_substeps,
        )
        odometry_substeps = jnp.clip(
            jnp.rint(info["_env_params"]["odometry_period"] / self.humans_dt).astype(jnp.int32),
            1,
            self.control_substeps,
        )
        return lidar_substeps, odometry_substeps

    @partial(jit, static_argnames=("self"))
    def _refresh_context_observation(self, state, info, noise_key):
        """Regenerate reset observations after a v2 context has been attached."""
        lidar_substeps, odometry_substeps = self._runtime_sensor_substeps(info)
        refreshed_info = info.copy()
        refreshed_info["substeps_from_last_scan"] %= lidar_substeps
        refreshed_info["substeps_from_last_odom_ref_scan"] %= odometry_substeps
        obs, humans_mask, obstacles_mask = self._get_obs(
            state, refreshed_info, jnp.zeros((2,)), noise_key
        )
        # The legacy reset built every stack entry before the context existed.
        # Do not retain those un-randomized scans behind the new contextual one.
        obs = jnp.repeat(obs[:1], self.n_stack, axis=0)
        refreshed_info["humans_visibility_mask"] = humans_mask
        refreshed_info["obstacles_visibility_mask"] = obstacles_mask
        refreshed_info["previous_obs"] = obs
        return obs, refreshed_info

    @partial(jit, static_argnames=("self"))
    def _reset_with_param_bounds(
        self,
        key,
        robot_lower,
        robot_upper,
        env_lower,
        env_upper,
        scenarios_prob=None,
    ):
        param_key, robot_key, env_key, reset_key = random.split(key, 4)
        del param_key
        robot_params = sample_context(
            robot_key, self.get_default_robot_params(), robot_lower, robot_upper
        )
        env_params = sample_context(
            env_key, self.get_default_env_params(), env_lower, env_upper
        )
        state, next_key, obs, info, outcome = self.reset(
            reset_key,
            scenarios_prob=scenarios_prob,
            visibility_chance=env_params["robot_visibility_probability"],
        )
        del obs
        info = self._attach_parameter_context(info, robot_params, env_params)
        observation_key, next_key = random.split(next_key)
        obs, info = self._refresh_context_observation(state, info, observation_key)
        return state, next_key, obs, info, robot_params, env_params, outcome

    @partial(jit, static_argnames=("self", "test", "reset_if_done"))
    def _step_with_param_bounds(
        self,
        state,
        info,
        robot_params,
        env_params,
        action,
        robot_lower,
        robot_upper,
        env_lower,
        env_upper,
        test=False,
        reset_if_done=False,
        reset_key=random.PRNGKey(0),
        env_key=random.PRNGKey(0),
        scenarios_prob=None,
    ):
        info = self._attach_parameter_context(info, robot_params, env_params)
        # Enforce the runtime triangular differential-drive action envelope.
        v_max = robot_params["v_max"]
        wheels_distance = jnp.maximum(robot_params["wheels_distance"], 1e-5)
        w_max = 2.0 * v_max / wheels_distance
        bounded_action = jnp.array([
            jnp.clip(action[0], 0.0, v_max),
            jnp.clip(action[1], -w_max, w_max),
        ])
        envelope = bounded_action[0] / jnp.maximum(v_max, 1e-5) + jnp.abs(
            bounded_action[1]
        ) / jnp.maximum(w_max, 1e-5)
        bounded_action = bounded_action / jnp.maximum(envelope, 1.0)
        result = self.step(
            state,
            info,
            bounded_action,
            test=test,
            reset_if_done=False,
            reset_key=reset_key,
            env_key=env_key,
            scenarios_prob=scenarios_prob,
            visibility_chance=env_params["robot_visibility_probability"],
        )
        next_state, next_obs, next_info, reward, outcome, (next_reset_key, next_env_key) = result

        if self.scenario != -1:
            def _reset_done(_):
                robot_key, env_param_key, episode_key = random.split(next_reset_key, 3)
                new_robot_params = sample_context(
                    robot_key, self.get_default_robot_params(), robot_lower, robot_upper
                )
                new_env_params = sample_context(
                    env_param_key, self.get_default_env_params(), env_lower, env_upper
                )
                reset_state, returned_key, reset_info = self._reset(
                    episode_key,
                    scenarios_prob=scenarios_prob,
                    visibility_chance=new_env_params["robot_visibility_probability"],
                )
                reset_info = self._attach_parameter_context(
                    reset_info, new_robot_params, new_env_params
                )
                observation_key, returned_key = random.split(returned_key)
                reset_obs, reset_info = self._refresh_context_observation(
                    reset_state, reset_info, observation_key
                )
                return (
                    reset_state,
                    reset_obs,
                    reset_info,
                    new_robot_params,
                    new_env_params,
                    returned_key,
                )

            def _keep(_):
                return (
                    next_state,
                    next_obs,
                    next_info,
                    robot_params,
                    env_params,
                    next_reset_key,
                )

            next_state, next_obs, next_info, robot_params, env_params, next_reset_key = lax.cond(
                reset_if_done & (~outcome["nothing"]),
                _reset_done,
                _keep,
                operand=None,
            )
        return (
            next_state,
            next_obs,
            next_info,
            robot_params,
            env_params,
            reward,
            outcome,
            (next_reset_key, next_env_key),
        )

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _validate_runtime_bounds(self, robot_lower, robot_upper, env_lower, env_upper):
        """Reject contexts the fixed-rate simulator cannot represent faithfully."""
        if float(robot_lower["wheels_distance"]) <= 0.0:
            raise ValueError("LaserNav robot_params require a positive wheels_distance")
        if (
            float(robot_lower["control_dt"]) != self.robot_dt
            or float(robot_upper["control_dt"]) != self.robot_dt
        ):
            raise ValueError("control_dt changes require constructing a matching LaserNav instance")
        for name in ("lidar_period", "odometry_period"):
            if (
                float(env_lower[name]) < self.humans_dt
                or float(env_upper[name]) > self.robot_dt
            ):
                raise ValueError(
                    f"{name} bounds must lie in [humans_dt, robot_dt] "
                    "because LaserNav stores one control interval of sensor history"
                )

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
        env_params=None,
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
            noise_key=noise_key,
            noise_params=env_params,
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
            info["time"] - info["substeps_from_last_scan"]*self.humans_dt
            - (info["_env_params"]["lidar_latency"] if "_env_params" in info else 0.0),
            info["time"] - info["substeps_from_last_odom_ref_scan"]*self.humans_dt
            - (info["_env_params"]["odometry_latency"] if "_env_params" in info else 0.0),
            info["time"],
            noise_key,
            info["_env_params"] if "_env_params" in info else None,
        )
        if "_robot_params" in info:
            current_obs = current_obs.at[3].set(info["_robot_params"]["radius"])
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
        robot_radius = info["_robot_params"]["radius"] if "_robot_params" in info else self.robot_radius
        robot_v_max = info["_robot_params"]["v_max"] if "_robot_params" in info else self.reward_function.v_max
        control_delay_mean = info["_robot_params"]["control_delay_mean"] if "_robot_params" in info else self.control_delay_mean
        control_delay_sigma = info["_robot_params"]["control_delay_std"] if "_robot_params" in info else self.control_delay_sigma
        ### Advance Environment noise key
        new_env_key, delay_key,_ = random.split(env_key, 3) 
        ### Robot goal update (next waypoint, if present)
        if self.scenario != -1: # Custom scenario, no automatic goal update
            info["robot_goal"], info["robot_goal_index"] = lax.cond(
                (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= robot_radius*3) & # Waypoint reached threshold is set to be higher
                (info['robot_goal_index'] < len(info['robot_goal_list'])-1) & # Check if current goal is not the last one
                (~(jnp.any(jnp.isnan(info['robot_goal_list'][info['robot_goal_index']+1])))), # Check if next goal is not NaN
                lambda _: (info['robot_goal_list'][info['robot_goal_index']+1], info['robot_goal_index']+1),
                lambda x: x,
                (info["robot_goal"], info["robot_goal_index"])
            )
        ### Compute robot delay
        info["robot_delay"] = jnp.clip(random.normal(delay_key) * control_delay_sigma + control_delay_mean, 0., self.actions_history_length * self.robot_dt) # Delay must be positive and lower than maximum history length * robot_dt
        ### Update state and info
        new_state, new_info, (state_history, humans_leg_state_history) = self._step(state, info, action) 
        ### Compute reward and outcome from the transition that was actually
        # executed.  Reward implementations historically integrate one constant
        # action from ``state``.  Reconstructing endpoint-equivalent velocities
        # keeps that public reward API intact while accounting for control delay,
        # actuator lag and acceleration limiting.
        human_displacements = new_state[:-1, :2] - state[:-1, :2]
        human_global_velocities = human_displacements / self.robot_dt
        human_theta = state[:-1, 4]
        human_body_velocities = jnp.stack(
            (
                jnp.cos(human_theta) * human_global_velocities[:, 0]
                + jnp.sin(human_theta) * human_global_velocities[:, 1],
                -jnp.sin(human_theta) * human_global_velocities[:, 0]
                + jnp.cos(human_theta) * human_global_velocities[:, 1],
            ),
            axis=-1,
        )
        reward_state = state.at[:-1, 2:4].set(human_body_velocities)
        delta_theta = wrap_angle(new_state[-1, 4] - state[-1, 4])
        effective_w = delta_theta / self.robot_dt
        robot_displacement = new_state[-1, :2] - state[-1, :2]
        mid_theta = state[-1, 4] + delta_theta / 2.0
        signed_chord = jnp.dot(
            robot_displacement,
            jnp.array([jnp.cos(mid_theta), jnp.sin(mid_theta)]),
        )
        effective_v = lax.cond(
            jnp.abs(delta_theta) > 1e-5,
            lambda: signed_chord * effective_w / (2.0 * jnp.sin(delta_theta / 2.0)),
            lambda: signed_chord / self.robot_dt,
        )
        effective_action = jnp.array([effective_v, effective_w])
        reward, outcome, reward_terms = self.reward_function.evaluate_transition(
            reward_state,
            effective_action,
            info,
            self.robot_dt,
            state_history=state_history,
        )
        ### Test outcome computation (during tests we check for actual collision or reaching goal)
        @jit
        def _test_outcome(val:tuple):
            state, info, outcome = val
            success, _ = self.reward_function.goal_reached_termination(
                state[-1,:2],
                robot_radius,
                info["robot_goal"],
            )
            collision_with_human, _ = self.reward_function.instant_human_collision_termination(
                state[-1,:2],
                robot_radius,
                state[:-1,:2],
                info["humans_parameters"][:,0]
            )
            collision_with_obstacle, _ = self.reward_function.instant_obstacle_collision_termination(
                state[-1,:2],
                robot_radius,
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
        lidar_substeps, odometry_substeps = self._runtime_sensor_substeps(new_info)
        new_info["substeps_from_last_scan"] = (
            new_info["substeps_from_last_scan"] + self.control_substeps
        ) % lidar_substeps
        new_info["substeps_from_last_odom_ref_scan"] = (
            (
                new_info["substeps_from_last_odom_ref_scan"]
                + self.control_substeps
                - new_info["substeps_from_last_scan"]
            )
            % odometry_substeps
        ) + new_info["substeps_from_last_scan"]
        gammas = jnp.array(tuple(reward_terms.keys()))
        rewards = jnp.array(tuple(reward_terms.values()))
        exponent = info["step"] * self.robot_dt * robot_v_max
        new_info["return"] += jnp.sum(jnp.power(gammas, exponent) * rewards)
        ### If done and reset_if_done, automatically reset the environment (available only if using standard scenarios)
        if self.scenario != -1: # Custom scenario, no automatic reset
            def _auto_reset(x):
                reset_state, returned_key, reset_info = self._reset(
                    x[1],
                    scenarios_prob=scenarios_prob,
                    visibility_chance=visibility_chance,
                )
                if "_robot_params" in x[2]:
                    reset_info = self._attach_parameter_context(
                        reset_info,
                        x[2]["_robot_params"],
                        x[2]["_env_params"],
                    )
                return reset_state, returned_key, reset_info
            new_state, reset_key, new_info = lax.cond(
                (reset_if_done) & (~(outcome["nothing"])),
                _auto_reset,
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

    @partial(jit, static_argnames=("self", "test", "reset_if_done"))
    def batch_step_with_param_bounds(
        self,
        states,
        infos,
        robot_params,
        env_params,
        actions,
        reset_keys,
        env_keys,
        robot_lower,
        robot_upper,
        env_lower,
        env_upper,
        test=False,
        reset_if_done=False,
        scenarios_prob=None,
    ):
        return vmap(
            LaserNav._step_with_param_bounds,
            in_axes=(
                None, 0, 0, 0, 0, 0,
                None, None, None, None,
                None, None, 0, 0, None,
            ),
        )(
            self,
            states,
            infos,
            robot_params,
            env_params,
            actions,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            test,
            reset_if_done,
            reset_keys,
            env_keys,
            scenarios_prob,
        )

    def batch_step_with_params(
        self,
        states,
        infos,
        robot_params,
        env_params,
        actions,
        reset_keys,
        env_keys,
        *,
        robot_param_bounds=None,
        env_param_bounds=None,
        test=False,
        reset_if_done=False,
        scenarios_prob=None,
    ):
        robot_nominal = validate_robot_params(self.get_default_robot_params())
        env_nominal = validate_env_params(self.get_default_env_params())
        robot_lower, robot_upper = bounds_from_nominal(robot_nominal, robot_param_bounds)
        env_lower, env_upper = bounds_from_nominal(env_nominal, env_param_bounds)
        self._validate_runtime_bounds(
            robot_lower, robot_upper, env_lower, env_upper
        )
        return self.batch_step_with_param_bounds(
            states,
            infos,
            robot_params,
            env_params,
            actions,
            reset_keys,
            env_keys,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            test=test,
            reset_if_done=reset_if_done,
            scenarios_prob=scenarios_prob,
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
    def batch_reset(self, keys, scenarios_prob=None, visibility_chance=0.):
        """Vectorized reset preserving the legacy defaults.

        ``scenarios_prob`` and ``visibility_chance`` are additive arguments used
        by curricula.  Previously the first rollout reset silently ignored both
        while auto-resets used them, so the first episodes came from a different
        distribution.
        """
        return vmap(LaserNav.reset, in_axes=(None, 0, None, None))(
            self,
            keys,
            scenarios_prob,
            visibility_chance,
        )

    def reset_with_params(
        self,
        key,
        robot_param_bounds=None,
        env_param_bounds=None,
        scenarios_prob=None,
    ):
        """Reset with per-episode contexts while retaining the legacy reset API."""
        robot_nominal = validate_robot_params(self.get_default_robot_params())
        env_nominal = validate_env_params(self.get_default_env_params())
        robot_lower, robot_upper = bounds_from_nominal(robot_nominal, robot_param_bounds)
        env_lower, env_upper = bounds_from_nominal(env_nominal, env_param_bounds)
        self._validate_runtime_bounds(
            robot_lower, robot_upper, env_lower, env_upper
        )
        return self._reset_with_param_bounds(
            key,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            scenarios_prob,
        )

    def batch_reset_with_params(
        self,
        keys,
        robot_param_bounds=None,
        env_param_bounds=None,
        scenarios_prob=None,
    ):
        robot_nominal = validate_robot_params(self.get_default_robot_params())
        env_nominal = validate_env_params(self.get_default_env_params())
        robot_lower, robot_upper = bounds_from_nominal(robot_nominal, robot_param_bounds)
        env_lower, env_upper = bounds_from_nominal(env_nominal, env_param_bounds)
        self._validate_runtime_bounds(
            robot_lower, robot_upper, env_lower, env_upper
        )
        return vmap(LaserNav._reset_with_param_bounds, in_axes=(None, 0, None, None, None, None, None))(
            self,
            keys,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            scenarios_prob,
        )

    def step_with_params(
        self,
        state,
        info,
        robot_params,
        env_params,
        action,
        *,
        robot_param_bounds=None,
        env_param_bounds=None,
        test=False,
        reset_if_done=False,
        reset_key=random.PRNGKey(0),
        env_key=random.PRNGKey(0),
        scenarios_prob=None,
    ):
        robot_params = validate_robot_params(robot_params)
        env_params = validate_env_params(env_params)
        robot_lower, robot_upper = bounds_from_nominal(robot_params, robot_param_bounds)
        env_lower, env_upper = bounds_from_nominal(env_params, env_param_bounds)
        return self._step_with_param_bounds(
            state,
            info,
            robot_params,
            env_params,
            action,
            robot_lower,
            robot_upper,
            env_lower,
            env_upper,
            test=test,
            reset_if_done=reset_if_done,
            reset_key=reset_key,
            env_key=env_key,
            scenarios_prob=scenarios_prob,
        )

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
