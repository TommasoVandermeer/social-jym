import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from .base_env import BaseEnv, SCENARIOS, ROBOT_KINEMATICS, ENVIRONMENTS

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
            kinematics='unicycle',
            max_cc_delay = 5.,
            ccso_n_static_humans:int = 3,
            thick_default_obstacle:bool = True,
            grid_map_computation:bool = False,
            grid_cell_size:float = 0.9, # Such parameter is suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT)
            grid_min_size:float = 18. # Such parameter is the minimum suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT) in order to always include all static obstacles, the robot and its goal.
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
            kinematics=kinematics,
            max_cc_delay=max_cc_delay,
            ccso_n_static_humans=ccso_n_static_humans,
            grid_map_computation=grid_map_computation,
            grid_cell_size=grid_cell_size,
            grid_min_size=grid_min_size,
            thick_default_obstacle=thick_default_obstacle,
            )
        ## Args validation
        assert reward_function.kinematics == self.kinematics, "The reward function's kinematics must be the same as the environment's kinematics."
        assert n_stack >=1, "The number of stacked observations must be at least 1."
        ## Env initialization
        self.n_stack = n_stack
        self.reward_function = reward_function
        self.environment = ENVIRONMENTS.index('lasernav')

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
        )
        # Previous observation initialization
        # info["previous_obs"] = jnp.stack([self._get_current_obs(initial_state, humans_parameters[:,0], static_obstacles[-1], jnp.zeros((2,)), random.PRNGKey(0)),]*self.n_stack, axis=0)
        info["previous_obs"] = vmap(self._get_current_obs, in_axes=(None,None,None,None,0))(
            initial_state,
            humans_parameters[:,0],
            static_obstacles[-1],
            jnp.zeros((2,)),
            random.split(noise_key, self.n_stack),
        )
        return info

    @partial(jit, static_argnames=("self"))
    def _get_current_obs(self, state:jnp.ndarray, humans_radii:jnp.ndarray, static_obstacles:jnp.ndarray, action:jnp.ndarray, noise_key:random.PRNGKey) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment, and the robot's action,
        this function computes the current observation of the state.

        args:
        - state: current state of the environment.
        - humans_radii: radii of the humans.
        - static_obstacles: static obstacles in the environment.
        - action: action to be taken by the robot (vx,vy) or (v,w).

        output:
        - current_obs: [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements]
        """
        measurements = self.get_lidar_measurements(
            state[-1, :2], # Lidar position (robot position)
            state[-1,4], # Lidar yaw angle (robot orientation)
            state[:-1, :2], # Human positions
            humans_radii,
            static_obstacles, 
            noise_key=noise_key
        )
        # Compute the current observation
        current_obs = jnp.array([
            *state[-1,:2], # Robot position
            state[-1,4], # Robot orientation
            self.robot_radius, # Robot radius
            *action, # Robot action (either (vx,vy) or (v,w))
            *measurements[:,0], # LiDAR measurements
        ])
        return current_obs

    @partial(jit, static_argnames=("self"))
    def _get_obs(self, state:jnp.ndarray, info:dict, action:jnp.ndarray, noise_key:random.PRNGKey) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment, and the robot's action,
        this function computes the observation of the current state (which is a stack of the last n_stack observations).

        args:
        - state: current state of the environment.
        - previous_obs: last observation of the environment.
        - info: dictionary containing additional information about the environment.
        - action: action to be taken by the robot (vx,vy) or (v,w).

        output:
        - obs (n_stack, lidar_num_rays + 6): Each stack [rx,ry,r_theta,r_radius,r_a1,r_a2,lidar_measurements].
        The first stack is the most recent one.
        """
        current_obs = self._get_current_obs(state, info["humans_parameters"][:,0], info["static_obstacles"][-1], action, noise_key)
        # Stack the current observation with the previous ones
        obs = jnp.vstack((current_obs,info["previous_obs"][:-1]))
        return obs
        
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
        obs = self._get_obs(state, info, action, env_key)
        new_env_key, _ = random.split(env_key) # Advance the env_key (we do it here to save the replicate the previous obs in previous_obs)
        reward, outcome = self.reward_function(state, action, info, self.robot_dt)
        ### Update state and info
        if self.robot_visible:
            if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.linalg.norm(action), 0., jnp.atan2(*jnp.flip(action)), 0.])]) # HSFM fictitious state
            elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], action[0], 0., state[-1,4], action[1]])]) # HSFM fictitious state
            new_state, new_info = lax.fori_loop(
                0,
                int(self.robot_dt/self.humans_dt),
                lambda _ , x: self._update_state_info(*x, action),
                (fictitious_state, info))
            # Overwrite the robot fictitious state with the real one
            new_state = new_state.at[-1,2:].set(jnp.array([
                0., 
                0., 
                new_state[-1,4] * int(self.kinematics == ROBOT_KINEMATICS.index('unicycle')), # If robot is holonomic 0 is passed as robot theta
                0.]))
        else:
            new_state, new_info = lax.fori_loop(
                0,
                int(self.robot_dt/self.humans_dt),
                lambda _ , x: self._update_state_info(*x, action),
                (state, info))
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
        new_info["return"] += pow(self.reward_function.gamma, info["step"] * self.robot_dt * self.reward_function.v_max) * reward
        new_info["previous_obs"] = obs
        ### If done and reset_if_done, automatically reset the environment (available only if using standard scenarios)
        if self.scenario != -1: # Custom scenario, no automatic reset
            new_state, reset_key, new_info = lax.cond(
                (reset_if_done) & (~(outcome["nothing"])),
                lambda x: self._reset(x[1], scenarios_prob=scenarios_prob),
                lambda x: x,
                (new_state, reset_key, new_info)
            )
        # TODO: Filter obstacles based on the robot position and grid cell decomposition of static obstacles
        return new_state, self._get_obs(new_state, new_info, action, new_env_key), new_info, reward, outcome, (reset_key, new_env_key)

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
    ):
        return vmap(LaserNav.step, in_axes=(None, 0, 0, 0, None, None, 0, 0, None))(
            self, 
            states, 
            infos, 
            actions, 
            test, 
            reset_if_done,
            reset_keys,
            env_keys,
            scenarios_prob,
        )
    
    @partial(jit, static_argnames=("self"))
    def reset(self, key:random.PRNGKey, scenarios_prob:jnp.ndarray=None) -> tuple:
        initial_state, key, info = self._reset(key, scenarios_prob=scenarios_prob)
        return \
            initial_state, \
            key, \
            info["previous_obs"], \
            info, \
            {"success": False, "collision_with_human": False, "collision_with_obstacle": False, "timeout": False, "nothing": True}
    
    @partial(jit, static_argnames=("self"))
    def batch_reset(self, keys):
        return vmap(LaserNav.reset, in_axes=(None,0))(self, keys)