import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from .base_env import BaseEnv, SCENARIOS, HUMAN_POLICIES, ROBOT_KINEMATICS, ENVIRONMENTS, wrap_angle

class SocialNav(BaseEnv):
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL.

    Suitable for policies: CADRL, SARL, SARL*, SARL-PPO, DIR-SAFE
    """
    def __init__(
            self, 
            robot_radius:float, 
            robot_dt:float, 
            humans_dt:float, 
            scenario:str, 
            n_humans:int, 
            reward_function:FunctionType,
            n_obstacles:int = 0,
            humans_policy='hsfm', 
            robot_visible=False, 
            circle_radius=7, 
            traffic_height=3, # traffic_height=3 # Conform with Social-Navigation-PyEnvs
            traffic_length=14,
            crowding_square_side=14,
            hybrid_scenario_subset=jnp.arange(0, len(SCENARIOS)-1, dtype=jnp.int32),
            lidar_angular_range=jnp.pi,
            lidar_max_dist=10.,
            lidar_num_rays=60,
            kinematics='holonomic',
            max_cc_delay = 5.,
            ccso_n_static_humans:int = 3,
            thick_default_obstacle:bool = False,
            grid_map_computation:bool = False,
            grid_cell_size:float = 0.9, # Such parameter is suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT)
            grid_min_size:float = 18. # Such parameter is the minimum suitable for the obstacles and scenarios defined (CC,Pat,Pet,RC,DCC,CCSO,CN,CT) in order to always include all static obstacles, the robot and its goal.
        ) -> None:
        ## Warnings 
        if n_obstacles > 0:
            print(f"\nWARNING: Obstacles have been added to the environment, but collision detection is not implemented yet (only with humans).\nThe robot must be able to avoid them by design.\n")
        ## BaseEnv initialization
        super().__init__(
            robot_radius=robot_radius,
            robot_dt=robot_dt,
            humans_dt=humans_dt,
            n_humans=n_humans,
            n_obstacles=n_obstacles,    
            scenario=scenario,
            humans_policy=humans_policy,
            robot_visible=robot_visible,
            circle_radius=circle_radius,
            traffic_height=traffic_height,
            traffic_length=traffic_length,
            crowding_square_side=crowding_square_side,
            hybrid_scenario_subset=hybrid_scenario_subset,
            lidar_angular_range=lidar_angular_range, 
            lidar_max_dist=lidar_max_dist, 
            lidar_num_rays=lidar_num_rays,
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
        ## Env initialization
        self.reward_function = reward_function
        self.environment = ENVIRONMENTS.index('socialnav')

    # --- Private methods ---

    def __repr__(self) -> str:
        return str(self.__dict__)

    @partial(jit, static_argnames=("self"))
    def _get_obs(self, state:jnp.ndarray, info:dict, action:jnp.ndarray) -> jnp.ndarray:
        """
        Given the current state, the additional information about the environment, and the robot's action,
        this function computes the observation of the current state.

        args:
        - state: current state of the environment.
        - info: dictionary containing additional information about the environment.
        - action: action to be taken by the robot (vx,vy) or (v,w).

        output:
        - obs: observation of the current state. It is in the form:
                [[human1_px, human1_py, human1_vx, human1_vy, human1_radius, padding],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius, padding],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius, padding],
                [robot_px, robot_py, robot_u1, robot_u2, robot_radius, robot_theta]].
        """
        obs = jnp.zeros((self.n_humans+1, 6))
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'): # In case of hsfm convert humans body velocities to linear velocities
            linear_velocities = jnp.zeros((self.n_humans, 2))
            linear_velocities = lax.fori_loop(0, self.n_humans, lambda i, lv: lv.at[i].set(jnp.matmul(jnp.array([[jnp.cos(state[i,4]), -jnp.sin(state[i,4])], [jnp.sin(state[i,4]), jnp.cos(state[i,4])]]), state[i,2:4])) , linear_velocities)
            obs = lax.fori_loop(0, self.n_humans, lambda i, obs: obs.at[i].set(jnp.array([state[i,0], state[i,1], linear_velocities[i,0], linear_velocities[i,1], info['humans_parameters'][i,0], 0.])), obs)
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'): # For sfm and orca policies the state already contains the linear velocities
            obs = lax.fori_loop(0, self.n_humans, lambda i, obs: obs.at[i].set(jnp.array([state[i,0], state[i,1], state[i,2], state[i,3], info['humans_parameters'][i,0], 0.])), obs)
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            obs = obs.at[-1].set(jnp.array([*state[-1,0:2], *action, self.robot_radius, 0.]))
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            obs = obs.at[-1].set(jnp.array([*state[-1,0:2], *action, self.robot_radius, state[-1,4]]))
        return obs

    # --- Public methods ---

    @partial(jit, static_argnames=("self"))
    def imitation_learning_step(self, state:jnp.ndarray, info:dict)-> tuple[jnp.ndarray, jnp.ndarray, dict, float, bool]:
        """
        Given an environment state and a dictionary containing additional information about the environment
        this function computes the next state, the observation, the reward, and whether the episode is done.
        The robot moves using the policy guiding humans.

        args:
        - state: jnp.ndarray containing the state of the environment.
        - info: dictionary containing additional information about the environment.

        output:
        - new_state: jnp.ndarray containing the updated state of the environment.
        - obs: observation of the new state.
        - info: dictionary containing additional information about the environment.
        - reward: reward obtained in the transition.
        - outcome: dictionary indicating whether the episode is in a terminal state or not.
        """
        ### Robot goal update (next waypoint, if present)
        info["robot_goal"], info["robot_goal_index"] = lax.cond(
            (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= self.robot_radius*3) & # Waypoint reached threshold is set to be higher
            (info['robot_goal_index'] < len(self.robot_goals_per_scenario[info["current_scenario"]])-1) & # Check if current goal is not the last one
            (~(jnp.any(jnp.isnan(self.robot_goals_per_scenario[info["current_scenario"]][info['robot_goal_index']+1])))), # Check if next goal is not NaN
            lambda _: (self.robot_goals_per_scenario[info["current_scenario"]][info['robot_goal_index']+1], info['robot_goal_index']+1),
            lambda x: x,
            (info["robot_goal"], info["robot_goal_index"])
        )
        old_info = info.copy()
        ### Update state
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            out = lax.fori_loop(
                0,
                int(self.robot_dt/self.humans_dt),
                lambda _ , x: self._update_state_info_imitation_learning(*x),
                (state, info))
            new_state, new_info = out
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            out = lax.fori_loop(
                0,
                int(self.robot_dt/self.humans_dt),
                lambda _ , x: self._update_state_info_imitation_learning(*x),
                (state[:,0:4], info))
            new_state, new_info = out
            new_state = jnp.pad(new_state, ((0,0),(0,2)))
        # Compute action by derivative of robot position and orientation
        dp = new_state[-1,0:2] - state[-1,0:2]
        if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
            action = dp / self.robot_dt
        elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
            if self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                new_state = new_state.at[-1,4].set(jnp.arctan2(*jnp.flip(dp)))
            action = jnp.array([jnp.linalg.norm(dp / self.robot_dt), wrap_angle(new_state[-1,4] - state[-1,4]) / self.robot_dt])
        ### Compute reward and outcome - WARNING: The old state is passed, not the updated one (but with the correct action applied)
        reward, outcome = self.reward_function(self._get_obs(state, old_info, action), old_info, self.robot_dt)
        ### Update step, time and return
        new_info["time"] += self.robot_dt
        new_info["step"] += 1
        new_info["return"] += pow(self.reward_function.gamma, info["step"] * self.robot_dt * self.reward_function.v_max) * reward
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, outcome
    
    @partial(jit, static_argnames=("self"))
    def step(
        self, 
        state:jnp.ndarray, 
        info:dict, 
        action:jnp.ndarray, 
        test:bool=False,
        reset_if_done:bool=False,
        reset_key:random.PRNGKey=random.PRNGKey(0),
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
        - reset_key: random.PRNGKey used to reset the environment. Only used if reset_if_done is True.
        """
        ### Robot goal update (next waypoint, if present)
        if self.scenario != -1: # Custom scenario, no automatic goal update
            info["robot_goal"], info["robot_goal_index"] = lax.cond(
                (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= self.robot_radius*3) & # Waypoint reached threshold is set to be higher
                (info['robot_goal_index'] < len(self.robot_goals_per_scenario[info["current_scenario"]])-1) & # Check if current goal is not the last one
                (~(jnp.any(jnp.isnan(self.robot_goals_per_scenario[info["current_scenario"]][info['robot_goal_index']+1])))), # Check if next goal is not NaN
                lambda _: (self.robot_goals_per_scenario[info["current_scenario"]][info['robot_goal_index']+1], info['robot_goal_index']+1),
                lambda x: x,
                (info["robot_goal"], info["robot_goal_index"])
            )
        ### Compute reward and outcome
        reward, outcome = self.reward_function(self._get_obs(state, info, action), info, self.robot_dt)
        ### Update state and info
        if self.robot_visible:
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                    fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.linalg.norm(action), 0., jnp.atan2(*jnp.flip(action)), 0.])]) # HSFM fictitious state
                elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                    fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], action[0], 0., state[-1,4], action[1]])]) # HSFM fictitious state
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                    fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], *action, 0., 0.])]) # SFM or ORCA fictitious state
                elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                    fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.cos(state[-1,4]) * action[0], jnp.sin(state[-1,4]) * action[0], state[-1,4], 0.])]) # SFM or ORCA fictitious state
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
        ### Test outcome computation (during tests we check for INSTANT collision or reaching goal)
        @jit
        def _test_outcome(val:tuple):
            state, info, outcome = val
            success, _ = self.reward_function.goal_reached_termination(
                state[-1,:2],
                self.robot_radius,
                info["robot_goal"],
            )
            failure, _ = self.reward_function.instant_collision_termination(
                state[-1,:2],
                self.robot_radius,
                state[:-1,:2],
                info["humans_parameters"][:,0]
            )
            outcome["success"] = success
            outcome["failure"] = (failure) & (~(success))
            outcome["timeout"] = jnp.all(jnp.array([outcome["timeout"], jnp.logical_not(outcome["failure"]), jnp.logical_not(outcome["success"])]))
            outcome["nothing"] = jnp.logical_not(jnp.any(jnp.array([outcome["success"], outcome["failure"], outcome["timeout"]])))
            return outcome
        outcome = lax.cond(test, lambda x: _test_outcome(x), lambda x: x[2], (new_state, info, outcome))
        ### Update time, step, return
        new_info["time"] += self.robot_dt
        new_info["step"] += 1
        new_info["return"] += pow(self.reward_function.gamma, info["step"] * self.robot_dt * self.reward_function.v_max) * reward
        ### If done and reset_if_done, automatically reset the environment (available only if using standard scenarios)
        if self.scenario != -1: # Custom scenario, no automatic reset
            new_state, reset_key, new_info = lax.cond(
                (reset_if_done) & (~(outcome["nothing"])),
                lambda x: self._reset(x[1]),
                lambda x: x,
                (new_state, reset_key, new_info)
            )
        # TODO: Filter obstacles based on the robot position and grid cell decomposition of static obstacles
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, outcome, reset_key

    @partial(jit, static_argnames=("self"))
    def batch_step(
        self, 
        states:jnp.ndarray, 
        infos:dict, 
        actions:jnp.ndarray, 
        reset_keys:jnp.ndarray, # This is moved upwards because a default value cannot be given.
        test:bool=False,
        reset_if_done:bool=False,
    ):
        return vmap(SocialNav.step, in_axes=(None, 0, 0, 0, None, None, 0))(
            self, 
            states, 
            infos, 
            actions, 
            test, 
            reset_if_done,
            reset_keys,
        )

    @partial(jit, static_argnames=("self"))
    def reset(self, key:random.PRNGKey) -> tuple:
        """
        Given a PRNG key, this function resets the environment to start a new episode returning a stochastic initial state.

        args:
        - key: PRNG key to initialize a random state

        output:
        - initial_state: initial full state of the environment. The state is a JAX array in the form:
                      [[*human1_state],
                       [*human2_state],
                       ...
                       [*humanN_state],
                       [robot_px, robot_py, pad, pad, robot_theta, pad]].
                      The length of each sub_array is 6.
        - new_key: PRNG key to be used in the next steps.
        - obs: observation of the initial state. The observation is a JAX array in the form:
               [[human1_px, human1_py, human1_vx, human1_vy, human1_radius],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius],
                [robot_px, robot_py, robot_ux, robot_uy, robot_radius]].
        - info: dictionary containing additional information about the environment.
        - outcome: dictionary indicating whether the episode is in a terminal state or not.
        """
        initial_state, key, info = self._reset(key)
        return initial_state, key, self._get_obs(initial_state, info, jnp.zeros((2,))), info, {"success": False, "failure": False, "timeout": False, "nothing": True}
    
    @partial(jit, static_argnames=("self"))
    def batch_reset(self, keys):
        return vmap(SocialNav.reset, in_axes=(None,0))(self, keys)

    @partial(jit, static_argnames=("self"))
    def reset_custom_episode(self, key:random.PRNGKey, custom_episode:dict) -> tuple:
        """
        Given a custom episode data, this function resets the environment to start a new episode with the given data.

        args:
        - key; PRNG key (NOT USED).
        - custom_episode: dictionary containing the custom episode data. Its keys are:
            full_state (jnp.array): initial full state of the environment. WARNING: The velocity of humans is always in the global frame (for hsfm you should be using the velocity on the body frame)
            humans_goal (jnp.array): goal positions of the humans.
            robot_goal (jnp.array): goal position of the robot.
            static_obstacles (jnp.array): positions of the static obstacles.
            scenario (int): scenario of the episode. Set to -1 for custom scenario.
            humans_radius (float): radius of the humans.
            humans_speed (float): max speed of the humans.

        output:
        - initial_state: initial full state of the environment. The state is a JAX array in the form:
                      [[*human1_state],
                       [*human2_state],
                       ...
                       [*humanN_state],
                       [robot_px, robot_py, robot_theta, pad..]].
                      The length of each sub_array is 6.
        - key: PRNG key to be used in the next steps. (SAME AS THE INPUT ONE)
        - obs: observation of the initial state. The observation is a JAX array in the form:
               [[human1_px, human1_py, human1_vx, human1_vy, human1_radius],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius],
                [robot_px, robot_py, robot_u1, robot_u2, robot_radius]].
        - info: dictionary containing additional information about the environment.
        - outcome: dictionary indicating whether the episode is in a terminal state or not.
        """
        ## Check input data is coherent with the environment
        # TODO: Verify input data is coheren with the environment: n_humans
        ## Reset the environment with the custom episode data
        full_state = jnp.array(custom_episode["full_state"])
        # If the humans policy is hsfm, convert the velocities to the body frame
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            full_state = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, x: x.at[i].set(jnp.array(
                    [x[i,0],
                    x[i,1],
                    *jnp.matmul(jnp.array([[jnp.cos(x[i,4]), -jnp.sin(x[i,4])], [jnp.sin(x[i,4]), jnp.cos(x[i,4])]]), x[i,2:4]),
                    x[i,4],
                    x[i,5]]))
                , full_state)
        humans_goal = jnp.array(custom_episode["humans_goal"])
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        humans_parameters = humans_parameters.at[:,0].set(jnp.array(custom_episode["humans_radius"]))
        humans_parameters = humans_parameters.at[:,2].set(jnp.array(custom_episode["humans_speed"]))
        robot_goal = jnp.array(custom_episode["robot_goal"])
        # Obstacles
        if self.n_obstacles == 0:
            static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan)
        else:
            static_obstacles = jnp.array(custom_episode["static_obstacles"])
        # Info
        info = self._init_info(
            full_state,
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=custom_episode["scenario"],
            humans_delay=jnp.zeros((self.n_humans,)),
        )
        return full_state, key, self._get_obs(full_state, info, jnp.zeros((2,))), info, {"success": False, "failure": False, "timeout": False, "nothing": True}
        