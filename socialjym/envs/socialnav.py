import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from jax.experimental import checkify
from functools import partial
from types import FunctionType

from socialjym.utils.aux_functions import is_multiple
from .base_env import BaseEnv, SCENARIOS, HUMAN_POLICIES, ROBOT_KINEMATICS

class SocialNav(BaseEnv):
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL.
    """
    def __init__(
            self, 
            robot_radius:float, 
            robot_dt:float, 
            humans_dt:float, 
            scenario:str, 
            n_humans:int, 
            reward_function:FunctionType,
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
            kinematics='holonomic'
        ) -> None:
        ## BaseEnv initialization
        super().__init__(
            robot_radius=robot_radius,
            humans_dt=humans_dt,
            n_humans=n_humans,
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
            kinematics=kinematics)
        ## Args validation
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        ## Env initialization
        self.robot_dt = robot_dt
        self.reward_function = reward_function

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
        - action: action to be taken by the robot.

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
        obs = obs.at[-1].set(jnp.array([*state[-1,0:2], *action, self.robot_radius, 0.]))
        return obs

    @partial(jit, static_argnames=("self"))
    def _reset(self, key:random.PRNGKey) -> tuple[jnp.ndarray, random.PRNGKey, dict]:
        key, subkey = random.split(key)
        if self.scenario == SCENARIOS.index('hybrid_scenario'):
            # Randomly choose a scenario between all then ones included in the hybrid_scenario subset
            randint = random.randint(subkey, shape=(), minval=0, maxval=len(self.hybrid_scenario_subset))
            scenario = self.hybrid_scenario_subset[randint]
            key, subkey = random.split(key)
        else:
            scenario = self.scenario
        full_state, info = lax.switch(scenario, [self._generate_circular_crossing_episode, 
                                                self._generate_parallel_traffic_episode,
                                                self._generate_perpendicular_traffic_episode,
                                                self._generate_robot_crowding_episode], subkey)
        return full_state, key, info
    
    @partial(jit, static_argnames=("self"))
    def _generate_circular_crossing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.zeros((self.n_humans+1, 2))
        disturbed_points = disturbed_points.at[-1].set(jnp.array([0, -self.circle_radius]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                new_angle = random.uniform(subkey, shape=(1,), minval=0, maxval=2*jnp.pi)
                disturbance = random.uniform(subkey, shape=(1,), minval=-0.1, maxval=0.5)
                new_point = jnp.squeeze((self.circle_radius + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + 0.1)))
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None)
                return (disturbed_points, key, valid)
            disturbed_points, key = for_val
            disturbed_points, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (disturbed_points, key, False))
            return (disturbed_points, key)
    
        final_for_val = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))
        disturbed_points, key = final_for_val
        goal_angles = jnp.arctan2(-disturbed_points[:,1], -disturbed_points[:,0])

        # Assign the humans' and robot's positions
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    goal_angles[i],
                    0.]))
                , full_state)
            # Robot
            full_state = full_state.at[self.n_humans].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:4], jnp.pi/2, *full_state[self.n_humans,5:]]))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    0,
                    0.]))
                , full_state)
            # Robot
            full_state = full_state.at[self.n_humans].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(self.circle_radius * jnp.array([jnp.cos(goal_angles[i]), jnp.sin(goal_angles[i])])),
            humans_goal)
        robot_goal = np.array([0., self.circle_radius])

        # Obstacles
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": SCENARIOS.index('circular_crossing')}
        return full_state, info
    
    @partial(jit, static_argnames=("self"))
    def _generate_parallel_traffic_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([-self.traffic_length/2 + 1, 0.])) # Conform with Social-Navigation-PyEnvs
        # disturbed_points = disturbed_points.at[-1].set(jnp.array([-self.traffic_length/2, 0.]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1)
                new_point = jnp.array([-self.traffic_length/2 + 1 + normalized_point[0] * self.traffic_length, -self.traffic_height/2 + normalized_point[1] * self.traffic_height])
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + 0.1)))
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None)
                return (disturbed_points, key, valid)
            disturbed_points, key = for_val
            disturbed_points, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (disturbed_points, key, False))
            return (disturbed_points, key)
    
        final_for_val = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))
        disturbed_points, key = final_for_val

        # Assign the humans' and robot's positions
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    jnp.pi,
                    0.]))
                , full_state)
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    0,
                    0.]))
                , full_state)
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([-self.traffic_length/2-3, disturbed_points[i,1]])),
            humans_goal)
        robot_goal = - disturbed_points[-1]

        # Obstacles
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": SCENARIOS.index('parallel_traffic')}
        return full_state, info

    @partial(jit, static_argnames=("self"))
    def _generate_perpendicular_traffic_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([0, self.traffic_length/2]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1)
                new_point = jnp.array([-self.traffic_length/2 + 1 + normalized_point[0] * self.traffic_length, -self.traffic_height/2 + normalized_point[1] * self.traffic_height])
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + 0.1)))
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None)
                return (disturbed_points, key, valid)
            disturbed_points, key = for_val
            disturbed_points, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (disturbed_points, key, False))
            return (disturbed_points, key)
    
        final_for_val = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))
        disturbed_points, key = final_for_val

        # Assign the humans' and robot's positions
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    jnp.pi,
                    0.]))
                , full_state)
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    0,
                    0.]))
                , full_state)
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([-self.traffic_length/2-3, disturbed_points[i,1]])),
            humans_goal)
        robot_goal = np.array([0, -self.traffic_length/2])

        # Obstacles
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": SCENARIOS.index('perpendicular_traffic')}
        return full_state, info

    @partial(jit, static_argnames=("self"))
    def _generate_robot_crowding_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+2, 2)) * -1000
        disturbed_points = disturbed_points.at[-2].set(jnp.array([self.crowding_square_side/2-1, 0.]))
        disturbed_points = disturbed_points.at[-1].set(jnp.array([-self.crowding_square_side/2-1, 0.])) # This is needed to make sure the robot has space to reach its goal
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1)
                new_point = jnp.array([-self.crowding_square_side/2 + normalized_point[0] * self.crowding_square_side, -self.crowding_square_side/2 + normalized_point[1] * self.crowding_square_side])
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + jnp.max(humans_parameters[:, -1]) + 0.1 + self.robot_radius)))
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None)
                return (disturbed_points, key, valid)
            disturbed_points, key = for_val
            disturbed_points, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (disturbed_points, key, False))
            return (disturbed_points, key)
    
        final_for_val = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))
        disturbed_points, key = final_for_val

        # Assign the humans' and robot's positions
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    jnp.pi,
                    0.]))
                , full_state)
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = lax.fori_loop(
                0, 
                self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                    [disturbed_points[i,0],
                    disturbed_points[i,1],
                    0.,
                    0.,
                    0,
                    0.]))
                , full_state)
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-2], *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(disturbed_points[i]),
            humans_goal)
        robot_goal = disturbed_points[-1]

        # Obstacles
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": SCENARIOS.index('robot_crowding')}
        return full_state, info

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
        info["time"] += self.robot_dt
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
        # Compute action by derivative of robot position
        action = (new_state[-1,0:2] - state[-1,0:2]) / self.robot_dt
        ### Compute reward and outcome - WARNING: The old state is passed, not the updated one (but with the correct action applied)
        reward, outcome = self.reward_function(self._get_obs(state, old_info, action), old_info, self.robot_dt)
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, outcome
    
    @partial(jit, static_argnames=("self"))
    def step(self, state:jnp.ndarray, info:dict, action:jnp.ndarray, test:bool=False)-> tuple[jnp.ndarray, jnp.ndarray, dict, float, bool]:
        """
        Given an environment state, a dictionary containing additional information about the environment, and an action,
        this function computes the next state, the observation, the reward, and whether the episode is done.

        args:
        - state: jnp.ndarray containing the state of the environment.
        - info: dictionary containing additional information about the environment.
        - action: action to be taken by the robot.

        output:
        - new_state: jnp.ndarray containing the updated state of the environment.
        - obs: observation of the new state.
        - info: dictionary containing additional information about the environment.
        - reward: reward obtained in the transition.
        - outcome: dictionary indicating whether the episode is in a terminal state or not.
        """
        info["time"] += self.robot_dt
        ### Compute reward and outcome
        reward, outcome = self.reward_function(self._get_obs(state, info, action), info, self.robot_dt)
        ### Update state and info
        if self.robot_visible:
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.linalg.norm(action), 0., jnp.atan2(*jnp.flip(action)), 0.])]) # HSFM fictitious state
                out = lax.fori_loop(
                    0,
                    int(self.robot_dt/self.humans_dt),
                    lambda _ , x: self._update_state_info(*x, action),
                    (fictitious_state, info))
                new_state, new_info = out
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                fictitious_state = jnp.vstack([state[0:self.n_humans][:,0:4], jnp.array([*state[-1,0:2], *action])]) # SFM or ORCA fictitious state
                out = lax.fori_loop(
                    0,
                    int(self.robot_dt/self.humans_dt),
                    lambda _ , x: self._update_state_info(*x, action),
                    (fictitious_state, info))
                new_state, new_info = out
                new_state = jnp.pad(new_state, ((0,0),(0,2)))
            new_state = new_state.at[-1,2:].set(jnp.array([0., 0., 0., 0.]))
        else:
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                out = lax.fori_loop(
                    0,
                    int(self.robot_dt/self.humans_dt),
                    lambda _ , x: self._update_state_info(*x, action),
                    (state, info))
                new_state, new_info = out
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                out = lax.fori_loop(
                    0,
                    int(self.robot_dt/self.humans_dt),
                    lambda _ , x: self._update_state_info(*x, action),
                    (state[:,0:4], info))
                new_state, new_info = out
                new_state = jnp.pad(new_state, ((0,0),(0,2)))
        ### Test outcome computation (during tests we check for actual collision or reaching goal)
        @jit
        def _test_outcome(val:tuple):
            state, info, outcome = val
            outcome["success"] = jnp.linalg.norm(state[-1,0:2] - info["robot_goal"]) < self.robot_radius
            outcome["failure"] = jnp.any(jnp.linalg.norm(state[0:self.n_humans,0:2] - state[-1,0:2], axis=1) < info["humans_parameters"][:,0])
            outcome["timeout"] = jnp.all(jnp.array([outcome["timeout"], jnp.logical_not(outcome["failure"]), jnp.logical_not(outcome["success"])]))
            outcome["nothing"] = jnp.logical_not(jnp.any(jnp.array([outcome["success"], outcome["failure"], outcome["timeout"]])))
            return outcome
        outcome = lax.cond(test, lambda x: _test_outcome(x), lambda x: x[2], (new_state, info, outcome))
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, outcome

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
                       [robot_px, robot_py, pad, pad..]].
                      The length of each sub_array is 6.
        - new_key: PRNG key to be used in the next steps.
        - obs: observation of the initial state. The observation is a JAX array in the form:
               [[human1_px, human1_py, human1_vx, human1_vy, human1_radius],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius],
                [robot_px, robot_py, robot_ux, robot_uy, robot_radius]].
        - info: dictionary containing additional information about the environment.
        """
        initial_state, key, info = self._reset(key)
        return initial_state, key, self._get_obs(initial_state, info, jnp.zeros((2,))), info
    
    @partial(jit, static_argnames=("self"))
    def reset_custom_episode(self, key:random.PRNGKey, custom_episode:dict) -> tuple:
        """
        Given a custom episode data, this function resets the environment to start a new episode with the given data.

        args:
        - key; PRNG key (NOT USED).
        - custom_episode: dictionary containing the custom episode data. Its keys are:
            full_state (np.array): initial full state of the environment. WARNING: The velocity of humans is always in the global frame (for hsfm you should be using the velocity on the body frame)
            humans_goal (np.array): goal positions of the humans.
            robot_goal (np.array): goal position of the robot.
            static_obstacles (np.array): positions of the static obstacles.
            scenario (int): scenario of the episode.
            humans_radius (float): radius of the humans.
            humans_speed (float): max speed of the humans.

        output:
        - initial_state: initial full state of the environment. The state is a JAX array in the form:
                      [[*human1_state],
                       [*human2_state],
                       ...
                       [*humanN_state],
                       [robot_px, robot_py, pad, pad..]].
                      The length of each sub_array is 6.
        - key: PRNG key to be used in the next steps. (SAME AS THE INPUT ONE)
        - obs: observation of the initial state. The observation is a JAX array in the form:
               [[human1_px, human1_py, human1_vx, human1_vy, human1_radius],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius],
                [robot_px, robot_py, robot_u1, robot_u2, robot_radius]].
        - info: dictionary containing additional information about the environment.
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
        static_obstacles = jnp.array(custom_episode["static_obstacles"])
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": custom_episode["scenario"]}
        return full_state, key, self._get_obs(full_state, info, jnp.zeros((2,))), info
        