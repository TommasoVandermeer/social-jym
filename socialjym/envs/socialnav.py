import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from jhsfm.hsfm import step as hsfm_humans_step
from jsfm.sfm import step as sfm_humans_step
from jhsfm.utils import get_standard_humans_parameters as hsfm_get_standard_humans_parameters
from jsfm.utils import get_standard_humans_parameters as sfm_get_standard_humans_parameters
from socialjym.utils.aux_functions import is_multiple, SCENARIOS, HUMAN_POLICIES
from .base_env import BaseEnv

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
            robot_discomfort_dist=0.2, 
            robot_visible=False, 
            circle_radius=7, 
            traffic_height=5, 
            traffic_length=14,
            crowding_square_side=14,
            time_limit=50,
            lidar_angular_range=jnp.pi,
            lidar_max_dist=10.,
            lidar_num_rays=60,
        ) -> None:
        ## Initialize the BaseEnv
        super().__init__(lidar_angular_range=lidar_angular_range, lidar_max_dist=lidar_max_dist, lidar_num_rays=lidar_num_rays)
        ## Args validation
        assert scenario in SCENARIOS, f"Invalid scenario. Choose one of {SCENARIOS}"
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        ## Env initialization
        # Configurable args
        self.robot_radius = robot_radius
        self.robot_dt = robot_dt
        self.humans_dt = humans_dt
        self.scenario = SCENARIOS.index(scenario)
        self.n_humans = n_humans
        self.humans_policy = HUMAN_POLICIES.index(humans_policy)
        if humans_policy == 'hsfm': 
            self.humans_step = hsfm_humans_step
            self.get_standard_humans_parameters = hsfm_get_standard_humans_parameters
        elif humans_policy == 'sfm':
            self.humans_step = sfm_humans_step
            self.get_standard_humans_parameters = sfm_get_standard_humans_parameters
        elif humans_policy == 'orca':
            raise NotImplementedError("ORCA policy is not implemented yet.")
        self.robot_discomfort_dist = robot_discomfort_dist
        self.robot_visible = robot_visible
        self.circle_radius = circle_radius
        self.traffic_height = traffic_height
        self.traffic_length = traffic_length
        self.crowding_square_side = crowding_square_side
        self.time_limit = time_limit
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
                [[human1_px, human1_py, human1_vx, human1_vy, human1_radius],
                [human2_px, human2_py, human2_vx, human2_vy, human2_radius],
                ...
                [humanN_px, humanN_py, humanN_vx, humanN_vy, humanN_radius],
                [robot_px, robot_py, robot_ux, robot_uy, robot_radius]].
        """
        obs = jnp.ones((self.n_humans+1, 5))
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'): # In case of hsfm convert humans body velocities to linear velocities
            linear_velocities = jnp.ones((self.n_humans, 2))
            linear_velocities = lax.fori_loop(0, self.n_humans, lambda i, lv: lv.at[i].set(jnp.matmul(jnp.array([[jnp.cos(state[i,4]), -jnp.sin(state[i,4])], [jnp.sin(state[i,4]), jnp.cos(state[i,4])]]), state[i,2:4])) , linear_velocities)
            obs = lax.fori_loop(0, self.n_humans, lambda i, obs: obs.at[i].set(jnp.array([state[i,0], state[i,1], linear_velocities[i,0], linear_velocities[i,1], info['humans_parameters'][i,0]])), obs)
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'): # For sfm and orca policies the state already contains the linear velocities
            obs = lax.fori_loop(0, self.n_humans, lambda i, obs: obs.at[i].set(jnp.array([state[i,0], state[i,1], state[i,2], state[i,3], info['humans_parameters'][i,0]])), obs)
        obs = obs.at[-1].set(jnp.array([*state[-1,0:2], *action, self.robot_radius]))
        return obs

    @partial(jit, static_argnames=("self"))
    def _reset(self, key:random.PRNGKey) -> tuple[jnp.ndarray, random.PRNGKey, dict]:
        key, subkey = random.split(key)
        if self.scenario == SCENARIOS.index('hybrid_scenario'):
            scenario = random.randint(subkey, shape=(), minval=0, maxval=4)
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
        disturbed_points = disturbed_points.at[-1].set(jnp.array([-self.traffic_length/2, 0.]))
        
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
        robot_goal = np.array([self.traffic_length/2, 0.])

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
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([self.crowding_square_side/2-1, 0.]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1)
                new_point = jnp.array([-self.crowding_square_side/2 + normalized_point[0] * self.crowding_square_side, -self.crowding_square_side/2 + normalized_point[1] * self.crowding_square_side])
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + 0.1 + self.robot_radius)))
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
            lambda i, humans_goal: humans_goal.at[i].set(disturbed_points[i]),
            humans_goal)
        robot_goal = np.array([-self.crowding_square_side/2-1, 0.])

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

    @partial(jit, static_argnames=("self"))
    def _update_humans_goal_and_state_for_each_scenario(self, state:jnp.ndarray, info:dict):

        @jit
        def _update_circular_crossing(val:tuple):
            info, state = val
            info["humans_goal"] = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, goals: lax.cond(
                    jnp.linalg.norm(state[i,0:2] - info["humans_goal"][i]) <= info["humans_parameters"][i,0], 
                    lambda x: x.at[i].set(-info["humans_goal"][i]), 
                    lambda x: x, 
                    goals),
                info["humans_goal"])
            return (info, state)
        
        @jit
        def _update_traffic_scenarios(val:tuple):
            info, state = val
            state = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, state: lax.cond(
                    jnp.linalg.norm(state[i,0:2] - info["humans_goal"][i]) <= info["humans_parameters"][i,0] + 2, 
                    lambda x: x.at[i,0:4].set(jnp.array([
                        jnp.max(jnp.append(x[:,0]+(jnp.max(jnp.append(info["humans_parameters"][:,0],self.robot_radius))*2)+(jnp.max(info["humans_parameters"][:,-1])*2)+0.05, self.traffic_length/2+1)), 
                        *x[i,1:4]])), 
                    lambda x: x, 
                    state),
                state)
            return (info, state)

        info_and_state = lax.switch(
            info["current_scenario"], 
            [_update_circular_crossing, 
            _update_traffic_scenarios, 
            _update_traffic_scenarios, 
            lambda x: x], 
            (info, state))
        info, state = info_and_state
            
        return info, state

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
        - done: boolean indicating whether the episode is done.
        """
        info["time"] += self.robot_dt
        humans_goal = info["humans_goal"]
        humans_parameters = info["humans_parameters"]
        static_obstacles = info["static_obstacles"]
        old_info = info.copy()
        ### Update state
        goals = jnp.vstack((humans_goal, info["robot_goal"]))
        parameters = jnp.vstack((humans_parameters, jnp.array([self.robot_radius, *self.get_standard_humans_parameters(1)[0,1:-1], 0.1]))) # Add safety space of 0.1 to robot
        if self.robot_visible:
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                new_state = lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt),
                                        state)
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                new_state = jnp.pad(lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt),
                                        state[:,0:4]),
                                    ((0,0),(0,2)))
        else:
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                new_state = lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x[0:self.n_humans], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles, self.humans_dt), 
                                                                self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[self.n_humans]]),
                                        state)
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                new_state = jnp.pad(lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x[0:self.n_humans], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles, self.humans_dt), 
                                                                self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[self.n_humans]]),
                                        state[:,0:4]),
                                    ((0,0),(0,2)))
        # Compute action by derivative of robot position
        action = (new_state[self.n_humans,0:2] - state[self.n_humans,0:2]) / self.robot_dt
        ### Update humans goal or state depending on the scenario
        new_info, new_state = self._update_humans_goal_and_state_for_each_scenario(new_state, info)
        ### Compute reward and done - WARNING: The old state is passed, not the updated one (but with the correct action applied)
        reward, done = self.reward_function(self._get_obs(state, old_info, action), old_info, self.robot_dt)
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, done
    
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
        - done: boolean indicating whether the episode is done.
        """
        info["time"] += self.robot_dt
        humans_goal = info["humans_goal"]
        humans_parameters = info["humans_parameters"]
        static_obstacles = info["static_obstacles"]
        ### Compute reward and done
        reward, done = self.reward_function(self._get_obs(state, info, action), info, self.robot_dt)
        ### Update state
        if self.robot_visible:
            goals = jnp.vstack((humans_goal, info["robot_goal"]))
            parameters = jnp.vstack((humans_parameters, jnp.array([self.robot_radius, *self.get_standard_humans_parameters(1)[0,1:]])))
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.linalg.norm(action), 0., jnp.atan2(*jnp.flip(action)), 0.])]) # HSFM fictitious state
                new_state = lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                                                                jnp.array([x[-1,0]+action[0]*self.humans_dt, x[-1,1]+action[1]*self.humans_dt, *x[-1,2:]])]),
                                        fictitious_state)
                new_state = new_state.at[self.n_humans,2:].set(jnp.array([0., 0., 0., 0.]))
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                fictitious_state = jnp.vstack([state[0:self.n_humans][:,0:4], jnp.array([*state[-1,0:2], *action])]) # SFM or ORCA fictitious state
                new_state = jnp.pad(lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                                                                jnp.array([x[-1,0]+action[0]*self.humans_dt, x[-1,1]+action[1]*self.humans_dt, *x[-1,2:]])]),
                                        fictitious_state),
                                    ((0,0),(0,2)))
                new_state = new_state.at[self.n_humans,2:].set(jnp.array([0., 0., 0., 0.]))
        else:
            ## Update humans
            if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
                new_state = lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x[0:self.n_humans], humans_goal, humans_parameters, static_obstacles, self.humans_dt), x[-1]]),
                                        state)
            elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
                new_state = jnp.pad(lax.fori_loop(0,
                                        int(self.robot_dt/self.humans_dt),
                                        lambda _ , x: jnp.vstack([self.humans_step(x[0:self.n_humans], humans_goal, humans_parameters, static_obstacles, self.humans_dt), x[-1]]),
                                        state[:,0:4]),
                                    ((0,0),(0,2)))
            ## Update robot
            new_state = new_state.at[self.n_humans,0:2].set(jnp.array([new_state[self.n_humans,0] + action[0] * self.robot_dt, new_state[self.n_humans,1] + action[1] * self.robot_dt]))
        ### Test done computation (during tests we check for actual collision or reaching goal)
        done = lax.cond(
            test,
            lambda _: jnp.any(jnp.array([
                jnp.linalg.norm(new_state[-1,0:2] - info["robot_goal"]) < self.robot_radius, # reaching goal
                jnp.any(jnp.linalg.norm(new_state[0:self.n_humans,0:2] - new_state[-1,0:2], axis=1) < info["humans_parameters"][:,0]), # collision
                jnp.any(info["time"] >= self.time_limit)])), # timeout
            lambda _: done,
            None)
        ### Update humans goal or state depending on the scenario
        new_info, new_state = self._update_humans_goal_and_state_for_each_scenario(new_state, info)
        return new_state, self._get_obs(new_state, new_info, action), new_info, reward, done

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
                      The length of each sub_array depends on the humans policy used (hsfm:6, sfm: 4, orca:4).
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
        