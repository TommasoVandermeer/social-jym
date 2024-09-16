import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, debug
from functools import partial
from types import FunctionType

from jhsfm.hsfm import step as humans_step
from jhsfm.utils import get_standard_humans_parameters
from socialjym.utils.aux_functions import is_multiple
from .base_env import BaseEnv

SCENARIOS = ["circular_crossing", "parallel_traffic", "hybrid_scenario"]
HUMAN_POLICIES = ["orca", "sfm", "hsfm"]

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
            traffic_height=3, 
            traffic_length=14,
            time_limit=50
        ) -> None:
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
        self.robot_discomfort_dist = robot_discomfort_dist
        self.robot_visible = robot_visible
        self.circle_radius = circle_radius
        self.traffic_height = traffic_height
        self.traffic_length = traffic_length
        self.time_limit = time_limit
        # Default args
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
        else: # For sfm and orca policies the state already contains the linear velocities
            obs = lax.fori_loop(0, self.n_humans, lambda i, obs: obs.at[i].set(jnp.array([state[i,0], state[i,1], state[i,2], state[i,3], info['humans_parameters'][i,0]])), obs)
        obs = obs.at[self.n_humans].set(jnp.array([*state[self.n_humans,0:2], *action, self.robot_radius]))
        return obs

    @partial(jit, static_argnames=("self"))
    def _reset(self, key:random.PRNGKey) -> tuple[jnp.ndarray, random.PRNGKey, dict]:
        key, subkey = random.split(key)
        full_state, info = lax.switch(self.scenario, [self._generate_circular_crossing_episode, 
                                                      self._generate_parallel_traffic_episode], subkey)
        return full_state, key, info
    
    @partial(jit, static_argnames=("self"))
    def _generate_circular_crossing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))

        ### DETERMINISTIC CIRCULAR CROSSING (spread humans evenly in the circle perimeter)
        ## Humans state, goals and parameters
        # humans_goal = jnp.zeros((self.n_humans, 2))
        # angle_width = (2 * jnp.pi) / (self.n_humans + 1)
        # full_state = lax.fori_loop(0, self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
        #                                     [self.circle_radius * jnp.sin((i+1) * angle_width),
        #                                     -self.circle_radius * jnp.cos((i+1) * angle_width),
        #                                     0.,
        #                                     0.,
        #                                     (jnp.pi/2) + (i+1) * angle_width,
        #                                     0.]))
        #              , full_state)
        # humans_goal = lax.fori_loop(0, self.n_humans, lambda i, humans_goal:
        #               humans_goal.at[i].set(jnp.array([-full_state[i,0], -full_state[i,1]]))
        #               , humans_goal)
        # humans_goal = humans_goal
        # humans_parameters = get_standard_humans_parameters(self.n_humans)

        ### STOCHASTIC CIRCULAR CROSSING
        ## Humans state, goals and parameters
        humans_goal = jnp.zeros((self.n_humans, 2))
        # TODO: Adjust parameters based on humans policy
        humans_parameters = get_standard_humans_parameters(self.n_humans)
        # Randomly generate the humans' positions
        min_dist = 2 * jnp.max(humans_parameters[:, 0]) + 0.1 # Calculate the minimum distance between humans
        min_angle = 2 * jnp.arcsin(min_dist / (2 * self.circle_radius)) # Calculate the minimum angular distance in radians for inter-point distance
        exclusion_angle = 3/2 * jnp.pi # Calculate the angular exclusion zone for distance from (0, -self.circle_radius)
        min_exclusion_angle = exclusion_angle - min_angle
        max_exclusion_angle = exclusion_angle + min_angle
        def is_valid(angles):
            angular_diffs = jnp.diff(jnp.concatenate([angles, angles[:1] + 2*jnp.pi]))
            valid_distances = jnp.all(angular_diffs >= min_angle)
            valid_from_south = jnp.all((angles < min_exclusion_angle) | (angles > max_exclusion_angle))
            return valid_distances & valid_from_south
        def loop_body(val):
            angles, key = val
            key, subkey = random.split(key)
            new_angles = random.uniform(subkey, shape=(self.n_humans,), minval=0, maxval=2*jnp.pi)
            new_angles = jnp.sort(new_angles)
            return lax.cond(is_valid(new_angles),
                                lambda _: (new_angles, key),
                                lambda _: (angles, key),
                                operand=None)
        angles = random.uniform(key, shape=(self.n_humans,), minval=0, maxval=2*jnp.pi) # Initialize with random angles
        angles = jnp.sort(angles)
        angles, key = lax.while_loop(lambda val: ~is_valid(val[0]),
                                    loop_body,
                                    (angles, key))
        key, subkey = random.split(key)
        disturbance = random.uniform(subkey, shape=(self.n_humans,), minval=-0.05, maxval=0.5) # Add some disturbance to the outer circle
        points = self.circle_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) # Convert angles to (x, y) coordinates on the circle
        disturbed_points = lax.fori_loop(0, self.n_humans, lambda i, points: points.at[i].set(points[i] + disturbance[i] * jnp.array([jnp.cos(angles[i]), jnp.sin(angles[i])])), points)
        
        # TODO: Modify full state based on humans policy
        # Assign the humans' positions and goals
        full_state = lax.fori_loop(
            0, 
            self.n_humans, lambda i, full_state: full_state.at[i].set(jnp.array(
                [disturbed_points[i,0],
                disturbed_points[i,1],
                0.,
                0.,
                angles[i] + jnp.pi,
                0.]))
            , full_state)
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([-points[i,0], -points[i,1]]))
            , humans_goal)
        # Robot state and goal
        full_state = full_state.at[self.n_humans].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:4], jnp.pi/2, *full_state[self.n_humans,5:]]))
        robot_goal = np.array([0., self.circle_radius])
        # Obstacles
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        # Info
        info = {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.}
        return full_state, info
    
    @partial(jit, static_argnames=("self"))
    def _generate_parallel_traffic_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        # TODO: Implement this method
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = get_standard_humans_parameters(self.n_humans)
        static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        robot_goal = np.array([self.traffic_length/2, 0])
        info = {"humans_goal": humans_goal, "robot_goal": robot_goal, "humans_parameters": humans_parameters, "static_obstacles": static_obstacles, "time": 0.}
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
        - done: boolean indicating whether the episode is done.
        """
        info["time"] += self.robot_dt
        humans_goal = info["humans_goal"]
        humans_parameters = info["humans_parameters"]
        static_obstacles = info["static_obstacles"]
        ### Compute reward and done
        reward, done = self.reward_function(self._get_obs(state, info, action), info, self.robot_dt)
        ### Update state
        # TODO: update humans depending on their policy
        goals = jnp.vstack((humans_goal, info["robot_goal"]))
        parameters = jnp.vstack((humans_parameters, jnp.array([self.robot_radius, *get_standard_humans_parameters(1)[0,1:-1], 0.2]))) # Add safety space of 0.1 to robot
        if self.robot_visible:
            new_state = lax.fori_loop(0,
                                      int(self.robot_dt/self.humans_dt),
                                      lambda _ , x: humans_step(x, goals, parameters, static_obstacles, self.humans_dt),
                                      state)
        else:
            new_state = lax.fori_loop(0,
                                      int(self.robot_dt/self.humans_dt),
                                      lambda _ , x: jnp.vstack([humans_step(x[0:self.n_humans], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles, self.humans_dt), 
                                                                humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[self.n_humans]]),
                                      state)
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'): # In case of hsfm convert robot body velocities to linear velocity
            action = (new_state[self.n_humans,0:2] - state[self.n_humans,0:2]) / self.robot_dt
            # action = jnp.matmul(jnp.array([[jnp.cos(state[self.n_humans,4]), -jnp.sin(state[self.n_humans,4])], [jnp.sin(state[self.n_humans,4]), jnp.cos(state[self.n_humans,4])]]), state[self.n_humans,2:4])
        else:
            action = state[self.n_humans,2:4]
        ### Update humans goal
        if self.scenario == SCENARIOS.index('circular_crossing'):
            info["humans_goal"] = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, goals: lax.cond(
                    jnp.linalg.norm(new_state[i,0:2] - info["humans_goal"][i]) <= info["humans_parameters"][i,0], 
                    lambda x: x.at[i].set(-x[i]), 
                    lambda x: x, 
                    goals),
                info["humans_goal"])
        elif self.scenario == SCENARIOS.index('parallel_traffic'):
            # TODO: update humans goal depending on each scenario
            pass
        return new_state, self._get_obs(new_state, info, action), info, reward, done
    
    @partial(jit, static_argnames=("self"))
    def step(self, state:jnp.ndarray, info:dict, action:jnp.ndarray)-> tuple[jnp.ndarray, jnp.ndarray, dict, float, bool]:
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
        # TODO: update humans depending on their policy
        if self.robot_visible:
            goals = jnp.vstack((humans_goal, info["robot_goal"]))
            parameters = jnp.vstack((humans_parameters, jnp.array([self.robot_radius, *get_standard_humans_parameters(1)[0,1:]])))
            fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2],*action,0.,0.])])
            new_state = lax.fori_loop(0,
                                      int(self.robot_dt/self.humans_dt),
                                      lambda _ , x: jnp.vstack([humans_step(x, goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                                                                jnp.array([x[-1,0]+action[0]*self.humans_dt, x[-1,1]+action[1]*self.humans_dt, *action, 0., 0.])]),
                                      fictitious_state)
            new_state = new_state.at[self.n_humans,2:].set(jnp.array([0., 0., 0., 0.]))
        else:
            ## Update humans
            new_state = lax.fori_loop(0,
                                      int(self.robot_dt/self.humans_dt),
                                      lambda _ , x: jnp.vstack([humans_step(x[0:self.n_humans], humans_goal, humans_parameters, static_obstacles, self.humans_dt), x[-1]]),
                                      state)
            ## Update robot
            new_state = new_state.at[self.n_humans,0:2].set(jnp.array([new_state[self.n_humans,0] + action[0] * self.robot_dt, new_state[self.n_humans,1] + action[1] * self.robot_dt]))
        ### Update humans goal
        if self.scenario == SCENARIOS.index('circular_crossing'):
            info["humans_goal"] = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, goals: lax.cond(
                    jnp.linalg.norm(new_state[i,0:2] - info["humans_goal"][i]) <= info["humans_parameters"][i,0], 
                    lambda x: x.at[i].set(-x[i]), 
                    lambda x: x, 
                    goals),
                info["humans_goal"])
        elif self.scenario == SCENARIOS.index('parallel_traffic'):
            # TODO: update humans goal depending on each scenario
            pass
        return new_state, self._get_obs(new_state, info, action), info, reward, done

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

