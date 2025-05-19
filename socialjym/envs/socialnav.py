import jax.numpy as jnp
from jax import random, jit, lax, debug, vmap
from functools import partial
from types import FunctionType

from socialjym.utils.aux_functions import is_multiple
from .base_env import BaseEnv, SCENARIOS, HUMAN_POLICIES, ROBOT_KINEMATICS, wrap_angle

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
            kinematics='holonomic',
            max_cc_delay = 5.,
            ccso_n_static_humans:int = 3,
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
            kinematics=kinematics,
            max_cc_delay=max_cc_delay,
            ccso_n_static_humans=ccso_n_static_humans,
            )
        ## Args validation
        assert humans_dt <= robot_dt, "The humans' time step must be less or equal than the robot's time step."
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        assert reward_function.kinematics == self.kinematics, "The reward function's kinematics must be the same as the environment's kinematics."
        if scenario == SCENARIOS.index('circular_crossing_with_static_obstacles') or (scenario == SCENARIOS.index('hybrid_scenario') and SCENARIOS.index('circular_crossing_with_static_obstacles') in hybrid_scenario_subset):
            assert n_humans > ccso_n_static_humans, "The number of static humans must be less than the total number of humans."
        ## Env initialization
        self.robot_dt = robot_dt
        self.reward_function = reward_function

    # --- Private methods ---

    def __repr__(self) -> str:
        return str(self.__dict__)

    @partial(jit, static_argnames=("self"))
    def _init_info(
        self,
        humans_goal:jnp.ndarray,
        robot_goal:jnp.ndarray,
        humans_parameters:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        current_scenario:int,
        humans_delay:jnp.ndarray,
    ) -> dict:
        """
        Initializes the info dictionary with the given parameters.

        args:
        - humans_goal: array of humans' goals.
        - robot_goal: array of robot's goal.
        - humans_parameters: array of humans' parameters.
        - static_obstacles: array of static obstacles.
        - current_scenario: current scenario index.
        - humans_delay: array of humans' delays.

        output:
        - info: dictionary containing the initialized values.
        """
        return {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": current_scenario,
            "humans_delay": humans_delay,
            "step": 0,
            "return": 0.,
        }

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
        full_state, info = lax.switch(
            scenario, 
            [
                self._generate_circular_crossing_episode, 
                self._generate_parallel_traffic_episode,
                self._generate_perpendicular_traffic_episode,
                self._generate_robot_crowding_episode,
                self._generate_delayed_circular_crossing_episode,
                self._generate_circular_crossing_with_static_obstacles_episode,
                self._generate_crowd_navigation_episode,
            ], 
            subkey
        )
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
    
        disturbed_points, key = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))
        goal_angles = jnp.arctan2(-disturbed_points[:,1], -disturbed_points[:,0])

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], goal_angles[:-1]))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[-1].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:4], jnp.pi/2, *full_state[self.n_humans,5:]]))

        # Assign the humans' and robot goals
        humans_goal = self.circle_radius * jnp.array([jnp.cos(goal_angles[:-1]), jnp.sin(goal_angles[:-1])]).T
        robot_goal = jnp.array([0., self.circle_radius])

        # Obstacles
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('circular_crossing'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
        return full_state, info
    
    @partial(jit, static_argnames=("self"))
    def _generate_delayed_circular_crossing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        key, subkey = random.split(key)
        full_state, info = self._generate_circular_crossing_episode(key)
        possible_delays = jnp.arange(0., self.max_cc_delay + self.robot_dt, self.robot_dt)
        info["humans_delay"] = info["humans_delay"].at[:].set(random.choice(subkey, possible_delays, shape=(self.n_humans,)))
        info["current_scenario"] = SCENARIOS.index('delayed_circular_crossing')
        # The next waypoint of humans is set to be its initial position
        info["humans_goal"] = info["humans_goal"].at[:].set(-info["humans_goal"])
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
            return disturbed_points, key
    
        disturbed_points, key = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.ones((self.n_humans,)) * jnp.pi))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
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
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('parallel_traffic'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
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
            return disturbed_points, key
    
        disturbed_points, key = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.ones((self.n_humans,)) * jnp.pi))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:4], -jnp.pi/2, *full_state[self.n_humans,5:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([-self.traffic_length/2-3, disturbed_points[i,1]])),
            humans_goal)
        robot_goal = jnp.array([0, -self.traffic_length/2])

        # Obstacles
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('perpendicular_traffic'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
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
            return disturbed_points, key
    
        disturbed_points, key = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key))

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-2], jnp.ones((self.n_humans,)) * jnp.pi))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-2], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-2], *full_state[self.n_humans,2:4], jnp.pi, *full_state[self.n_humans,5:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(disturbed_points[i]),
            humans_goal)
        robot_goal = disturbed_points[-1]

        # Obstacles
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('robot_crowding'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
        return full_state, info

    @partial(jit, static_argnames=("self"))
    def _generate_circular_crossing_with_static_obstacles_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        inner_circle_radius = self.circle_radius - 3.

        # Assign radius and max velocity to static obstacles
        @jit
        def _overwrite_radius_and_vel(
                parameters:jnp.ndarray, 
                idx:int, 
                radius:float, 
                max_vel:float, 
                key:random.PRNGKey
            ) -> jnp.ndarray:
            parameters = lax.cond(
                idx < (self.ccso_n_static_humans),
                lambda _: parameters.at[0:3].set(jnp.array([
                    jnp.squeeze(radius + random.uniform(key, shape=(1,), minval=-0.2, maxval=0.2)),
                    parameters[1], 
                    max_vel
                ])),
                lambda _: parameters,
                None
            )
            return parameters
        key, subkey = random.split(key)
        subkeys = random.split(subkey, num=self.n_humans)
        humans_parameters = vmap(_overwrite_radius_and_vel, in_axes=(0, 0, None, None, 0))(
            humans_parameters, 
            jnp.arange(self.n_humans), 
            1., 
            0., 
            subkeys)

        # Randomly generate the humans' positions
        disturbed_points = jnp.zeros((self.n_humans+1, 2))
        disturbed_points = disturbed_points.at[-1].set(jnp.array([0, -self.circle_radius]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid, inner_circle_radius = while_val
                key, subkey = random.split(key)
                new_angle = lax.cond(
                    i < (self.ccso_n_static_humans),
                    lambda _: (jnp.pi / int(self.ccso_n_static_humans)) * (-0.5 + 2 * i + random.uniform(subkey, shape=(1,), minval=-0.25, maxval=0.25)),
                    lambda _: random.uniform(subkey, shape=(1,), minval=0, maxval=2*jnp.pi),  # 2 * jnp.pi * (i - self.ccso_n_static_humans) / (self.n_humans - self.ccso_n_static_humans) + random.uniform(subkey, shape=(1,), minval=-0.05, maxval=0.05),
                    None
                )
                key, subkey = random.split(key)
                disturbance = lax.cond(
                    i < (self.ccso_n_static_humans),
                    lambda _: random.uniform(subkey, shape=(2,), minval=-0.1, maxval=0.1),
                    lambda _: random.uniform(subkey, shape=(2,), minval=-0.35, maxval=0.35),
                    None
                )
                new_point = lax.cond(
                    i < (self.ccso_n_static_humans),
                    lambda _: inner_circle_radius * jnp.squeeze(jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)])) + disturbance,
                    lambda _: self.circle_radius * jnp.squeeze(jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)])) + disturbance,
                    None
                )
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1) - (jnp.append(humans_parameters[:, 0], self.robot_radius) + humans_parameters[i, 0] + 0.2)
                valid = jnp.all(differences >= 0)
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None
                )
                return (disturbed_points, key, valid, inner_circle_radius)
            disturbed_points, key, inner_circle_radius = for_val
            disturbed_points, key, _, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (disturbed_points, key, False, inner_circle_radius))
            return (disturbed_points, key, inner_circle_radius)
    
        disturbed_points, key, _ = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, key, inner_circle_radius))
        goal_angles = jnp.arctan2(-disturbed_points[:,1], -disturbed_points[:,0])

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], goal_angles[:-1]))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[-1].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:4], jnp.pi/2, *full_state[self.n_humans,5:]]))

        # Assign the humans' and robot goals
        @jit
        def _set_humans_goal(idx:int, goal_angle:float, point:jnp.ndarray) -> jnp.ndarray:
            goal = lax.cond(
                idx < (self.ccso_n_static_humans),
                lambda _: point,
                lambda _: self.circle_radius * jnp.array([jnp.cos(goal_angle), jnp.sin(goal_angle)]).T,
                None
            )
            return goal
        humans_goal = vmap(_set_humans_goal, in_axes=(0, 0, 0))(
            jnp.arange(self.n_humans), 
            goal_angles[:-1], 
            disturbed_points[:-1]
        )
        robot_goal = jnp.array([0., self.circle_radius])

        # Obstacles
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('circular_crossing_with_static_obstacles'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
        return full_state, info

    @partial(jit, static_argnames=("self"))
    def _generate_crowd_navigation_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        positions = jnp.ones((self.n_humans+1, 2)) * -1000
        positions = positions.at[-1].set(jnp.array([0, -self.circle_radius]))
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                points, key, valid = while_val
                key, subkey = random.split(key)
                new_angle = random.uniform(subkey, shape=(1,), minval=0, maxval=2*jnp.pi)
                key, subkey = random.split(key)
                new_distance = random.uniform(subkey, shape=(1,), minval=0., maxval=self.circle_radius)
                new_point = jnp.squeeze(new_distance * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
                differences = jnp.linalg.norm(points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]))))
                points = lax.cond(
                    valid,
                    lambda _: points.at[i].set(new_point),
                    lambda _: points,
                    operand=None)
                return (points, key, valid)
            points, key = for_val
            points, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[2]), _while_body, (points, key, False))
            return (points, key)
        positions, key = lax.fori_loop(0, self.n_humans, _fori_body, (positions, key))
        
        @jit 
        def _goal_comp(position:jnp.ndarray, subkey:random.PRNGKey) -> jnp.ndarray:
            position_angle = jnp.atan2(position[1], position[0])
            new_angle = wrap_angle(random.uniform(subkey, shape=(), minval=position_angle-jnp.pi/4, maxval=position_angle+jnp.pi/4))
            new_distance = random.uniform(subkey, shape=(1,), minval=0., maxval=self.circle_radius)
            return jnp.squeeze(new_distance * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
        key, subkey = random.split(key)
        subkeys = random.split(subkey, num=self.n_humans)
        human_goals = vmap(_goal_comp, in_axes=(0,0))(
            positions[:-1], 
            subkeys,
        )
        goal_angles = jnp.arctan2(human_goals[:,1], human_goals[:,0])

        # Assign the humans' and robot's positions
        @jit
        def _set_state(position:jnp.ndarray, theta:float) -> jnp.ndarray:
            return jnp.array([
                position[0],
                position[1],
                0.,
                0.,
                theta,
                0.
            ])
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(positions[:-1], goal_angles))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(positions[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[-1].set(jnp.array([0., -self.circle_radius, *full_state[self.n_humans,2:4], jnp.pi/2, *full_state[self.n_humans,5:]]))

        # Assign the humans' and robot goals
        humans_goal = self.circle_radius * jnp.array([jnp.cos(goal_angles), jnp.sin(goal_angles)]).T
        robot_goal = jnp.array([0., self.circle_radius])

        # Obstacles
        static_obstacles = jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan) # dummy obstacles
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=SCENARIOS.index('crowd_navigation'),
            humans_delay=jnp.zeros((self.n_humans,)),
        )
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
        ### Test outcome computation (during tests we check for actual collision or reaching goal)
        @jit
        def _test_outcome(val:tuple):
            state, info, outcome = val
            outcome["success"] = jnp.linalg.norm(state[-1,0:2] - info["robot_goal"]) < self.robot_radius
            outcome["failure"] = jnp.all(jnp.array([jnp.any(jnp.linalg.norm(state[0:self.n_humans,0:2] - state[-1,0:2], axis=1) < (info["humans_parameters"][:,0] + self.robot_radius)), jnp.logical_not(outcome["success"])]))
            outcome["timeout"] = jnp.all(jnp.array([outcome["timeout"], jnp.logical_not(outcome["failure"]), jnp.logical_not(outcome["success"])]))
            outcome["nothing"] = jnp.logical_not(jnp.any(jnp.array([outcome["success"], outcome["failure"], outcome["timeout"]])))
            return outcome
        outcome = lax.cond(test, lambda x: _test_outcome(x), lambda x: x[2], (new_state, info, outcome))
        ### If done and reset_if done, automatically reset the environment
        new_state, reset_key, new_info = lax.cond(
            (reset_if_done) & (~(outcome["nothing"])),
            lambda x: self._reset(x[1]),
            lambda x: x,
            (new_state, reset_key, new_info)
        )
        ### Update time, step, return
        new_info["time"] += self.robot_dt
        new_info["step"] += 1
        new_info["return"] += pow(self.reward_function.gamma, info["step"] * self.robot_dt * self.reward_function.v_max) * reward
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
            scenario (int): scenario of the episode.
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
        static_obstacles = jnp.array(custom_episode["static_obstacles"])
        # Info
        info = self._init_info(
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=custom_episode["scenario"],
            humans_delay=jnp.zeros((self.n_humans,)),
        )
        return full_state, key, self._get_obs(full_state, info, jnp.zeros((2,))), info, {"success": False, "failure": False, "timeout": False, "nothing": True}
        