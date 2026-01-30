from abc import ABC, abstractmethod
from functools import partial
from jax import jit, vmap, lax, random, debug
import jax.numpy as jnp

from jhsfm.hsfm import step as hsfm_humans_step
from jsfm.sfm import step as sfm_humans_step
from jorca.orca import step as orca_humans_step
from jhsfm.utils import get_standard_humans_parameters as hsfm_get_standard_humans_parameters
from jsfm.utils import get_standard_humans_parameters as sfm_get_standard_humans_parameters
from jorca.utils import get_standard_humans_parameters as orca_get_standard_humans_parameters

SCENARIOS = [
    "circular_crossing", 
    "parallel_traffic", 
    "perpendicular_traffic", 
    "robot_crowding", 
    "delayed_circular_crossing",
    "circular_crossing_with_static_obstacles",
    "crowd_navigation",
    "corner_traffic",
    "hybrid_scenario" # Make sure to update this list (if new scenarios are added) but always leave the last element as "hybrid_scenario"
] 
HUMAN_POLICIES = [
    "orca",
    "sfm", 
    "hsfm"
]
ROBOT_KINEMATICS = [
    "holonomic",
    "unicycle"
]
ENVIRONMENTS = [
    "socialnav",
    "lasernav",
]
EPSILON = 1e-5 # Small value to avoid math overflow

@jit
def wrap_angle(theta:float) -> float:
    """
    This function wraps the angle to the interval [-pi, pi]
    
    args:
    - theta: angle to be wrapped
    
    output:
    - wrapped_theta: angle wrapped to the interval [-pi, pi]
    """
    wrapped_theta = lax.cond(
        theta == jnp.pi,
        lambda x: x,
        lambda x: (x + jnp.pi) % (2 * jnp.pi) - jnp.pi,
        theta)
    return wrapped_theta

@jit
def is_multiple(number:float, dividend:float, tolerance:float=1e-7) -> bool:
    """
    Checks if a number (also a float) is a multiple of another number within a given tolerance error.
    """
    mod = number % dividend
    return jnp.any(jnp.array([abs(mod) <= tolerance,abs(dividend - mod) <= tolerance]))

@jit
def roto_translate_pose_and_vel(position, orientation, velocity, ref_position, ref_orientation):
    """Roto-translate a 2D pose and a velocity to a given reference pose."""
    c, s = jnp.cos(-ref_orientation), jnp.sin(-ref_orientation)
    R = jnp.array([[c, -s],
                [s,  c]])
    translated_position = position - ref_position
    rotated_position = R @ translated_position
    rotated_orientation = orientation - ref_orientation
    rotated_velocity = R @ velocity
    return rotated_position, rotated_orientation, rotated_velocity

@jit
def roto_translate_poses_and_vels(positions, orientations, velocities, ref_position, ref_orientation):
    """Roto-translate a batch of 2D poses and velocities to a given reference pose."""
    return vmap(roto_translate_pose_and_vel, in_axes=(0, 0, 0, None, None))(positions, orientations, velocities, ref_position, ref_orientation)

@jit
def roto_translate_obstacle_segments(obstacle_segments, ref_position, ref_orientation):
    # Translate segments to robot frame
    obstacle_segments = obstacle_segments.at[:, :, 0].set(obstacle_segments[:, :, 0] - ref_position[0])
    obstacle_segments = obstacle_segments.at[:, :, 1].set(obstacle_segments[:, :, 1] - ref_position[1])
    # Rotate segments by -ref_orientation
    c, s = jnp.cos(-ref_orientation), jnp.sin(-ref_orientation)
    rot = jnp.array([[c, -s], [s, c]])
    obstacle_segments = jnp.einsum('ij,klj->kli', rot, obstacle_segments)
    return obstacle_segments

@jit
def roto_translate_obstacles(obstacles, ref_positions, ref_orientations):
    return vmap(roto_translate_obstacle_segments, in_axes=(0, None, None))(obstacles, ref_positions, ref_orientations)

@jit
def thicken_obstacles(obstacles, thickness):
    """
    Transform a line segment obstacle into a rectangle with given thickness.
    args:
    - obstacles: jnp.ndarray of shape (..., 1, 2, 2), representing line segments defined by start and end points.
    - thickness: float, thickness of the obstacle rectangle.
    """
    p1 = obstacles[..., 0, 0, :] 
    p2 = obstacles[..., 0, 1, :]
    v = p2 - p1
    len_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
    u = v / (len_v + 1e-6)
    n = jnp.stack([-u[..., 1], u[..., 0]], axis=-1)
    offset = n * (thickness / 2.0)
    c1 = p1 + offset
    c2 = p1 - offset
    c3 = p2 - offset
    c4 = p2 + offset
    seg1 = jnp.stack([c1, c2], axis=-2)
    seg2 = jnp.stack([c2, c3], axis=-2)
    seg3 = jnp.stack([c3, c4], axis=-2)
    seg4 = jnp.stack([c4, c1], axis=-2)
    thick_obstacles = jnp.stack([seg1, seg2, seg3, seg4], axis=-3)
    return thick_obstacles

class BaseEnv(ABC):
    """
    Base class for social navigation environments.
    Defines all the scenarios, ray casting, hidden reset and info initialization.
    """
    def __init__(
        self,
        robot_radius:float, 
        robot_dt:float,
        humans_dt:float, 
        scenario:str, 
        n_humans:int, 
        n_obstacles:int,
        humans_policy:str, 
        robot_visible:bool, 
        circle_radius:float, 
        traffic_height:float,
        traffic_length:float,
        crowding_square_side:float,
        hybrid_scenario_subset: jnp.ndarray,
        lidar_angular_range:float,
        lidar_max_dist:float,
        lidar_num_rays:int,
        lidar_noise:bool,
        lidar_noise_fixed_std:float,
        lidar_noise_proportional_std:float,
        lidar_salt_and_pepper_prob:float,
        kinematics:str,
        max_cc_delay:float,
        ccso_n_static_humans:int,
        grid_map_computation:bool,
        grid_cell_size:float,
        grid_min_size:float,
        thick_default_obstacle:bool
    ) -> None:
        ## Args validation
        assert scenario in SCENARIOS or scenario is None, f"Invalid scenario. Choose one of {SCENARIOS}, or None for custom scenario."
        if scenario is None:
            print("\nWARNING: Custom scenario is selected. Make sure to implement the 'reset_custom_episode' method in the derived class (not 'reset').\n")
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert kinematics in ROBOT_KINEMATICS, f"Invalid robot kinematics. Choose one of {ROBOT_KINEMATICS}"
        if grid_map_computation:
            assert grid_cell_size > 0, "There should be at least one obstacle (also padding obstacles) to enable grid map computation."
        assert humans_dt <= robot_dt, "The humans' time step must be less or equal than the robot's time step."
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        if scenario == SCENARIOS.index('circular_crossing_with_static_obstacles') or (scenario == SCENARIOS.index('hybrid_scenario') and SCENARIOS.index('circular_crossing_with_static_obstacles') in hybrid_scenario_subset):
            assert n_humans > ccso_n_static_humans, "The number of static humans must be less than the total number of humans."
        ## Env initialization
        self.robot_dt = robot_dt
        self.robot_radius = robot_radius
        self.humans_dt = humans_dt
        if scenario is None:
            self.scenario = -1  # Custom scenario
        else:
            self.scenario = SCENARIOS.index(scenario)
        self.n_humans = n_humans
        self.n_obstacles = n_obstacles
        self.humans_policy = HUMAN_POLICIES.index(humans_policy)
        if humans_policy == 'hsfm': 
            self.humans_step = hsfm_humans_step
            self.get_standard_humans_parameters = hsfm_get_standard_humans_parameters
        elif humans_policy == 'sfm':
            self.humans_step = sfm_humans_step
            self.get_standard_humans_parameters = sfm_get_standard_humans_parameters
        elif humans_policy == 'orca':
            self.humans_step = orca_humans_step
            self.get_standard_humans_parameters = orca_get_standard_humans_parameters
            assert self.n_obstacles == 0, "ORCA human model does not support avoidance of static obstacles yet.\n"
            print("\nWARNING: ORCA human model (JORCA library) might still be buggy.")
            print("WARNING: ORCA human model is not properly optimized (JORCA library), RL training could be seriously slowed down. It is recommended to use it only for evaluation purposes.\n")
        self.robot_visible = robot_visible
        self.circle_radius = circle_radius
        self.traffic_height = traffic_height
        self.traffic_length = traffic_length
        self.crowding_square_side = crowding_square_side
        self.hybrid_scenario_subset = hybrid_scenario_subset
        self.lidar_angular_range = lidar_angular_range
        self.lidar_max_dist = lidar_max_dist
        self.lidar_num_rays = lidar_num_rays
        self.lidar_noise = lidar_noise
        self.lidar_noise_fixed_std = lidar_noise_fixed_std
        self.lidar_noise_proportional_std = lidar_noise_proportional_std
        self.lidar_salt_and_pepper_prob = lidar_salt_and_pepper_prob
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        self.max_cc_delay = max_cc_delay
        self.ccso_n_static_humans = ccso_n_static_humans
        self.thick_default_obstacle = thick_default_obstacle
        # Global planning parameters
        if grid_map_computation:
            print("\nWARNING: Grid map computation is enabled. This will slow down the simulation, especially if many static obstacles are present.\n")
        self.grid_map_computation = grid_map_computation
        self.grid_cell_size = grid_cell_size
        self.grid_min_size = grid_min_size
        ## Static obstacles initialization
        self.static_obstacles_per_scenario = jnp.array([
            [ # Circular crossing
                [[[0.75, -2*self.circle_radius/3],[2, -2*self.circle_radius/3+1.5*self.circle_radius/7]]],
                [[[-0.75, -2*self.circle_radius/3+2*self.circle_radius/7],[-2, -2*self.circle_radius/3+3.5*self.circle_radius/7]]],
                [[[0.75, -2*self.circle_radius/3+4*self.circle_radius/7],[2, -2*self.circle_radius/3+5.5*self.circle_radius/7]]],
                [[[-0.75, -2*self.circle_radius/3+6*self.circle_radius/7],[-2, -2*self.circle_radius/3+7.5*self.circle_radius/7]]],
                [[[0.75, -2*self.circle_radius/3+8*self.circle_radius/7],[2, -2*self.circle_radius/3+9.5*self.circle_radius/7]]],
            ], 
            [ # Parallel traffic
                [[[-self.traffic_length/2-1, self.traffic_height/2 + 0.3],[self.traffic_length/2-0.5, self.traffic_height/2 + 0.3]]],
                [[[-self.traffic_length/2-1, -(self.traffic_height/2 + 0.3)],[self.traffic_length/2-0.5, -(self.traffic_height/2 + 0.3)]]],
                [[[-1.,0],[1.,0.]]],
                [[[-self.traffic_length/4-0.5,self.traffic_height/4],[-self.traffic_length/4+0.5,self.traffic_height/4]]],
                [[[self.traffic_length/4-0.5,self.traffic_height/4],[self.traffic_length/4+0.5,self.traffic_height/4]]],
            ], 
            [ # Perpendicular traffic
                [[[-self.traffic_length/8, self.traffic_length/2 +1],[-self.traffic_length/8, self.traffic_height/2+0.5]]],
                [[[self.traffic_length/8, self.traffic_length/2 +1],[self.traffic_length/8, self.traffic_height/2+0.5]]],
                [[[-1.,0],[1.,0.]]],
                [[[0., -self.traffic_height/2-0.5],[0., -self.traffic_height/2-2]]],
                [[[-0.5,-self.traffic_length/2+0.6],[0.5,-self.traffic_length/2+0.6]]],
            ], 
            [ # Robot crowding
                [[[-1.,0],[1.,0.]]],
                [[[self.crowding_square_side/4, 1],[self.crowding_square_side/4-1, -1]]],
                [[[-self.crowding_square_side/4, -1],[-self.crowding_square_side/4-1, 1]]],
                [[[-self.crowding_square_side/2, 2],[-self.crowding_square_side/2-1, 0.5]]],
                [[[-self.crowding_square_side/2, -2],[-self.crowding_square_side/2-1, -0.5]]],
            ], 
            [ # Delayed circular crossing
                [[[1.5*self.circle_radius/7 * jnp.cos(2*jnp.pi/5), 1.5*self.circle_radius/7 * jnp.sin(2*jnp.pi/5)],[3.5*self.circle_radius/7*jnp.cos(2*jnp.pi/5), 3.5*self.circle_radius/7*jnp.sin(2*jnp.pi/5)]]],
                [[[1.5*self.circle_radius/7 * jnp.cos((2*jnp.pi/5)*2), 1.5*self.circle_radius/7 * jnp.sin((2*jnp.pi/5)*2)],[3.5*self.circle_radius/7*jnp.cos((2*jnp.pi/5)*2), 3.5*self.circle_radius/7*jnp.sin((2*jnp.pi/5)*2)]]],
                [[[1.5*self.circle_radius/7 * jnp.cos((2*jnp.pi/5)*3), 1.5*self.circle_radius/7 * jnp.sin((2*jnp.pi/5)*3)],[3.5*self.circle_radius/7*jnp.cos((2*jnp.pi/5)*3), 3.5*self.circle_radius/7*jnp.sin((2*jnp.pi/5)*3)]]],
                [[[1.5*self.circle_radius/7 * jnp.cos((2*jnp.pi/5)*4), 1.5*self.circle_radius/7 * jnp.sin((2*jnp.pi/5)*4)],[3.5*self.circle_radius/7*jnp.cos((2*jnp.pi/5)*4), 3.5*self.circle_radius/7*jnp.sin((2*jnp.pi/5)*4)]]],
                [[[1.5*self.circle_radius/7 * jnp.cos((2*jnp.pi/5)*5), 1.5*self.circle_radius/7 * jnp.sin((2*jnp.pi/5)*5)],[3.5*self.circle_radius/7*jnp.cos((2*jnp.pi/5)*5), 3.5*self.circle_radius/7*jnp.sin((2*jnp.pi/5)*5)]]],
            ], 
            [ # Circular crossing with static obstacles (this scenario is already challenging enough, so we do not add more static obstacles)
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
            ], 
            [ # Crowd navigation
                [[[-self.circle_radius/2, -self.circle_radius/2],[-self.circle_radius/2+1, -self.circle_radius/2+1]]],
                [[[0., -self.circle_radius/2-1],[0., -self.circle_radius/2+2]]],
                [[[0., self.circle_radius/2-1],[0., self.circle_radius/2+2]]],
                [[[self.circle_radius/2, self.circle_radius/2],[self.circle_radius/2-1, self.circle_radius/2-1]]],
                [[[-0.5, self.circle_radius-1],[0.5, self.circle_radius-1]]],
            ],
            [ # Corner traffic
                [[[self.traffic_length/2-self.traffic_height/2-0.3, 0.],[self.traffic_length/2-self.traffic_height/2-0.3, self.traffic_length/2-self.traffic_height/2-0.3]]],
                [[[self.traffic_length/2+self.traffic_height/2+0.3, 0.],[self.traffic_length/2+self.traffic_height/2+0.3, self.traffic_length/2+self.traffic_height/2+0.3]]],
                [[[self.traffic_length/2-0.25,self.traffic_length/2+0.25],[self.traffic_length/2+0.25,self.traffic_length/2-0.25]]],
                [[[0.,self.traffic_length/2-self.traffic_height/2-0.3],[self.traffic_length/2-self.traffic_height/2-0.3, self.traffic_length/2-self.traffic_height/2-0.3]]],
                [[[0.,self.traffic_length/2+self.traffic_height/2+0.3],[self.traffic_length/2+self.traffic_height/2+0.3, self.traffic_length/2+self.traffic_height/2+0.3]]],
            ],
        ])
        if thick_default_obstacle:
            self.static_obstacles_per_scenario = thicken_obstacles(self.static_obstacles_per_scenario, thickness=0.1)
        if n_obstacles > 5:
            assert self.scenario == -1, "Standard scenarios with more than 5 obstacles are not supported yet. Only with custom scenarios."
        ## Robot goals initialization
        self.robot_goals_per_scenario = jnp.array([
            [[0., self.circle_radius],[jnp.nan, jnp.nan]], # Circular crossing
            [[self.traffic_length/2-1, 0.],[jnp.nan, jnp.nan]], # Parallel traffic
            [[0., -self.traffic_length/2],[jnp.nan, jnp.nan]], # Perpendicular traffic
            [[-self.crowding_square_side/2-1, 0.],[jnp.nan, jnp.nan]], # Robot crowding
            [[0., self.circle_radius],[jnp.nan, jnp.nan]], # Delayed circular crossing
            [[0., self.circle_radius],[jnp.nan, jnp.nan]], # Circular crossing with static obstacles
            [[0., self.circle_radius],[jnp.nan, jnp.nan]], # Crowd navigation
            [[self.traffic_length/2-self.traffic_height/4, self.traffic_length/2-self.traffic_height/4],[self.traffic_length/2, 1.]], # Corner traffic
        ])
        ## Possible delays for delayed circular crossing scenario
        self.possible_delays = jnp.arange(0., self.max_cc_delay + self.robot_dt, self.robot_dt)

    # --- Abstract methods --- #

    @abstractmethod
    def _get_obs(self, state):
        pass

    @abstractmethod
    def reset(self, key):
        pass

    @abstractmethod
    def step(self, env_state, action):
        pass

    # --- Private methods --- #

    @partial(jit, static_argnames=("self"))
    def _reset(self, key:random.PRNGKey, scenarios_prob:jnp.ndarray=None) -> tuple[jnp.ndarray, random.PRNGKey, dict]:
        key, scen_key, flip_key, noise_key = random.split(key, 4)
        if self.scenario == SCENARIOS.index('hybrid_scenario'):
            # Randomly choose a scenario between all then ones included in the hybrid_scenario subset
            randint = random.choice(scen_key, a=len(self.hybrid_scenario_subset), p=scenarios_prob)
            scenario = self.hybrid_scenario_subset[randint]
            key, scen_key = random.split(key)
        else:
            scenario = self.scenario
        full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, humans_delay = lax.switch(
            scenario, 
            [
                self._generate_circular_crossing_episode, 
                self._generate_parallel_traffic_episode,
                self._generate_perpendicular_traffic_episode,
                self._generate_robot_crowding_episode,
                self._generate_delayed_circular_crossing_episode,
                self._generate_circular_crossing_with_static_obstacles_episode,
                self._generate_crowd_navigation_episode,
                self._generate_corner_traffic_episode,
            ], 
            scen_key
        )
        full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles, is_x_flipped, is_y_flipped = self._random_flip(
            full_state, 
            humans_goal, 
            robot_goal, 
            static_obstacles, 
            self.robot_goals_per_scenario[scenario],
            flip_key
        )
        info = self._init_info(
            full_state,
            humans_goal=humans_goal,
            robot_goal=robot_goal,
            robot_goal_list=robot_goal_list,
            humans_parameters=humans_parameters,
            static_obstacles=static_obstacles,
            current_scenario=scenario,
            humans_delay=humans_delay,
            is_x_flipped=is_x_flipped,
            is_y_flipped=is_y_flipped,
            noise_key=noise_key,
        )
        if self.grid_map_computation: # Compute the grid map of static obstacles for global planning
            info['grid_cells'], info['occupancy_grid'] = self.build_grid_map_and_occupancy(full_state, info)
        return full_state, key, info

    @partial(jit, static_argnames=("self"))
    def _random_flip(
        self, 
        full_state:jnp.ndarray, 
        humans_goal:jnp.ndarray,
        robot_goal:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        robot_goal_list:jnp.ndarray,
        key:random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, bool]:
        """
        Randomly flips the environment along the x-axis and y-axis with 50% probability.

        args:
        - full_state: array of shape (n_humans+1, 5) representing the state of the robot and humans.
        - humans_goal: array of shape (n_humans, 2) representing the goals of the humans.
        - robot_goal: array of shape (2,) representing the goal of the robot.
        - robot_goal_list: array of shape (n_waypoints, 2) representing the list of waypoints for the robot.
        - static_obstacles: array of shape (n_humans+1, n_obstacles, 1, 2, 2) representing the static obstacles.
        - key: random.PRNGKey for randomness.

        output:
        - full_state: possibly flipped full_state.
        - humans_goal: possibly flipped humans_goal.
        - robot_goal: possibly flipped robot_goal.
        - robot_goal_list: possibly flipped robot_goal_list.
        - static_obstacles: possibly flipped static_obstacles.
        - flip_x: boolean indicating if a flip along the x-axis was performed.
        - flip_y: boolean indicating if a flip along the y-axis was performed.
        """
        def _flip_y(state, humans_goal, robot_goal, robot_goal_list, static_obstacles):
            state = state.at[:, 1].set(-state[:, 1]) # Flip y position
            state = state.at[:, 4].set(-state[:, 4]) # Flip orientation
            humans_goal = humans_goal.at[:, 1].set(-humans_goal[:, 1]) # Flip humans' goals
            robot_goal = robot_goal.at[1].set(-robot_goal[1]) # Flip robot's goal
            robot_goal_list = robot_goal_list.at[:, 1].set(-robot_goal_list[:, 1]) # Flip robot's waypoint list
            static_obstacles = static_obstacles.at[:, :, :, :, 1].set(-static_obstacles[:, :, :, :, 1]) # Flip static obstacles
            return state, humans_goal, robot_goal, robot_goal_list, static_obstacles
        def _flip_x(state, humans_goal, robot_goal, robot_goal_list, static_obstacles):
            state = state.at[:, 0].set(-state[:, 0]) # Flip x position
            state = state.at[:, 4].set(vmap(wrap_angle)(jnp.pi - state[:, 4])) # Flip orientation
            humans_goal = humans_goal.at[:, 0].set(-humans_goal[:, 0]) # Flip humans' goals
            robot_goal = robot_goal.at[0].set(-robot_goal[0]) # Flip robot's goal
            robot_goal_list = robot_goal_list.at[:, 0].set(-robot_goal_list[:, 0]) # Flip robot's waypoint list
            static_obstacles = static_obstacles.at[:, :, :, :, 0].set(-static_obstacles[:, :, :, :, 0]) # Flip static obstacles
            return state, humans_goal, robot_goal, robot_goal_list, static_obstacles
        x_key, y_key = random.split(key)
        flip_x = random.bernoulli(y_key, p=0.5)
        full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles = lax.cond(
            flip_x,
            _flip_x,
            lambda s, h, r, rl, so: (s, h, r, rl, so),
            full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles
        )
        flip_y = random.bernoulli(x_key, p=0.5)
        full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles = lax.cond(
            flip_y,
            _flip_y,
            lambda s, h, r, rl, so: (s, h, r, rl, so),
            full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles
        )
        return full_state, humans_goal, robot_goal, robot_goal_list, static_obstacles, flip_x, flip_y

    @partial(jit, static_argnames=("self"))
    def _init_info(
        self,
        full_state:jnp.ndarray,
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
        Initializes the info dictionary with the given parameters.

        args:
        - full_state: initial state of the environment. (UNUSED HERE)
        - humans_goal: array of humans' goals.
        - robot_goal: array of robot's goal.
        - humans_parameters: array of humans' parameters.
        - static_obstacles: array of static obstacles.
        - current_scenario: current scenario index.
        - humans_delay: array of humans' delays.
        - noise_key: random.PRNGKey for noise generation. (UNUSED HERE)

        output:
        - info: dictionary containing the initialized values.
        """
        return {
            "humans_goal": humans_goal, 
            "robot_goal": robot_goal, 
            "robot_goal_index": 0, # If robot has a waypoint list, this is the index of the next waypoint to reach
            "robot_goal_list": robot_goal_list, # If robot has a waypoint list, this is the list of waypoints
            "humans_parameters": humans_parameters, 
            "static_obstacles": static_obstacles, 
            "time": 0.,
            "current_scenario": current_scenario,
            "humans_delay": humans_delay,
            "step": 0,
            "return": 0.,
            "is_x_flipped": is_x_flipped,
            "is_y_flipped": is_y_flipped,
        }

    @partial(jit, static_argnames=("self"))
    def _init_obstacles(self, key:random.PRNGKey, scenario:int) -> jnp.ndarray:
        if self.n_obstacles == 0:
            return jnp.full((self.n_humans+1, 1, 1, 2, 2), jnp.nan)
        else:
            obstacles = self.static_obstacles_per_scenario[scenario]
            perm = random.permutation(key, obstacles.shape[0])
            shuffled_obstacles = obstacles[perm]
            picked_obstacles = shuffled_obstacles[:self.n_obstacles]
            # TODO: Filter obstacles based on the robot position and grid cell decomposition of static obstacles
            return jnp.repeat(jnp.array([picked_obstacles]), self.n_humans+1, axis=0)

    @partial(jit, static_argnames=("self"))
    def _init_robot_goal(self, scenario:int) -> jnp.ndarray:
        """
        Initializes the robot's goal based on the current scenario.

        args:
        - scenario: current scenario index.

        output:
        - robot_goal: array containing the robot's goal.
        """
        return self.robot_goals_per_scenario[scenario][0]
    
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
        robot_goal = self._init_robot_goal(SCENARIOS.index('circular_crossing'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('circular_crossing'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))
    
    @partial(jit, static_argnames=("self"))
    def _generate_delayed_circular_crossing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        key, subkey = random.split(key)
        full_state, humans_goal, _, humans_parameters, _, _ = self._generate_circular_crossing_episode(key)
        robot_goal = self._init_robot_goal(SCENARIOS.index('delayed_circular_crossing'))
        static_obstacles=self._init_obstacles(key, SCENARIOS.index('delayed_circular_crossing'))
        humans_delay = random.choice(subkey, self.possible_delays, shape=(self.n_humans,))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, humans_delay

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
                new_point = jnp.array([-self.traffic_length/2 + 3 + normalized_point[0] * (self.traffic_length - 1), -self.traffic_height/2 + normalized_point[1] * self.traffic_height])
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
        robot_goal = self._init_robot_goal(SCENARIOS.index('parallel_traffic'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('parallel_traffic'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

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
        robot_goal = self._init_robot_goal(SCENARIOS.index('perpendicular_traffic'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('perpendicular_traffic'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

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
        robot_goal = self._init_robot_goal(SCENARIOS.index('robot_crowding'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('robot_crowding'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

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
                    lambda _: (jnp.pi / (int(self.ccso_n_static_humans) + 1e-5)) * (-0.5 + 2 * i + random.uniform(subkey, shape=(1,), minval=-0.25, maxval=0.25)),
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
        robot_goal = self._init_robot_goal(SCENARIOS.index('circular_crossing_with_static_obstacles'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('circular_crossing_with_static_obstacles'))
        # Info
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

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
        robot_goal = self._init_robot_goal(SCENARIOS.index('crowd_navigation'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('crowd_navigation'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))
    
    @partial(jit, static_argnames=("self"))
    def _generate_corner_traffic_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([1., self.traffic_length/2])) # Conform with Social-Navigation-PyEnvs
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1) - 0.5
                new_point = jnp.array([self.traffic_length/2 + normalized_point[0] * self.traffic_height, self.traffic_length/4 + normalized_point[1] * (self.traffic_length/2 - 1)])
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
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.ones((self.n_humans,)) * jnp.pi/2))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([disturbed_points[i,0],disturbed_points[i,0]])),
            humans_goal)
        robot_goal = self._init_robot_goal(SCENARIOS.index('corner_traffic'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('corner_traffic'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _human_ray_intersect(self, direction:jnp.ndarray, human_position:jnp.ndarray, lidar_position:jnp.ndarray, human_radius:float) -> float:
        s = lidar_position - human_position
        b = jnp.dot(s, direction)
        c = jnp.dot(s, s) - human_radius**2
        h = b * b - c
        distance = lax.cond(
            h < 0,
            lambda x: x,
            lambda x: lax.cond(
                - b - jnp.sqrt(h) < 0,
                lambda y: y,
                lambda _: - b - jnp.sqrt(h),
                x),
            self.lidar_max_dist)
        return distance        
    
    @partial(jit, static_argnames=("self"))
    def _batch_human_ray_intersect(self, direction:jnp.ndarray, human_positions:jnp.ndarray, lidar_position:jnp.ndarray, human_radiuses:float) -> jnp.ndarray:
        humans_distances = vmap(BaseEnv._human_ray_intersect, in_axes=(None,None,0,None,0))(self, direction, human_positions, lidar_position, human_radiuses)
        shortest_distance_index = jnp.argmin(humans_distances)
        return humans_distances[shortest_distance_index], shortest_distance_index

    @partial(jit, static_argnames=("self"))
    def _segment_ray_intersect(self, p1:jnp.ndarray, p2:jnp.ndarray, lidar_position:jnp.ndarray, direction:jnp.ndarray) -> float:
        @jit
        def _is_nan(_):
            return self.lidar_max_dist
        @jit
        def _not_nan(data):
            p1, p2, lidar_position, direction = data
            v1 = lidar_position - p1
            v2 = p2 - p1
            v3 = jnp.array([-direction[1], direction[0]])
            dot = jnp.dot(v2, v3)
            t1 = jnp.cross(v2, v1) / dot
            t2 = jnp.dot(v1, v3) / dot
            distance = lax.cond(
                (dot != 0) & (t1 >= 0) & (t2 >= 0) & (t2 <= 1),
                lambda x: jnp.linalg.norm(direction * t1),
                lambda x: self.lidar_max_dist,
                None)
            return distance
        return lax.cond(
            jnp.any(jnp.isnan(jnp.array([p1, p2]))),
            _is_nan,
            _not_nan,
            (p1, p2, lidar_position, direction)
        )
    
    @partial(jit, static_argnames=("self"))
    def _obstacle_ray_intersect(self, direction:jnp.ndarray, obstacle:jnp.ndarray, lidar_position:jnp.ndarray) -> float:
        distances = vmap(BaseEnv._segment_ray_intersect, in_axes=(None,0,0,None,None))(self, obstacle[:,0,:], obstacle[:,1,:], lidar_position, direction)
        shortest_distance_index = jnp.argmin(distances)
        return distances[shortest_distance_index], shortest_distance_index

    @partial(jit, static_argnames=("self"))
    def _batch_obstacle_ray_intersect(self, direction:jnp.ndarray, obstacles:jnp.ndarray, lidar_position:jnp.ndarray) -> float:
        distances, collision_idxs = vmap(BaseEnv._obstacle_ray_intersect, in_axes=(None,None,0,None))(self, direction, obstacles, lidar_position)
        shortest_distance_index = jnp.argmin(distances)
        return distances[shortest_distance_index], jnp.array([shortest_distance_index, collision_idxs[shortest_distance_index]])

    @partial(jit, static_argnames=("self"))
    def _ray_cast(self, angle:float, lidar_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray, static_obstacles:jnp.ndarray) -> float:
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        measurement1, human_collision_idx = self._batch_human_ray_intersect(direction, human_positions, lidar_position, human_radiuses)
        measurement2, obstacles_collision_idx = self._batch_obstacle_ray_intersect(direction, static_obstacles, lidar_position)
        min_dist = jnp.min(jnp.array([measurement1, measurement2]))
        # Compute final collision index
        @jit
        def _collided(x):
            min_dist, measurement1, human_collision_idx, obstacles_collision_idx = x
            is_human_collision = (min_dist == measurement1)
            human_collision_idx = lax.cond(
                is_human_collision,
                lambda x: x,
                lambda _: jnp.array(-1, dtype=jnp.int32),
                human_collision_idx,
            )
            obstacle_collision_idx = lax.cond(
                is_human_collision,
                lambda _: jnp.array([-1, -1], dtype=jnp.int32),
                lambda x: x,
                obstacles_collision_idx,
            )
            return min_dist, human_collision_idx, obstacle_collision_idx
        return lax.cond(
            min_dist < self.lidar_max_dist,
            _collided,
            lambda x: (x[0], jnp.array(-1, dtype=jnp.int32), jnp.array([-1, -1], dtype=jnp.int32)),
            (min_dist, measurement1, human_collision_idx, obstacles_collision_idx)
        )

    @partial(jit, static_argnames=("self"))
    def _scenario_based_state_post_update(self, state:jnp.ndarray, info:dict):

        @jit
        def _update_circular_crossing(val:tuple):
            @jit
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float) -> jnp.ndarray:
                goal = lax.cond(
                    jnp.linalg.norm(position - goal) <= radius,
                    lambda x: -x,
                    lambda x: x,
                    goal)
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0))(state[:-1,0:2], info["humans_goal"], info["humans_parameters"][:,0])
            return (info, state)
        
        @jit
        def _update_delayed_circular_crossing(val:tuple):
            @jit
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, delay:float, time:float) -> jnp.ndarray:
                goal = lax.cond(
                    jnp.all(jnp.array([jnp.linalg.norm(position - goal) <= radius, time >= delay])),
                    lambda x: -x,
                    lambda x: x,
                    goal)
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0,0,None))(state[:-1,0:2], info["humans_goal"], info["humans_parameters"][:,0], info["humans_delay"], info["time"])
            return (info, state)
        
        @jit
        def _update_traffic_scenarios(val:tuple):
            @jit
            def _update_human_state_and_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, positions:jnp.ndarray, radiuses:jnp.ndarray, safety_spaces:jnp.ndarray, is_x_flipped:bool) -> tuple:
                flip_x = lax.cond(is_x_flipped,lambda _: -1.,lambda _: 1.,None)
                position, goal = lax.cond(
                    # jnp.linalg.norm(position - goal) <= radius + 2,
                    jnp.linalg.norm(position - goal) <= 3, # Compliant with Social-Navigation-PyEnvs
                    lambda _: (
                        jnp.array([
                        # flip_x * jnp.max(jnp.append(positions[:,0]+(jnp.max(jnp.append(radiuses,self.robot_radius))*2)+(jnp.max(safety_spaces)*2)+0.05, self.traffic_length/2+1)), 
                        flip_x * jnp.max(jnp.append(positions[:,0] + (jnp.max(jnp.append(radiuses, self.robot_radius))*2)+(jnp.max(safety_spaces)*2), self.traffic_length/2)), # Compliant with Social-Navigation-PyEnvs
                        jnp.clip(position[1], -self.traffic_height/2, self.traffic_height/2)]
                        ),
                        jnp.array([goal[0], position[1]]),
                    ),
                    lambda x: x,
                    (position, goal))
                return position, goal
            info, state = val
            new_positions, new_goals = vmap(_update_human_state_and_goal, in_axes=(0,0,0,None,None,None, None))(
                state[:-1,0:2], 
                info["humans_goal"], 
                info["humans_parameters"][:,0], 
                state[:,0:2], 
                info["humans_parameters"][:,0], 
                info["humans_parameters"][:,-1],
                info['is_x_flipped']
            )
            state = state.at[:-1,0:2].set(new_positions)
            info["humans_goal"] = info["humans_goal"].at[:].set(new_goals)
            return info, state
        
        @jit
        def _update_circular_crossing_with_static_obstacles(val:tuple):
            @jit
            def _update_human_goal(idx:int, position:jnp.ndarray, goal:jnp.ndarray, radius:float) -> jnp.ndarray:
                goal = lax.cond(
                    (jnp.linalg.norm(position - goal) <= radius) & (idx >= self.ccso_n_static_humans),
                    lambda x: -x,
                    lambda x: x,
                    goal)
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0,0))(
                jnp.arange(self.n_humans), 
                state[:-1,0:2], 
                info["humans_goal"], 
                info["humans_parameters"][:,0])
            return (info, state)
        
        @jit
        def _update_crowd_navigation(val:tuple):
            @jit
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float) -> jnp.ndarray:
                @jit
                def _set_new_goal(position, goal):
                    key = random.PRNGKey(jnp.array(jnp.linalg.norm(goal)*1000, int))
                    key1, key2 = random.split(key)
                    position_angle = jnp.atan2(position[1], position[0])
                    new_angle = wrap_angle(random.uniform(key1, shape=(), minval=position_angle-jnp.pi/4, maxval=position_angle+jnp.pi/4))
                    new_distance = random.uniform(key2, shape=(1,), minval=0., maxval=self.circle_radius)
                    new_goal = jnp.squeeze(new_distance * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
                    return new_goal
                goal = lax.cond(
                    jnp.linalg.norm(position - goal) <= radius,
                    lambda x: _set_new_goal(*x),
                    lambda x: x[1],
                    (position, goal))
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0))(state[:-1,0:2], info["humans_goal"], info["humans_parameters"][:,0])
            return (info, state)
        
        @jit
        def _update_corner_traffic(val:tuple):
            @jit
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, is_x_flipped:bool, is_y_flipped:bool) -> jnp.ndarray:
                flip_x = lax.cond(is_x_flipped,lambda _: -1.,lambda _: 1.,None)
                flip_y = lax.cond(is_y_flipped,lambda _: -1.,lambda _: 1.,None)
                goal = lax.cond(
                    jnp.linalg.norm(position - goal) <= radius+0.1,
                    lambda x: lax.cond(
                        jnp.abs(x[0])==jnp.abs(x[1]),
                        lambda y: lax.cond(
                            position[1] * flip_y < position[0] * flip_x,
                            lambda z: jnp.array([0., jnp.max(jnp.abs(z)) * flip_y]),
                            lambda z: jnp.array([jnp.max(jnp.abs(z)) * flip_x, 0.]),
                            y,
                        ),
                        lambda y: jnp.array([jnp.max(jnp.abs(y)) * flip_x, jnp.max(jnp.abs(y)) * flip_y]),
                        x,
                    ),
                    lambda x: x,
                    goal)
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0,None,None))(state[:-1,0:2], info["humans_goal"], info["humans_parameters"][:,0], info["is_x_flipped"], info["is_y_flipped"])
            return (info, state)

        if self.scenario != -1:  # If not custom scenario
            new_info, new_state = lax.switch(
                info["current_scenario"], 
                [
                    _update_circular_crossing, 
                    _update_traffic_scenarios, 
                    _update_traffic_scenarios, 
                    lambda x: x,
                    _update_delayed_circular_crossing,
                    _update_circular_crossing_with_static_obstacles,
                    _update_crowd_navigation,
                    _update_corner_traffic,
                ], 
                (info, state),
            )
        else:
            new_info = info
            new_state = state
        return new_info, new_state

    @partial(jit, static_argnames=("self"))
    def _update_state_info(
        self, 
        state:jnp.ndarray, 
        info:dict,
        action:jnp.ndarray
    ) -> tuple:
        """
        This function updates the state and the info of the environment given the current state, the info and the action taken by the robot.
        The state shape is ((n_humans+1,6). The last row of the state matrix corresponds to the robot state, which must be given in the correct form based 
        on the human motion model used.

        args:
        - state ((n_humans+1,6): jnp.ndarray containing the state of the environment.
        - info (dict): dictionary containing the information of the environment.
        - action (2,): jnp.ndarray containing the action taken by the robot.

        output:
        - new_state ((n_humans+1,6): jnp.ndarray containing the new state of the environment.
        """
        goals = jnp.vstack((info["humans_goal"], info["robot_goal"]))
        second_parameter = 80. if self.humans_policy == HUMAN_POLICIES.index("hsfm") or self.humans_policy == HUMAN_POLICIES.index("sfm") else 5.  # Mass if HSFM or SFM, time horizon if ORCA
        parameters = jnp.vstack((info["humans_parameters"], jnp.array([self.robot_radius, second_parameter, *self.get_standard_humans_parameters(1)[0,2:]])))
        static_obstacles = info["static_obstacles"]
        ## Humans update
        if self.humans_policy == HUMAN_POLICIES.index("hsfm"):
            if self.robot_visible:
                new_state = jnp.vstack(
                    [self.humans_step(state, goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                    state[-1]])
            else:
                new_state = jnp.vstack(
                    [self.humans_step(state[0:self.n_humans], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles[0:self.n_humans], self.humans_dt), 
                    state[-1]])
        elif self.humans_policy == HUMAN_POLICIES.index("sfm") or self.humans_policy == HUMAN_POLICIES.index("orca"):
            if self.robot_visible:
                new_state = jnp.vstack(
                    [self.humans_step(state[:,0:4], goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                    state[-1,0:4]])
            else:
                new_state = jnp.vstack(
                    [self.humans_step(state[0:self.n_humans,0:4], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles[0:self.n_humans], self.humans_dt), 
                    state[-1,0:4]])
            new_state = jnp.pad(new_state, ((0,0),(0,2)))
        ## Robot update
        if self.kinematics == ROBOT_KINEMATICS.index("holonomic"):
            new_state = new_state.at[-1,0:2].set(jnp.array([
                state[-1,0]+action[0]*self.humans_dt, 
                state[-1,1]+action[1]*self.humans_dt]))
        elif self.kinematics == ROBOT_KINEMATICS.index("unicycle"):
            new_state = lax.cond(
                jnp.abs(action[1]) > EPSILON,
                lambda x: x.at[-1].set(jnp.array([
                    state[-1,0]+(action[0]/action[1])*(jnp.sin(state[-1,4]+action[1]*self.humans_dt)-jnp.sin(state[-1,4])),
                    state[-1,1]+(action[0]/action[1])*(jnp.cos(state[-1,4])-jnp.cos(state[-1,4]+action[1]*self.humans_dt)),
                    *state[-1,2:4],
                    wrap_angle(state[-1,4]+action[1]*self.humans_dt),
                    state[-1,5]])),
                lambda x: x.at[-1].set(jnp.array([
                    state[-1,0]+action[0]*self.humans_dt*jnp.cos(state[-1,4]),
                    state[-1,1]+action[0]*self.humans_dt*jnp.sin(state[-1,4]),
                    *state[-1,2:]])),
                new_state)
            if self.robot_visible and (self.humans_policy == HUMAN_POLICIES.index("sfm") or self.humans_policy == HUMAN_POLICIES.index("orca")):
                new_state = new_state.at[-1,2:4].set(jnp.array([action[0] * jnp.cos(new_state[-1,4]), action[0] * jnp.sin(new_state[-1,4])]))
        ## Post update stuff
        new_info, new_state = self._scenario_based_state_post_update(new_state, info)
        return (new_state, new_info)

    @partial(jit, static_argnames=("self"))
    def _update_state_info_imitation_learning(
        self,
        state:jnp.ndarray, 
        info:dict
    ) -> tuple:
        """
        This function updates the state and the info of the environment given the current state and the info.
        The state shape depends on the human motion model used ((n_humans+1,6) for hsfm and (n_humans+1,4) for sfm).
        The last row of the state matrix corresponds to the robot state, which must be given in the correct form based on the human motion model used.
        Using this function the robot state will be updated using the same policy used for the humans.

        args:
        - state ((n_humans+1,6) or (n_humans+1,4)): jnp.ndarray containing the state of the environment.
        - info (dict): dictionary containing the information of the environment.

        output:
        - new_state ((n_humans+1,6) or (n_humans+1,4)): jnp.ndarray containing the new state of the environment.
        """
        goals = jnp.vstack((info["humans_goal"], info["robot_goal"]))
        second_parameter = 80. if self.humans_policy == HUMAN_POLICIES.index("hsfm") or self.humans_policy == HUMAN_POLICIES.index("sfm") else 5. # Mass if HSFM or SFM, time horizon if ORCA
        parameters = jnp.vstack((info["humans_parameters"], jnp.array([self.robot_radius, second_parameter, *self.get_standard_humans_parameters(1)[0,2:-1], 0.1]))) # Add safety space of 0.1 to robot
        static_obstacles = info["static_obstacles"]
        if self.robot_visible:
            new_state = self.humans_step(state, goals, parameters, static_obstacles, self.humans_dt)
        else:
            new_state = jnp.vstack([
                self.humans_step(state[0:-1], goals[0:-1], parameters[0:-1], static_obstacles[0:-1], self.humans_dt), 
                self.humans_step(state, goals, parameters, static_obstacles, self.humans_dt)[-1]])
        new_info, new_state = self._scenario_based_state_post_update(new_state, info)
        return (new_state, new_info)

    # --- Public methods ---

    def get_parameters(self):
        """
        This function returns the parameters of the environment as a dictionary.

        output:
        - params: dictionary containing the parameters of the environment.
        """
        params = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                params[key] = value
        return params
    
    @partial(jit, static_argnames=("self"))
    def batch_ray_cast(self, angles:float, lidar_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray, static_obstacles:jnp.ndarray) -> jnp.ndarray:
        """
        This function performs a batch ray cast for the given angles and lidar position.

        args:
        - angles (num_rays,): jnp.ndarray containing the angles of the rays.
        - lidar_position (2,): jnp.ndarray containing the x and y coordinates of the lidar.
        - human_positions (self.n_humans,2): jnp.ndarray containing the x and y coordinates of the humans.
        - human_radiuses (self.n_humans,): jnp.ndarray containing the radius of the humans.
        - static_obstacles (self.n_obstacles, m, 2, 2): jnp.ndarray containing the static obstacles as line segments (m is the number of segments per obstacle).

        output:
        - measurements (num_rays,): jnp.ndarray containing the distances of the rays.
        - human_collision_idxs (num_rays,): jnp.ndarray containing the indexes of the humans collided by the rays (-1 if no collision).
        - obstacle_collision_idxs (num_rays,2): jnp.ndarray containing the indexes of the obstacles and segments collided by the rays (-1 if no collision).
        """
        return vmap(BaseEnv._ray_cast, in_axes=(None,0,None,None,None,None))(self, angles, lidar_position, human_positions, human_radiuses, static_obstacles)

    @partial(jit, static_argnames=("self"))
    def get_lidar_measurements(
        self, 
        lidar_position:jnp.ndarray, 
        lidar_yaw:float,  
        human_positions:jnp.ndarray, 
        human_radiuses:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        noise_key=random.PRNGKey(0)
    ) -> jnp.ndarray:
        """
        Given the current state of the environment, the robot orientation and the additional information about the environment,
        this function computes the lidar measurements of the robot. The lidar measurements are given as a set of distances and angles (in the global frame) for each ray.

        args:
        - lidar_position (2,): jnp.ndarray containing the x and y coordinates of the lidar.
        - lidar_yaw (1,): float containing the orientation of the lidar.
        - human_positions (self.n_humans,2): jnp.ndarray containing the x and y coordinates of the humans.
        - human_radiuses (self.n_humans,): jnp.ndarray containing the radius of the humans.
        - static_obstacles (self.n_obstacles, m, 2, 2): jnp.ndarray containing the static obstacles as line segments (m is the number of segments per obstacle).

        output:
        - lidar_output (self.lidar_num_rays,2): jnp.ndarray containing the lidar measurements of the robot and the angle (IN THE GLOBAL FRAME) for each ray.
          WARNING: the angles are in the global frame, not in the robot frame.
        """
        angles = jnp.linspace(lidar_yaw - self.lidar_angular_range/2, lidar_yaw + self.lidar_angular_range/2, self.lidar_num_rays)
        measurements, _, _ = self.batch_ray_cast(angles, lidar_position, human_positions, human_radiuses, static_obstacles)
        if self.lidar_noise:
            measurements = self.add_lidar_noise(measurements,noise_key)
        lidar_output = jnp.stack((measurements, angles), axis=-1)
        return lidar_output
    
    @partial(jit, static_argnames=("self"))
    def add_lidar_noise(self, measurements:jnp.ndarray, noise_key:random.PRNGKey) -> jnp.ndarray:
        """
        Add noise and salt-and-pepper to the given lidar measurements.

        args:
        - measurements (self.lidar_num_rays,): jnp.ndarray containing the lidar measurements of the robot.
        - noise_key: jax.random.PRNGKey for randomness.

        output:
        - noisy_measurements (self.lidar_num_rays,): jnp.ndarray containing the noisy lidar measurements of the robot.
        """
        beam_dropout_key, noise_key = random.split(noise_key)
        ## Gaussian noise to LiDAR scans + Beam dropout
        sigma = self.lidar_noise_fixed_std + self.lidar_noise_proportional_std * measurements 
        noise = random.normal(noise_key, shape=measurements.shape) * sigma 
        noisy_distances = jnp.clip(measurements + noise, 0., self.lidar_max_dist)
        is_dropout = random.bernoulli(beam_dropout_key, p=self.lidar_salt_and_pepper_prob, shape=measurements.shape)
        noisy_distances = jnp.where(is_dropout, self.lidar_max_dist, noisy_distances) 
        return noisy_distances

    @partial(jit, static_argnames=("self"))
    def object_visibility(self, rc_humans_positions, humans_radii, rc_static_obstacles, epsilon=1e-5):
        """
        Assess which humans and static obstacles are visible from the robot's perspective.

        params:
        - rc_humans_positions: (n_humans, 2) array of humans positions IN ROBOT-CENTRIC FRAME
        - humans_radii: (n_humans,) array of humans radii
        - rc_static_obstacles: (n_obstacles, 2, 2) array of static obstacle segments IN ROBOT-CENTRIC FRAME

        returns:
        - visible_humans_mask: (n_humans,) boolean array indicating which humans are visible
        - visible_static_obstacles_mask: (n_obstacles,n_segments) boolean array indicating which static obstacle segments are visible
        """
        ### Compute ordered array of all objects endpoint angles
        ## Humans
        humans_versors = rc_humans_positions / jnp.linalg.norm(rc_humans_positions, axis=1, keepdims=True)  # Shape: (n_humans, 2)
        left_versors = humans_versors @ jnp.array([[0, 1], [-1, 0]])  # Rotate by +90 degrees
        humans_left_edge_points = rc_humans_positions + (humans_radii[:, None] - epsilon) * left_versors  # Shape: (n_humans, 2)
        humans_right_edge_points = rc_humans_positions - (humans_radii[:, None] - epsilon) * left_versors  # Shape: (n_humans, 2)
        humans_left_angles = jnp.arctan2(humans_left_edge_points[:,1], humans_left_edge_points[:,0]) # Shape: (n_humans,)
        humans_right_angles = jnp.arctan2(humans_right_edge_points[:,1], humans_right_edge_points[:,0]) # Shape: (n_humans,)
        humans_edge_angles = jnp.concatenate((humans_left_angles, humans_right_angles))  # Shape: (2*n_humans,)
        ## Obstacles
        obstacle_segments = rc_static_obstacles.reshape((self.n_obstacles*self.static_obstacles_per_scenario.shape[2], 2, 2))  # Shape: (n_obstacles*n_segments, 2, 2)
        obstacle_first_edge_points = obstacle_segments[:,0,:]  # Shape: (n_obstacles*n_segments, 2)
        obstacle_second_edge_points = obstacle_segments[:,1,:]  # Shape: (n_obstacles*n_segments, 2)
        first_to_second_versors = obstacle_second_edge_points - obstacle_first_edge_points / jnp.linalg.norm(obstacle_second_edge_points - obstacle_first_edge_points, axis=1, keepdims=True)  # Shape: (n_obstacles*n_segments, 2)
        obstacle_first_edge_points = obstacle_first_edge_points + (epsilon * first_to_second_versors)  # Shape: (n_obstacles*n_segments, 2)
        obstacle_second_edge_points = obstacle_second_edge_points - (epsilon * first_to_second_versors)  # Shape: (n_obstacles*n_segments, 2)
        obstacle_first_edge_angles = jnp.arctan2(obstacle_first_edge_points[:,1], obstacle_first_edge_points[:,0])  # Shape: (n_obstacles*n_segments,)
        obstacle_second_edge_angles = jnp.arctan2(obstacle_second_edge_points[:,1], obstacle_second_edge_points[:,0])  # Shape: (n_obstacles*n_segments,)
        obstacle_edge_angles = jnp.append(obstacle_first_edge_angles, obstacle_second_edge_angles)  # Shape: (2*n_obstacles*n_segments,)
        ## Merge and sort all edge angles
        all_edge_angles = jnp.concatenate((humans_edge_angles, obstacle_edge_angles))  # Shape: (2*n_humans + 2*n_obstacles*n_segments,)
        sorted_all_edge_angles = jnp.sort(all_edge_angles)
        # Wrap around for midpoint computation
        sorted_all_edge_angles = jnp.append(sorted_all_edge_angles, sorted_all_edge_angles[0])  # Shape: (2*n_humans + 2*n_obstacles*n_segments + 1,)
        ### Compute midpoint angles between consecutive object endpoints
        sorted_all_verors = jnp.array([jnp.cos(sorted_all_edge_angles), jnp.sin(sorted_all_edge_angles)]).T  # Shape: (2*n_humans + 2*n_obstacles*n_segments + 1, 2)
        midpoint_verors = (sorted_all_verors[:-1] + sorted_all_verors[1:])  # Shape: (2*n_humans + 2*n_obstacles*n_segments, 2)
        midpoint_verors = midpoint_verors / jnp.linalg.norm(midpoint_verors, axis=1, keepdims=True)  # Normalize
        midpoint_angles = jnp.arctan2(midpoint_verors[:,1], midpoint_verors[:,0])  # Shape: (2*n_humans + 2*n_obstacles*n_segments,)
        all_angles = jnp.concatenate((all_edge_angles, midpoint_angles)) # Shape: (4*n_humans + 4*n_obstacles*n_segments,)
        ### Ray-cast all computed angles and assess visibility of all objects (human_collision_idxs shape: (n_rays,), obstacle_collision_idxs shape: (n_rays, 2))
        _, human_collision_idxs, obstacle_collision_idxs = self.batch_ray_cast(
            all_angles,
            jnp.array([0., 0.]),
            rc_humans_positions,
            humans_radii,
            rc_static_obstacles
        )
        humans_visibility_mask = vmap(lambda idx: jnp.any(human_collision_idxs == idx))(jnp.arange(self.n_humans))  # Shape: (n_humans,)
        @jit
        def segment_visibility(obstacle_idx, segment_idx, obstacle_collision_idxs):
            return jnp.any(jnp.all(obstacle_collision_idxs == jnp.array([obstacle_idx, segment_idx]), axis=1))
        @jit
        def obstacle_segments_visibility(obstacle_idx, segment_idxs, obstacle_collision_idxs):
            return vmap(segment_visibility, in_axes=(None, 0, None))(obstacle_idx, segment_idxs, obstacle_collision_idxs)
        obstacles_visibility_mask = vmap(obstacle_segments_visibility, in_axes=(0, None, None))(
            jnp.arange(self.n_obstacles), 
            jnp.arange(self.static_obstacles_per_scenario.shape[2]), 
            obstacle_collision_idxs
        ) # Shape: (n_obstacles, n_segments)
        return humans_visibility_mask, obstacles_visibility_mask
    
    @partial(jit, static_argnames=("self"))
    def batch_object_visibility(self, batch_rc_humans_positions, batch_humans_radii, batch_rc_static_obstacles, epsilon=1e-5):
        """
        Compute object visibility with respect to the robot for a batch of frames.

        params:
        - batch_rc_humans_positions: (batch_size, n_humans, 2) array of humans positions IN ROBOT-CENTRIC FRAME
        - batch_humans_radii: (batch_size, n_humans) array of humans radii
        - batch_rc_static_obstacles: (batch_size, n_obstacles, n_segments, 2, 2) array of static obstacle segments IN ROBOT-CENTRIC FRAME
        """
        return vmap(BaseEnv.object_visibility, in_axes=(None,0,0,0,None))(
            self,
            batch_rc_humans_positions,
            batch_humans_radii,
            batch_rc_static_obstacles,
            epsilon
        )

    @partial(jit, static_argnames=("self"))
    def humans_inside_lidar_range(self, positions, radii):
        return jnp.linalg.norm(positions, axis=-1) - radii <= self.lidar_max_dist
    
    @partial(jit, static_argnames=("self"))
    def batch_humans_inside_lidar_range(self, batch_rc_positions, batch_radii):
        return vmap(BaseEnv.humans_inside_lidar_range, in_axes=(None,0,0))(
            self,
            batch_rc_positions,
            batch_radii
        )

    @partial(jit, static_argnames=("self"))
    def robot_centric_transform(
        self, 
        humans_positions,
        humans_orientations,
        humans_velocities,
        static_obstacles,
        robot_position,
        robot_orientation,
        robot_goal,
    ):
        rc_humans_positions, rc_humans_orientations, rc_humans_velocities = roto_translate_poses_and_vels(
            humans_positions,
            humans_orientations,
            humans_velocities,
            robot_position,
            robot_orientation,
        )
        rc_static_obstacles = roto_translate_obstacles(
            static_obstacles,
            robot_position,
            robot_orientation,
        )
        rc_robot_goal, _, _ = roto_translate_pose_and_vel(
            robot_goal,
            jnp.array([0.0]),
            jnp.array([0.0,0.0]),
            robot_position,
            robot_orientation,
        )
        return rc_humans_positions, rc_humans_orientations, rc_humans_velocities, rc_static_obstacles, rc_robot_goal

    @partial(jit, static_argnames=("self"))
    def batch_robot_centric_transform(
        self, 
        humans_positions,
        humans_orientations,
        humans_velocities,
        static_obstacles,
        robot_positions,
        robot_orientations,
        robot_goals,
    ):
        """
        Compute robot-centric transformations for a batch of frames.

        params:
        - humans_positions: (batch_size, n_humans, 2) array of humans positions
        - humans_orientations: (batch_size, n_humans) array of humans orientations
        - humans_velocities: (batch_size, n_humans, 2) array of humans velocities
        - static_obstacles: (batch_size, n_obstacles, n_segments, 2, 2) array of static obstacle segments
        - robot_positions: (batch_size, 2) array of robot positions
        - robot_orientations: (batch_size,) array of robot orientations
        - robot_goals: (batch_size, 2) array of robot goals
        """
        return vmap(BaseEnv.robot_centric_transform, in_axes=(None,0,0,0,0,0,0,0))(
            self,
            humans_positions,
            humans_orientations,
            humans_velocities,
            static_obstacles,
            robot_positions,
            robot_orientations,
            robot_goals,
        )

    @partial(jit, static_argnames=("self"))
    def get_grid_map_center(self, state, info):
        """
        Computes the center of the grid map based on the current state and info of the environment.

        parameters:
        - state: Current state of the environment (robot + humans)
        - info: Additional information from the environment

        returns:
        - center: Array of shape (2,) containing the (x, y) coordinates of the grid map center
        """
        # center = jnp.nanmean(jnp.vstack((jnp.reshape(info['static_obstacles'][-1], (self.n_obstacles * 2,-1)), state[-1,:2], info['robot_goal'])), axis=0)
        center = jnp.nanmean(jnp.vstack((jnp.reshape(self.static_obstacles_per_scenario[info['current_scenario']], (10,-1)), state[-1,:2], info['robot_goal'])), axis=0)
        return center

    @partial(jit, static_argnames=("self"))
    def build_grid_map_and_occupancy(self, state, info, epsilon=1e-5):
        """
        Builds a square grid map centered around the robot and computes the occupancy grid based on static obstacles.

        parameters:
        - state: Current state of the environment (robot + humans)
        - info: Additional information from the environment

        returns:
        - grid_cells: Array of shape (n_x, n_y, 2) containing the (x, y) coordinates of each grid cell center. n_x and n_y depend on the fixed grid size defined by cell_size and min_grid_size.
        - occupancy_grid: Boolean array of shape (n_x, n_y), where True indicates an occupied cell
        - edges: Array of shape (n_cells, n_cells) representing the edges matrix for pathfinding
        """
        cell_size = self.grid_cell_size # Grid cell size (in meters)
        min_grid_size = self.grid_min_size # Grid minimum size (in meters)
        center = self.get_grid_map_center(state, info)
        dists_vector = jnp.concatenate([-jnp.arange(0, min_grid_size/2 + cell_size, cell_size)[::-1][:-1],jnp.arange(0, min_grid_size/2 + cell_size, cell_size)])
        grid_center_x, grid_center_y = jnp.meshgrid(dists_vector + center[0], dists_vector + center[1])
        n_x = grid_center_x.shape[0]
        n_y = grid_center_y.shape[1]
        grid_cells = jnp.array(jnp.vstack((grid_center_x.flatten(), grid_center_y.flatten())).T)
        @jit
        def _edge_intersects_cell(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
            @jit
            def _not_nan_obs(val:tuple):
                x1, y1, x2, y2, xmin, xmax, ymin, ymax = val
                dx = x2 - x1
                dy = y2 - y1
                p = jnp.array([-dx, dx, -dy, dy])
                q = jnp.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1])
                @jit
                def loop_body(i, tup):
                    t, p, q = tup
                    t0, t1 = t
                    t0, t1 = lax.switch(
                        (jnp.sign(p[i])+1).astype(jnp.int32),
                        [
                            lambda t: lax.cond(q[i]/p[i] > t[1], lambda _: (2.,1.), lambda x: (jnp.max(jnp.array([x[0],q[i]/p[i]])), x[1]), t),  # p[i] < 0
                            lambda t: lax.cond(q[i] < 0, lambda _: (2.,1.), lambda x: x, t),  # p[i] == 0
                            lambda t: lax.cond(q[i]/p[i] < t[0], lambda _: (2.,1.), lambda x: (x[0], jnp.min(jnp.array([x[1],q[i]/p[i]]))), t),  # p[i] > 0
                        ],
                        (t0, t1),
                    )
                    # debug.print("t0: {x}, t1: {y}, switch_case: {z}", x=t0, y=t1, z=(jnp.sign(p[i])+1).astype(jnp.int32))
                    return ((t0, t1), p ,q)
                t, p, q = lax.fori_loop(
                    0, 
                    4,
                    loop_body,
                    ((0., 1.), p, q),
                )
                t0, t1 = t
                inside_or_intersects = ~(t0 > t1)
                return inside_or_intersects
            @jit
            def _nan_obs(val:tuple):
                # If the obstacle is NaN, it means it doesn't exist, so it cannot intersect the cell
                return False
            return lax.cond(
                jnp.any(jnp.isnan(jnp.array([x1, y1, x2, y2]))), 
                _nan_obs,
                _not_nan_obs, 
                (x1, y1, x2, y2, xmin, xmax, ymin, ymax)
            )
        @jit
        def _obstacle_intersects_cell(obstacle, xmin, xmax, ymin, ymax):
            return jnp.any(vmap(_edge_intersects_cell, in_axes=(0,0,0,0,None,None,None,None))(obstacle[:,0,0], obstacle[:,0,1], obstacle[:,1,0], obstacle[:,1,1], xmin, xmax, ymin, ymax))
        @jit
        def _is_cell_occupied(obstacles, xmin, xmax, ymin, ymax):
            return jnp.any(vmap(_obstacle_intersects_cell, in_axes=(0, None, None, None, None))(obstacles, xmin, xmax, ymin, ymax))
        @jit
        def _build_occupancy_vector(obstacles, xmins, xmaxs, ymins, ymaxs):
            """
            Returns a boolean array of shape (n_cells,) indicating whether each cell is occupied (True) or free (False).

            parameters:
            - obstacles: Array of shape (n_obstacles, n_edges, 2, 2) representing the line segments of the obstacles
            - xmins, xmaxs, ymins, ymaxs: Arrays of shape (n_cells,) representing the boundaries of each grid cell

            returns:
            - occupancy_vector: Boolean array of shape (n_cells,), where True indicates an occupied cell
            """
            return vmap(_is_cell_occupied, in_axes=(None, 0, 0, 0, 0))(obstacles, xmins, xmaxs, ymins, ymaxs)
        # Prepare obstacle segments
        occupancy_vector = _build_occupancy_vector(
            info['static_obstacles'][-1],
            grid_cells[:,0] - cell_size/2 - epsilon,
            grid_cells[:,0] + cell_size/2 + epsilon,
            grid_cells[:,1] - cell_size/2 - epsilon,
            grid_cells[:,1] + cell_size/2 + epsilon,
        )
        grid_cells = jnp.stack((grid_center_x, grid_center_y), axis=-1)
        occupancy_grid = jnp.reshape(occupancy_vector, (n_x, n_y))
        return grid_cells, occupancy_grid
    
    @partial(jit, static_argnames=("self"))
    def get_grid_size(self):
        """
        Computes the size of the grid map based on the cell size and minimum grid size.

        returns:
        - n_x: Number of cells in the x direction
        - n_y: Number of cells in the y direction
        """
        cell_size = self.grid_cell_size # Grid cell size (in meters)
        min_grid_size = self.grid_min_size # Grid minimum size (in meters)
        dists_vector = jnp.concatenate([-jnp.arange(0, min_grid_size/2 + cell_size, cell_size)[::-1][:-1],jnp.arange(0, min_grid_size/2 + cell_size, cell_size)])
        n_x = dists_vector.shape[0]
        n_y = dists_vector.shape[0]
        return n_x, n_y