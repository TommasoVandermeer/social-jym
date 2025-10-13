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

class BaseEnv(ABC):
    def __init__(
        self,
        robot_radius:float, 
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
        kinematics:str,
        max_cc_delay:float,
        ccso_n_static_humans:int,
        grid_map_computation:bool,
        grid_cell_size:float,
        grid_min_size:float,
    ) -> None:
        ## Args validation
        assert scenario in SCENARIOS or scenario is None, f"Invalid scenario. Choose one of {SCENARIOS}, or None for custom scenario."
        if scenario is None:
            print("\nWARNING: Custom scenario is selected. Make sure to implement the 'reset_custom_episode' method in the derived class (not 'reset').\n")
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert kinematics in ROBOT_KINEMATICS, f"Invalid robot kinematics. Choose one of {ROBOT_KINEMATICS}"
        if grid_map_computation:
            assert grid_cell_size > 0, "There should be at least one obstacle (also padding obstacles) to enable grid map computation."
        ## Env initialization
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
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        self.max_cc_delay = max_cc_delay
        self.ccso_n_static_humans = ccso_n_static_humans
        # Global planning parameters
        if grid_map_computation:
            print("\nWARNING: Grid map computation is enabled. This will slow down the simulation, especially if many static obstacles are present.\n")
        self.grid_map_computation = grid_map_computation
        self.grid_cell_size = grid_cell_size
        self.grid_min_size = grid_min_size

    # --- Private methods ---

    @abstractmethod
    def _get_obs(self, state):
        pass

    @abstractmethod
    def _reset(self, key):
        pass

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
        return jnp.min(vmap(BaseEnv._human_ray_intersect, in_axes=(None,None,0,None,0))(self, direction, human_positions, lidar_position, human_radiuses))

    @partial(jit, static_argnames=("self"))
    def _ray_cast(self, angle:float, lidar_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray) -> float:
        # TODO: add obstacles ray casting
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        measurement = self._batch_human_ray_intersect(direction, human_positions, lidar_position, human_radiuses)
        return measurement

    @partial(jit, static_argnames=("self"))
    def _batch_ray_cast(self, angles:float, lidar_position:jnp.ndarray, human_positions:jnp.ndarray, human_radiuses:jnp.ndarray) -> jnp.ndarray:
        return vmap(BaseEnv._ray_cast, in_axes=(None,0,None,None,None))(self, angles, lidar_position, human_positions, human_radiuses)

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
            def _update_human_state_and_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, positions:jnp.ndarray, radiuses:jnp.ndarray, safety_spaces:jnp.ndarray) -> tuple:
                position, goal = lax.cond(
                    # jnp.linalg.norm(position - goal) <= radius + 2,
                    jnp.linalg.norm(position - goal) <= 3, # Compliant with Social-Navigation-PyEnvs
                    lambda _: (
                        jnp.array([
                        # jnp.max(jnp.append(positions[:,0]+(jnp.max(jnp.append(radiuses,self.robot_radius))*2)+(jnp.max(safety_spaces)*2)+0.05, self.traffic_length/2+1)), 
                        jnp.max(jnp.append(positions[:,0] + (jnp.max(jnp.append(radiuses, self.robot_radius))*2)+(jnp.max(safety_spaces)*2), self.traffic_length/2)), # Compliant with Social-Navigation-PyEnvs
                        jnp.clip(position[1], -self.traffic_height/2, self.traffic_height/2)]
                        ),
                        jnp.array([goal[0], position[1]]),
                    ),
                    lambda x: x,
                    (position, goal))
                return position, goal
            info, state = val
            new_positions, new_goals = vmap(_update_human_state_and_goal, in_axes=(0,0,0,None,None,None))(
                state[:-1,0:2], 
                info["humans_goal"], 
                info["humans_parameters"][:,0], 
                state[:,0:2], 
                info["humans_parameters"][:,0], 
                info["humans_parameters"][:,-1])
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
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float) -> jnp.ndarray:
                goal = lax.cond(
                    jnp.linalg.norm(position - goal) <= radius+0.1,
                    lambda x: lax.cond(
                        x[0]==x[1],
                        lambda y: lax.cond(
                            position[1] < position[0],
                            lambda z: jnp.array([0., jnp.max(z)]),
                            lambda z: jnp.array([jnp.max(z), 0.]),
                            y,
                        ),
                        lambda y: jnp.array([jnp.max(y),jnp.max(y)]),
                        x,
                    ),
                    lambda x: x,
                    goal)
                return goal
            info, state = val
            info["humans_goal"] = vmap(_update_human_goal, in_axes=(0,0,0))(state[:-1,0:2], info["humans_goal"], info["humans_parameters"][:,0])
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

    @abstractmethod
    def reset(self, key):
        pass

    @abstractmethod
    def reset_custom_episode(self, key, episode):
        pass

    @abstractmethod
    def step(self, env_state, action):
        pass

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
    def get_lidar_measurements(
        self, 
        lidar_position:jnp.ndarray, 
        lidar_yaw:float,  
        human_positions:jnp.ndarray, 
        human_radiuses:jnp.ndarray
    ) -> jnp.ndarray:
        """
        Given the current state of the environment, the robot orientation and the additional information about the environment,
        this function computes the lidar measurements of the robot.

        args:
        - lidar_position (2,): jnp.ndarray containing the x and y coordinates of the lidar.
        - lidar_yaw (1,): float containing the orientation of the lidar.
        - human_positions (self.n_humans,2): jnp.ndarray containing the x and y coordinates of the humans.
        - human_radiuses (self.n_humans,): jnp.ndarray containing the radius of the humans.

        output:
        - lidar_output (self.lidar_num_rays,2): jnp.ndarray containing the lidar measurements of the robot and the angle for each ray.
        """
        angles = jnp.linspace(lidar_yaw - self.lidar_angular_range/2, lidar_yaw + self.lidar_angular_range/2, self.lidar_num_rays)
        measurements = self._batch_ray_cast(angles, lidar_position, human_positions, human_radiuses)
        lidar_output = jnp.stack((measurements, angles), axis=-1)
        return lidar_output
    
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