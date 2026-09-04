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
    # Social scenarios
    "circular_crossing", 
    "parallel_traffic", 
    "perpendicular_traffic", 
    "robot_crowding", 
    "delayed_circular_crossing",
    "circular_crossing_with_static_obstacles",
    "crowd_navigation",
    # Realistic scenarios (testing)
    "corner_traffic", # Double waypoint
    "door_crossing", # Double waypoint
    "crowd_chasing",
    # Navigation scenarios
    "turn_l",
    "narrow_passage",
    "slalom",
    "random_obstacle",
    "narrow_corridor",
    "random_room",
    "t_corridor",
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
ROBOT_VELOCITY_DYNAMICS = [
    "first_order_system",
    "coupled_slew_rate"
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

@partial(jit, static_argnames=("N", "max_attempts"))
def generate_wall_noise(
    key: random.PRNGKey,
    obstacles: jnp.ndarray,    # Shape: (O, 1, 2, 2)
    robot_position: jnp.ndarray, # Shape: (2,)
    robot_goal: jnp.ndarray,     # Shape: (2,)
    N: int,                    # Numero di micro-segmenti da generare
    noise_length: float = 0.4,  # Lunghezza del micro-segmento (es. 5 cm)
    noise_offset: float = 0.05,  # Spostamento ortogonale dal muro (es. 2 cm)
    safe_distance: float = 1.0,  # Distanza minima da robot e goal
    max_attempts: int = 5       # Limite loop per evitare loop infiniti nel JIT
) -> jnp.ndarray:
    walls = obstacles.reshape(-1, 2, 2)
    num_walls = walls.shape[0]
    def generate_single_noise(k_init):
        def cond_fn(state):
            step, _, valid, _ = state
            return (step < max_attempts) & (~valid)
        def body_fn(state):
            step, k_curr, _, _ = state
            k1, k2, k3, k4, k_next = random.split(k_curr, 5)
            w_idx = random.randint(k1, shape=(), minval=0, maxval=num_walls)
            wall = walls[w_idx] # (2, 2)
            p1, p2 = wall[0], wall[1]
            t = random.uniform(k2, shape=())
            c = p1 + t * (p2 - p1)
            dist_robot = jnp.linalg.norm(c - robot_position)
            dist_goal = jnp.linalg.norm(c - robot_goal)
            valid = (dist_robot >= safe_distance) & (dist_goal >= safe_distance)
            wall_vec = p2 - p1
            wall_angle = jnp.arctan2(wall_vec[1], wall_vec[0])
            angle_noise = random.uniform(k3, shape=(), minval=-jnp.pi/4, maxval=jnp.pi/4)
            final_angle = wall_angle + angle_noise
            dir_vec = jnp.array([jnp.cos(final_angle), jnp.sin(final_angle)])
            normal_vec = jnp.array([-jnp.sin(wall_angle), jnp.cos(wall_angle)])
            c_offset = c + normal_vec * random.uniform(k4, shape=(), minval=-noise_offset, maxval=noise_offset)
            seg = jnp.stack([
                c_offset - (noise_length / 2) * dir_vec,
                c_offset + (noise_length / 2) * dir_vec
            ])
            return step + 1, k_next, valid, seg
        init_state = (0, k_init, False, jnp.zeros((2, 2)))
        _, _, is_valid, final_seg = lax.while_loop(cond_fn, body_fn, init_state)
        return jnp.where(is_valid, final_seg, jnp.full((2, 2), jnp.nan))
    keys = random.split(key, N)
    noise_segments = vmap(generate_single_noise)(keys)
    return noise_segments[:, None, :, :]

@partial(jit, static_argnames=("n_humans"))
def get_humans_standard_leg_parameters(n_humans):
    single_human_leg_params = jnp.array([
        0.5, # step length
        0.75, # legs base percent (percent of human radius at which legs are attached to the body horizontally)
        0.6, # step duration
        0.12, # leg radius (ALWAYS KEEP LAST)
    ])
    return jnp.repeat(single_human_leg_params[None], n_humans, axis=0)

@partial(jit, static_argnames=("humans_policy"))
def init_single_human_leg_state(random_key, human_state, human_radius, legs_parameters, humans_policy):
    if humans_policy == HUMAN_POLICIES.index('hsfm'):
        orientation = human_state[4]
    else:
        orientation = jnp.arctan2(human_state[3], human_state[2])
    px, py = human_state[0], human_state[1]
    key1, key2 = random.split(random_key)
    stance_leg = random.bernoulli(key1, 0.5)
    stance_leg_phase = random.uniform(key2, minval=0., maxval=.5)
    swing_leg_phase = stance_leg_phase + 0.5
    def _left_is_stance():
        left_x  = px + legs_parameters[1]*human_radius*jnp.cos(orientation+jnp.pi/2) - (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.cos(orientation)
        left_y  = py + legs_parameters[1]*human_radius*jnp.sin(orientation+jnp.pi/2) - (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.sin(orientation)
        right_x = px + legs_parameters[1]*human_radius*jnp.cos(orientation-jnp.pi/2) + (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.cos(orientation)
        right_y = py + legs_parameters[1]*human_radius*jnp.sin(orientation-jnp.pi/2) + (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.sin(orientation)
        return jnp.array([left_x, left_y, stance_leg_phase, right_x, right_y, swing_leg_phase])
    def _right_is_stance():
        left_x  = px + legs_parameters[1]*human_radius*jnp.cos(orientation+jnp.pi/2) + (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.cos(orientation)
        left_y  = py + legs_parameters[1]*human_radius*jnp.sin(orientation+jnp.pi/2) + (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.sin(orientation)
        right_x = px + legs_parameters[1]*human_radius*jnp.cos(orientation-jnp.pi/2) - (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.cos(orientation)
        right_y = py + legs_parameters[1]*human_radius*jnp.sin(orientation-jnp.pi/2) - (legs_parameters[0]/2) * 2 * stance_leg_phase * jnp.sin(orientation)
        return jnp.array([left_x, left_y, swing_leg_phase, right_x, right_y, stance_leg_phase])
    return lax.cond(stance_leg == 1, _left_is_stance, _right_is_stance)

@partial(jit, static_argnames=("dt","humans_policy"))
def update_single_human_leg(human_state, human_leg_state, human_radius, dt, legs_parameters, humans_policy):
    if humans_policy == HUMAN_POLICIES.index('hsfm'):
        orientation = human_state[4]
    else:
        orientation = jnp.arctan2(human_state[3], human_state[2])
    position = human_state[0:2]
    com_left  = position + legs_parameters[1]*human_radius*jnp.array([jnp.cos(orientation+jnp.pi/2), jnp.sin(orientation+jnp.pi/2)])
    com_right = position + legs_parameters[1]*human_radius*jnp.array([jnp.cos(orientation-jnp.pi/2), jnp.sin(orientation-jnp.pi/2)])

    def stance(lx, ly): return lx, ly
    def swing(com, lx, ly, phase):
        alpha = (phase - 0.5) * 2
        tx = com[0] + (legs_parameters[0]/2)*jnp.cos(orientation)
        ty = com[1] + (legs_parameters[0]/2)*jnp.sin(orientation)
        return (1-alpha)*lx + alpha*tx, (1-alpha)*ly + alpha*ty

    lx, ly = lax.cond(
        human_leg_state[2] < 0.5,
        lambda: stance(human_leg_state[0], human_leg_state[1]),
        lambda: swing(com_left, human_leg_state[0], human_leg_state[1], human_leg_state[2]))
    rx, ry = lax.cond(
        human_leg_state[5] < 0.5,
        lambda: stance(human_leg_state[3], human_leg_state[4]),
        lambda: swing(com_right, human_leg_state[3], human_leg_state[4], human_leg_state[5]))
    return jnp.array([lx, ly, (human_leg_state[2]+dt/legs_parameters[2])%1.0, rx, ry, (human_leg_state[5]+dt/legs_parameters[2])%1.0])

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
        robot_visible:bool, # If set to None, it can be modified at each new episode (Curriculum setting)
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
        velocity_dynamics:str,
        tau_action_0:float, # Used for First Order System velocity dynamics
        tau_action_1:float, # Used for First Order System velocity dynamics
        wheels_max_linear_acceleration:float, # Used for Coupled Slew Rate velocity dynamics
        wheels_distance:float, # Used for Coupled Slew Rate velocity dynamics
        control_delay_mean:float,
        control_delay_sigma:float,
        lidar_salt_and_pepper_prob:float,
        kinematics:str,
        max_cc_delay:float,
        ccso_n_static_humans:int,
        ccso_static_humans_radius_mean:float,
        ccso_static_humans_radius_std:float,
        grid_map_computation:bool,
        grid_cell_size:float,
        grid_min_size:float,
        thick_default_obstacle:bool,
        obstacles_noise:float,
        noisy_walls:bool,
        leg_dynamics:bool,
    ) -> None:
        ## Args validation
        assert (scenario in SCENARIOS) or (scenario is None) or (scenario in ['training_scenario','testing_scenario']), f"Invalid scenario. Choose one of {SCENARIOS}, None for custom scenario, or training_scenario for a mixture of scenarios for training, or testing_scenario for a mixture of scenarios for testing."
        if scenario is None:
            print("\nWARNING: Custom scenario is selected. Make sure to implement the 'reset_custom_episode' method in the derived class (not 'reset').\n")
        if scenario == 'training_scenario':
            scenario = 'hybrid_scenario'
            hybrid_scenario_subset = jnp.array([0,1,2,3,4,6]) # Skip CCSO
        if scenario == 'testing_scenario':
            scenario = 'hybrid_scenario'
            hybrid_scenario_subset = jnp.array([7,8,9]) # Double waypoint scenarios
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert kinematics in ROBOT_KINEMATICS, f"Invalid robot kinematics. Choose one of {ROBOT_KINEMATICS}"
        if grid_map_computation:
            assert grid_cell_size > 0, "There should be at least one obstacle (also padding obstacles) to enable grid map computation."
        assert humans_dt <= robot_dt, "The humans' time step must be less or equal than the robot's time step."
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        if scenario == SCENARIOS.index('circular_crossing_with_static_obstacles') or (scenario == SCENARIOS.index('hybrid_scenario') and SCENARIOS.index('circular_crossing_with_static_obstacles') in hybrid_scenario_subset):
            assert n_humans > ccso_n_static_humans, "The number of static humans must be less than the total number of humans."
        assert tau_action_0 >= 0., "Time constant of first order system for action 0 should be greater or equal to zero."
        assert tau_action_1 >= 0., "Time constant of first order system for action 1 should be greater or equal to zero."
        assert control_delay_mean >= 0., "Mean control delay should be greater or equal to zero."
        assert control_delay_sigma >= 0., "Control delay sigma should be greater or equal to zero."
        assert velocity_dynamics in ROBOT_VELOCITY_DYNAMICS, f"Robot velocity dynamics must be one of {ROBOT_VELOCITY_DYNAMICS}"
        if velocity_dynamics == "coupled_slew_rate":
            assert kinematics == "unicycle"
        assert obstacles_noise <= 0.2, "Obstacles noise should be kept low (<=0.2) as otherwise episodes construction might be too random"
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
        self.robot_velocity_dynamics = ROBOT_VELOCITY_DYNAMICS.index(velocity_dynamics)
        self.tau_action_0 = tau_action_0
        self.tau_action_1 = tau_action_1
        self.wheels_max_linear_acceleration = wheels_max_linear_acceleration
        self.wheels_distance = wheels_distance
        self.control_delay_mean = control_delay_mean
        self.control_delay_sigma = control_delay_sigma
        self.actions_history_length = jnp.max(jnp.array([jnp.ceil((self.control_delay_mean + 3 * self.control_delay_sigma)/self.robot_dt), 2], dtype=jnp.int32))
        self.action_0_dynamics = tau_action_0 > 0.
        self.action_1_dynamics = tau_action_1 > 0.
        self.limited_acceleration = wheels_max_linear_acceleration < jnp.inf
        self.kinematics = ROBOT_KINEMATICS.index(kinematics)
        self.max_cc_delay = max_cc_delay
        self.ccso_n_static_humans = ccso_n_static_humans
        self.ccso_static_humans_radius_mean = ccso_static_humans_radius_mean
        self.ccso_static_humans_radius_std = ccso_static_humans_radius_std
        self.thick_default_obstacle = thick_default_obstacle
        self.n_segments = 4 if self.thick_default_obstacle else 1
        # Global planning parameters
        if grid_map_computation:
            print("\nWARNING: Grid map computation is enabled. This will slow down the simulation, especially if many static obstacles are present.\n")
        self.grid_map_computation = grid_map_computation
        self.grid_cell_size = grid_cell_size
        self.grid_min_size = grid_min_size
        self.leg_dynamics = leg_dynamics
        self.obstacles_noise = obstacles_noise
        self.noisy_walls = noisy_walls
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
                # [[[-1.,0],[1.,0.]]],
                # [[[-self.traffic_length/4-0.5,self.traffic_height/4],[-self.traffic_length/4+0.5,self.traffic_height/4]]],
                # [[[self.traffic_length/4-0.5,self.traffic_height/4],[self.traffic_length/4+0.5,self.traffic_height/4]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
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
            [ # Door crossing
                [[[0., 0.75],[0., 2.5]]],
                [[[0., -0.75],[0., -2.5]]],
                [[[-5., 2.5],[5., 2.5]]],
                [[[-5., -2.5],[5., -2.5]]],
                [[[2.5, 0.5],[2.5, -0.5]]],
            ],
            [ # Crowd chasing
                [[[-self.traffic_length/2-3, self.traffic_height/2 + 0.7],[self.traffic_length/2+3, self.traffic_height/2 + 0.7]]],
                [[[-self.traffic_length/2-3, -(self.traffic_height/2 + 0.7)],[self.traffic_length/2+3, -(self.traffic_height/2 + 0.7)]]],
                [[[-1.,0],[1.,0.]]],
                [[[-self.traffic_length/2-3, self.traffic_height/2 + 0.7],[-self.traffic_length/2-3, -(self.traffic_height/2 + 0.7)]]],
                [[[self.traffic_length/2+3, self.traffic_height/2 + 0.7],[self.traffic_length/2+3, -(self.traffic_height/2 + 0.7)]]],
            ],
            [ # L-turn
                [[[-1.,-5.],[1.,-5.]]],
                [[[1.,-5.],[1.,3.]]],
                [[[1.,3.],[-3., 3.]]],
                [[[-3., 1.],[-1., 1.]]],
                [[[-1., 1.],[-1., -5.]]],
            ],
            [ # Narrow passage
                [[[-3., -4.],[3., -4.]]],
                [[[3., -4.],[3., 2.]]],
                [[[3., 2.],[0.65, 2.]]],
                [[[-0.65, 2.],[-3., 2.]]],
                [[[-3., 2.],[-3., -4.]]],
            ],
            [ # Slalom
                [[[-1.5, 8.],[-1.5, -5.]]],
                [[[-1.5, -5.],[1.5, -5.]]],
                [[[1.5, -5.],[1.5, 8.]]],
                [[[-1.5, -2.],[0., -2.]]],
                [[[1.5, 2.],[0., 2.]]],
            ],
            [ # Random obstacle
                [[[-6.,-6.],[-6.,6.]]],
                [[[-6.,6.],[6.,6.]]],
                [[[6.,6.],[6.,-6.]]],
                [[[6.,-6.],[-6.,-6.]]],
                [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
            ],
            [ # Narrow corridor
                [[[-0.9,6.],[-0.9,-6.]]],
                [[[0.9,6.],[0.9,1.2]]],
                [[[0.9,1.2],[1.2,0.]]],
                [[[1.2,0.],[0.9,-1.2]]],
                [[[0.9,-1.2],[0.9,-6.]]],
            ],
            [ # Random room
                [[[0.5,0.5],[0.5,-0.5]]],
                [[[0.5,-0.5],[-0.5,-0.5]]],
                [[[-0.5,-0.5],[-0.5,0.5]]],
                [[[-0.5,0.5],[0.5,0.5]]],
                [[[-0.1,0.],[0.1,0.]]],
            ],
            [ # T-corridor
                [[[-2.,1.],[2,1.]]],
                [[[-2.,-.5],[-.5,-.5]]],
                [[[-.5,-.5],[-.5,-2]]],
                [[[.5,-2.],[.5,-.5]]],
                [[[.5,-.5],[2.,-.5]]],
            ],
        ])
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
            [[self.traffic_length/2+self.traffic_height/4, self.traffic_length/2+self.traffic_height/4],[self.traffic_length/2, 1.]], # Corner traffic
            [[0., 0.],[5., 0.]], # Door crossing
            [[self.traffic_length/2-1, 0.],[jnp.nan, jnp.nan]], # Crowd chasing
            [[-2., 2.],[jnp.nan, jnp.nan]], # L-turn
            [[0., 3.],[jnp.nan, jnp.nan]], # Narrow passage
            [[0.,7.],[jnp.nan, jnp.nan]], # Slalom
            [[0.,5.],[jnp.nan, jnp.nan]], # Random obstacle
            [[0.7, 0.],[jnp.nan, jnp.nan]], # Narrow corridor
            [[0., 1.],[jnp.nan, jnp.nan]], # Random room
            [[0., -1.],[jnp.nan, jnp.nan]], # T-corridor
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
    def _reset(self, key:random.PRNGKey, scenarios_prob:jnp.ndarray=None, visibility_chance:float=0.) -> tuple[jnp.ndarray, random.PRNGKey, dict]:
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
                self._generate_door_crossing_episode,
                self._generate_crowd_chasing_episode,
                self._generate_l_turn_episode,
                self._generate_narrow_passage_episode,
                self._generate_slalom_episode,
                self._generate_random_obstacle_episode,
                self._generate_narrow_corridor_episode,
                self._generate_random_room_episode,
                self._generate_t_corridor_episode,
            ], 
            scen_key
        )
        if self.noisy_walls:
            key, subkey = random.split(key, 2)
            noisy_walls = generate_wall_noise(
                subkey,
                static_obstacles, 
                full_state[-1,:2], 
                robot_goal, 
                5,                    
            )
            static_obstacles = jnp.concatenate([static_obstacles, noisy_walls], axis=0)
        if self.thick_default_obstacle:
            static_obstacles = thicken_obstacles(static_obstacles, thickness=0.1)
        static_obstacles = jnp.repeat(jnp.array([static_obstacles]), self.n_humans+1, axis=0)
        # TODO: Filter obstacles based on the robot position and grid cell decomposition of static obstacles
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
            visibility_chance=visibility_chance,
        )
        if self.grid_map_computation: # Compute the grid map of static obstacles for global planning
            info['grid_cells'], info['occupancy_grid'] = self.build_grid_map_and_occupancy(full_state, info)
            info['grid_cells_size'] = self.grid_cell_size
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
        visibility_chance:float=0.,
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
        leg_param_key, leg_init_key, visibility_key = random.split(noise_key, 3)
        leg_parameters = get_humans_standard_leg_parameters(self.n_humans)
        leg_state = vmap(init_single_human_leg_state, in_axes=(0, 0, 0, 0, None))(
                random.split(leg_init_key, self.n_humans),
                full_state[0:self.n_humans], 
                humans_parameters[:, 0], 
                leg_parameters,
                self.humans_policy,
            )
        # Visibility computation
        if self.robot_visible is not None:
            visibility_chance = float(self.robot_visible)
        visibility = jnp.fill_diagonal(
            jnp.ones((self.n_humans+1,self.n_humans+1), dtype=jnp.bool), 
            jnp.zeros((self.n_humans+1,), dtype=jnp.bool), 
            inplace=False
        ).at[:-1,-1].set(random.bernoulli(visibility_key, p=visibility_chance, shape=(self.n_humans,)))
        info = {
            "humans_goal": humans_goal,
            "visibility": visibility,
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
            "action_history": jnp.zeros((self.actions_history_length, 2)), # History of taken actions, used for modelling delays
            "robot_delay": 0., # Current delay of the robot, used for delayed action execution
            "intermediate_states": jnp.repeat(full_state[None], int(self.robot_dt/self.humans_dt), axis=0), # Used to store the intermediate states between two robot steps
            "humans_leg_parameters": leg_parameters,
            "humans_leg_state": leg_state,
            "intermediate_leg_states": jnp.repeat(leg_state[None], int(self.robot_dt/self.humans_dt), axis=0), # Used to store the intermediate leg states between two robot steps
        }
        return info

    @partial(jit, static_argnames=("self"))
    def _init_obstacles(self, key:random.PRNGKey, scenario:int) -> jnp.ndarray:
        if self.n_obstacles == 0:
            return jnp.full((1, 1, 2, 2), jnp.nan)
        else:
            perm_key, noise_key = random.split(key)
            # Pick obstacles
            obstacles = self.static_obstacles_per_scenario[scenario]
            perm = random.permutation(perm_key, obstacles.shape[0])
            shuffled_obstacles = obstacles[perm]
            picked_obstacles = shuffled_obstacles[:self.n_obstacles]
            # Add small random noise to vertices
            noise = random.uniform(noise_key, (self.n_obstacles,1,2,2)) * self.obstacles_noise
            noised_obstacles = picked_obstacles + noise
            return noised_obstacles

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
                    jnp.squeeze(radius + random.uniform(key, shape=(1,), minval=-self.ccso_static_humans_radius_std, maxval=self.ccso_static_humans_radius_std)),
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
            self.ccso_static_humans_radius_mean, 
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
    def _generate_door_crossing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([-4., 0.])) 
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, humans_goal, key, valid = while_val
                key, room_key, noise_key, goal_key = random.split(key, 4)
                room = random.bernoulli(room_key)
                normalized_point = random.uniform(noise_key, shape=(2,), minval=-1, maxval=1)
                new_point = jnp.array([
                    -2.5 + room * 5 + normalized_point[0] * 2.4, 
                    normalized_point[1] * 2.4,
                ])
                differences = jnp.linalg.norm(disturbed_points - new_point, axis=1)
                valid = jnp.all(differences >= (2 * (jnp.max(humans_parameters[:, 0]) + 0.1)))
                disturbed_points = lax.cond(
                    valid,
                    lambda _: disturbed_points.at[i].set(new_point),
                    lambda _: disturbed_points,
                    operand=None)
                normalized_goal_point = random.uniform(noise_key, shape=(2,), minval=-1, maxval=1)
                humans_goal = humans_goal.at[i].set(jnp.array([
                    -2.5 + room * 5 + normalized_goal_point[0] * 2.4, 
                    normalized_goal_point[1] * 2.4,
                ]))
                return (disturbed_points, humans_goal, key, valid)
            disturbed_points, humans_goal, key = for_val
            disturbed_points, humans_goal, key, _ = lax.while_loop(lambda val: jnp.logical_not(val[3]), _while_body, (disturbed_points, humans_goal, key, False))
            return disturbed_points, humans_goal, key
    
        disturbed_points, humans_goal, key = lax.fori_loop(0, self.n_humans, _fori_body, (disturbed_points, jnp.empty((self.n_humans,2)), key))

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
            key, subkey = random.split(key)
            random_orientations = random.uniform(subkey, shape=(self.n_humans,), minval=-1, maxval=1) * jnp.pi
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], random_orientations))
        elif self.humans_policy == HUMAN_POLICIES.index('sfm') or self.humans_policy == HUMAN_POLICIES.index('orca'):
            # Humans
            full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:]]))

        # Assign robot goals
        robot_goal = self._init_robot_goal(SCENARIOS.index('door_crossing'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('door_crossing'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _generate_crowd_chasing_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_goal = jnp.zeros((self.n_humans, 2))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)

        # Randomly generate the humans' positions
        disturbed_points = jnp.ones((self.n_humans+1, 2)) * -1000
        disturbed_points = disturbed_points.at[-1].set(jnp.array([-self.traffic_length/2 + 1, 0.])) 
        
        @jit
        def _fori_body(i:int, for_val:tuple):
            @jit 
            def _while_body(while_val:tuple):
                disturbed_points, key, valid = while_val
                key, subkey = random.split(key)
                normalized_point = random.uniform(subkey, shape=(2,), minval=0, maxval=1)
                new_point = jnp.array([-self.traffic_length/2 + 2 + normalized_point[0] * (self.traffic_length/2), -self.traffic_height/2 + normalized_point[1] * self.traffic_height])
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
        # Humans
        full_state = full_state.at[:-1].set(vmap(_set_state, in_axes=(0, 0))(disturbed_points[:-1], jnp.zeros((self.n_humans,))))
        # Robot
        full_state = full_state.at[self.n_humans].set(jnp.array([*disturbed_points[-1], *full_state[self.n_humans,2:]]))

        # Assign the humans' and robot goals
        humans_goal = lax.fori_loop(
            0, 
            self.n_humans, 
            lambda i, humans_goal: humans_goal.at[i].set(jnp.array([self.traffic_length/2+3, disturbed_points[i,1]])),
            humans_goal)
        robot_goal = self._init_robot_goal(SCENARIOS.index('crowd_chasing'))

        # Obstacles
        static_obstacles = self._init_obstacles(key, SCENARIOS.index('crowd_chasing'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _generate_l_turn_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, obs_key, rob_key, rob_goal_key = random.split(key, 4)
        points = _gen_human(random.split(hum_key, self.n_humans))
        humans_goal = points
        full_state = full_state.at[:-1,:2].set(points)
        # Robot
        robot_position = jnp.array([0., -3.5]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([0.4, 0.9])
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., jnp.pi/2, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('turn_l'))
        robot_goal = robot_goal + random.uniform(rob_goal_key, (2,),minval=-1.) * jnp.array([0.5, 0.5])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('turn_l'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _generate_narrow_passage_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, obs_key, rob_key, delay_key = random.split(key, 4)
        points = _gen_human(random.split(hum_key, self.n_humans))
        points = points.at[-1].set(jnp.array([0., 1.65]) + random.uniform(hum_key, (2,),minval=-1.) * jnp.array([0., 0.4]))
        humans_goal = points
        humans_goal = humans_goal.at[-1].set(jnp.array([0., 5.]))
        full_state = full_state.at[:-1,:2].set(points)
        humans_delay = random.choice(delay_key, self.possible_delays, shape=(self.n_humans,))
        # Robot
        robot_position = jnp.array([0., -3.]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([2., 0.5])
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., jnp.pi/2, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('narrow_passage'))
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('narrow_passage'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, humans_delay
    
    @partial(jit, static_argnames=("self"))
    def _generate_slalom_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, obs_key, rob_key, rob_orient_key, rob_goal_key = random.split(key, 5)
        points = _gen_human(random.split(hum_key, self.n_humans))
        humans_goal = points
        full_state = full_state.at[:-1,:2].set(points)
        # Robot
        robot_position = jnp.array([0., -4.]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([0.4, 0.4])
        robot_orientation = jnp.pi/2  + random.uniform(rob_orient_key, (), minval=-1.) * jnp.pi/4
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., robot_orientation, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('slalom'))
        robot_goal = robot_goal + random.uniform(rob_goal_key, (2,),minval=-1.) * jnp.array([0.5, 0.5])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('slalom'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))
    
    @partial(jit, static_argnames=("self"))
    def _generate_random_obstacle_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, hum_goal_key, obs_key, random_obs_key, rob_key, rob_orient_key, rob_goal_key = random.split(key, 7)
        points = _gen_human(random.split(hum_key, self.n_humans))
        points = points.at[-1].set(random.uniform(hum_key, (2,),minval=-1.) * jnp.array([4.,2.]))
        humans_goal = points
        humans_goal = humans_goal.at[-1].set(random.uniform(hum_goal_key, (2,),minval=-1.) * jnp.array([4.,2.]))
        full_state = full_state.at[:-1,:2].set(points)
        # Robot
        robot_position = jnp.array([0., -5.]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([4, 0.1])
        robot_orientation = jnp.pi/2  + random.uniform(rob_orient_key, (), minval=-1.) * jnp.pi/4
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., robot_orientation, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('random_obstacle'))
        robot_goal = robot_goal + random.uniform(rob_goal_key, (2,),minval=-1.) * jnp.array([4, 0.1])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('random_obstacle'))
        if self.n_obstacles > 0:
            key1, key2, key3 = random.split(random_obs_key, 3)
            center = random.uniform(key1, (2,), minval=-1) * jnp.array([4, 2])
            orientation = random.uniform(key2, (), minval=-1.) * jnp.pi
            length = random.uniform(key3, (), minval=1.5, maxval=1.5)
            displacement = jnp.array([jnp.cos(orientation), jnp.sin(orientation)]) * length / 2
            vertex1 = center + displacement
            vertex2 = center - displacement
            # Spawn random obstacle in place of last one
            static_obstacles = static_obstacles.at[-1].set(jnp.array(
                [[vertex1,vertex2]]
            ))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))
    
    @partial(jit, static_argnames=("self"))
    def _generate_narrow_corridor_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, hum_goal_key, obs_key, rob_key, rob_side_key, rob_orient_key, rob_goal_key = random.split(key, 7)
        points = _gen_human(random.split(hum_key, self.n_humans))
        points = points.at[-1].set(jnp.array([0.,4.]) + random.uniform(hum_key, (2,),minval=-1.) * jnp.array([0.05, 0.5]))
        humans_goal = points
        humans_goal = humans_goal.at[-1].set(jnp.array([0.,-4.]) + random.uniform(hum_key, (2,),minval=-1.) * jnp.array([0.05, 0.5]))
        full_state = full_state.at[:-1,:2].set(points)
        # Robot
        side = random.bernoulli(rob_side_key) * 2 - 1
        robot_position = jnp.array([0., side * 5]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([0.05, 0.5])
        robot_orientation = jnp.pi/2  + random.uniform(rob_orient_key, (), minval=-1.) * jnp.pi/4
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., robot_orientation, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('narrow_corridor'))
        robot_goal = robot_goal + random.uniform(rob_goal_key, (2,),minval=-1.) * jnp.array([0.05, 0.05])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('narrow_corridor'))
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _generate_random_room_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, obs_key, rob_key, rob_orient_key, room_key = random.split(key, 5)
        points = _gen_human(random.split(hum_key, self.n_humans))
        humans_goal = points
        full_state = full_state.at[:-1,:2].set(points)
        # Room
        room_side = random.uniform(room_key) * 4 + 5
        # Robot
        robot_position = jnp.array([0., -room_side/2 + 1.5]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([0.1, 0.1])
        robot_orientation = jnp.pi/2  + random.uniform(rob_orient_key, (), minval=-1.) * jnp.pi/2
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., robot_orientation, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('random_room')) * room_side / 2 + jnp.array([0.,-1.])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('random_room')) * room_side
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _generate_t_corridor_episode(self, key:random.PRNGKey) -> tuple[jnp.ndarray, dict]:
        full_state = jnp.zeros((self.n_humans+1, 6))
        humans_parameters = self.get_standard_humans_parameters(self.n_humans)
        # Humans (Randomly generate the humans' positions far away)
        @vmap
        def _gen_human(key):
            new_angle = random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
            disturbance = random.uniform(key, shape=(1,), minval=-0.1, maxval=0.5)
            new_point = jnp.squeeze((1_000 + disturbance) * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)]))
            return new_point
        hum_key, obs_key, rob_key, rob_orient_key, width_key, height_key, goal_key = random.split(key, 7)
        points = _gen_human(random.split(hum_key, self.n_humans))
        humans_goal = points
        full_state = full_state.at[:-1,:2].set(points)
        # Room
        width_scale = random.uniform(width_key, minval=1., maxval=4.)
        height_scale = random.uniform(height_key, minval=1., maxval=3.)
        # Incoming human 
        full_state = full_state.at[0,:2].set(jnp.array([width_scale,0.]) + random.uniform(hum_key, (2,),minval=-1.) * jnp.array([0.1, 0.1]))
        full_state = full_state.at[0,4].set(jnp.pi)
        humans_goal = humans_goal.at[0].set(jnp.array([-width_scale,0.]))
        # Robot
        robot_position = jnp.array([-width_scale * 1.5, 0.]) + random.uniform(rob_key, (2,),minval=-1.) * jnp.array([0.1, 0.1])
        robot_orientation = 0  + random.uniform(rob_orient_key, (), minval=-1.) * jnp.pi/2
        full_state = full_state.at[-1].set(jnp.array([*robot_position, 0., 0., robot_orientation, 0.]))
        # Robot goal
        robot_goal = self._init_robot_goal(SCENARIOS.index('t_corridor')) * height_scale * 1.5 + random.uniform(goal_key, (2,),minval=-1.) * jnp.array([0.15, 0.15])
        # Obstacles
        static_obstacles = self._init_obstacles(obs_key, SCENARIOS.index('t_corridor'))
        static_obstacles = static_obstacles.at[:,:,:,0].set(static_obstacles[:,:,:,0] * width_scale)
        static_obstacles = static_obstacles.at[:,:,:,1].set(static_obstacles[:,:,:,1] * height_scale)
        return full_state, humans_goal, robot_goal, humans_parameters, static_obstacles, jnp.zeros((self.n_humans,))

    @partial(jit, static_argnames=("self"))
    def _human_ray_intersect(self, direction:jnp.ndarray, human_position:jnp.ndarray, lidar_position:jnp.ndarray, human_radius:float) -> float:
        s = lidar_position - human_position
        b = jnp.dot(s, direction)
        c = jnp.dot(s, s) - human_radius**2
        h = b * b - c  
        sqrt_h = jnp.sqrt(jnp.maximum(h, 0.0))
        t = -b - sqrt_h
        valid_intersection = (h >= 0.0) & (t > 0.0)
        return jnp.where(valid_intersection, t, self.lidar_max_dist)    
    
    @partial(jit, static_argnames=("self"))
    def _batch_human_ray_intersect(self, direction:jnp.ndarray, human_positions:jnp.ndarray, lidar_position:jnp.ndarray, human_radiuses:float) -> jnp.ndarray:
        humans_distances = vmap(BaseEnv._human_ray_intersect, in_axes=(None,None,0,None,0))(self, direction, human_positions, lidar_position, human_radiuses)
        shortest_distance_index = jnp.nanargmin(humans_distances)
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
        shortest_distance_index = jnp.nanargmin(distances)
        return distances[shortest_distance_index], shortest_distance_index

    @partial(jit, static_argnames=("self"))
    def _batch_obstacle_ray_intersect(self, direction:jnp.ndarray, obstacles:jnp.ndarray, lidar_position:jnp.ndarray) -> float:
        distances, collision_idxs = vmap(BaseEnv._obstacle_ray_intersect, in_axes=(None,None,0,None))(self, direction, obstacles, lidar_position)
        shortest_distance_index = jnp.nanargmin(distances)
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
        def _update_delayed_episodes(val:tuple):
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
            def _update_human_state_and_goal_leg_dynamics(position:jnp.ndarray, goal:jnp.ndarray, radius:float, legs_state:jnp.ndarray, positions:jnp.ndarray, radiuses:jnp.ndarray, safety_spaces:jnp.ndarray, is_x_flipped:bool) -> tuple:
                flip_x = lax.cond(is_x_flipped,lambda _: -1.,lambda _: 1.,None)
                new_position, new_goal = lax.cond(
                    jnp.linalg.norm(position - goal) <= 3, 
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
                transition = new_position - position
                new_legs_state = jnp.array([
                    legs_state[0] + transition[0], 
                    legs_state[1] + transition[1], 
                    legs_state[2], 
                    legs_state[3] + transition[0], 
                    legs_state[4] + transition[1], 
                    legs_state[5],
                ])
                return new_position, new_goal, new_legs_state
            @jit
            def _update_human_state_and_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, positions:jnp.ndarray, radiuses:jnp.ndarray, safety_spaces:jnp.ndarray, is_x_flipped:bool) -> tuple:
                flip_x = lax.cond(is_x_flipped,lambda _: -1.,lambda _: 1.,None)
                position, goal = lax.cond(
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
            if self.leg_dynamics:
                new_positions, new_goals, info["humans_leg_state"] = vmap(_update_human_state_and_goal_leg_dynamics, in_axes=(0,0,0,0,None,None,None, None))(
                    state[:-1,0:2], 
                    info["humans_goal"], 
                    info["humans_parameters"][:,0], 
                    info["humans_leg_state"],
                    state[:,0:2], 
                    info["humans_parameters"][:,0], 
                    info["humans_parameters"][:,-1],
                    info['is_x_flipped'],
                )
            else:
                new_positions, new_goals = vmap(_update_human_state_and_goal, in_axes=(0,0,0,None,None,None, None))(
                    state[:-1,0:2], 
                    info["humans_goal"], 
                    info["humans_parameters"][:,0], 
                    state[:,0:2], 
                    info["humans_parameters"][:,0], 
                    info["humans_parameters"][:,-1],
                    info['is_x_flipped'],
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

        @jit
        def _update_door_crossing(val:tuple):
            @jit
            def _update_human_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float) -> jnp.ndarray:
                @jit
                def _set_new_goal(position, goal):
                    key = random.PRNGKey(jnp.array(jnp.linalg.norm(position)*1000, int))
                    key1, key2, key3 = random.split(key, 3)
                    door_goal = random.bernoulli(key1, p=0.1)
                    normalized_goal_point = random.uniform(key2, shape=(2,), minval=-1, maxval=1)
                    new_room = random.bernoulli(key3)
                    case = jnp.argmax(jnp.array([
                        (goal[0] < 0) & ~(door_goal), # Left room
                        goal[0] == 0 & ~(door_goal), # Door goal
                        goal[0] > 0 & ~(door_goal), # Right room
                        door_goal, # Set new goal as door
                    ]))
                    new_goal = lax.switch(
                        case,
                        [
                            lambda: jnp.array([
                                -2.5 + normalized_goal_point[0] * 2.4, 
                                normalized_goal_point[1] * 2.4,
                            ]), # Remain in left room
                            lambda: jnp.array([
                                -2.5 + new_room * 5 + normalized_goal_point[0] * 2.4, 
                                normalized_goal_point[1] * 2.4,
                            ]), # Pick one room randomly
                            lambda: jnp.array([
                                2.5 + normalized_goal_point[0] * 2.4, 
                                normalized_goal_point[1] * 2.4,
                            ]), # Remain in right room
                            lambda: jnp.array([
                                0.,
                                0.
                            ]), # Set door goal
                        ]
                    )
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
        def _update_crowd_chasing(val:tuple):
            @jit
            def _update_human_state_and_goal(position:jnp.ndarray, goal:jnp.ndarray, radius:float, positions:jnp.ndarray, radiuses:jnp.ndarray, safety_spaces:jnp.ndarray, is_x_flipped:bool) -> tuple:
                flip_x = lax.cond(is_x_flipped,lambda _: -1.,lambda _: 1.,None)
                position, goal = lax.cond(
                    # jnp.linalg.norm(position - goal) <= radius + 2,
                    jnp.linalg.norm(position - goal) <= 3,
                    lambda _: (
                        jnp.array([
                        flip_x * jnp.min(jnp.append(positions[:,0] + (jnp.max(jnp.append(radiuses, self.robot_radius))*2)+(jnp.max(safety_spaces)*2), -self.traffic_length/2)),
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

        if self.scenario != -1:  # If not custom scenario
            new_info, new_state = lax.switch(
                info["current_scenario"], 
                [
                    _update_circular_crossing, 
                    _update_traffic_scenarios, 
                    _update_traffic_scenarios, 
                    lambda x: x,
                    _update_delayed_episodes,
                    _update_circular_crossing_with_static_obstacles,
                    _update_crowd_navigation,
                    _update_corner_traffic,
                    _update_door_crossing,
                    _update_crowd_chasing,
                    lambda x: x,
                    _update_delayed_episodes,
                    lambda x: x,
                    lambda x: x,
                    lambda x: x,
                    lambda x: x,
                    lambda x: x,
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
        action:jnp.ndarray,
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
            if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.linalg.norm(state[-1,2:4]), 0., jnp.atan2(*jnp.flip(state[-1,2:4])), 0.])]) # HSFM fictitious state
            elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], state[-1,2], 0., state[-1,4], state[-1,3]])]) # HSFM fictitious state
            new_state = jnp.vstack(
                [self.humans_step(fictitious_state, info["visibility"], goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                state[-1]])
        elif self.humans_policy == HUMAN_POLICIES.index("sfm") or self.humans_policy == HUMAN_POLICIES.index("orca"):
            if self.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], *state[-1,2:4], 0., 0.])]) # SFM or ORCA fictitious state
            elif self.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                fictitious_state = jnp.vstack([state[0:self.n_humans], jnp.array([*state[-1,0:2], jnp.cos(state[-1,4]) * state[-1,2], jnp.sin(state[-1,4]) * state[-1,2], state[-1,4], 0.])]) # SFM or ORCA fictitious state
            new_state = jnp.vstack(
                [self.humans_step(fictitious_state[:,0:4], info["visibility"], goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                state[-1,0:4]])
            new_state = jnp.pad(new_state, ((0,0),(0,2)))
            new_state = new_state.at[-1,4:].set(state[-1,4:])
        ## Robot update
        # Compute delayed action
        robot_velocity = lax.cond(
            info["robot_delay"] == 0,
            lambda: action,
            lambda: info["action_history"][(info["robot_delay"] // self.robot_dt).astype(jnp.int32)],
        )
        # Apply velocity dynamics
        if self.robot_velocity_dynamics == ROBOT_VELOCITY_DYNAMICS.index("first_order_system"):
            if self.action_0_dynamics:
                alpha = jnp.exp(-self.humans_dt/self.tau_action_0)
                robot_velocity = robot_velocity.at[0].set(alpha * state[-1,2] + (1 - alpha) * robot_velocity[0])
            if self.action_1_dynamics:
                alpha = jnp.exp(-self.humans_dt/self.tau_action_1)
                robot_velocity = robot_velocity.at[1].set(alpha * state[-1,3] + (1 - alpha) * robot_velocity[1])
        elif self.robot_velocity_dynamics == ROBOT_VELOCITY_DYNAMICS.index("coupled_slew_rate"):
            if self.limited_acceleration:
                a_req = (robot_velocity[0] - state[-1,2]) / self.humans_dt
                alpha_req = (robot_velocity[1] - state[-1,3]) / self.humans_dt
                effort = abs(a_req) + (self.wheels_distance / 2.0) * abs(alpha_req)
                scale = lax.cond(
                    (effort > self.wheels_max_linear_acceleration) & (effort > 1e-6),
                    lambda: self.wheels_max_linear_acceleration / effort, 
                    lambda: 1.0,
                )
                robot_velocity = robot_velocity.at[0].set(state[-1,2] + (a_req * scale) * self.humans_dt)
                robot_velocity = robot_velocity.at[1].set(state[-1,3] + (alpha_req * scale) * self.humans_dt)
        # Apply position dynamics
        if self.kinematics == ROBOT_KINEMATICS.index("holonomic"):
            new_state = new_state.at[-1,0:4].set(jnp.array([
                state[-1,0]+state[-1,2]*self.humans_dt, 
                state[-1,1]+state[-1,3]*self.humans_dt,
                *robot_velocity,
            ]))
        elif self.kinematics == ROBOT_KINEMATICS.index("unicycle"):
            new_state = lax.cond(
                jnp.abs(state[-1,3]) > EPSILON,
                lambda x: x.at[-1].set(jnp.array([
                    state[-1,0]+(state[-1,2]/state[-1,3])*(jnp.sin(state[-1,4]+state[-1,3]*self.humans_dt)-jnp.sin(state[-1,4])),
                    state[-1,1]+(state[-1,2]/state[-1,3])*(jnp.cos(state[-1,4])-jnp.cos(state[-1,4]+state[-1,3]*self.humans_dt)),
                    *robot_velocity,
                    wrap_angle(state[-1,4]+state[-1,3]*self.humans_dt),
                    state[-1,5]
                ])),
                lambda x: x.at[-1].set(jnp.array([
                    state[-1,0]+state[-1,2]*self.humans_dt*jnp.cos(state[-1,4]),
                    state[-1,1]+state[-1,2]*self.humans_dt*jnp.sin(state[-1,4]),
                    *robot_velocity,
                    *state[-1,4:]
                ])),
                new_state)
        ## Legs dynamics
        if self.leg_dynamics:
            info["humans_leg_state"] = vmap(update_single_human_leg, in_axes=(0, 0, 0, None, 0, None))(
                new_state[0:self.n_humans], 
                info["humans_leg_state"], 
                info["humans_parameters"][:, 0],
                self.humans_dt,
                info["humans_leg_parameters"],
                self.humans_policy,
            )
        ## Post update stuff
        new_info, new_state = self._scenario_based_state_post_update(new_state, info)
        new_info["robot_delay"] = jnp.max(jnp.array([0., info["robot_delay"] - self.humans_dt]))
        return (new_state, new_info)

    @partial(jit, static_argnames=("self"))
    def _step(
        self,
        state:jnp.ndarray,
        info:dict,
        action:jnp.ndarray,
    ):
        def scan_step(carry, _):
            curr_state, curr_info = carry
            new_state, new_info = self._update_state_info(curr_state, curr_info, action)
            return (new_state, new_info), (new_state, new_info["humans_leg_state"])
        (new_state, new_info), (state_history, humans_leg_state_history) = lax.scan(
            f=scan_step,
            init=(state, info),
            xs=None,
            length=int(self.robot_dt/self.humans_dt)
        )
        return new_state, new_info, (state_history, humans_leg_state_history)

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
        new_state = self.humans_step(state, info["visibility"], goals, parameters, static_obstacles, self.humans_dt)
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
        human_legs_positions:jnp.ndarray,
        human_radii:jnp.ndarray,
        human_legs_radii:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        noise_key=random.PRNGKey(0)
    ) -> jnp.ndarray:
        """
        Given the current state of the environment, the robot orientation and the additional information about the environment,
        this function computes the lidar measurements of the robot. The lidar measurements are given as a set of distances and angles (in the global frame) for each ray.
        If LEG_DYNAMICS = False: the LiDAR rays will collide with the humans, which are modeled as circles with radius given by human_radii and positions given by human_positions.
        If LEG_DYNAMICS = True: the LiDAR rays will collide with the legs of the humans, which are modeled as circles with radius given by human_legs_radii and positions given by human_legs_positions.
        NOTICE: in the current implementation, to compute LiDAR measurements with legs, we feed humans'legs positions and radii in the downstream functions as if they were humans.

        args:
        - lidar_position (2,): jnp.ndarray containing the x and y coordinates of the lidar.
        - lidar_yaw (1,): float containing the orientation of the lidar.
        - human_positions (self.n_humans,2): jnp.ndarray containing the x and y coordinates of the humans.
        - human_legs_positions (self.n_humans,4): jnp.ndarray containing the x and y coordinates of the humans legs (only used if leg_dynamics is True).
        - human_radii (self.n_humans,): jnp.ndarray containing the radius of the humans.
        - human_legs_radii (self.n_humans,): jnp.ndarray containing the radius of the humans' legs (only used if leg_dynamics is True).
        - static_obstacles (self.n_obstacles, m, 2, 2): jnp.ndarray containing the static obstacles as line segments (m is the number of segments per obstacle).

        output:
        - lidar_output (self.lidar_num_rays,2): jnp.ndarray containing the lidar measurements of the robot and the angle (IN THE GLOBAL FRAME) for each ray.
          WARNING: the angles are in the global frame, not in the robot frame.
        - human_visibility_mask (self.n_humans,): boolean jnp.ndarray indicating which humans are visible by the LiDAR (i.e. at least one ray collides with them).
        - obstacles_visibility_mask (self.n_obstacles, m): boolean jnp.ndarray indicating which static obstacle segments are visible by the LiDAR (i.e. at least one ray collides with them).
        """
        angles = jnp.linspace(lidar_yaw - self.lidar_angular_range/2, lidar_yaw + self.lidar_angular_range/2, self.lidar_num_rays)
        if self.leg_dynamics:
            # To compute the LiDAR measurements with leg dynamics, we treat the legs as separate entities that can occlude the rays. Therefore, we need to reshape the human legs positions and radii to be fed into the ray casting function as if they were humans.
            human_positions = jnp.reshape(human_legs_positions, (self.n_humans*2, 2))
            human_radii = jnp.repeat(human_legs_radii, 2)
        measurements, human_collision_idxs, obstacle_collision_idxs = self.batch_ray_cast(angles, lidar_position, human_positions, human_radii, static_obstacles)
        if self.leg_dynamics:
            # With leg dynamics, we consider a human visible if at least one of the legs is collided by a ray.
            humans_visibility_mask = vmap(lambda idx: (jnp.any(human_collision_idxs == 2 * idx) | jnp.any(human_collision_idxs == 2 * idx + 1)))(jnp.arange(self.n_humans))  # Shape: (n_humans,)
        else:
            humans_visibility_mask = vmap(lambda idx: jnp.any(human_collision_idxs == idx))(jnp.arange(self.n_humans))  # Shape: (n_humans,)
        @jit
        def segment_visibility(obstacle_idx, segment_idx, obstacle_collision_idxs):
            return jnp.any(jnp.all(obstacle_collision_idxs == jnp.array([obstacle_idx, segment_idx]), axis=1))
        @jit
        def obstacle_segments_visibility(obstacle_idx, segment_idxs, obstacle_collision_idxs):
            return vmap(segment_visibility, in_axes=(None, 0, None))(obstacle_idx, segment_idxs, obstacle_collision_idxs)
        obstacles_visibility_mask = vmap(obstacle_segments_visibility, in_axes=(0, None, None))(
            jnp.arange(self.n_obstacles), 
            jnp.arange(self.n_segments), 
            obstacle_collision_idxs
        ) # Shape: (n_obstacles, n_segments)
        if self.lidar_noise:
            measurements = self.add_lidar_noise(measurements,noise_key)
        lidar_output = jnp.stack((measurements, angles), axis=-1)
        return lidar_output, humans_visibility_mask, obstacles_visibility_mask
    
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
        obstacle_segments = rc_static_obstacles.reshape(((self.n_obstacles+5*self.noisy_walls)*self.n_segments, 2, 2))  # Shape: (n_obstacles*n_segments, 2, 2)
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
        sorted_all_versors = jnp.array([jnp.cos(sorted_all_edge_angles), jnp.sin(sorted_all_edge_angles)]).T  # Shape: (2*n_humans + 2*n_obstacles*n_segments + 1, 2)
        midpoint_versors = (sorted_all_versors[:-1] + sorted_all_versors[1:])  # Shape: (2*n_humans + 2*n_obstacles*n_segments, 2)
        midpoint_versors = midpoint_versors / jnp.linalg.norm(midpoint_versors, axis=1, keepdims=True)  # Normalize
        midpoint_angles = jnp.arctan2(midpoint_versors[:,1], midpoint_versors[:,0])  # Shape: (2*n_humans + 2*n_obstacles*n_segments,)
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
            jnp.arange(self.n_obstacles+5*self.noisy_walls), 
            jnp.arange(self.n_segments), 
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
        # Two conditions: 
        #   distance to origin minus radius less than lidar max distance
        #   angle within lidar angular range (assumed to be centered at 0, i.e., robot facing right of the robot along x-axis)
        return (jnp.linalg.norm(positions, axis=-1) - radii <= self.lidar_max_dist) & (jnp.abs(jnp.arctan2(positions[:,1], positions[:,0])) <= self.lidar_angular_range / 2)
    
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