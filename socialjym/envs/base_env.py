from abc import ABC, abstractmethod
from functools import partial
from jax import jit, vmap, lax, debug
import jax.numpy as jnp

from jhsfm.hsfm import step as hsfm_humans_step
from jsfm.sfm import step as sfm_humans_step
from jhsfm.utils import get_standard_humans_parameters as hsfm_get_standard_humans_parameters
from jsfm.utils import get_standard_humans_parameters as sfm_get_standard_humans_parameters

SCENARIOS = [
    "circular_crossing", 
    "parallel_traffic", 
    "perpendicular_traffic", 
    "robot_crowding", 
    "hybrid_scenario"] # Make sure to update this list (if new scenarios are added) but always leave the last element as "hybrid_scenario"
HUMAN_POLICIES = [
    "orca", # TODO: Implement JORCA (Jax based ORCA)
    "sfm", 
    "hsfm"]
ROBOT_KINEMATICS = [
    "holonomic",
    "unicycle"]

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
        kinematics:str
    ) -> None:
        ## Args validation
        assert scenario in SCENARIOS, f"Invalid scenario. Choose one of {SCENARIOS}"
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert kinematics in ROBOT_KINEMATICS, f"Invalid robot kinematics. Choose one of {ROBOT_KINEMATICS}"
        ## Env initialization
        self.robot_radius = robot_radius
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
            @jit
            def _true_cond_bodi(i:int, info:dict, state:jnp.ndarray):
                state = state.at[i,0:4].set(jnp.array([
                        # jnp.max(jnp.append(state[:,0]+(jnp.max(jnp.append(info["humans_parameters"][:,0],self.robot_radius))*2)+(jnp.max(info["humans_parameters"][:,-1])*2)+0.05, self.traffic_length/2+1)), 
                        jnp.max(jnp.append(state[:,0]+(jnp.max(jnp.append(info["humans_parameters"][:,0],self.robot_radius))*2)+(jnp.max(info["humans_parameters"][:,-1])*2), self.traffic_length/2)), # Compliant with Social-Navigation-PyEnvs
                        jnp.clip(state[i,1], -self.traffic_height/2, self.traffic_height/2),
                        *state[i,2:4]]))
                info["humans_goal"] = info["humans_goal"].at[i].set(jnp.array([info["humans_goal"][i,0], state[i,1]]))
                return (info, state)

            info, state = val
            out = lax.fori_loop(
                0, 
                self.n_humans, 
                lambda i, val: lax.cond(
                    # jnp.linalg.norm(val[1][i,0:2] - val[0]["humans_goal"][i]) <= val[0]["humans_parameters"][i,0] + 2, 
                    jnp.linalg.norm(val[1][i,0:2] - val[0]["humans_goal"][i]) <= 3, # Compliant with Social-Navigation-PyEnvs
                    lambda x: _true_cond_bodi(i, x[0], x[1]),
                    lambda x: x, 
                    val),
                (info, state))
            info, state = out
            return (info, state)

        info_and_state = lax.switch(
            info["current_scenario"], 
            [_update_circular_crossing, 
            _update_traffic_scenarios, 
            _update_traffic_scenarios, 
            lambda x: x], 
            (info, state))
        new_info, new_state = info_and_state
            
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
        parameters = jnp.vstack((info["humans_parameters"], jnp.array([self.robot_radius, 80., *self.get_standard_humans_parameters(1)[0,2:]])))
        static_obstacles = info["static_obstacles"]
        # Humans update
        if self.humans_policy == HUMAN_POLICIES.index("hsfm"):
            if self.robot_visible:
                new_state = jnp.vstack(
                    [self.humans_step(state, goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                    state[-1]])
            else:
                new_state = jnp.vstack(
                    [self.humans_step(state[0:self.n_humans], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles, self.humans_dt), 
                    state[-1]])
        elif self.humans_policy == HUMAN_POLICIES.index("sfm") or self.humans_policy == HUMAN_POLICIES.index("orca"):
            if self.robot_visible:
                new_state = jnp.vstack(
                    [self.humans_step(state[:,0:4], goals, parameters, static_obstacles, self.humans_dt)[0:self.n_humans], 
                    state[-1,0:4]])
            else:
                new_state = jnp.vstack(
                    [self.humans_step(state[0:self.n_humans,0:4], goals[0:self.n_humans], parameters[0:self.n_humans], static_obstacles, self.humans_dt), 
                    state[-1,0:4]])
            new_state = jnp.pad(new_state, ((0,0),(0,2)))
        # Robot update
        if self.kinematics == ROBOT_KINEMATICS.index("holonomic"):
            new_state = new_state.at[-1,0:2].set(jnp.array([
                state[-1,0]+action[0]*self.humans_dt, 
                state[-1,1]+action[1]*self.humans_dt]))
        elif self.kinematics == ROBOT_KINEMATICS.index("unicycle"):
            new_state = lax.cond(
                action[1] != 0,
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
        parameters = jnp.vstack((info["humans_parameters"], jnp.array([self.robot_radius, 80., *self.get_standard_humans_parameters(1)[0,2:-1], 0.1]))) # Add safety space of 0.1 to robot
        static_obstacles = info["static_obstacles"]
        if self.robot_visible:
            new_state = self.humans_step(state, goals, parameters, static_obstacles, self.humans_dt)
        else:
            new_state = jnp.vstack([
                self.humans_step(state[0:-1], goals[0:-1], parameters[0:-1], static_obstacles, self.humans_dt), 
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