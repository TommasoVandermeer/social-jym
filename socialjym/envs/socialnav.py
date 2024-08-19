import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from functools import partial
from jhsfm.hsfm import step
from jhsfm.utils import get_standard_humans_parameters
from socialjym.utils.aux_functions import is_multiple

from .base_env import BaseEnv

SCENARIOS = ["circular_crossing", "parallel_traffic"]
HUMAN_POLICIES = ["orca", "sfm", "hsfm"]

class SocialNav(BaseEnv):
    """
    A simple OpenAI gym-like environment based on JAX to train mobile robots for social navigation tasks 
    through RL.
    """
    def __init__(self, robot_radius:float, robot_dt:float, humans_dt:float, scenario:str, n_humans:int, humans_policy='hsfm', 
                 robot_visible=False, circle_radius=7, traffic_height=3, traffic_length=14) -> None:
        # Args validation
        assert scenario in SCENARIOS, f"Invalid scenario. Choose one of {SCENARIOS}"
        assert humans_policy in HUMAN_POLICIES, f"Invalid human policy. Choose one of {HUMAN_POLICIES}"
        assert is_multiple(robot_dt, humans_dt), "The robot's time step must be a multiple of the humans' time step."
        # Env initialization
        self.robot_radius = robot_radius
        self.robot_dt = robot_dt
        self.humans_dt = humans_dt
        self.scenario = SCENARIOS.index(scenario)
        self.n_humans = n_humans
        self.humans_policy = SCENARIOS.index(scenario)
        self.robot_visible = robot_visible
        self.circle_radius = circle_radius
        self.traffic_height = traffic_height
        self.traffic_length = traffic_length

    # --- Private methods ---

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _get_obs(self, state):
        return state

    @partial(jit, static_argnames=("self"))
    def _reset(self, key):
        key, subkey = random.split(key)
        full_state = lax.switch(self.scenario, [self._generate_circular_crossing_episode, 
                                                self._generate_parallel_traffic_episode], subkey)
        return full_state, key

    @partial(jit, static_argnames=("self"))
    def _reset_if_done(self, env_state, done):
        pass
    
    @partial(jit, static_argnames=("self"))
    def _get_reward_done(self, new_state):
        pass
    
    @partial(jit, static_argnames=("self"))
    def _generate_circular_crossing_episode(self, key):
        full_state = jnp.zeros((self.n_humans+1, 6))
        # Humans state, goals and parameters
        humans_goal = jnp.zeros((self.n_humans, 2))
        angle_width = (2 * jnp.pi) / (self.n_humans)
        full_state = lax.fori_loop(0, self.n_humans, lambda i, full_state: full_state.at[i].set(
                                            jnp.array([self.circle_radius * jnp.cos(i * angle_width),
                                            self.circle_radius * jnp.sin(i * angle_width),
                                            0.,
                                            0.,
                                            -jnp.pi + i * angle_width,
                                            0.]))
                     , full_state)
        humans_goal = lax.fori_loop(0, self.n_humans, lambda i, humans_goal:
                      humans_goal.at[i].set(jnp.array([-full_state[i,0], -full_state[i,1]]))
                      , humans_goal)
        self.humans_goal = humans_goal
        self.humans_parameters = get_standard_humans_parameters(self.n_humans)
        # Robot state and goal
        full_state = full_state.at[self.n_humans].set(jnp.array([0., -self.circle_radius, jnp.nan, jnp.nan, jnp.nan, jnp.nan]))
        self.robot_goal = jnp.array([0., self.circle_radius])
        # Obstacles
        self.static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles
        return full_state
    
    @partial(jit, static_argnames=("self"))
    def _generate_parallel_traffic_episode(self, key):
        # TODO: Implement this method
        full_state = jnp.zeros((self.n_humans+1, 6))
        return full_state

    # --- Public methods ---

    @partial(jit, static_argnames=("self"))
    def step(self, state, action) -> tuple:
        pass

    def reset(self, key) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given a PRNG key, this function resets the environment to start a new episode returning a stochastic initial state.

        args:
        - key: PRNG key to initialize a random state

        output:
        - full_state: initial full state of the environment. The state is a JAX array in the form:
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
        """
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)