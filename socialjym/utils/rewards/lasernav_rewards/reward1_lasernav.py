from jax import jit, lax, vmap
import jax.numpy as jnp
from functools import partial

from socialjym.utils.rewards.base_reward import BaseReward
from socialjym.envs.base_env import ROBOT_KINEMATICS, HUMAN_POLICIES
from socialjym.utils.terminations.robot_human_collision import InstantRobotHumanCollision, IntervalRobotHumanCollision
from socialjym.utils.terminations.robot_obstacle_collision import InstantRobotObstacleCollision
from socialjym.utils.terminations.robot_reached_goal import RobotReachedGoal
from socialjym.utils.terminations.timeout import Timeout
from jhsfm.hsfm import get_linear_velocity

class Reward1LaserNav(BaseReward):
    def __init__(
        self, 
        robot_radius: float,
        gamma: float = 0.99,
        v_max: float = 1.0,
        goal_reward: float = 1.0,
        collision_penalty: float = -0.25,
        discomfort_distance: float = 0.2,
        time_limit: float = 50.,
    ) -> None:
        super().__init__(gamma)
        
        # Check input parameters
        assert v_max > 0, "v_max must be positive"
        assert goal_reward > 0, "goal_reward must be positive"
        assert collision_penalty < 0, "collision_penalty must be negative"
        assert discomfort_distance > 0, "discomfort_distance must be positive"
        assert time_limit > 0, "time_limit must be positive"

        # Initialize reward parameters
        self.type = "lasernav_reward1"
        self.v_max = v_max
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.discomfort_distance = discomfort_distance
        self.time_limit = time_limit
        self.kinematics = ROBOT_KINEMATICS.index('unicycle')
        self.robot_radius = robot_radius
        self.humans_policy = HUMAN_POLICIES.index('hsfm')
        
        # Define terminations
        self.interval_human_collision_termination = IntervalRobotHumanCollision()
        self.instant_human_collision_termination = InstantRobotHumanCollision()
        self.instant_obstacle_collision_termination = InstantRobotObstacleCollision()
        self.goal_reached_termination = RobotReachedGoal()
        self.timeout = Timeout(time_limit)

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray,
        info: dict, 
        dt: float
    ) -> tuple[float, dict]:
        """
        Calcola il reward per l'ambiente LaserNav.
        Reward denso basato su:
        - Raggiungimento goal (+)
        - Collisione con umani/ostacoli (-)
        - Discomfort distance da umani (-)
        - Time penalty implicita (opzionale, qui non inclusa ma aggiungibile)

        args:
        - state: current state of the environment (n_humans+1, 6)
        - action: (v, w) for unicycle
        - info: dictionary containing additional information about the environment
        - dt: time step of the simulation
        """
        
        # Estrai stato corrente
        robot_pos = state[-1, :2]
        robot_yaw = state[-1, 4]
        humans_pos = state[:-1, :2]
        robot_goal = info["robot_goal"]
        humans_radiuses = info["humans_parameters"][:, 0]
        robot_radius = self.robot_radius
        time = info["time"]

        # 1. Calcola posizione futura del Robot (Modello Unicycle)
        next_robot_pos = lax.cond(
            action[1] != 0,
            lambda x: x.at[:].set(jnp.array([
                x[0] + (action[0]/action[1]) * (jnp.sin(robot_yaw + action[1] * dt) - jnp.sin(robot_yaw)),
                x[1] + (action[0]/action[1]) * (jnp.cos(robot_yaw) - jnp.cos(robot_yaw + action[1] * dt))
            ])),
            lambda x: x.at[:].set(jnp.array([
                x[0] + action[0] * dt * jnp.cos(robot_yaw),
                x[1] + action[0] * dt * jnp.sin(robot_yaw)
            ])),
            robot_pos
        )

        # 2. Calcola posizione futura degli Umani
        if self.humans_policy == HUMAN_POLICIES.index('hsfm'):
            humans_orientations = state[:-1, 4]
            humans_velocities = vmap(get_linear_velocity)(humans_orientations, state[:-1, 2:4])
            next_humans_pos = humans_pos + humans_velocities * dt
        else:
            next_humans_pos = humans_pos + state[:-1, 2:4] * dt

        # 3. Rilevamento Collisioni
        
        # A. Collisione con Umani (Intervallo)
        collision_with_human, human_collision_info = self.interval_human_collision_termination(
            robot_pos, 
            next_robot_pos,
            robot_radius,
            humans_pos,
            next_humans_pos,
            humans_radiuses
        )

        # B. Collisione con Ostacoli Statici (Istantanea)
        collision_with_obstacle, _ = self.instant_obstacle_collision_termination(
            next_robot_pos,
            robot_radius,
            info['static_obstacles'][-1],
        )

        # Unione delle failure
        failure = collision_with_human | collision_with_obstacle

        # 4. Rilevamento Discomfort (Solo Umani)
        # Se non c'è collisione, controlliamo se siamo troppo vicini
        min_human_distance = human_collision_info['min_distance']
        discomfort = jnp.all(jnp.array([
            ~failure, # Non conta se c'è già stata collisione
            min_human_distance < self.discomfort_distance
        ]))

        # 5. Raggiungimento Goal
        reached_goal, _ = self.goal_reached_termination(
            next_robot_pos,
            robot_radius,
            robot_goal,
        )

        # 6. Timeout
        timeout, _ = self.timeout(time)

        # 7. Calcolo Reward
        reward = 0.0

        # + Reward per Goal
        reward = lax.cond(
            (~failure) & reached_goal, 
            lambda r: r + self.goal_reward, 
            lambda r: r, 
            reward
        )
        
        # - Penalty per Collisione (Umano o Ostacolo)
        reward = lax.cond(
            failure, 
            lambda r: r + self.collision_penalty, 
            lambda r: r, 
            reward
        )
        
        # - Penalty per Discomfort (Proporzionale alla vicinanza)
        # Formula: -0.5 * dt * (soglia - distanza_reale)
        reward = lax.cond(
            discomfort, 
            lambda r: r - 0.5 * dt * (self.discomfort_distance - min_human_distance), 
            lambda r: r, 
            reward
        )
        
        # + Reward di avvicinamento al goal (Shaping opzionale ma consigliato per training veloce)
        # dist_curr = jnp.linalg.norm(robot_pos - robot_goal)
        # dist_next = jnp.linalg.norm(next_robot_pos - robot_goal)
        # reward += 2.0 * (dist_curr - dist_next) # Premia se ti avvicini

        # 8. Definizione Outcome
        outcome = {
            "nothing": ~((failure) | (reached_goal) | (timeout)),
            "success": (~(failure)) & (reached_goal),
            "collision_with_human": collision_with_human,
            "collision_with_obstacle": collision_with_obstacle,
            "timeout": timeout & (~(failure)) & (~(reached_goal))
        }

        return reward, outcome