import jax.numpy as jnp
from jax import lax, jit, random, vmap, debug
from jax_tqdm import loop_tqdm
from tqdm import tqdm
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import pickle as pkl
import os
from datetime import date

from socialjym.envs.base_env import BaseEnv, SCENARIOS, HUMAN_POLICIES, ROBOT_KINEMATICS, ENVIRONMENTS, wrap_angle
from socialjym.policies.base_policy import BasePolicy

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
def batch_roto_translate_poses_and_vels(positions, orientations, velocities, ref_positions, ref_orientations):
    """Roto-translate a batch of 2D poses and velocities to a batch of given reference poses."""
    return vmap(roto_translate_poses_and_vels, in_axes=(0, 0, 0, 0, 0))(positions, orientations, velocities, ref_positions, ref_orientations)

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
def batch_roto_translate_obstacles(obstacles, ref_positions, ref_orientations):
    return vmap(roto_translate_obstacles, in_axes=(0, 0, 0))(obstacles, ref_positions, ref_orientations)

@jit
def linear_decay(start:float, end:float, current_iteration:int, decay_rate:float) -> float:
    value = lax.cond(current_iteration < decay_rate, lambda x: start + (end - start) / decay_rate * x, lambda x: end, current_iteration)
    return value

@jit
def binary_to_decimal(binary:jnp.array) -> int:
    decimal = lax.fori_loop(
        0,
        len(binary),
        lambda i, x: x + (2 ** i) * binary[len(binary) - 1 -i],
        0)
    return decimal

def decimal_to_binary(decimal:int, n_bits:int) -> jnp.array:
    binary = jnp.zeros((n_bits,), dtype=bool)
    for bit in range(n_bits):
        binary = binary.at[n_bits - bit - 1].set(bool(decimal % 2))
        decimal = decimal // 2
    return binary

def plot_state(
        ax:Axes, 
        time:float, 
        full_state:jnp.ndarray, 
        humans_policy:str, 
        scenario:int, 
        humans_radiuses:np.ndarray, 
        robot_radius:float, 
        circle_radius=7, 
        traffic_height=3, 
        traffic_length=14, 
        crowding_square_side=14,
        plot_time=True,
        kinematics:str='holonomic',
        xlims:list=None,
        ylims:list=None,
    ) -> None:
    """
    Plots a given single state of the environment.

    args:
    - ax: matplotlib.axes.Axes. The axes to plot the state.
    - time: float. The time of the state to be plotted.
    - full_state: shape=(n_humans+1, 6), dtype=jnp.float32
        The state of the environment to be plotted. The last row of full_state corresponds to the robot state.
    - humans_policy: str, one of ['orca', 'sfm', 'hsfm']
    - humans_radiuses: np.ndarray, shape=(n_humans,), dtype=np.float32
    - circle_radius: float
    - robot_radius: float

    output:
    - None
    """
    n_humans = len(humans_radiuses)
    colors = list(mcolors.TABLEAU_COLORS.values())
    num = int(time) if (time).is_integer() else (time)
    # Humans
    for h in range(len(full_state)-1): 
        if humans_policy == 'hsfm': 
            head = plt.Circle((full_state[h,0] + np.cos(full_state[h,4]) * humans_radiuses[h], full_state[h,1] + np.sin(full_state[h,4]) * humans_radiuses[h]), 0.1, color=colors[h%len(colors)], zorder=1)
            ax.add_patch(head)
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor=colors[h%len(colors)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        if plot_time: 
            ax.text(full_state[h,0],full_state[h,1], f"{num}", color=colors[h%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
        elif (not plot_time) and (n_humans < 11): 
            ax.text(full_state[h,0],full_state[h,1], f"{h}", color=colors[h%len(colors)], va="center", ha="center", size=10, zorder=1, weight='bold')
        # else: ax.text(full_state[h,0],full_state[h,1], f"{h}", color=colors[h%len(colors)], va="center", ha="center", size=10, zorder=1, weight='bold')
    # Robot
    if kinematics == 'unicycle':
        head = plt.Circle((full_state[-1,0] + np.cos(full_state[-1,4]) * robot_radius, full_state[-1,1] + np.sin(full_state[-1,4]) * robot_radius), 0.1, color='black', zorder=1)
        ax.add_patch(head)
    circle = plt.Circle((full_state[-1,0],full_state[-1,1]), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
    ax.add_patch(circle)
    # Time/Label
    if plot_time: 
        ax.text(full_state[-1,0],full_state[-1,1], f"{num}", color="black", va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=3, weight='bold')
    else: 
        ax.text(full_state[-1,0],full_state[-1,1], f"R", color="black", va="center", ha="center", size=10, zorder=3, weight='bold')
    # Set axis limits and labels
    if scenario == SCENARIOS.index('circular_crossing') or scenario == SCENARIOS.index('delayed_circular_crossing') or scenario == SCENARIOS.index('circular_crossing_with_static_obstacles') or scenario == SCENARIOS.index('crowd_navigation'): 
        ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
    elif scenario == SCENARIOS.index('parallel_traffic'): 
        ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_height-3,traffic_height+3])
    elif scenario == SCENARIOS.index('perpendicular_traffic'): 
        ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_length/2,traffic_length/2])
    elif scenario == SCENARIOS.index('robot_crowding'): 
        ax.set(xlabel='X',ylabel='Y',xlim=[-crowding_square_side/2-1.5,crowding_square_side/2+1.5],ylim=[-crowding_square_side/2-1.5,crowding_square_side/2+1.5])
    elif scenario == SCENARIOS.index('corner_traffic'):
        ax.set(xlabel='X',ylabel='Y',xlim=[-2,traffic_length/2+traffic_height/2+2],ylim=[-2,traffic_length/2+traffic_height/2+2])
    elif scenario is None:
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlabel='X',ylabel='Y',xlim=xlims,ylim=ylims)

def plot_trajectory(ax:Axes, agents_positions:jnp.ndarray, humans_goal:jnp.ndarray, robot_goal:jnp.ndarray):
    colors = list(mcolors.TABLEAU_COLORS.values())
    n_agents = len(agents_positions[0])
    for h in range(n_agents): 
        ax.plot(agents_positions[:,h,0], agents_positions[:,h,1], color=colors[h%len(colors)] if h < n_agents - 1 else "red", linewidth=1, zorder=0)
        if h < n_agents - 1: ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=colors[h%len(colors)], zorder=2)
        else: ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)

def plot_lidar_measurements(ax:Axes, lidar_measurements:jnp.ndarray, robot_state:jnp.ndarray, robot_radius:float):
    for i in range(len(lidar_measurements)):
        ax.plot(
            [robot_state[0], robot_state[0] + lidar_measurements[i,0] * jnp.cos(lidar_measurements[i,1])], 
            [robot_state[1], robot_state[1] + lidar_measurements[i,0] * jnp.sin(lidar_measurements[i,1])], 
            color="black", 
            linewidth=0.5, 
            zorder=0)

def initialize_metrics_dict(n_trials:int, dims:tuple=()) -> dict:
    metrics = {
        "successes": 0 if len(dims) == 0 else jnp.zeros(dims),
        "collisions": 0 if len(dims) == 0 else jnp.zeros(dims),
        "collisions_with_human": 0 if len(dims) == 0 else jnp.zeros(dims),
        "collisions_with_obstacle": 0 if len(dims) == 0 else jnp.zeros(dims),
        "timeouts": 0 if len(dims) == 0 else jnp.zeros(dims),
        "returns": jnp.empty((*dims, n_trials,)),
        "times_to_goal": jnp.empty((*dims, n_trials,)),
        "average_speed": jnp.empty((*dims, n_trials,)),
        "average_acceleration": jnp.empty((*dims, n_trials,)),
        "average_jerk": jnp.empty((*dims, n_trials,)),
        "average_angular_speed": jnp.empty((*dims, n_trials,)),
        "average_angular_acceleration": jnp.empty((*dims, n_trials,)),
        "average_angular_jerk": jnp.empty((*dims, n_trials,)),
        "min_distance": jnp.empty((*dims, n_trials,)),
        "space_compliance": jnp.empty((*dims, n_trials,)),
        "episodic_spl": jnp.empty((*dims, n_trials,)),
        "path_length": jnp.empty((*dims, n_trials,)),
        "scenario": jnp.empty((*dims, n_trials,), dtype=int),
        "feasible_actions_rate": jnp.empty((*dims, n_trials,)),
    }
    return metrics

def print_average_metrics(n_trials:int, metrics:dict) -> None:
    print("RESULTS")
    print(f"Success rate: {round(metrics['successes']/n_trials,2):.2f}")
    print(f"Collision rate: {round(metrics['collisions']/n_trials,2):.2f}")
    print(f" - of which with humans: {round(metrics['collisions_with_human']/n_trials,2):.2f}")
    print(f" - of which with obstacles: {round(metrics['collisions_with_obstacle']/n_trials,2):.2f}")
    print(f"Timeout rate: {round(metrics['timeouts']/n_trials,2):.2f}")
    print(f"Average return: {round(jnp.mean(metrics['returns']),2):.2f}")
    print(f"SPL: {round(jnp.mean(metrics['episodic_spl']),2):.2f}")
    print(f"Average time to goal: {round(jnp.nanmean(metrics['times_to_goal']),2):.2f} s")
    print(f"Average path length: {round(jnp.nanmean(metrics['path_length']),2):.2f} m")
    print(f"Average speed: {round(jnp.nanmean(metrics['average_speed']),2):.2f} m/s")
    print(f"Average acceleration: {round(jnp.nanmean(metrics['average_acceleration']),2):.2f} m/s^2")
    print(f"Average jerk: {round(jnp.nanmean(metrics['average_jerk']),2):.2f} m/s^3")
    print(f"Average space compliance: {round(jnp.nanmean(metrics['space_compliance']),2):.2f}")
    print(f"Average minimum distance to humans: {round(jnp.nanmean(metrics['min_distance']),2):.2f} m")
    print(f"Average angular speed: {round(jnp.nanmean(metrics['average_angular_speed']),2):.2f} rad/s")
    print(f"Average angular acceleration: {round(jnp.nanmean(metrics['average_angular_acceleration']),2):.2f} rad/s^2")
    print(f"Average angular jerk: {round(jnp.nanmean(metrics['average_angular_jerk']),2):.2f} rad/s^3")
    print(f"Average feasible actions rate: {round(jnp.nanmean(metrics['feasible_actions_rate']),2):.2f}")

@partial(jit, static_argnames=["environment","robot_dt", "robot_radius", "ccso_n_static_humans", "max_steps", "personal_space"])
def compute_episode_metrics(
    environment:int,
    # Saving variables
    metrics:dict,
    episode_idx:int, # Index of the current episode
    # Episode data
    initial_robot_position:jnp.ndarray,
    all_states:jnp.ndarray, 
    all_actions:jnp.ndarray, 
    outcome:dict,
    episode_steps:int,
    end_info:dict,
    # Metric params
    max_steps:float,
    personal_space:float,
    # Env params
    robot_dt:float,
    robot_radius:float,
    ccso_n_static_humans:int,
    robot_specs:dict = {'kinematics': 1, 'v_max': 1.0, 'wheels_distance': 0.7, 'dt': 0.25, 'radius': 0.3},
) -> dict:
    robot_goal = end_info["robot_goal"]
    ## Update metrics
    metrics["successes"] = lax.cond(outcome["success"], lambda x: x + 1, lambda x: x, metrics["successes"])
    if environment == ENVIRONMENTS.index('socialnav'):
        failure = outcome['collision']
        metrics["collisions_with_human"] = lax.cond(failure, lambda x: x + 1, lambda x: x, metrics["collisions_with_human"])
    elif environment == ENVIRONMENTS.index('lasernav'):
        failure = outcome['collision_with_human'] | outcome['collision_with_obstacle']
        metrics["collisions_with_human"] = lax.cond(outcome['collision_with_human'], lambda x: x + 1, lambda x: x, metrics["collisions_with_human"])
        metrics["collisions_with_obstacle"] = lax.cond(outcome['collision_with_obstacle'], lambda x: x + 1, lambda x: x, metrics["collisions_with_obstacle"])
    metrics["collisions"] = lax.cond(failure, lambda x: x + 1, lambda x: x, metrics["collisions"])
    metrics["timeouts"] = lax.cond(outcome["timeout"], lambda x: x + 1, lambda x: x, metrics["timeouts"])
    metrics["returns"] = metrics["returns"].at[episode_idx].set(end_info["return"])
    metrics["scenario"] = metrics["scenario"].at[episode_idx].set(end_info["current_scenario"])
    path_length = lax.fori_loop(0, episode_steps-1, lambda p, val: val + jnp.linalg.norm(all_states[p+1, -1, :2] - all_states[p, -1, :2]), 0.)
    metrics["episodic_spl"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.min(jnp.array([1.,jnp.linalg.norm(robot_goal-initial_robot_position)/path_length]))), lambda x: x.at[episode_idx].set(0.), metrics["episodic_spl"])
    # Metrics computed only if the episode is successful
    metrics["path_length"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(path_length), lambda x: x.at[episode_idx].set(jnp.nan), metrics["path_length"])
    metrics["times_to_goal"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set((episode_steps-1) * robot_dt), lambda x: x.at[episode_idx].set(jnp.nan), metrics["times_to_goal"])
    speeds = lax.fori_loop(
        0, 
        max_steps, 
        lambda s, x: lax.cond(
            s < episode_steps-1,
            lambda y: y.at[s].set((all_states[s+1, -1, :2] - all_states[s, -1, :2]) / robot_dt),
            lambda y: y.at[s].set(jnp.array([jnp.nan,jnp.nan])), 
            x),
        jnp.empty((max_steps, 2)))
    metrics["average_speed"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.linalg.norm(speeds, axis=1))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_speed"])
    accelerations = lax.fori_loop(
        0,
        max_steps,
        lambda a, x: lax.cond(
            a < episode_steps-2,
            lambda y: y.at[a].set((speeds[a+1] - speeds[a]) / robot_dt),
            lambda y: y.at[a].set(jnp.array([jnp.nan,jnp.nan])),
            x),
        jnp.empty((max_steps, 2)))
    metrics["average_acceleration"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.linalg.norm(accelerations, axis=1))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_acceleration"])
    jerks = lax.fori_loop(
        0,
        max_steps,
        lambda j, x: lax.cond(
            j < episode_steps-3,
            lambda y: y.at[j].set((accelerations[j+1] - accelerations[j]) / robot_dt),
            lambda y: y.at[j].set(jnp.array([jnp.nan,jnp.nan])),
            x),
        jnp.empty((max_steps, 2)))
    metrics["average_jerk"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.linalg.norm(jerks, axis=1))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_jerk"])
    angular_speeds = lax.fori_loop(
        0,
        max_steps,
        lambda s, x: lax.cond(
            s < episode_steps-1,
            lambda y: y.at[s].set(wrap_angle(all_states[s+1, -1, 4] - all_states[s, -1, 4]) / robot_dt),
            lambda y: y.at[s].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["average_angular_speed"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.abs(angular_speeds))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_angular_speed"])
    angular_accelerations = lax.fori_loop(
        0,
        max_steps,
        lambda a, x: lax.cond(
            a < episode_steps-2,
            lambda y: y.at[a].set((angular_speeds[a+1] - angular_speeds[a]) / robot_dt),
            lambda y: y.at[a].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["average_angular_acceleration"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.abs(angular_accelerations))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_angular_acceleration"])
    angular_jerks = lax.fori_loop(
        0,
        max_steps,
        lambda j, x: lax.cond(
            j < episode_steps-3,
            lambda y: y.at[j].set((angular_accelerations[j+1] - angular_accelerations[j]) / robot_dt),
            lambda y: y.at[j].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["average_angular_jerk"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(jnp.abs(angular_jerks))), lambda x: x.at[episode_idx].set(jnp.nan), metrics["average_angular_jerk"])
    min_distances = lax.fori_loop(
        0,
        max_steps,
        lambda m, x: lax.cond(
            m < episode_steps,
            lambda y: lax.cond(
                end_info["current_scenario"] == SCENARIOS.index('circular_crossing_with_static_obstacles'), # Static humans in this scenario are not used to compute min_distance and space_compliance
                lambda z: z.at[m].set(jnp.min(jnp.linalg.norm(all_states[m, ccso_n_static_humans:-1, :2] - all_states[m, -1, :2], axis=1) - end_info["humans_parameters"][ccso_n_static_humans:,0]) - robot_radius),
                lambda z: z.at[m].set(jnp.min(jnp.linalg.norm(all_states[m, :-1, :2] - all_states[m, -1, :2], axis=1) - end_info["humans_parameters"][:,0]) - robot_radius),
                y),
            lambda y: y.at[m].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["min_distance"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmin(min_distances)), lambda x: x.at[episode_idx].set(jnp.nan), metrics["min_distance"])
    space_compliances = lax.fori_loop(
        0,
        max_steps,
        lambda s, x: lax.cond(
            s < episode_steps,
            lambda y: y.at[s].set(min_distances[s] > personal_space),
            lambda y: y.at[s].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["space_compliance"] = lax.cond(outcome["success"], lambda x: x.at[episode_idx].set(jnp.nanmean(space_compliances)), lambda x: x.at[episode_idx].set(jnp.nan), metrics["space_compliance"])
    # Only computed for unicycle robot kinematics during successful episodes
    all_actions = all_actions.at[:].set(jnp.round(all_actions, 3))
    out_of_boundary_conditions = \
        (jnp.abs(all_actions[:,1]) - (2 * (robot_specs["v_max"] - all_actions[:,0]) / robot_specs["wheels_distance"]) > 1e-2) | \
        (all_actions[:,0] < 0) | \
        (all_actions[:,0] > robot_specs["v_max"])
    # debug.print("Not feasible actions: {x}", x=jnp.where(out_of_boundary_conditions[:, None], all_actions, jnp.full_like(all_actions, jnp.nan)))
    feasible_actions = lax.fori_loop(
        0,
        max_steps,
        lambda f, x: lax.cond(
            f < episode_steps - 1,
            lambda y: y.at[f].set(~out_of_boundary_conditions[f]),
            lambda y: y.at[f].set(jnp.nan),
            x),
        jnp.empty((max_steps,)))
    metrics["feasible_actions_rate"] = lax.cond(
        (robot_specs["kinematics"] == ROBOT_KINEMATICS.index("unicycle")) & outcome["success"],
        lambda x: x.at[episode_idx].set(jnp.nansum(feasible_actions)/(episode_steps-1)),
        lambda x: x.at[episode_idx].set(jnp.nan),
        metrics["feasible_actions_rate"],
    )
    return metrics

def test_k_trials(
    k: int, 
    random_seed: int, 
    env: BaseEnv, 
    policy: BasePolicy, 
    model_params: dict, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    personal_space:float=0.5,
    custom_episodes:dict=None,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests a policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - policy: BasePolicy. The policy to test.
    - model_params: dict. The parameters of the policy model.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - personal_space: float. A parameter used to compute space compliance.
    - custom_episodes: dict. A dictionary containing custom episodes OF THE STANDARD SCENARIOS (not fully custom episodes) to test the policy on. If None, the environment will be reset normally.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """

    if isinstance(policy, str) and policy == "imitation_learning": # The robot will move as humans in the environment
        imitation_learning = True
    else:
        imitation_learning = False
        actor_critic_policy = (policy.name == "SARL-PPO") or (policy.name == "DIRSAFE")

    if policy.name == "SARL*" and policy.use_planner:
        assert env.grid_map_computation, "SARL* policy requires the environment to compute the grid map."

    # Since jax does not allow to loop over a dict, we have to decompose it in singular jax numpy arrays
    if custom_episodes is not None:
        assert len(custom_episodes) == k, "The number of custom episodes must be equal to the number of trials."
        custom_trials = True
        custom_states = jnp.array([custom_episodes[i]["full_state"] for i in range(k)])
        custom_robot_goals = jnp.array([custom_episodes[i]["robot_goal"] for i in range(k)])
        custom_humans_goals = jnp.array([custom_episodes[i]["humans_goal"] for i in range(k)])
        custom_static_obstacles = jnp.array([custom_episodes[i]["static_obstacles"] for i in range(k)])
        custom_scenario = jnp.array([custom_episodes[i]["scenario"] for i in range(k)])
        custom_humans_radius = jnp.array([custom_episodes[i]["humans_radius"] for i in range(k)])
        custom_humans_speed = jnp.array([custom_episodes[i]["humans_speed"] for i in range(k)])
    else:
        custom_trials = False
        # Dummy variables
        custom_states = jnp.empty((k, env.n_humans+1, 6))
        custom_robot_goals = jnp.empty((k, 2))
        custom_humans_goals = jnp.empty((k, env.n_humans, 2))
        custom_static_obstacles = jnp.empty((k, env.n_humans+1, env.n_obstacles, 1, 2, 2))
        custom_scenario = jnp.empty((k,), dtype=int)
        custom_humans_radius = jnp.empty((k, env.n_humans))
        custom_humans_speed = jnp.empty((k, env.n_humans))

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):   
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            state, obs, info, outcome, policy_key, steps, all_actions, all_states = while_val
            # Make a step in the environment
            if imitation_learning:
                old_state = state.copy()
                state, obs, info, _, outcome = env.imitation_learning_step(state,info)
                dp = state[-1,0:2] - old_state[-1,0:2]
                if env.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                    action = dp / env.robot_dt
                elif env.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                    if env.humans_policy == HUMAN_POLICIES.index('sfm') or env.humans_policy == HUMAN_POLICIES.index('orca'):
                        state = state.at[-1,4].set(jnp.arctan2(*jnp.flip(dp)))
                    action = jnp.array([jnp.linalg.norm(dp / env.robot_dt), wrap_angle(state[-1,4] - old_state[-1,4]) / env.robot_dt])
            else:
                if actor_critic_policy:
                    action, policy_key, _, _, _ = policy.act(policy_key, obs, info, model_params, sample=False)
                else:
                    action, policy_key, _, _ = policy.act(policy_key, obs, info, model_params, 0.)
                state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            # Update step counter
            steps += 1
            return state, obs, info, outcome, policy_key, steps, all_actions, all_states

        ## Retrieve data from the tuple
        seed, metrics = for_val
        policy_key, reset_key = vmap(random.PRNGKey)(jnp.zeros(2, dtype=int) + seed) # We don't care if we generate two identical keys, they operate differently
        ## Reset the environment
        state, reset_key, obs, info, init_outcome = lax.cond(
            custom_trials, 
            lambda x: env.reset_custom_episode(
                x,
                {"full_state": custom_states[i], 
                 "robot_goal": custom_robot_goals[i], 
                 "humans_goal": custom_humans_goals[i], 
                 "static_obstacles": custom_static_obstacles[i], 
                 "scenario": custom_scenario[i], 
                 "humans_radius": custom_humans_radius[i], 
                 "humans_speed": custom_humans_speed[i]}),
            lambda x: env.reset(x), 
            reset_key)
        # state, reset_key, obs, info, init_outcome = env.reset(reset_key)
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val_init = (state, obs, info, init_outcome, policy_key, 0, all_actions, all_states)
        _, _, end_info, outcome, policy_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[3]["nothing"] == True, _while_body, while_val_init)
        ## Update metrics
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': policy.v_max, 'wheels_distance': policy.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Execute k tests
    if env.scenario == SCENARIOS.index("circular_crossing_with_static_obstacles"):
        print(f"\nExecuting {k} tests with {env.n_humans - env.ccso_n_static_humans} dynamic humans and {env.ccso_n_static_humans} static humans...")
    else:
        print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    _, metrics = lax.fori_loop(0, k, _fori_body, (random_seed, metrics))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def test_k_custom_trials(
    k: int, 
    random_seed: int, 
    env: BaseEnv, 
    policy: BasePolicy, 
    model_params: dict, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    custom_episodes:dict,
    personal_space:float=0.5,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests a policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - policy: BasePolicy. The policy to test.
    - model_params: dict. The parameters of the policy model.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - custom_episode: dictionary containing the custom episode data. Its keys are:
        full_state (jnp.array): initial full state of the environment. WARNING: The velocity of humans is always in the global frame (for hsfm you should be using the velocity on the body frame)
        humans_goal (jnp.array): goal positions of the humans.
        robot_goals (jnp.array): final goal position of the robot.
        static_obstacles (jnp.array): positions of the static obstacles.
        humans_radius (float): radius of the humans.
        humans_speed (float): max speed of the humans.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """

    ### Assert data correctness
    assert env.scenario == -1, "The environment must be an environment with custom episodes."
    assert list(custom_episodes.keys()) == ["full_state", "humans_goal", "robot_goals", "static_obstacles", "humans_radius", "humans_speed"], "Invalid keys in custom_episodes. Expected keys: ['full_state', 'humans_goal', 'robot_goals', 'static_obstacles', 'humans_radius', 'humans_speed']"
    for key, value in custom_episodes.items():
        assert key in ["full_state", "humans_goal", "robot_goals", "static_obstacles", "humans_radius", "humans_speed"], f"Invalid key {key} in custom_episodes."
        assert value.shape[0] == k, f"Invalid shape for {key} in custom_episodes. Expected shape ({k}, ...), got {value.shape}."

    ### Check if the policy is imitation learning or PPO
    if isinstance(policy, str) and policy == "imitation_learning": # The robot will move as humans in the environment
        imitation_learning = True
    else:
        imitation_learning = False
        actor_critic_policy = (policy.name == "SARL-PPO") or (policy.name == "DIRSAFE")

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):   
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            episode_idx, state, obs, info, outcome, policy_key, steps, all_actions, all_states = while_val
            # Update robot goal
            info["robot_goal"], info["robot_goal_index"] = lax.cond(
                (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= env.robot_radius*2) & # Waypoint reached threshold is set to be higher
                (info['robot_goal_index'] < len(custom_episodes["robot_goals"][episode_idx])-1) & # Check if current goal is not the last one
                (~(jnp.any(jnp.isnan(custom_episodes["robot_goals"][episode_idx,info['robot_goal_index']+1])))), # Check if next goal is not NaN
                lambda _: (custom_episodes["robot_goals"][episode_idx,info['robot_goal_index']+1], info['robot_goal_index']+1),
                lambda x: x,
                (info["robot_goal"], info["robot_goal_index"])
            )
            # Update humans goal
            info["humans_goal"] = lax.fori_loop(
                0, 
                env.n_humans, 
                lambda h, x: lax.cond(
                    jnp.linalg.norm(state[h,:2] - info["humans_goal"][h]) <= info["humans_parameters"][h,0],
                    lambda y: lax.cond(
                        jnp.all(info["humans_goal"][h] == custom_episodes["humans_goal"][episode_idx,h]),
                        lambda z: z.at[h].set(custom_episodes["full_state"][episode_idx,h,:2]),
                        lambda z: z.at[h].set(custom_episodes["humans_goal"][episode_idx,h]),
                        y,
                    ),
                    lambda y: y,
                    x
                ),
                info["humans_goal"],
            )
            # Make a step in the environment
            if imitation_learning:
                old_state = state.copy()
                state, obs, info, _, outcome = env.imitation_learning_step(state,info)
                dp = state[-1,0:2] - old_state[-1,0:2]
                if env.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                    action = dp / env.robot_dt
                elif env.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                    if env.humans_policy == HUMAN_POLICIES.index('sfm') or env.humans_policy == HUMAN_POLICIES.index('orca'):
                        state = state.at[-1,4].set(jnp.arctan2(*jnp.flip(dp)))
                    action = jnp.array([jnp.linalg.norm(dp / env.robot_dt), wrap_angle(state[-1,4] - old_state[-1,4]) / env.robot_dt])
            else:
                if actor_critic_policy:
                    action, policy_key, _, _, _ = policy.act(policy_key, obs, info, model_params, sample=False)
                else:
                    action, policy_key, _, _ = policy.act(policy_key, obs, info, model_params, 0.)
                state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            # Update step counter
            steps += 1
            return episode_idx, state, obs, info, outcome, policy_key, steps, all_actions, all_states

        ## Retrieve data from the tuple
        seed, metrics = for_val
        policy_key = random.PRNGKey(seed) # We don't care if we generate two identical keys, they operate differently
        ## Reset the environment
        state, _, obs, info, init_outcome = env.reset_custom_episode(
            random.PRNGKey(0), # Not used, but required by the function
            {
                "full_state": custom_episodes["full_state"][i],
                "robot_goal": custom_episodes["robot_goals"][i,0],
                "humans_goal": custom_episodes["humans_goal"][i],
                "static_obstacles": custom_episodes["static_obstacles"][i],
                "scenario": -1,
                "humans_radius": custom_episodes["humans_radius"][i],
                "humans_speed": custom_episodes["humans_speed"][i],
            }
        )
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val_init = (i, state, obs, info, init_outcome, policy_key, 0, all_actions, all_states)
        _, _, _, end_info, outcome, policy_key, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[4]["nothing"] == True, _while_body, while_val_init)
        ## Update metrics
        metrics["waypoint_reached"] = metrics["waypoint_reached"].at[i].set(end_info["robot_goal_index"])
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': policy.v_max, 'wheels_distance': policy.wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Execute k tests
    print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    _, metrics = lax.fori_loop(0, k, _fori_body, (random_seed, metrics))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def interpolate_obstacle_segments(obstacles, points_per_meter=10):
    point_list = []
    for obs in np.array(obstacles):
        for edge in obs:
            p0, p1 = edge
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            length = np.linalg.norm(p1 - p0)
            n_points = max(2, int(np.ceil(length * points_per_meter)))
            t = np.linspace(0, 1, n_points)
            points = (1 - t)[:, None] * p0 + t[:, None] * p1
            point_list.append(points)
    if point_list:
        return np.vstack(point_list)
    else:
        return np.empty((0, 2))

def interpolate_humans_boundaries(humans_pose, humans_radiuses, points_per_human=10):
    point_list = []
    for pose, radius in zip(humans_pose, humans_radiuses):
        angle = jnp.linspace(0, 2 * jnp.pi, points_per_human)
        x = pose[0] + radius * jnp.cos(angle)
        y = pose[1] + radius * jnp.sin(angle)
        point_list.append(jnp.stack((x, y), axis=-1))
    return jnp.concatenate(point_list, axis=0)

def test_k_trials_dwa(
    k: int,
    random_seed: int, 
    env: BaseEnv, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    robot_vmax:float=1.0,
    robot_wheels_distance:float=0.7,
    personal_space:float=0.5,
    custom_episodes:dict=None,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests a policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """
    try:
        import dwa
    except ImportError:
        raise ImportError("DWA package is not installed. Please install it to use this function.\nYou can install it with 'pip3 install dynamic-window-approach'.\n Checkout https://github.com/goktug97/DynamicWindowApproach")
    
    assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "DWA can only be used with unicycle robots."

    # Since jax does not allow to loop over a dict, we have to decompose it in singular jax numpy arrays
    if custom_episodes is not None:
        assert len(custom_episodes) == k, "The number of custom episodes must be equal to the number of trials."
        custom_trials = True
        custom_states = jnp.array([custom_episodes[i]["full_state"] for i in range(k)])
        custom_robot_goals = jnp.array([custom_episodes[i]["robot_goal"] for i in range(k)])
        custom_humans_goals = jnp.array([custom_episodes[i]["humans_goal"] for i in range(k)])
        custom_static_obstacles = jnp.array([custom_episodes[i]["static_obstacles"] for i in range(k)])
        custom_scenario = jnp.array([custom_episodes[i]["scenario"] for i in range(k)])
        custom_humans_radius = jnp.array([custom_episodes[i]["humans_radius"] for i in range(k)])
        custom_humans_speed = jnp.array([custom_episodes[i]["humans_speed"] for i in range(k)])
    else:
        custom_trials = False
        # Dummy variables
        custom_states = jnp.empty((k, env.n_humans+1, 6))
        custom_robot_goals = jnp.empty((k, 2))
        custom_humans_goals = jnp.empty((k, env.n_humans, 2))
        custom_static_obstacles = jnp.empty((k, env.n_humans+1, env.n_obstacles, 1, 2, 2))
        custom_scenario = jnp.empty((k,), dtype=int)
        custom_humans_radius = jnp.empty((k, env.n_humans))
        custom_humans_speed = jnp.empty((k, env.n_humans))
    
    def _fori_body(i:int, for_val:tuple):   
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud = while_val
            # Construct point cloud
            humans_point_cloud = interpolate_humans_boundaries(obs[:-1,:2], info['humans_parameters'][:,0])
            point_cloud = jnp.concatenate((obstacles_point_cloud, humans_point_cloud), axis=0)
            # Make a step in the environment
            action = jnp.array(dwa.planning(tuple(map(float, np.append(obs[-1,:2],obs[-1,5]))), tuple(map(float, obs[-1,2:4])), tuple(map(float, info['robot_goal'])), np.array(point_cloud, dtype=np.float32), dwa_config))
            state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            # Update step counter
            steps += 1
            return state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud

        ## Retrieve data from the tuple
        seed, metrics, dwa_config = for_val
        reset_key = random.PRNGKey(seed) 
        ## Reset the environment
        state, reset_key, obs, info, outcome = lax.cond(
            custom_trials, 
            lambda x: env.reset_custom_episode(
                x,
                {"full_state": custom_states[i], 
                 "robot_goal": custom_robot_goals[i], 
                 "humans_goal": custom_humans_goals[i], 
                 "static_obstacles": custom_static_obstacles[i], 
                 "scenario": custom_scenario[i], 
                 "humans_radius": custom_humans_radius[i], 
                 "humans_speed": custom_humans_speed[i]}),
            lambda x: env.reset(x), 
            reset_key)
        # Construc obstacles point cloud
        obstacles_point_cloud = interpolate_obstacle_segments(info["static_obstacles"][-1])
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val = (state, obs, info, outcome, 0, all_actions, all_states, obstacles_point_cloud)
        while outcome["nothing"]:
            state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud = _while_body(while_val)
            while_val = (state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud)
        _, _, end_info, outcome, episode_steps, all_actions, all_states, _ = while_val
        ## Update metrics
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': robot_vmax, 'wheels_distance': robot_wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Define DWA configuration
    dwa_config = dwa.Config(
        max_speed=robot_vmax,
        min_speed=0.0,
        max_yawrate=2*robot_vmax/robot_wheels_distance,
        dt = env.robot_dt,
        max_accel=4,
        max_dyawrate=4,
        predict_time = .5,
        velocity_resolution = 0.1, # Discretization of the velocity space
        yawrate_resolution = np.radians(1.0), # Discretization of the yawrate space
        heading = 0.04,
        clearance = 0.2,
        velocity = 0.2,
        base=[-env.robot_radius, -env.robot_radius, env.robot_radius, env.robot_radius],  # [x_min, y_min, x_max, y_max] in meters
    )
    # Execute k tests
    if env.scenario == SCENARIOS.index("circular_crossing_with_static_obstacles"):
        print(f"\nExecuting {k} tests with {env.n_humans - env.ccso_n_static_humans} dynamic humans and {env.ccso_n_static_humans} static humans...")
    else:
        print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    for i in tqdm(range(k), desc="Testing DWA policy"):
        random_seed, metrics = _fori_body(i, (random_seed, metrics, dwa_config))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def test_k_custom_trials_dwa(
    k: int,
    random_seed: int, 
    env: BaseEnv, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    custom_episodes:dict,
    robot_vmax:float=1.0,
    robot_wheels_distance:float=0.7,
    personal_space:float=0.5,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests a policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """
    try:
        import dwa
    except ImportError:
        raise ImportError("DWA package is not installed. Please install it to use this function.\nYou can install it with 'pip3 install dynamic-window-approach'.\n Checkout https://github.com/goktug97/DynamicWindowApproach")
    
    ### Assert data correctness
    assert env.scenario == -1, "The environment must be an environment with custom episodes."
    assert list(custom_episodes.keys()) == ["full_state", "humans_goal", "robot_goals", "static_obstacles", "humans_radius", "humans_speed"], "Invalid keys in custom_episodes. Expected keys: ['full_state', 'humans_goal', 'robot_goals', 'static_obstacles', 'humans_radius', 'humans_speed']"
    for key, value in custom_episodes.items():
        assert key in ["full_state", "humans_goal", "robot_goals", "static_obstacles", "humans_radius", "humans_speed"], f"Invalid key {key} in custom_episodes."
        assert value.shape[0] == k, f"Invalid shape for {key} in custom_episodes. Expected shape ({k}, ...), got {value.shape}."
    assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "DWA can only be used with unicycle robots."

    def _fori_body(i:int, for_val:tuple):   
        # Construct point cloud functions

        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            episode_idx, state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud = while_val
            # Update robot goal
            info["robot_goal"], info["robot_goal_index"] = lax.cond(
                (jnp.linalg.norm(state[-1,:2] - info["robot_goal"]) <= env.robot_radius*2) & # Waypoint reached threshold is set to be higher
                (info['robot_goal_index'] < len(custom_episodes["robot_goals"][episode_idx])-1) & # Check if current goal is not the last one
                (~(jnp.any(jnp.isnan(custom_episodes["robot_goals"][episode_idx,info['robot_goal_index']+1])))), # Check if next goal is not NaN
                lambda _: (custom_episodes["robot_goals"][episode_idx,info['robot_goal_index']+1], info['robot_goal_index']+1),
                lambda x: x,
                (info["robot_goal"], info["robot_goal_index"])
            )
            # Update humans goal
            info["humans_goal"] = lax.fori_loop(
                0, 
                env.n_humans, 
                lambda h, x: lax.cond(
                    jnp.linalg.norm(state[h,:2] - info["humans_goal"][h]) <= info["humans_parameters"][h,0],
                    lambda y: lax.cond(
                        jnp.all(info["humans_goal"][h] == custom_episodes["humans_goal"][episode_idx,h]),
                        lambda z: z.at[h].set(custom_episodes["full_state"][episode_idx,h,:2]),
                        lambda z: z.at[h].set(custom_episodes["humans_goal"][episode_idx,h]),
                        y,
                    ),
                    lambda y: y,
                    x
                ),
                info["humans_goal"],
            )
            # Construct point cloud
            humans_point_cloud = interpolate_humans_boundaries(obs[:-1,:2], info['humans_parameters'][:,0])
            point_cloud = jnp.concatenate((obstacles_point_cloud, humans_point_cloud), axis=0)
            # Make a step in the environment
            action = jnp.array(dwa.planning(tuple(map(float, np.append(obs[-1,:2],obs[-1,5]))), tuple(map(float, obs[-1,2:4])), tuple(map(float, info['robot_goal'])), np.array(point_cloud, dtype=np.float32), dwa_config))
            state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            # Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            # Update step counter
            steps += 1
            return episode_idx, state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud

        ## Retrieve data from the tuple
        seed, metrics, dwa_config = for_val
        ## Reset the environment
        state, _, obs, info, outcome = env.reset_custom_episode(
            random.PRNGKey(0), # Not used, but required by the function
            {
                "full_state": custom_episodes["full_state"][i],
                "robot_goal": custom_episodes["robot_goals"][i,0],
                "humans_goal": custom_episodes["humans_goal"][i],
                "static_obstacles": custom_episodes["static_obstacles"][i],
                "scenario": -1,
                "humans_radius": custom_episodes["humans_radius"][i],
                "humans_speed": custom_episodes["humans_speed"][i],
            }
        )
        initial_robot_position = state[-1,:2]
        # Construct obstacles point cloud
        obstacles_point_cloud = interpolate_obstacle_segments(info["static_obstacles"][-1])
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val = (i, state, obs, info, outcome, 0, all_actions, all_states, obstacles_point_cloud)
        while outcome["nothing"]:
            i, state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud = _while_body(while_val)
            while_val = (i, state, obs, info, outcome, steps, all_actions, all_states, obstacles_point_cloud)
        _, _, _, end_info, outcome, episode_steps, all_actions, all_states, _ = while_val
        ## Update metrics
        metrics["waypoint_reached"] = metrics["waypoint_reached"].at[i].set(end_info["robot_goal_index"])
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': robot_vmax, 'wheels_distance': robot_wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Define DWA configuration
    dwa_config = dwa.Config(
        max_speed=robot_vmax,
        min_speed=0.0,
        max_yawrate=2*robot_vmax/robot_wheels_distance,
        dt = env.robot_dt,
        max_accel=4,
        max_dyawrate=4,
        predict_time = .5,
        velocity_resolution = 0.1, # Discretization of the velocity space
        yawrate_resolution = np.radians(1.0), # Discretization of the yawrate space
        heading = 0.04,
        clearance = 0.2,
        velocity = 0.2,
        base=[-env.robot_radius, -env.robot_radius, env.robot_radius, env.robot_radius],  # [x_min, y_min, x_max, y_max] in meters
    )
    # Execute k tests
    print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    for i in tqdm(range(k), desc="Testing DWA policy"):
        random_seed, metrics = _fori_body(i, (random_seed, metrics, dwa_config))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def test_k_trials_sfm(
    k: int,
    random_seed: int, 
    env: BaseEnv, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    robot_vmax:float=1.0,
    robot_wheels_distance:float=0.7,
    personal_space:float=0.5,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests the SFM policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - robot_vmax: float. The maximum speed of the robot.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """
    
    try:
        from jsfm.sfm import single_update
        from jhsfm.hsfm import get_linear_velocity
        from jsfm.utils import get_standard_humans_parameters
    except ImportError:
        raise ImportError("JSFM or JHSFM package is not installed. Please install it to following instructions at https://github.com/TommasoVandermeer/social-jym/")

    # assert isinstance(env, SocialNav), "The environment must be an instance of SocialNav."

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):  
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            prev_state, state, obs, info, outcome, steps, all_actions, all_states = while_val
            temp = jnp.copy(state)
            ### COMPUTE ROBOT ACTION ###
            if env.humans_policy == HUMAN_POLICIES.index('hsfm'):
                # Setup humans parameters
                parameters = get_standard_humans_parameters(env.n_humans+1)
                parameters = parameters.at[-1,0].set(env.robot_radius) # Set robot radius
                parameters = parameters.at[-1,2].set(robot_vmax) # Set robot max speed
                parameters = parameters.at[:-1,0].set(info["humans_parameters"][:,0]) # Set humans radius
                parameters = parameters.at[:-1,2].set(info["humans_parameters"][:,2]) # Set humans max_speed
                # Convert HSFM state to SFM state
                humans_lin_vel = vmap(get_linear_velocity, in_axes=(0, 0))(state[:-1,4], state[:-1,2:4])
                feed_state = jnp.copy(state[:,:4])
                feed_state = feed_state.at[:-1,2:4].set(humans_lin_vel)
            elif env.humans_policy == HUMAN_POLICIES.index('sfm'):
                # Setup humans parameters
                parameters = jnp.vstack((info["humans_parameters"], jnp.array([env.robot_radius, 80., robot_vmax, *get_standard_humans_parameters(1)[0,3:]])))
                # Setup robot state for SFM
                humans_lin_vel = state[:-1,2:4]
                feed_state = jnp.copy(state[:,:4])
            elif env.humans_policy == HUMAN_POLICIES.index('orca'):
                # Setup humans parameters
                parameters = get_standard_humans_parameters(env.n_humans+1)
                parameters = parameters.at[-1,0].set(env.robot_radius) # Set robot radius
                parameters = parameters.at[-1,2].set(robot_vmax) # Set robot max speed
                parameters = parameters.at[:-1,0].set(info["humans_parameters"][:,0]) # Set humans radius
                parameters = parameters.at[:-1,2].set(info["humans_parameters"][:,2]) # Set humans max_speed
                # Setup robot state for SFM
                humans_lin_vel = state[:-1,2:4]
                feed_state = jnp.copy(state[:,:4])
            # Set tobot feed state
            feed_state = feed_state.at[-1,2:4].set((state[-1,0:2]-prev_state[-1,0:2])/env.robot_dt)
            # Step the robot in SFM environment (doing it with substeps to avoid instabilities)
            @jit
            def _substep(i, feed_state):
                new_robot_state = single_update(
                    -1,
                    feed_state, 
                    info["robot_goal"], 
                    parameters,
                    info["static_obstacles"][-1],
                    env.humans_dt
                )
                feed_state = feed_state.at[:-1,0].set(feed_state[:-1,0] + humans_lin_vel[:,0] * env.humans_dt)
                feed_state = feed_state.at[:-1,1].set(feed_state[:-1,1] + humans_lin_vel[:,1] * env.humans_dt)
                feed_state = feed_state.at[-1].set(new_robot_state)
                return feed_state
            new_feed_state = lax.fori_loop(0, int(env.robot_dt/env.humans_dt), _substep, feed_state)
            new_robot_state = new_feed_state[-1]
            # Compute action
            new_velocity = (new_robot_state[0:2] - state[-1,0:2]) / env.robot_dt
            if env.kinematics == ROBOT_KINEMATICS.index('unicycle'):
                action = jnp.array([jnp.linalg.norm(new_velocity), wrap_angle(jnp.atan2(new_robot_state[3], new_robot_state[2]) - state[-1,4]) / env.robot_dt])
            elif env.kinematics == ROBOT_KINEMATICS.index('holonomic'):
                action = new_robot_state[2:4]
            ### STEP THE ENVIRONMENT ###
            state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            ### Update prev_state
            prev_state = jnp.copy(temp)
            ### Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            ### Update step counter
            steps += 1
            return prev_state, state, obs, info, outcome, steps, all_actions, all_states
        
        ## Retrieve data from the tuple
        seed, metrics = for_val
        reset_key = random.PRNGKey(seed)
        ## Reset the environment
        state, reset_key, obs, info, init_outcome = env.reset(reset_key)
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val_init = (state, state, obs, info, init_outcome, 0, all_actions, all_states)
        _, _, _, end_info, outcome, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[4]["nothing"] == True, _while_body, while_val_init)
        ## Update metrics
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': robot_vmax, 'wheels_distance': robot_wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Execute k tests
    if env.scenario == SCENARIOS.index("circular_crossing_with_static_obstacles"):
        print(f"\nExecuting {k} tests with {env.n_humans - env.ccso_n_static_humans} dynamic humans and {env.ccso_n_static_humans} static humans...")
    else:
        print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    _, metrics = lax.fori_loop(0, k, _fori_body, (random_seed, metrics))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def test_k_trials_hsfm(
    k: int,
    random_seed: int, 
    env: BaseEnv, 
    time_limit: float, # WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage
    robot_vmax:float=1.0,
    robot_wheels_distance:float=0.7,
    personal_space:float=0.5,
    print_avg_metrics:bool=True
) -> tuple:
    """
    This function tests the HSFM policy in a given environment for k trials and outputs a series of metrics.

    args:
    - k: int. The number of trials to execute.
    - random_seed: int. The random seed to use for the execution.
    - env: BaseEnv. The environment to test the policy in.
    - time_limit: float. The maximum time limit for each trial. WARNING: This does not effectively modifies the max length of a trial, it is just used to shape array sizes for data storage.
    - robot_vmax: float. The maximum speed of the robot.
    - personal_space: float. A parameter used to compute space compliance.

    output:
    - metrics: dict. A dictionary containing the metrics of the tests.
    """
    
    try:
        from jhsfm.hsfm import single_update, get_linear_velocity
        from jhsfm.utils import get_standard_humans_parameters
    except ImportError:
        raise ImportError("JHSFM package is not installed. Please install it to following instructions at https://github.com/TommasoVandermeer/social-jym/")

    # assert isinstance(env, SocialNav), "The environment must be an instance of SocialNav."
    assert env.kinematics == ROBOT_KINEMATICS.index('unicycle'), "HSFM can only be used with unicycle robots."

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):  
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            prev_state, state, obs, info, outcome, steps, all_actions, all_states = while_val
            temp = jnp.copy(state)
            ### COMPUTE ROBOT ACTION ###
            if env.humans_policy == HUMAN_POLICIES.index('hsfm'):
                # Setup humans parameters
                parameters = jnp.vstack((info["humans_parameters"], jnp.array([env.robot_radius, 80., robot_vmax, *get_standard_humans_parameters(1)[0,3:]])))
                # Setup robot state for HSFM
                humans_lin_vel = vmap(get_linear_velocity, in_axes=(0, 0))(state[:-1,4], state[:-1,2:4])
                feed_state = jnp.copy(state)
            elif env.humans_policy == HUMAN_POLICIES.index('sfm') or env.humans_policy == HUMAN_POLICIES.index('orca'):
                # Setup humans parameters
                parameters = get_standard_humans_parameters(env.n_humans+1)
                parameters = parameters.at[-1,0].set(env.robot_radius) # Set robot radius
                parameters = parameters.at[-1,2].set(robot_vmax) # Set robot max speed
                parameters = parameters.at[:-1,0].set(info["humans_parameters"][:,0]) # Set humans radius
                parameters = parameters.at[:-1,2].set(info["humans_parameters"][:,2]) # Set humans max_speed
                # Convert SFM/ORCA state to HSFM state
                humans_lin_vel = state[:-1,2:4]
                feed_state = jnp.copy(state)
                humans_theta = jnp.arctan2(humans_lin_vel[:,1], humans_lin_vel[:,0])
                previous_humans_lin_vel = prev_state[:-1,2:4]
                previous_humans_theta = jnp.arctan2(previous_humans_lin_vel[:,1], previous_humans_lin_vel[:,0])
                @jit
                def _get_body_and_ang_velocity(lin_vel, theta, previous_theta):
                    rotational_matrix = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
                    angular_velocity = wrap_angle(theta - previous_theta) / env.humans_dt
                    return rotational_matrix @ lin_vel, angular_velocity
                humans_body_vel, humans_angular_vel = vmap(_get_body_and_ang_velocity, in_axes=(0,0,0))(
                    humans_lin_vel, 
                    humans_theta, 
                    previous_humans_theta
                )
                feed_state = feed_state.at[:-1,2:4].set(humans_body_vel)
                feed_state = feed_state.at[:-1,4].set(humans_theta)
                feed_state = feed_state.at[:-1,5].set(humans_angular_vel)
            # Set robot feed state
            linear_velocity = (state[-1,:2]-prev_state[-1,0:2])/env.robot_dt
            robot_theta = state[-1,4] 
            rotational_matrix = jnp.array([[jnp.cos(robot_theta), jnp.sin(robot_theta)], [-jnp.sin(robot_theta), jnp.cos(robot_theta)]])
            body_velocity = rotational_matrix @ linear_velocity
            feed_state = feed_state.at[-1,2:4].set(body_velocity)
            angular_velocity = wrap_angle(robot_theta - prev_state[-1,4]) / env.robot_dt
            feed_state = feed_state.at[-1,5].set(angular_velocity)
            # Step the robot in HSFM environment (doing it with substeps to avoid instabilities)
            @jit
            def _substep(i, feed_state):
                new_robot_state = single_update(
                    -1,
                    feed_state, 
                    info["robot_goal"], 
                    parameters,
                    info["static_obstacles"][-1],
                    env.humans_dt
                )
                feed_state = feed_state.at[:-1,0].set(feed_state[:-1,0] + humans_lin_vel[:,0] * env.humans_dt)
                feed_state = feed_state.at[:-1,1].set(feed_state[:-1,1] + humans_lin_vel[:,1] * env.humans_dt)
                feed_state = feed_state.at[-1].set(new_robot_state)
                return feed_state
            new_feed_state = lax.fori_loop(0, int(env.robot_dt/env.humans_dt), _substep, feed_state)
            new_robot_state = new_feed_state[-1]
            # Compute action
            new_velocity = (new_robot_state[:2] - state[-1,:2]) / env.robot_dt
            action = jnp.array([jnp.linalg.norm(new_velocity), wrap_angle(new_robot_state[4] - state[-1,4]) / env.robot_dt])
            ### STEP THE ENVIRONMENT ###
            state, obs, info, _, outcome, _ = env.step(state,info,action,test=True)
            ### Update prev_state
            prev_state = jnp.copy(temp)
            ### Save data
            all_actions = all_actions.at[steps].set(action)
            all_states = all_states.at[steps].set(state)
            ### Update step counter
            steps += 1
            return prev_state, state, obs, info, outcome, steps, all_actions, all_states
        
        ## Retrieve data from the tuple
        seed, metrics = for_val
        reset_key = random.PRNGKey(seed)
        ## Reset the environment
        state, reset_key, obs, info, init_outcome = env.reset(reset_key)
        initial_robot_position = state[-1,:2]
        ## Episode loop
        all_actions = jnp.empty((int(time_limit/env.robot_dt)+1, 2))
        all_states = jnp.empty((int(time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        while_val_init = (state, state, obs, info, init_outcome, 0, all_actions, all_states)
        _, _, _, end_info, outcome, episode_steps, all_actions, all_states = lax.while_loop(lambda x: x[4]["nothing"] == True, _while_body, while_val_init)
        ## Update metrics
        metrics = compute_episode_metrics(
            environment=env.environment,
            metrics=metrics,
            episode_idx=i, 
            initial_robot_position=initial_robot_position, 
            all_states=all_states, 
            all_actions=all_actions, 
            outcome=outcome, 
            episode_steps=episode_steps, 
            end_info=end_info, 
            max_steps=int(time_limit/env.robot_dt)+1, 
            personal_space=personal_space,
            robot_dt=env.robot_dt,
            robot_radius=env.robot_radius,
            ccso_n_static_humans=env.ccso_n_static_humans,
            robot_specs={'kinematics': env.kinematics, 'v_max': robot_vmax, 'wheels_distance': robot_wheels_distance, 'dt': env.robot_dt, 'radius': env.robot_radius},
        )
        seed += 1
        return seed, metrics
    
    # Initialize metrics
    metrics = initialize_metrics_dict(k)
    # Execute k tests
    if env.scenario == SCENARIOS.index("circular_crossing_with_static_obstacles"):
        print(f"\nExecuting {k} tests with {env.n_humans - env.ccso_n_static_humans} dynamic humans and {env.ccso_n_static_humans} static humans...")
    else:
        print(f"\nExecuting {k} tests with {env.n_humans} humans...")
    _, metrics = lax.fori_loop(0, k, _fori_body, (random_seed, metrics))
    # Print results
    if print_avg_metrics:
        print_average_metrics(k, metrics)
    return metrics

def animate_trajectory(
    states:jnp.ndarray, 
    humans_radiuses:np.ndarray, 
    robot_radius:float, 
    humans_policy:str,
    robot_goal:np.ndarray,
    scenario:int,
    robot_dt:float=0.25,
    static_obstacles:jnp.ndarray=None,
    lidar_measurements:jnp.ndarray=None,
    kinematics:str='holonomic',
    action_space_params:jnp.ndarray=None,
    action_space_aside:bool=False,
    vmax:float=None,
    wheels_distance:float=None,
    save:bool=False,
    save_path:str=None,
    figsize:tuple=None,
    ) -> None:

    if action_space_params is not None:
        assert kinematics == 'unicycle', "Action space parameters are only available for unicycle kinematics."
        assert vmax is not None, "vmax must be provided if action space parameters are used."
        assert wheels_distance is not None, "wheels_distance must be provided if action space parameters are used."

    if save:
        assert save_path is not None, "save_path must be provided if save is True."
        if not save_path.endswith('.mp4'):
            save_path += '.mp4'
        if os.path.exists(save_path):
            print(f"Warning: {save_path} already exists. It will be overwritten.")

    # TODO: Add a progress bar
    
    if action_space_params is not None and action_space_aside:
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300, gridspec_kw={'width_ratios': [4, 1]})
        fig.subplots_adjust(right=0.95, left=0.1, bottom=0.1, top=0.85)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        fig.subplots_adjust(right=0.78, top=0.90, bottom=0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=[-10,10],ylim=[-10,10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if (static_obstacles is not None) and (scenario is None):
        xlims = [jnp.nanmin(static_obstacles[:,:,:,0]), jnp.nanmax(static_obstacles[:,:,:,0])]
        ylims = [jnp.nanmin(static_obstacles[:,:,:,1]), jnp.nanmax(static_obstacles[:,:,:,1])]
        ax.autoscale(enable=False)
    else:
        xlims = None
        ylims = None

    def animate(frame):
        ax.clear()
        if action_space_params is not None and action_space_aside:
            ax.legend(
                title=f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))}",
                handles=[
                    Line2D([0], [0], color='white', marker='o', markersize=7, markerfacecolor='red', markeredgecolor='black', linewidth=2, label='Robot'), 
                    Line2D([0], [0], color='white', marker='o', markersize=7, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans'),
                    Line2D([0], [0], color='white', marker='*', markersize=7, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Goal')
                ],
                loc='lower center',    
                bbox_to_anchor=(0.5, 1.0),
                fontsize=7,
                title_fontsize=7,
            )
        else:
            ax.set_title(f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))} - Humans policy: {humans_policy.upper()}", weight='bold')
            ax.legend(
                handles=[
                    Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='red', markeredgecolor='black', linewidth=2, label='Robot'), 
                    Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans'),
                    Line2D([0], [0], color='white', marker='*', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Goal')
                ],
                bbox_to_anchor=(0.99, 0.5), 
                loc='center left',
            )
        if len(robot_goal.shape) == 1:
            ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)
        else:
            ax.scatter(robot_goal[frame,0], robot_goal[frame,1], marker="*", color="red", zorder=2)
        plot_state(ax, frame*robot_dt, states[frame], humans_policy, scenario, humans_radiuses, robot_radius, plot_time=False, kinematics=kinematics, xlims=xlims, ylims=ylims)
        if lidar_measurements is not None:
            plot_lidar_measurements(ax, lidar_measurements[frame], states[frame][-1], robot_radius)
        if static_obstacles is not None:
            if static_obstacles.shape[1] > 1: # Polygon obstacles
                for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
            else: # One segment obstacles
                for o in static_obstacles: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
        if action_space_params is not None and frame < len(action_space_params):
            new_alpha, new_beta, new_gamma = action_space_params[frame]
            if not action_space_aside:
                ax.add_artist(plt.Rectangle(
                    (states[frame,-1,0] - robot_radius, states[frame,-1,1] - new_alpha*robot_dt**2*new_gamma*vmax/(4*wheels_distance) - robot_radius), 
                    new_alpha*vmax*robot_dt + 2 * robot_radius, 
                    2*robot_radius + (new_alpha*robot_dt**2*vmax/(4*wheels_distance) * (new_beta + new_gamma)), 
                    rotation_point=(float(states[frame,-1,0]), float(states[frame,-1,1])), 
                    angle=jnp.rad2deg(states[frame,-1,4]), 
                    color='green', 
                    fill=False, 
                    zorder=3, 
                    linewidth=2,
                    linestyle='--'
                ))
            else:
                axes[1].clear()
                axes[1].set_xlabel("$v$ (m/s)")
                axes[1].set_ylabel("$\omega$ (rad/s)")
                axes[1].set_xlim(-0.1, vmax + 0.1)
                axes[1].set_ylim(-2*vmax/wheels_distance - 0.3, 2*vmax/wheels_distance + 0.3)
                axes[1].set_xticks(jnp.arange(0, vmax+0.2, 0.2))
                axes[1].set_xticklabels([round(i,1) for i in np.arange(0, vmax, 0.2)] + [r"$\overline{v}$"])
                axes[1].set_yticks(np.arange(-2,3,1).tolist() + [2*vmax/wheels_distance,-2*vmax/wheels_distance])
                axes[1].set_yticklabels([round(i) for i in np.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
                axes[1].grid()
                axes[1].add_patch(
                    plt.Polygon(
                        [   
                            [0,2*vmax/wheels_distance],
                            [0,-2*vmax/wheels_distance],
                            [vmax,0],
                        ],
                        closed=True,
                        fill=True,
                        edgecolor='red',
                        facecolor='lightcoral',
                        linewidth=2,
                        zorder=2,
                        label='Feasible action space'
                    ),
                )
                axes[1].add_patch(
                    plt.Polygon(
                        [   
                            [0,(2*vmax/wheels_distance)*new_beta],
                            [0,(-2*vmax/wheels_distance)*new_gamma],
                            [new_alpha*vmax,0],
                        ],
                        closed=True,
                        fill=True,
                        edgecolor='green',
                        facecolor='lightgreen',
                        linewidth=2,
                        zorder=3,
                        label='Collision-free action space'
                    ),
                )
                axes[1].legend(fontsize=7, bbox_to_anchor=(0.95, 1.12))

    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=len(states))
    
    if save:
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)

    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused

    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()

def save_policy_params(
    policy_name:str,
    policy_params: dict, 
    train_env_params:dict, 
    reward_params:dict,
    hyperparameters:dict, 
    path:str,
    filename=None
) -> None:

    if os.path.exists(path) == False:
        os.makedirs(path)

    if filename is None:
        today = date.today().strftime('%d_%m_%Y')
        filename = f"{policy_name}_k{train_env_params['kinematics']}_nh{train_env_params['n_humans']}_hp{train_env_params['humans_policy']}_s{train_env_params['scenario']}_r{reward_params['type'][-1]}_{today}.pkl"
    else:
        filename = f"{filename}.pkl"
        
    with open(os.path.join(path, filename), 'wb') as f:
        pkl.dump({
            "policy_name": policy_name,
            "policy_params": policy_params,
            "train_env_params": {k: train_env_params[k] for k in set(list(train_env_params.keys())) - set(['reward_function'])},
            "reward_params": reward_params,
            "hyperparameters": hyperparameters}, f)

def load_socialjym_policy(
    path:str, 
) -> dict:
    with open(os.path.join(path), 'rb') as f:
        trained_policy = pkl.load(f)
        policy_params = trained_policy["policy_params"]
        return policy_params

def load_crowdnav_policy(
        policy_name:str,
        path:str) -> tuple:
    
    import torch

    ## Get Value Network Parameters (weights and biases)
    vnet_params = {}
    if policy_name == "cadrl":
        l = 0
        for k, v in torch.load(path, weights_only=True).items():
            if l%2 == 0:
                vnet_params[f"mlp/~/linear_{l//2}"] = {}
                vnet_params[f"mlp/~/linear_{l//2}"]["w"] = jnp.array(v.detach().cpu().numpy().T)
            else:
                vnet_params[f"mlp/~/linear_{l//2}"]["b"] = jnp.array(v.detach().cpu().numpy().T)
            l += 1
    elif policy_name == "sarl":
        l = 0
        for k, v in torch.load(path, weights_only=True).items():
            if l%2 == 0:
                if l == 0: vnet_params["value_network/~/mlp1/~/linear_0"] = {}
                if l == 2: vnet_params["value_network/~/mlp1/~/linear_1"] = {}
                if l == 4: vnet_params["value_network/~/mlp2/~/linear_0"] = {}
                if l == 6: vnet_params["value_network/~/mlp2/~/linear_1"] = {}
                if l == 8: vnet_params["value_network/~/attention/~/linear_0"] = {}
                if l == 10: vnet_params["value_network/~/attention/~/linear_1"] = {}
                if l == 12: vnet_params["value_network/~/attention/~/linear_2"] = {}
                if l == 14: vnet_params["value_network/~/mlp3/~/linear_0"] = {}
                if l == 16: vnet_params["value_network/~/mlp3/~/linear_1"] = {}
                if l == 18: vnet_params["value_network/~/mlp3/~/linear_2"] = {}
                if l == 20: vnet_params["value_network/~/mlp3/~/linear_3"] = {}
                vnet_params[list(vnet_params.keys())[l//2]]["w"] = jnp.array(v.detach().cpu().numpy().T)
            else:
                vnet_params[list(vnet_params.keys())[l//2]]["b"] = jnp.array(v.detach().cpu().numpy().T)
            l += 1
    return vnet_params
