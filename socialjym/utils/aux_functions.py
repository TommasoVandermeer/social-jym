import jax.numpy as jnp
from jax import lax, jit, random, vmap, debug
from jax_tqdm import loop_tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from socialjym.envs.base_env import BaseEnv
from socialjym.policies.base_policy import BasePolicy

@jit
def epsilon_scaling_decay(epsilon_start:float, epsilon_end:float, current_episode:int, decay_rate:float) -> float:
    epsilon = lax.cond(current_episode < decay_rate, lambda x: epsilon_start + (epsilon_end - epsilon_start) / decay_rate * x, lambda x: epsilon_end, current_episode)
    return epsilon

@jit
def is_multiple(number:float, dividend:float, tolerance:float=1e-7) -> bool:
    """
    Checks if a number (also a float) is a multiple of another number within a given tolerance error.
    """
    mod = number % dividend
    return jnp.any(jnp.array([abs(mod) <= tolerance,abs(dividend - mod) <= tolerance]))

@jit
def point_to_line_distance(point:jnp.ndarray, line_start:jnp.ndarray, line_end:jnp.ndarray) -> float:
    """
    Computes the distance between a point and a line defined by two points.

    args:
    - point: jnp.ndarray, shape=(2,), dtype=jnp.float32. The point to compute the distance from.
    - line_start: jnp.ndarray, shape=(2,), dtype=jnp.float32. The starting point of the line.
    - line_end: jnp.ndarray, shape=(2,), dtype=jnp.float32. The ending point of the line.

    output:
    - distance: float. The distance between the point and the line.
    """
    x = point[0]
    y = point[1]
    x1 = line_start[0]
    y1 = line_start[1]
    x2 = line_end[0]
    y2 = line_end[1]
    dx = x2 - x1
    dy = y2 - y1

    u = lax.cond(
        jnp.all(jnp.array([dx == 0, dy == 0])), 
        lambda _: 0., 
        lambda _: jnp.squeeze(((x - x1) * dx + (y - y1) * dy) / jnp.linalg.norm(line_end - line_start)**2), 
        None)

    # Clamp u to [0,1]
    u = lax.cond(u < 0, lambda x: 0., lambda x: x, u)
    u = lax.cond(u > 1, lambda x: 1., lambda x: x, u)

    closest_point = jnp.array([x1 + u * dx, y1 + u * dy])
    closest_distance = jnp.linalg.norm(closest_point - point)

    return closest_distance

@jit
def batch_point_to_line_distance(points:jnp.ndarray, line_starts:jnp.ndarray, line_ends:jnp.ndarray) -> jnp.ndarray:
    return vmap(point_to_line_distance, in_axes=(0, 0, 0))(points, line_starts, line_ends)

def plot_state(
        ax:Axes, 
        time:float, 
        full_state:tuple, 
        humans_policy:str, 
        scenario:str, 
        humans_radiuses:np.ndarray, 
        robot_radius:float, 
        circle_radius=7, 
        traffic_height=3, 
        traffic_length=14, 
        plot_time=True):
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
    colors = list(mcolors.TABLEAU_COLORS.values())
    num = int(time) if (time).is_integer() else (time)
    if scenario == 'circular_crossing': ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
    elif scenario == 'parallel_traffic': ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_height-1,traffic_height+1])
    # Humans
    for h in range(len(full_state)-1): 
        if humans_policy == 'hsfm': 
            head = plt.Circle((full_state[h,0] + np.cos(full_state[h,4]) * humans_radiuses[h], full_state[h,1] + np.sin(full_state[h,4]) * humans_radiuses[h]), 0.1, color=colors[h%len(colors)], zorder=1)
            ax.add_patch(head)
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor=colors[h%len(colors)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        if plot_time: ax.text(full_state[h,0],full_state[h,1], f"{num}", color=colors[h%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
        else: ax.text(full_state[h,0],full_state[h,1], f"{h}", color=colors[h%len(colors)], va="center", ha="center", size=10, zorder=1, weight='bold')
    # Robot
    circle = plt.Circle((full_state[-1,0],full_state[-1,1]), robot_radius, edgecolor="red", facecolor="red", fill=True, zorder=1)
    ax.add_patch(circle)
    if plot_time: ax.text(full_state[-1,0],full_state[-1,1], f"{num}", color=colors[(len(full_state)-1)%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
    else: ax.text(full_state[-1,0],full_state[-1,1], f"R", color="black", va="center", ha="center", size=10, zorder=1, weight='bold')

def plot_trajectory(ax:Axes, all_states:jnp.ndarray, humans_goal:jnp.ndarray, robot_goal:jnp.ndarray):
    colors = list(mcolors.TABLEAU_COLORS.values())
    n_agents = len(all_states[0])
    for h in range(n_agents): 
        ax.plot(all_states[:,h,0], all_states[:,h,1], color=colors[h%len(colors)] if h < n_agents - 1 else "red", linewidth=1, zorder=0)
        if h < n_agents - 1: ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=colors[h%len(colors)], zorder=2)
        else: ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)

def test_k_trials(k: int, random_seed: int, env: BaseEnv, policy: BasePolicy, model_params: dict) -> tuple:

    @loop_tqdm(k)
    @jit
    def _fori_body(i:int, for_val:tuple):
         
        @jit
        def _while_body(while_val:tuple):
            # Retrieve data from the tuple
            state, obs, info, done, policy_key, steps, all_states, all_dones, all_rewards = while_val
            # Make a step in the environment
            action, policy_key, _ = policy.act(policy_key, obs, info, model_params, 0.)
            state, obs, info, reward, done = env.step(state,info,action) 
            # Save data
            all_states = all_states.at[steps].set(state)
            all_dones = all_dones.at[steps].set(done)
            all_rewards = all_rewards.at[steps].set(reward)
            # Update step counter
            steps += 1
            return state, obs, info, done, policy_key, steps, all_states, all_dones, all_rewards

        # Retrieve data from the tuple
        reset_key, policy_key, metrics = for_val
        # Reset the environment
        state, reset_key, obs, info = env.reset(reset_key)
        # Episode loop
        all_states = jnp.empty((int(env.time_limit/env.robot_dt)+1, env.n_humans+1, 6))
        all_dones = jnp.empty((int(env.time_limit/env.robot_dt)+1,))
        all_rewards = jnp.empty((int(env.time_limit/env.robot_dt)+1,))
        while_val_init = (state, obs, info, False, policy_key, 0, all_states, all_dones, all_rewards)
        _, _, _, _, policy_key, episode_steps, all_states, all_dones, all_rewards = lax.while_loop(lambda x: x[3] == False, _while_body, while_val_init)
        # Update metrics
        @jit
        def _compute_state_value_for_body(j:int, t:int, value:float):
            value += pow(policy.gamma, (j-t) * policy.dt * policy.v_max) * all_rewards[j]
            return value 
        success = (all_rewards[episode_steps-1] == 1)
        metrics["successes"] = lax.cond(success, lambda x: x + 1, lambda x: x, metrics["successes"])
        metrics["collisions"] = lax.cond(all_rewards[episode_steps-1] == -0.25, lambda x: x + 1, lambda x: x, metrics["collisions"])
        metrics["timeouts"] = lax.cond(jnp.all(jnp.array([all_rewards[episode_steps-1] != 1., all_rewards[episode_steps-1] != -0.25])), lambda x: x + 1, lambda x: x, metrics["timeouts"])
        metrics["returns"] = metrics["returns"].at[i].set(lax.fori_loop(0, episode_steps, lambda k, val: _compute_state_value_for_body(k, 0, val), 0.))
        metrics["times_to_goal"] = lax.cond(success, lambda x: x.at[i].set((episode_steps-1) * env.robot_dt), lambda x: x.at[i].set(jnp.nan), metrics["times_to_goal"])
        return reset_key, policy_key, metrics
    
    # Initialize the random keys
    policy_key, reset_key = vmap(random.PRNGKey)(jnp.arange(2) + random_seed)
    # Initialize metrics
    metrics = {
        "successes": 0, 
        "collisions": 0, 
        "timeouts": 0, 
        "returns": jnp.empty((k,)),
        "times_to_goal": jnp.empty((k,))}
    # Execute k tests
    print(f"Executing {k} tests with {env.n_humans} humans...")
    _, _, metrics = lax.fori_loop(0, k, _fori_body, (reset_key, policy_key, metrics))
    print(f"Success rate: {round(metrics['successes']/k, 2)}")
    print(f"Collision rate: {round(metrics['collisions']/k, 2)}")
    print(f"Timeout rate: {round(metrics['timeouts']/k, 2)}")
    print(f"Average return: {round(jnp.mean(metrics['returns']), 2)}")
    print(f"Average time to goal: {round(jnp.nanmean(metrics['times_to_goal']), 2)}")
    return metrics
                
def animate_trajectory(
    states:jnp.ndarray, 
    humans_radiuses:np.ndarray, 
    robot_radius:float, 
    humans_policy:str,
    robot_goal:np.ndarray,
    robot_dt:float=0.25,
    ) -> None:
    # TODO: Improve this method: 
    #           - add a progress bar,
    #           - add scenario and HMM args to plot accordingly.

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.78, top=0.90, bottom=0.05)
    ax.set_aspect('equal')
    ax.set(xlim=[-10,10],ylim=[-10,10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def animate(frame):
        ax.clear()
        ax.set_title(f"Time: {'{:.2f}'.format(round(frame*robot_dt,2))} - Humans policy: {humans_policy.upper()}", weight='bold')
        ax.legend(
            handles=[
                Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Robot'), 
                Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='blue', linewidth=2, label='Humans'),
                Line2D([0], [0], color='white', marker='*', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Goal')],
            bbox_to_anchor=(0.99, 0.5), loc='center left')
        ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)
        plot_state(ax, frame*robot_dt, states[frame], humans_policy, "circular_crossing", humans_radiuses, robot_radius, plot_time=False)

    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=len(states))
    
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused

    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()