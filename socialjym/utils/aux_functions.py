import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes

def is_multiple(number, dividend, tolerance=1e-7) -> bool:
    """
    Checks if a number (also a float) is a multiple of another number within a given tolerance error.
    """
    mod = number % dividend
    return (abs(mod) <= tolerance) or (abs(dividend - mod) <= tolerance)

def plot_state(ax:Axes, time:float, full_state:tuple, humans_policy:str, scenario:str, humans_radiuses:np.ndarray, robot_radius:float, circle_radius=7, traffic_height=3, traffic_length=14):
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
        ax.text(full_state[h,0],full_state[h,1], f"{num}", color=colors[h%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
    # Robot
    circle = plt.Circle((full_state[-1,0],full_state[-1,1]), robot_radius, edgecolor="red", facecolor="red", fill=True, zorder=1)
    ax.add_patch(circle)
    ax.text(full_state[-1,0],full_state[-1,1], f"{num}", color=colors[(len(full_state)-1)%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')

def plot_trajectory(ax:Axes, all_states:jnp.ndarray, humans_goal:jnp.ndarray, robot_goal:jnp.ndarray):
    colors = list(mcolors.TABLEAU_COLORS.values())
    n_agents = len(all_states[0])
    for h in range(n_agents): 
        ax.plot(all_states[:,h,0], all_states[:,h,1], color=colors[h%len(colors)] if h < n_agents - 1 else "red", linewidth=1, zorder=0)
        if h < n_agents - 1: ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=colors[h%len(colors)], zorder=2)
        else: ax.scatter(robot_goal[0], robot_goal[1], marker="*", color="red", zorder=2)