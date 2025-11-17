import jax.numpy as jnp
import matplotlib.pyplot as plt
import os 
import pickle
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset_2.pkl'), 'rb') as file:
    raw_data = pickle.load(file)

# data = {
#     "episode_starts": jnp.zeros((n_steps,), dtype=bool),
#     "lidar_measurements": jnp.zeros((n_steps,lidar_num_rays,2)),
#     "humans_positions": jnp.zeros((n_steps,n_humans,2)),
#     "humans_velocities": jnp.zeros((n_steps,n_humans,2)),
#     "humans_orientations": jnp.zeros((n_steps,n_humans)),
#     "humans_radii": jnp.zeros((n_steps,n_humans)),
#     "robot_positions": jnp.zeros((n_steps,2)),
#     "robot_orientations": jnp.zeros((n_steps,)),
#     "robot_actions": jnp.zeros((n_steps,2)),
#     "robot_goals": jnp.zeros((n_steps,2)),
#     "static_obstacles": jnp.zeros((n_steps,n_obstacles,1,2,2)),
# }

n_steps = raw_data['humans_positions'].shape[0]

rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
p_visualization_threshold = 0.01
fig, ax = plt.subplots(1,1,figsize=(14,10))
fig.subplots_adjust(left=0.05, right=0.90, wspace=0.13)
def animate(frame):
    ax.clear()
    ax.set(xlim=[-10,10], ylim=[-10,10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y', labelpad=-13)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(
        handles=[
            Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='red', markeredgecolor='black', linewidth=2, label='Robot'), 
            Line2D([0], [0], color='white', marker='o', markersize=10, markerfacecolor='blue', markeredgecolor='black', linewidth=2, label='Humans'),
            Line2D([0], [0], color='white', marker='*', markersize=10, markerfacecolor='red', markeredgecolor='red', linewidth=2, label='Goal')
        ],
        bbox_to_anchor=(0.99, 0.5), 
        loc='center left',
    )
    # Plot humans
    for h in range(len(raw_data["humans_positions"][frame])):
        head = plt.Circle((raw_data["humans_positions"][frame][h,0] + jnp.cos(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h], raw_data["humans_positions"][frame][h,1] + jnp.sin(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h]), 0.1, color='black', zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((raw_data["humans_positions"][frame][h,0], raw_data["humans_positions"][frame][h,1]), raw_data['humans_radii'][frame][h], edgecolor='black', facecolor='blue', fill=True, zorder=1)
        ax.add_patch(circle)
    # Plot robot
    head = plt.Circle((raw_data["robot_positions"][frame][0] + jnp.cos(raw_data["robot_orientations"][frame]) * 0.3, raw_data["robot_positions"][frame][1] + jnp.sin(raw_data["robot_orientations"][frame]) * 0.3), 0.1, color='black', zorder=1)
    ax.add_patch(head)
    circle = plt.Circle((raw_data["robot_positions"][frame][0], raw_data["robot_positions"][frame][1]), 0.3, edgecolor="black", facecolor="red", fill=True, zorder=3)
    ax.add_patch(circle)
    # Plot robot goal
    ax.scatter(raw_data["robot_goals"][frame,0], raw_data["robot_goals"][frame,1], marker="*", color="red", zorder=2)
    # Plot static obstacles
    for i, o in enumerate(raw_data["static_obstacles"][frame]):
        for j, s in enumerate(o):
            color = 'black' 
            linestyle = 'solid' 
            alpha = 1
            ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
anim = FuncAnimation(fig, animate, interval=0.25*1000, frames=n_steps)
anim.paused = False
def toggle_pause(self, *args, **kwargs):
    if anim.paused: anim.resume()
    else: anim.pause()
    anim.paused = not anim.paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)
plt.show()