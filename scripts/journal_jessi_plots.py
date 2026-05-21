import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward
from socialjym.policies.jessi import JESSI

### PARAMETERS
L = 0.7 # Distance between the wheels of the robot
v_max = 1. # Maximum linear velocity of the robot
dt = 0.75
radius = 0.3
n_actions_per_dim = 50 # Number of actions per dimension to plot the action space boundaries
n_points_per_per_circle = 50 # Number of points to plot the circles around the envelope points
obstacles = jnp.array([
    [[-0.2, 0.4],[-0.2, 0.37]],
    [[-0.2, 0.37],[0.67, 0.37]],
    [[0.67, 0.37],[0.67, -0.4]],
    [[0.67, -0.4],[0.7, -0.4]],
    [[0.7, -0.4],[0.7, 0.4]],
    [[0.7, 0.4],[-0.2, 0.4]],
])
lidar_num_rays = 20
lidar_angular_range = jnp.pi

### UTILS
w_max = 2*v_max/L # Maximum angular velocity of the robot, given the maximum linear velocity and the distance between the wheels

### FIG1: action_space.eps
figure, ax = plt.subplots(1,1, figsize=(10, 3))
figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25)
ax.add_patch(
    plt.Polygon(
        [   
            [w_max,0],
            [-w_max,0],
            [0.,v_max],
        ],
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=2,
    ),
)
ax.set_xticks([w_max, 0., -w_max])
ax.set_xticklabels([r"$\overline{\omega}$", "0", r"$-\overline{\omega}$"])
ax.set_yticks([0.,v_max])
ax.set_yticklabels(["0", r"$\overline{v}$"])
ax.set_ylim(-0.1, v_max + 0.1)
ax.set_xlim(-w_max - 0.3, w_max + 0.3)
ax.set_ylabel("$v$ (m/s)", labelpad=-15)
ax.set_xlabel("$\omega$ (rad/s)", labelpad=-5)
ax.plot([-10, 10], [0, 0], color='black', linewidth=3, zorder=5)
ax.text(0, 0.1, "$v \geq 0$", zorder=5, verticalalignment='bottom', horizontalalignment='center')
ax.plot([-(w_max)*2, w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
# ax.text(-2.5 , L, r"$\omega \leq \frac{\overline{\omega}}{\overline{v}}v - \overline{\omega}$", zorder=5, verticalalignment='center', horizontalalignment='left')
ax.text(-2.5 , L, r"$\omega \geq \frac{2(v-\overline{v})}{L}$", zorder=5, verticalalignment='center', horizontalalignment='left')
ax.plot([(w_max)*2, -w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
# ax.text(2.5 , L, r"$\omega \leq \overline{\omega} - \frac{\overline{\omega}}{\overline{v}}v$", zorder=5, verticalalignment='center', horizontalalignment='right')
ax.text(2.5 , L, r"$\omega \leq \frac{2(\overline{v}-v)}{L}$", zorder=5, verticalalignment='center', horizontalalignment='right')
ax.grid()
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space.eps'), format='eps')
plt.close()

### FIG2: action_space_2.eps
figure, ax = plt.subplots(1,1, figsize=(10, 3))
figure.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.27)
ax.add_patch(
    plt.Polygon(
        [   
            [w_max,0],
            [-w_max,0],
            [0.,v_max],
        ],
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=2,
    ),
)
ax.set_xticks([w_max, 0., -w_max])
ax.set_xticklabels([r"$\frac{2 \bar{\phi} \rho}{L}$", "0", r"$-\frac{2 \bar{\phi} \rho}{L}$"])
ax.set_yticks([0.,v_max])
ax.set_yticklabels(["0", r"$\bar{\phi} \rho$"])
ax.set_ylim(-0.1, v_max + 0.1)
ax.set_xlim(-w_max - 0.3, w_max + 0.3)
ax.set_ylabel("$v$ (m/s)", labelpad=-15)
ax.set_xlabel("$\omega$ (rad/s)", labelpad=-5)
ax.plot([-10, 10], [0, 0], color='black', linewidth=3, zorder=5)
ax.text(0, 0.1, "$v \geq 0$", zorder=5, verticalalignment='bottom', horizontalalignment='center')
ax.plot([-(w_max)*2, w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
ax.text(-2.5 , L, r"$\omega \geq \frac{2(v-\bar{\phi} \rho)}{L}$", zorder=5, verticalalignment='center', horizontalalignment='left')
ax.plot([(w_max)*2, -w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
ax.text(2.5 , L, r"$\omega \leq \frac{2(\bar{\phi} \rho-v)}{L}$", zorder=5, verticalalignment='center', horizontalalignment='right')
ax.grid()
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_2.eps'), format='eps')
plt.close()

### FIG3: action_space_bounding_1.eps & action_space_bounding_2.eps & action_space_bounding_3.eps & action_space_bounding_4.eps
policy = JESSI(
    radius, 
    v_max, 
    dt, 
    L,
    5,
    lidar_angular_range,
    10.,
    lidar_num_rays,
)
env_params = {
    'n_stack': 5,
    'lidar_num_rays': lidar_num_rays,
    'lidar_angular_range': lidar_angular_range,
    'lidar_max_dist': 10.,
    'n_humans': 1, #5,
    'n_obstacles': 0, #5,
    'robot_radius': 0.3,
    'robot_dt': dt,
    'humans_dt': 0.01,
    'robot_visible': True,
    'scenario': 'hybrid_scenario',
    'reward_function': DummyReward(robot_radius=0.3, time_limit=10),
    'kinematics': 'unicycle',
}
env = LaserNav(**env_params)
vs = jnp.concatenate([
    jnp.zeros(n_actions_per_dim),
    jnp.linspace(0, v_max, n_actions_per_dim),
    jnp.linspace(0, v_max, n_actions_per_dim),
])
ws = jnp.concatenate([
    jnp.linspace(-w_max, w_max, n_actions_per_dim),
    jnp.linspace(-w_max, 0, n_actions_per_dim),
    jnp.linspace(w_max, 0, n_actions_per_dim),
])
actions = jnp.stack((vs, ws), axis=-1)
displacements = jnp.array([[
    a[0]/a[1] * jnp.sin(a[1]*dt) if a[1] != 0 else a[0]*dt,
    a[0]/a[1] * (1 - jnp.cos(a[1]*dt)) if a[1] != 0 else 0,
] for a in actions])
envelope_points = jnp.array([[
    c + jnp.array([radius * jnp.cos(theta), radius * jnp.sin(theta)]) for theta in jnp.linspace(0, 2*jnp.pi, n_points_per_per_circle)
] for c in displacements]).reshape(-1, 2)
hull = ConvexHull(envelope_points)
angles = jnp.linspace(-lidar_angular_range/2, lidar_angular_range/2, lidar_num_rays)
directions = jnp.array([
    jnp.array([jnp.cos(angle), jnp.sin(angle)]) for angle in angles
])
dist, _ = vmap(env._obstacle_ray_intersect, in_axes=(0, None, None))(
    directions,
    obstacles,
    jnp.array([0., 0.]),
)
collision_points = jnp.array([dist[i] * directions[i] for i in range(lidar_num_rays)])
figure, ax = plt.subplots(1,1,figsize=(11, 8))
figure.subplots_adjust(left=0.08, right=1, top=0.99, bottom=0.10, wspace=0.1)
ax.set_aspect('equal')
ax.fill(envelope_points[hull.vertices, 0], envelope_points[hull.vertices, 1], facecolor='lightcoral', edgecolor='red', zorder=2)
ax.fill(obstacles[:,:,0],obstacles[:,:,1], facecolor='black', edgecolor='black', zorder=7)
ax.add_artist(plt.Circle((0, 0), radius, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
for i in range(lidar_num_rays):
    ax.plot([0, collision_points[i, 0]], [0, collision_points[i, 1]], color='blue', linewidth=1, zorder=5)
ax.scatter(collision_points[:,0], collision_points[:,1], color='blue', s=150, zorder=8, marker='x')
ax.set_xlim(-radius - 0.05, v_max * dt + radius + 0.05)
ax.set_ylim(-0.55, 0.55)
ax.set_xlabel("$\Delta x$ (m)")
ax.set_ylabel("$\Delta y$ (m)", labelpad=-5)
ax.grid()
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_1.eps'), format='eps')
plt.close()
figure, ax = plt.subplots(1,1,figsize=(4, 8))
figure.subplots_adjust(left=0.18, right=0.97, top=0.98, bottom=0.10, wspace=0.1)
ax.add_patch(
    plt.Polygon(
        [   
            [0,w_max],
            [0,-w_max],
            [v_max,0],
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
ax.set_xlim(-0.1, v_max + 0.1)
ax.set_ylim(-w_max - 0.1, w_max + 0.1)
ax.set_xlabel("$v$ (m/s)")
ax.set_ylabel("$\omega$ (rad/s)", labelpad=-20)
ax.grid()
ax.set_xticks(jnp.arange(0, v_max+0.5, 0.5))
ax.set_xticklabels([round(i,1) for i in jnp.arange(0, v_max, 0.5)] + [r"$\overline{v}$"])
ax.set_yticks(jnp.arange(-2,3,1).tolist() + [w_max,-w_max])
ax.set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_2.eps'), format='eps')
plt.close()
alpha, beta, gamma = policy.bound_action_space(collision_points)
vs = jnp.concatenate([
    jnp.zeros(n_actions_per_dim),
    jnp.linspace(0, alpha * v_max, n_actions_per_dim),
    jnp.linspace(0, alpha * v_max, n_actions_per_dim),
])
ws = jnp.concatenate([
    jnp.linspace(-gamma * w_max, beta * w_max, n_actions_per_dim),
    jnp.linspace(-gamma * w_max, 0, n_actions_per_dim),
    jnp.linspace(beta * w_max, 0, n_actions_per_dim),
])
actions = jnp.stack((vs, ws), axis=-1)
displacements = jnp.array([[
    a[0]/a[1] * jnp.sin(a[1]*dt) if a[1] != 0 else a[0]*dt,
    a[0]/a[1] * (1 - jnp.cos(a[1]*dt)) if a[1] != 0 else 0,
] for a in actions])
envelope_points = jnp.array([[
    c + jnp.array([radius * jnp.cos(theta), radius * jnp.sin(theta)]) for theta in jnp.linspace(0, 2*jnp.pi, n_points_per_per_circle)
] for c in displacements]).reshape(-1, 2)
hull = ConvexHull(envelope_points)
figure, ax = plt.subplots(1,1,figsize=(9, 8))
figure.subplots_adjust(left=0.10, right=1, top=0.99, bottom=0.10, wspace=0.1)
ax.set_aspect('equal')
ax.fill(envelope_points[hull.vertices, 0], envelope_points[hull.vertices, 1], facecolor='lightgreen', edgecolor='green', zorder=2)
ax.fill(obstacles[:,:,0],obstacles[:,:,1], facecolor='black', edgecolor='black', zorder=7)
ax.add_artist(plt.Circle((0, 0), radius, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
for i in range(lidar_num_rays):
    ax.plot([0, collision_points[i, 0]], [0, collision_points[i, 1]], color='blue', linewidth=1, zorder=5)
ax.scatter(collision_points[:,0], collision_points[:,1], color='blue', s=150, zorder=8, marker='x')
ax.set_xlim(-radius - 0.05, alpha * v_max * dt + radius + 0.05)
ax.set_ylim(-0.55, 0.45)
ax.set_xlabel("$\Delta x$ (m)")
ax.set_ylabel("$\Delta y$ (m)", labelpad=-5)
ax.grid()
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_3.eps'), format='eps')
plt.close()