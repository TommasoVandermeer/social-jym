import jax.numpy as jnp
from jax import vmap, random
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
from jhsfm.hsfm import get_linear_velocity

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

### FIG1: action_space.eps
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space.eps')):
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
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_2.eps')):
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
if (not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_bounding_1.eps'))) or \
   (not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_bounding_2.eps'))) or \
   (not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_bounding_3.eps'))) or \
   (not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_bounding_4.eps'))) or \
   (not os.path.exists(os.path.join(os.path.dirname(__file__), 'action_space_bounding_summary.eps'))):
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
    figure, ax = plt.subplots(1,1,figsize=(11, 8))
    figure.subplots_adjust(left=0.10, right=1, top=0.99, bottom=0.10, wspace=0.1)
    ax.set_aspect('equal')
    ax.fill(envelope_points[hull.vertices, 0], envelope_points[hull.vertices, 1], facecolor='lightgreen', edgecolor='green', zorder=2)
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
    figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_3.eps'), format='eps')
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
            label='Original set'
        ),
    )
    ax.add_patch(
        plt.Polygon(
            [   
                [0,beta*w_max],
                [0,-gamma*w_max],
                [alpha*v_max,0],
            ],
            closed=True,
            fill=True,
            edgecolor='green',
            facecolor='lightgreen',
            linewidth=2,
            zorder=2,
            label='Feasible set'
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
    ax.legend(fontsize=16.5, loc='upper right')
    figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_4.eps'), format='eps')
    plt.close()

### FIG4: action_space_bounding_summary.eps
    figure = plt.figure(figsize=(13, 8.25))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.1)
    gs = figure.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], wspace=0., hspace=0.)
    ax00 = figure.add_subplot(gs[0, 0])
    ax01 = figure.add_subplot(gs[0, 1])
    ax10 = figure.add_subplot(gs[1, 0])
    ax11 = figure.add_subplot(gs[1, 1])
    ax_right = figure.add_subplot(gs[:, 2])
    pos = ax_right.get_position()
    ax_right.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    for ax in (ax00, ax01, ax10, ax11):
        ax.set_aspect('equal')
        ax.set_xlim(-radius - 0.05, v_max * dt + radius + 0.05)
        ax.set_ylim(-0.55, 0.65)
        ax.grid()
        ax.add_artist(plt.Circle((0, 0), radius, color='black', fill=False, zorder=100, linewidth=2, linestyle='--'))
        ax.scatter(collision_points[:,0], collision_points[:,1], color='blue', s=150, zorder=101, marker='x')
    for ax in (ax00, ax10):
        ax.set_ylabel("$\Delta y$ (m)", labelpad=-5)
    for ax in (ax10, ax11):
        ax.set_xlabel("$\Delta x$ (m)")
    for ax in (ax00, ax01):
        ax.set_xticklabels([])
    for ax in (ax01, ax11):
        ax.set_yticklabels([])
    ax_right.set_xlim(-0.1, v_max + 0.1)
    ax_right.set_ylim(-w_max - 0.1, w_max + 0.1)
    ax_right.set_xlabel("$v$ (m/s)")
    ax_right.set_ylabel("$\omega$ (rad/s)", labelpad=-25)
    ax_right.grid()
    ax_right.set_xticks(jnp.arange(0, v_max+0.5, 0.5))
    ax_right.set_xticklabels([round(i,1) for i in jnp.arange(0, v_max, 0.5)] + [r"$\overline{v}$"])
    ax_right.set_yticks(jnp.arange(-2,3,1).tolist() + [w_max,-w_max])
    ax_right.set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
    ## AX (0,0)
    ax00.text(-0.3, 0.58, "Unbounded displacements", verticalalignment='center', horizontalalignment='left', fontsize=18, fontweight='bold')
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
    ax00.set_aspect('equal')
    ax00.fill(envelope_points[hull.vertices, 0], envelope_points[hull.vertices, 1], facecolor='lightcoral', edgecolor='red', zorder=2)
    ax00.fill(obstacles[:,:,0],obstacles[:,:,1], facecolor='black', edgecolor='black', zorder=7)
    for i in range(lidar_num_rays):
        ax00.plot([0, collision_points[i, 0]], [0, collision_points[i, 1]], color='blue', linewidth=1, zorder=5)
    ## AX (0,1)
    ax01.text(-0.3, 0.58, "Bounded displacements", verticalalignment='center', horizontalalignment='left', fontsize=18, fontweight='bold')
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
    ax01.set_aspect('equal')
    ax01.fill(envelope_points[hull.vertices, 0], envelope_points[hull.vertices, 1], facecolor='lightgreen', edgecolor='green', zorder=2)
    ax01.fill(obstacles[:,:,0],obstacles[:,:,1], facecolor='black', edgecolor='black', zorder=7)
    for i in range(lidar_num_rays):
        ax01.plot([0, collision_points[i, 0]], [0, collision_points[i, 1]], color='blue', linewidth=1, zorder=5)
    ## AX (:,2)
    ax_right.add_patch(
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
            label='Original set'
        ),
    )
    ax_right.add_patch(
        plt.Polygon(
            [   
                [0,beta*w_max],
                [0,-gamma*w_max],
                [alpha*v_max,0],
            ],
            closed=True,
            fill=True,
            edgecolor='green',
            facecolor='lightgreen',
            linewidth=2,
            zorder=2,
            label='Feasible set'
        ),
    )
    # ax_right.legend(fontsize=16.5, loc='upper right')
    ## AX (1,0)
    ax10.text(-0.3, 0.58, r"Stage 1: Reduce $\alpha$", verticalalignment='center', horizontalalignment='left', fontsize=18, fontweight='bold')
    ax10.add_artist(
        plt.Rectangle(
            (-radius, -dt**2*v_max**2/(4*L) - radius), 
            v_max*dt + 2 * radius, 
            2*radius + (dt**2*v_max**2/(4*L) * 2), 
            color='red', 
            fill=False, 
            zorder=3, 
            linewidth=2,
        )
    )
    ax10.add_artist(
        plt.Rectangle(
            (-radius,-radius), 
            v_max*dt + 2 * radius, 
            2*radius, 
            edgecolor='red', 
            fill=False, 
            zorder=3, 
            linewidth=2,
        )
    )
    ax10.add_artist(
        plt.Rectangle(
            (0,-radius), 
            v_max*dt + radius, 
            2*radius, 
            edgecolor='red', 
            fill=True, 
            facecolor='lightgrey',
            zorder=3, 
            linewidth=2
        )
    )
    def is_inside_box(point, box):
        x, y = point
        x_min, y_min = box[0]
        x_max, y_max = box[1]
        return x_min <= x <= x_max and y_min <= y <= y_max
    is_in = [is_inside_box(p, [(0, -radius), (v_max*dt + radius, radius)]) for p in collision_points]
    for i, p in enumerate(collision_points):
        ax10.scatter(p[0], p[1], color='darkgreen' if is_in[i] else 'blue', s=150, zorder=101, marker='x')
    def segment(ax, xy0, xy1, label, label_pos=None, color='black'):
        ax.plot([xy0[0],xy1[0]], [xy0[1],xy1[1]], color=color, zorder=8, linewidth=2)
        if xy0[0] == xy1[0]: # Vertical segment
            marker = '_'
            label_pos = label_pos if label_pos is not None else (xy0[0] + 0.03, (xy0[1] + xy1[1]) / 2)
            ax.text(label_pos[0], label_pos[1], label, verticalalignment='center', horizontalalignment='left', color=color, zorder=8, fontsize=16)
        elif xy0[1] == xy1[1]: # Horizontal segment
            marker = '|'
            label_pos = label_pos if label_pos is not None else ((xy0[0] + xy1[0]) / 2, xy0[1]+0.01)
            ax.text(label_pos[0], label_pos[1], label, verticalalignment='bottom', horizontalalignment='center', color=color, zorder=8, fontsize=16)
        else: # Diagonal segment
            marker = 'x'
            label_pos = label_pos if label_pos is None else ((xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2)
            ax.text(label_pos[0], label_pos[1], label, verticalalignment='center', horizontalalignment='center', color=color, zorder=8, fontsize=16)
        ax.scatter([xy0[0],xy1[0]], [xy0[1],xy1[1]], color=color, s=50, zorder=8, marker=marker)
    segment(ax10, [0.,0.], [0.,-radius], '$r$', color='black')
    segment(ax10, [radius,0.], [v_max*dt + radius,0.], '$\Delta x_{\max}$', label_pos=(0.85,0.01), color='black')
    ax10.text(-radius/2, 0, r'$\mathcal{B}_0$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    ax10.text(-radius/2, -radius-(dt**2*v_max/(4*L))/2, r'$\mathcal{B}_{\gamma}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    ax10.text(-radius/2, radius+(dt**2*v_max/(4*L))/2+0.02, r'$\mathcal{B}_{\beta}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    ax10.text((radius + v_max * dt)/2, -0.05, r'$\mathcal{B}_{\alpha}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    ## AX (1,1)
    ax11.text(-0.3, 0.58, r"Stage 2: Reduce $\beta$ and $\gamma$", verticalalignment='center', horizontalalignment='left', fontsize=18, fontweight='bold')
    ax11.add_artist(
        plt.Rectangle(
            (-radius, -alpha*dt**2*v_max**2/(4*L) - radius), 
            alpha*v_max*dt + 2 * radius, 
            2*radius + (alpha*dt**2*v_max**2/(4*L) * 2), 
            color='red', 
            fill=False, 
            zorder=3, 
            linewidth=2,
        )
    )
    ax11.add_artist(
        plt.Rectangle(
            (-radius,radius), 
            alpha*v_max*dt + 2 * radius, 
            (alpha*dt**2*v_max/(4*L)), 
            edgecolor='red', 
            fill=True, 
            facecolor='lightgrey',
            zorder=3, 
            linewidth=2,
        )
    )
    ax11.add_artist(
        plt.Rectangle(
            (-radius,-alpha*dt**2*v_max/(4*L) - radius), 
            alpha*v_max*dt + 2 * radius, 
            (alpha*dt**2*v_max/(4*L)), 
            edgecolor='red', 
            fill=True, 
            facecolor='lightgrey', 
            zorder=3, 
            linewidth=2,
        )
    )
    is_in = [is_inside_box(p, [(-radius, radius), (alpha*v_max*dt + radius, alpha*dt**2*v_max/(4*L) + radius)]) or \
            is_inside_box(p, [(-radius, -alpha*dt**2*v_max/(4*L) - radius), (alpha*v_max*dt + radius, - radius)]) for p in collision_points]
    for i, p in enumerate(collision_points):
        ax11.scatter(p[0], p[1], color='darkgreen' if is_in[i] else 'blue', s=150, zorder=101, marker='x')
    segment(ax11, [radius,0.], [alpha*v_max*dt + radius,0.], r'$\tilde{\Delta} x_{\max}$', color='black')
    segment(ax11, [0.,0.], [0.,-radius], '$r$', color='black')
    segment(
        ax11, 
        [(alpha*v_max*dt + 2*radius)/2-radius-0.3,-radius], 
        [(alpha*v_max*dt + 2*radius)/2-radius-0.3,-radius-(alpha*dt**2*v_max/(4*L))], 
        r'$|\Delta y_{\min}|$', 
        label_pos=((alpha*v_max*dt + 2*radius)/2-radius-0.3,-radius-(alpha*dt**2*v_max/(4*L))-0.05), 
        color='black'
    )
    segment(
        ax11, 
        [(alpha*v_max*dt + 2*radius)/2-radius-0.3,+radius], 
        [(alpha*v_max*dt + 2*radius)/2-radius-0.3,+radius+(alpha*dt**2*v_max/(4*L))], 
        r'$\Delta y_{\max}$', 
        label_pos=((alpha*v_max*dt + 2*radius)/2-radius-0.3,+radius+(alpha*dt**2*v_max/(4*L))+0.05), 
        color='black'
    )
    ax11.text(-radius + 0.05, -radius-(gamma*alpha*dt**2*v_max/(4*L))/2 , r'$\mathcal{B}_{\overline{\gamma}}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    ax11.text(-radius + 0.05, radius+(beta*alpha*dt**2*v_max/(4*L))/2 , r'$\mathcal{B}_{\overline{\beta}}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=16)
    figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space_bounding_summary.eps'), format='eps')
    plt.close()

### FIG4: scenarios.eps
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'scenarios.eps')):
    env_params = {
        'n_stack': 5,
        'lidar_num_rays': lidar_num_rays,
        'lidar_angular_range': lidar_angular_range,
        'lidar_max_dist': 10.,
        'n_humans': 5, #5,
        'n_obstacles': 5, #5,
        'robot_radius': 0.3,
        'robot_dt': dt,
        'humans_dt': 0.01,
        'robot_visible': True,
        'scenario': 'parallel_traffic',
        'reward_function': DummyReward(robot_radius=0.3, time_limit=10),
        'kinematics': 'unicycle',
    }
    env = LaserNav(**env_params)
    key = random.PRNGKey(0)
    state, key, obs, info, outcome = env.reset(key)
    # PLOT INITIAL STATE
    figure, ax = plt.subplots(1,1, figsize=(9, 4))
    figure.subplots_adjust(left=0.09, right=0.96, top=0.99, bottom=0.15)
    ax.set(xlim=[-10,10], ylim=[-4,4])
    ax.set_xlabel('X', labelpad=-2)
    ax.set_ylabel('Y', labelpad=-2)
    ax.set_aspect('equal', adjustable='box')
    # Plot humans
    humans_positions = state[:-1,:2]
    humans_orientations = state[:-1,4]
    humans_poses = jnp.concatenate([humans_positions, humans_orientations[:,None]], axis=-1)
    humans_body_velocities = state[:-1,2:4]
    humans_velocities = vmap(get_linear_velocity, in_axes=(0,0))(
        humans_orientations,
        humans_body_velocities,
    )
    for h in range(len(humans_poses)):
        color = 'blue'
        alpha = 0.6
        head = plt.Circle((humans_poses[h,0] + jnp.cos(humans_poses[h,2]) * info['humans_parameters'][h,0], humans_poses[h,1] + jnp.sin(humans_poses[h,2]) * info['humans_parameters'][h,0]), 0.1, color='black', alpha=alpha, zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((humans_poses[h,0], humans_poses[h,1]), info['humans_parameters'][h,0], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
        ax.add_patch(circle)
    # Plot human velocities
    for h in range(len(humans_poses)):
        color = 'blue'
        alpha = 0.6
        ax.arrow(
            humans_poses[h,0],
            humans_poses[h,1],
            humans_velocities[h,0],
            humans_velocities[h,1],
            head_width=0.15,
            head_length=0.15,
            fc=color,
            ec=color,
            alpha=alpha,
            zorder=30,
        )
    # Plot robot
    robot_position = state[-1,:2]
    head = plt.Circle((robot_position[0] + policy.robot_radius * jnp.cos(state[-1,4]), robot_position[1] + policy.robot_radius * jnp.sin(state[-1,4])), 0.1, color='black', zorder=1)
    ax.add_patch(head)
    circle = plt.Circle((robot_position[0], robot_position[1]), policy.robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
    ax.add_patch(circle)
    # Plot robot goal
    ax.plot(
        info['robot_goal'][0],
        info['robot_goal'][1],
        marker='*',
        markersize=7,
        color='red',
        zorder=5,
    )
    # Plot static obstacles
    if info['static_obstacles'][-1].shape[1] > 1: # Polygon obstacles
        for o in info['static_obstacles'][-1]: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in info['static_obstacles'][-1]: ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    figure.savefig(os.path.join(os.path.dirname(__file__), "episode_example.eps"), format='eps')