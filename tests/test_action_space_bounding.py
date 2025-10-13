import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax, jit, vmap, debug
from scipy.spatial import ConvexHull

vmax = 1.
wheels_distance = 0.7
samples = 205
dt = .75
robot_radius = 0.3
# Control parameters
alpha = 1.
beta = 1.
gamma = 1.
# Obstacles
obstacles = jnp.array([
    [[[0.52,0.31],[-0.40,0.31]],[[-0.40,0.31],[-0.40,0.33]],[[-0.40,0.33],[0.52,0.33]],[[0.52,0.33],[0.52,0.31]]],
    [[[0.50,-0.50],[0.50,0.31]],[[0.50,0.31],[0.52,0.31]],[[0.52,0.31],[0.52,-0.50]],[[0.52,-0.50],[0.50,-0.50]]],
])
obstacles = obstacles.at[:,:,:,1].set(obstacles[:,:,:,1] + 0.05)  # Shift obstacles down by 0.05
obstacles = obstacles.at[:,:,:,0].set(obstacles[:,:,:,0] + 0.2) # Shift obstacles right by 0.2

# Exact integration of the action space function
@jit
def exact_integration_of_action_space(x:jnp.ndarray, action:jnp.ndarray) -> jnp.ndarray:
    @jit
    def exact_integration_with_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + action[0] * jnp.cos(x[2]) * dt)
        x = x.at[1].set(x[1] + action[0] * jnp.sin(x[2]) * dt)
        return x
    @jit
    def exact_integration_with_non_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + (action[0]/action[1]) * (jnp.sin(x[2] + action[1] * dt) - jnp.sin(x[2])))
        x = x.at[1].set(x[1] + (action[0]/action[1]) * (jnp.cos(x[2]) - jnp.cos(x[2] + action[1] * dt)))
        x = x.at[2].set(x[2] + action[1] * dt)
        return x
    x = lax.cond(
        action[1] != 0,
        exact_integration_with_non_zero_omega,
        exact_integration_with_zero_omega,
        x)
    return x
angular_speeds = jnp.linspace(-vmax/(wheels_distance/2), vmax/(wheels_distance/2), 2*samples-1, endpoint=True)
speeds = jnp.linspace(0, vmax, samples, endpoint=True)
unconstrained_action_space = jnp.empty((len(angular_speeds)*len(speeds),2))
unconstrained_action_space = lax.fori_loop(
    0,
    len(angular_speeds),
    lambda i, x: lax.fori_loop(
        0,
        len(speeds),
        lambda j, y: lax.cond(
            jnp.all(jnp.array([i<len(angular_speeds)-j, i>=j])),
            lambda z: z.at[i*len(speeds)+j].set(jnp.array([speeds[j],angular_speeds[i]])),
            lambda z: z.at[i*len(speeds)+j].set(jnp.array([jnp.nan,jnp.nan])),
            y),
        x),
    unconstrained_action_space)
# Segment-rectangle intersection function
@jit
def segment_rectangle_intersection(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
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
    intersection_point_0 = lax.switch(
        jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t0 == 0), (inside_or_intersects) & (t0 > 0)])),
        [
            lambda _: jnp.array([jnp.nan, jnp.nan]),
            lambda _: jnp.array([x1, y1]),
            lambda _: jnp.array([x1 + t0 * dx, y1 + t0 * dy]),
        ],
        None,
    )
    intersection_point_1 = lax.switch(
        jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t1 == 1), (inside_or_intersects) & (t1 < 1)])),
        [
            lambda _: jnp.array([jnp.nan, jnp.nan]),
            lambda _: jnp.array([x2, y2]),
            lambda _: jnp.array([x1 + t1 * dx, y1 + t1 * dy]),
        ],
        None,
    )
    return inside_or_intersects, intersection_point_0, intersection_point_1


### Plot the action space (Vx, Vy)
action_space = unconstrained_action_space[~jnp.isnan(unconstrained_action_space).any(axis=1)]
v_w = vmap(exact_integration_of_action_space, in_axes=(0, 0))(jnp.zeros((len(action_space),3)), action_space) / dt
# plt.scatter(v_w[:,0], v_w[:,1])
positive_w = action_space[action_space[:,1] > 0][:,1]
negative_w = action_space[action_space[:,1] < 0][:,1]
second_order_taylor_approximation_positive_vx = alpha * (vmax - (wheels_distance / 2) * positive_w)
second_order_taylor_approximation_negative_vx = alpha * (vmax + (wheels_distance / 2) * negative_w)
second_order_taylor_approximation_positive_vy = (alpha * dt / 2) * (vmax * positive_w - (wheels_distance / 2) * jnp.square(positive_w))
second_order_taylor_approximation_negative_vy = (alpha * dt / 2) * (vmax * negative_w + (wheels_distance / 2) * jnp.square(negative_w))
max_approximated_vy = jnp.array([
    alpha * vmax / 2,
    alpha * dt * vmax**2 / (4 * wheels_distance),
])
min_approximated_vy = jnp.array([
    alpha * vmax / 2,
    -alpha * dt * vmax**2 / (4 * wheels_distance),
])
# plt.scatter(second_order_taylor_approximation_positive_vx, second_order_taylor_approximation_positive_vy, color='red')
# plt.scatter(second_order_taylor_approximation_negative_vx, second_order_taylor_approximation_negative_vy, color='red')
# plt.scatter(max_approximated_vy[0], max_approximated_vy[1], color='green')
# plt.scatter(min_approximated_vy[0], min_approximated_vy[1], color='green')
# plt.xlabel("Vx")
# plt.ylabel("Vy")
# plt.show()

### Plot the position displacements (\Delta x, \Delta y)
from matplotlib import rc, rcParams
font = {'weight' : 'regular',
        'size'   : 38}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
figure, ax = plt.subplots(1, 2, figsize=(25.61, 13.61), gridspec_kw={'width_ratios': [2, 1]})
figure.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.07, wspace=0.1)
figure2, ax2 = plt.subplots(1,1,figsize=(11, 8))
figure2.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.15, wspace=0.1)
figure3, ax3 = plt.subplots(1,1,figsize=(5.5, 8))
figure3.subplots_adjust(left=0.19, right=0.97, top=0.98, bottom=0.15, wspace=0.1)
figure4, ax4 = plt.subplots(1,1,figsize=(11, 8))
figure4.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.15, wspace=0.1)
figure5, ax5 = plt.subplots(1,1,figsize=(5.5, 8))
figure5.subplots_adjust(left=0.19, right=0.97, top=0.98, bottom=0.15, wspace=0.1)
pxy_theta = v_w * dt
# plt.scatter(pxy_theta[:,0], pxy_theta[:,1], s=5, color='pink')
second_order_taylor_approximation_positive_dx = second_order_taylor_approximation_positive_vx * dt
second_order_taylor_approximation_negative_dx = second_order_taylor_approximation_negative_vx * dt
second_order_taylor_approximation_positive_dy = second_order_taylor_approximation_positive_vy * dt
second_order_taylor_approximation_negative_dy = second_order_taylor_approximation_negative_vy * dt
# Plot initial configuration
ax[0].add_artist(plt.Circle((0, 0), robot_radius, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
ax2.add_artist(plt.Circle((0, 0), robot_radius, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
ax4.add_artist(plt.Circle((0, 0), robot_radius, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
# Plot convex hull of all possible new configurations
circles = [(x, y, robot_radius)for x, y in zip(second_order_taylor_approximation_positive_dx[::20], second_order_taylor_approximation_positive_dy[::20])] + [(x, y, robot_radius) for x, y in zip(second_order_taylor_approximation_negative_dx[::20], second_order_taylor_approximation_negative_dy[::20])]
points = []
for x, y, r in circles:
    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    cx = x + r * jnp.cos(theta)
    cy = y + r * jnp.sin(theta)
    points.extend(zip(cx, cy))
points = jnp.array(points)
hull = ConvexHull(points)
ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], facecolor='lightcoral', edgecolor='red', zorder=2)
ax2.fill(points[hull.vertices, 0], points[hull.vertices, 1], facecolor='lightcoral', edgecolor='red', zorder=2)
# Plot obstacles
for i, o in enumerate(obstacles): 
    if o.shape[0] == 1:  # Single segment obstacle
        ax[0].plot(o[0,:,0], o[0,:,1], color='black', linewidth=3, zorder=7)
        ax2.plot(o[0,:,0], o[0,:,1], color='black', linewidth=3, zorder=7)
        ax4.plot(o[0,:,0], o[0,:,1], color='black', linewidth=3, zorder=7)
    else:  # Multiple segments obstacle
        ax[0].fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=7)
        ax2.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=7)
        ax4.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=7)
## Plot displacement boundary rectangle 
ax[0].add_artist(
    plt.Rectangle(
        (-robot_radius, -alpha*dt**2*gamma*vmax**2/(4*wheels_distance) - robot_radius), 
        alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (alpha*dt**2*vmax**2/(4*wheels_distance) * (beta + gamma)), 
        color='red', 
        fill=False, 
        zorder=3, 
        linewidth=3,
        label=r'Possible configurations at time $t + \Delta t$'
    )
)
ax2.add_artist(
    plt.Rectangle(
        (-robot_radius, -alpha*dt**2*gamma*vmax**2/(4*wheels_distance) - robot_radius), 
        alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (alpha*dt**2*vmax**2/(4*wheels_distance) * (beta + gamma)), 
        color='red', 
        fill=False, 
        zorder=3, 
        linewidth=3,
        label=r'Possible configurations at time $t + \Delta t$'
    )
)
# Lower ALPHA
intersection_points = []
for o, obs in enumerate(obstacles):
    for s, segment in enumerate(obs):
        intersection, intersection_point_0, intersection_point_1 = segment_rectangle_intersection(
            segment[0][0], 
            segment[0][1], 
            segment[1][0], 
            segment[1][1],
            0. + 1e-6, # xmin
            alpha * vmax * dt + robot_radius - 1e-6, # xmax
            -robot_radius + 1e-6, # ymin
            robot_radius - 1e-6, # ymax
        )
        intersection_points.append(intersection_point_0)
        intersection_points.append(intersection_point_1)
        # ax[0].scatter(intersection_point_0[0], intersection_point_0[1], color='blue', s=10, zorder=4)
        # ax[0].scatter(intersection_point_1[0], intersection_point_1[1], color='blue', s=10, zorder=4)
intersection_points = jnp.array(intersection_points)
alpha_intersection_points = jnp.copy(intersection_points)
min_x = jnp.nanmin(intersection_points[:,0])
new_alpha = lax.cond(
    ~jnp.isnan(min_x),
    lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (vmax * dt),
    lambda _: alpha,
    None,
)
# Lower BETA
intersection_points = []
for o, obs in enumerate(obstacles):
    for s, segment in enumerate(obs):
        intersection, intersection_point_0, intersection_point_1 = segment_rectangle_intersection(
            segment[0][0], 
            segment[0][1], 
            segment[1][0], 
            segment[1][1],
            -robot_radius + 1e-6, # xmin
            new_alpha * vmax * dt + robot_radius - 1e-6, # xmax
            robot_radius + 1e-6, # ymin
            robot_radius + (new_alpha*dt**2*beta*vmax**2/(4*wheels_distance)) - 1e-6, # ymax
        )
        intersection_points.append(intersection_point_0)
        intersection_points.append(intersection_point_1)
        # ax[0].scatter(intersection_point_0[0], intersection_point_0[1], color='blue', s=10, zorder=4)
        # ax[0].scatter(intersection_point_1[0], intersection_point_1[1], color='blue', s=10, zorder=4)
intersection_points = jnp.array(intersection_points)
beta_intersection_points = jnp.copy(intersection_points)
min_y = jnp.nanmin(intersection_points[:,1])
new_beta = lax.cond(
    ~jnp.isnan(min_y),
    lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
    lambda _: beta,
    None,
)
# Lower GAMMA
intersection_points = []
for o, obs in enumerate(obstacles):
    for s, segment in enumerate(obs):
        intersection, intersection_point_0, intersection_point_1 = segment_rectangle_intersection(
            segment[0][0], 
            segment[0][1], 
            segment[1][0], 
            segment[1][1],
            -robot_radius + 1e-6, # xmin
            new_alpha * vmax * dt + robot_radius - 1e-6, # xmax
            -robot_radius - (new_alpha*dt**2*gamma*vmax**2/(4*wheels_distance)) + 1e-6, # ymin
            -robot_radius - 1e-6, # ymax
        )
        intersection_points.append(intersection_point_0)
        intersection_points.append(intersection_point_1)
        # ax[0].scatter(intersection_point_0[0], intersection_point_0[1], color='blue', s=10, zorder=4)
        # ax[0].scatter(intersection_point_1[0], intersection_point_1[1], color='blue', s=10, zorder=4)
intersection_points = jnp.array(intersection_points)
gamma_intersection_points = jnp.copy(intersection_points)
max_y = jnp.nanmax(intersection_points[:,1])
new_gamma = lax.cond(
    ~jnp.isnan(max_y),
    lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
    lambda _: gamma,
    None,
)
print(f"New alpha (for collision avoidance): {new_alpha}")
print(f"New beta (for collision avoidance): {new_beta}")
print(f"New gamma (for collision avoidance): {new_gamma}")
ax[0].add_artist(
    plt.Rectangle(
        (-robot_radius, -new_alpha*dt**2*new_gamma*vmax**2/(4*wheels_distance) - robot_radius), 
        new_alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance) * (new_beta + new_gamma)), 
        color='green', 
        fill=False, 
        zorder=8, 
        linewidth=3,
        label=r'Collision-free configurations at time $t + \Delta t$',
    ),
)
ax4.add_artist(
    plt.Rectangle(
        (-robot_radius, -new_alpha*dt**2*new_gamma*vmax**2/(4*wheels_distance) - robot_radius), # + 0.008,
        new_alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance) * (new_beta + new_gamma)), # - 0.008, 
        color='green', 
        fill=False, 
        zorder=8, 
        linewidth=3,
        label=r'Collision-free configurations at time $t + \Delta t$',
    ),
)
# Plot new position displacements and new taylor approximation
import numpy as np
def sample_from_triangle(A, B, C, n_samples=1):
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)
    # Reflect if outside the triangle
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    samples = (1 - u - v)[:, None] * A + u[:, None] * B + v[:, None] * C
    return jnp.array(samples)
constrained_action_space = sample_from_triangle(
    jnp.array([0, 2*new_beta*vmax/wheels_distance]),
    jnp.array([0, -2*new_gamma*vmax/wheels_distance]),
    jnp.array([new_alpha*vmax, 0]),
    n_samples=2*samples**2
)
new_pxy_theta = vmap(exact_integration_of_action_space, in_axes=(0, 0))(jnp.zeros((len(constrained_action_space),3)), constrained_action_space)
# ax[0].scatter(new_pxy_theta[:,0], new_pxy_theta[:,1], s=5, color='yellow')
circles = [(x, y, robot_radius) for x, y in zip(new_pxy_theta[::25,0], new_pxy_theta[::25,1])] + [(new_alpha * vmax * dt, 0, robot_radius)]
points = []
for x, y, r in circles:
    if y == 0:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)
    elif y < 0:
        theta = jnp.linspace(jnp.pi, 2 * jnp.pi, 80)
    else:
        theta = jnp.linspace(0, jnp.pi, 80)
    cx = x + r * jnp.cos(theta)
    cy = y + r * jnp.sin(theta)
    points.extend(zip(cx, cy))
points = jnp.array(points)
hull = ConvexHull(points)
ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], facecolor='lightgreen', edgecolor='green', zorder=2)
ax4.fill(points[hull.vertices, 0], points[hull.vertices, 1], facecolor='lightgreen', edgecolor='green', zorder=2)
# Set plot specs
ax[0].set_xlim(-robot_radius - 0.05, vmax * dt + robot_radius + 0.05)
ax[0].set_ylim(-0.5, 0.5)
ax[0].axis("equal")
ax[0].set_xlabel("$x$ (m)")
ax[0].set_ylabel("$y$ (m)")
ax2.set_aspect('equal',adjustable='box')
ax2.set_xlim(-robot_radius - 0.05, vmax * dt + robot_radius + 0.05)
ax2.set_ylim(-0.52, 0.52)
ax2.set_xlabel("$x$ (m)")
ax2.set_ylabel("$y$ (m)", labelpad=-35)
ax2.set_xticks(jnp.arange(-0.2, vmax * dt + robot_radius, 0.2), labels=[round(i,1) for i in jnp.arange(-0.2, vmax * dt + robot_radius, 0.2).tolist()] )
ax2.grid(zorder=1)
ax4.set_aspect('equal',adjustable='box')
ax4.set_xlim(-robot_radius - 0.05, vmax * dt + robot_radius + 0.05)
ax4.set_ylim(-0.52, 0.52)
ax4.set_xlabel("$x$ (m)")
ax4.set_ylabel("$y$ (m)", labelpad=-35)
ax4.set_xticks(jnp.arange(-0.2, vmax * dt + robot_radius, 0.2), labels=[round(i,1) for i in jnp.arange(-0.2, vmax * dt + robot_radius, 0.2).tolist()] )
ax4.grid(zorder=1)
h, l = ax[0].get_legend_handles_labels()
import matplotlib.patches as mpatches
h[0] = mpatches.Patch(edgecolor='red', fill=True, facecolor='lightcoral', linewidth=2, label='Next feasible configurations envelope/box')
h[1] = mpatches.Patch(edgecolor='green', fill=True, facecolor='lightgreen', linewidth=2, label='Next collision-free configurations envelope/box')
h.append(mpatches.Patch(color='black', label='Obstacles'))
l.append('Obstacles')
ax[0].legend(h, l)
ax[0].grid(zorder=1)
### Plot original action space (v, w) and the new bounded action space
ax[1].add_patch(
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
ax[1].add_patch(
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
ax[1].set_xlim(-0.1, vmax + 0.1)
ax[1].set_ylim(-2*vmax/wheels_distance - 0.3, 2*vmax/wheels_distance + 0.3)
ax[1].set_xlabel("$v$ (m/s)")
ax[1].set_ylabel("$\omega$ (rad/s)")
ax[1].grid()
ax[1].legend()
ax[1].set_xticks(jnp.arange(0, vmax+0.2, 0.2))
ax[1].set_xticklabels([round(i,1) for i in np.arange(0, vmax, 0.2)] + [r"$\overline{v}$"])
ax[1].set_yticks(np.arange(-2,3,1).tolist() + [2*vmax/wheels_distance,-2*vmax/wheels_distance])
ax[1].set_yticklabels([round(i) for i in np.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
ax3.add_patch(
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
ax5.add_patch(
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
ax5.add_patch(
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
ax3.set_xlim(-0.1, vmax + 0.1)
ax3.set_ylim(-2*vmax/wheels_distance - 0.1, 2*vmax/wheels_distance + 0.1)
ax3.set_xlabel("$v$ (m/s)")
ax3.set_ylabel("$\omega$ (rad/s)", labelpad=-35)
ax3.grid()
ax3.set_xticks(jnp.arange(0, vmax+0.5, 0.5))
ax3.set_xticklabels([round(i,1) for i in np.arange(0, vmax, 0.5)] + [r"$\overline{v}$"])
ax3.set_yticks(np.arange(-2,3,1).tolist() + [2*vmax/wheels_distance,-2*vmax/wheels_distance])
ax3.set_yticklabels([round(i) for i in np.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
ax5.set_xlim(-0.1, vmax + 0.1)
ax5.set_ylim(-2*vmax/wheels_distance - 0.1, 2*vmax/wheels_distance + 0.1)
ax5.set_xlabel("$v$ (m/s)")
ax5.set_ylabel("$\omega$ (rad/s)", labelpad=-35)
ax5.grid()
ax5.set_xticks(jnp.arange(0, vmax+0.5, 0.5))
ax5.set_xticklabels([round(i,1) for i in np.arange(0, vmax, 0.5)] + [r"$\overline{v}$"])
ax5.set_yticks(np.arange(-2,3,1).tolist() + [2*vmax/wheels_distance,-2*vmax/wheels_distance])
ax5.set_yticklabels([round(i) for i in np.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
import os
figure.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding.eps"), format='eps')
figure2.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_1.eps"), format='eps')
figure3.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_2.eps"), format='eps')
figure4.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_3.eps"), format='eps')
figure5.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_4.eps"), format='eps')
# plt.show()



### Plot collision-free rectangle algorithm
from matplotlib import rc
font = {'weight' : 'regular',
        'size'   : 27}
rc('font', **font)
figure, ax = plt.subplots(1, 2, figsize=(25.61, 8.7))
figure.subplots_adjust(left=0.06, right=0.95, top=0.96, bottom=0.08, wspace=0.)
# Plot initial configuration
ax[0].add_artist(plt.Circle((0, 0), robot_radius, color='black', fill=False, zorder=4, linewidth=2, linestyle='--'))
ax[1].add_artist(plt.Circle((0, 0), robot_radius, color='black', fill=False, zorder=4, linewidth=2, linestyle='--'))
# Plot obstacles
for i, o in enumerate(obstacles): 
    if o.shape[0] == 1:  # Single segment obstacle
        ax[0].plot(o[0,:,0], o[0,:,1], color='black', linewidth=3, zorder=7)
    else:  # Multiple segments obstacle
        ax[0].fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=7)
# Plot obstacles
for i, o in enumerate(obstacles): 
    if o.shape[0] == 1:  # Single segment obstacle
        ax[1].plot(o[0,:,0], o[0,:,1], color='black', linewidth=3, zorder=7)
    else:  # Multiple segments obstacle
        ax[1].fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=7)
# Plot bounding rectangles
ax[0].add_artist(
    plt.Rectangle(
        (-robot_radius, -alpha*dt**2*gamma*vmax**2/(4*wheels_distance) - robot_radius), 
        alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (alpha*dt**2*vmax**2/(4*wheels_distance) * (beta + gamma)), 
        color='red', 
        fill=False, 
        zorder=3, 
        linewidth=3,
    )
)
ax[0].add_artist(
    plt.Rectangle(
        (-robot_radius,-robot_radius), 
        alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius, 
        edgecolor='red', 
        fill=False, 
        zorder=3, 
        linewidth=3,
    )
)
ax[0].add_artist(
    plt.Rectangle(
        (0,-robot_radius), 
        alpha*vmax*dt + robot_radius, 
        2*robot_radius, 
        edgecolor='red', 
        fill=True, 
        facecolor='lightgrey',
        zorder=3, 
        linewidth=3
    )
)
ax[1].add_artist(
    plt.Rectangle(
        (-robot_radius, -new_alpha*dt**2*gamma*vmax**2/(4*wheels_distance) - robot_radius), 
        new_alpha*vmax*dt + 2 * robot_radius, 
        2*robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance) * (beta + gamma)), 
        color='red', 
        fill=False, 
        zorder=3, 
        linewidth=3,
    )
)
ax[1].add_artist(
    plt.Rectangle(
        (-robot_radius,robot_radius), 
        new_alpha*vmax*dt + 2 * robot_radius, 
        (new_alpha*dt**2*vmax/(4*wheels_distance) * (beta)), 
        edgecolor='red', 
        fill=True, 
        facecolor='lightgrey',
        zorder=3, 
        linewidth=3,
    )
)
ax[1].add_artist(
    plt.Rectangle(
        (-robot_radius,-new_alpha*dt**2*gamma*vmax/(4*wheels_distance) - robot_radius), 
        new_alpha*vmax*dt + 2 * robot_radius, 
        (new_alpha*dt**2*vmax/(4*wheels_distance) * (gamma)), 
        edgecolor='red', 
        fill=True, 
        facecolor='lightgrey', 
        zorder=3, 
        linewidth=3,
    )
)
# Plot computed intersection points
alpha_intersection_points = alpha_intersection_points[~jnp.isnan(alpha_intersection_points).any(axis=1)]
beta_intersection_points = beta_intersection_points[~jnp.isnan(beta_intersection_points).any(axis=1)]
gamma_intersection_points = gamma_intersection_points[~jnp.isnan(gamma_intersection_points).any(axis=1)]
# Remove NaN points
ax[0].fill(alpha_intersection_points[:,0], alpha_intersection_points[:,1], color='blue', zorder=8, label=r'$\mathcal{I}_{\alpha}$')
ax[1].fill(beta_intersection_points[:,0], beta_intersection_points[:,1], color='green', zorder=8, label=r'$\mathcal{I}_{\beta}$')
ax[1].fill(gamma_intersection_points[:,0], gamma_intersection_points[:,1], color='pink', zorder=8, label=r'$\mathcal{I}_{\gamma}$')
# Plot segments and labels for algorithm understanding
def segment(ax, xy0, xy1, label, label_pos=None, color='black'):
    ax.plot([xy0[0],xy1[0]], [xy0[1],xy1[1]], color=color, zorder=8, linewidth=2)
    if xy0[0] == xy1[0]: # Vertical segment
        marker = '_'
        label_pos = label_pos if label_pos is not None else (xy0[0] + 0.01, (xy0[1] + xy1[1]) / 2)
        ax.text(label_pos[0], label_pos[1], label, verticalalignment='center', horizontalalignment='left', color=color, zorder=8)
    elif xy0[1] == xy1[1]: # Horizontal segment
        marker = '|'
        label_pos = label_pos if label_pos is not None else ((xy0[0] + xy1[0]) / 2, xy0[1]+0.01)
        ax.text(label_pos[0], label_pos[1], label, verticalalignment='bottom', horizontalalignment='center', color=color, zorder=8)
    else: # Diagonal segment
        marker = 'x'
        label_pos = label_pos if label_pos is None else ((xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2)
        ax.text(label_pos[0], label_pos[1], label, verticalalignment='center', horizontalalignment='center', color=color, zorder=8)
    ax.scatter([xy0[0],xy1[0]], [xy0[1],xy1[1]], color=color, s=50, zorder=8, marker=marker)
segment(ax[0], [0.,0.], [0.,-robot_radius], '$r$', color='black')
segment(ax[1], [0.,0.], [0.,-robot_radius], '$r$', color='black')
segment(ax[0], [robot_radius,0.], [alpha*vmax*dt + robot_radius,0.], '$\Delta x_{\max}$', label_pos=(0.85,0.01), color='black')
segment(ax[1], [robot_radius,0.], [new_alpha*vmax*dt + robot_radius,0.], r'$\tilde{\Delta} x_{\max}$', color='black')
segment(
    ax[1], 
    [(new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,-robot_radius], 
    [(new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,-robot_radius-(new_alpha*dt**2*vmax/(4*wheels_distance) * (gamma))], 
    r'$|\Delta y_{\min}|$', 
    label_pos=((new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,-robot_radius-(new_alpha*dt**2*vmax/(4*wheels_distance) * (gamma))-0.05), 
    color='black'
)
segment(
    ax[1], 
    [(new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,+robot_radius], 
    [(new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,+robot_radius+(new_alpha*dt**2*vmax/(4*wheels_distance) * (beta))], 
    r'$\Delta y_{\max}$', 
    label_pos=((new_alpha*vmax*dt + 2*robot_radius)/2-robot_radius,+robot_radius+(new_alpha*dt**2*vmax/(4*wheels_distance) * (beta))+0.05), 
    color='black'
)
# Boxes labels
ax[0].text(-robot_radius/2, 0, r'$\mathcal{B}_0$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
ax[0].text(-robot_radius + (2*robot_radius + vmax * dt)/2, -robot_radius-(alpha*dt**2*vmax/(4*wheels_distance))/2, r'$\mathcal{B}_{\gamma}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
ax[0].text(-robot_radius + (2*robot_radius + vmax * dt)/2, robot_radius+(alpha*dt**2*vmax/(4*wheels_distance))/2+0.02, r'$\mathcal{B}_{\beta}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
ax[0].text((robot_radius + vmax * dt)/2, -0.05, r'$\mathcal{B}_{\alpha}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
ax[1].text(-robot_radius + 0.05, -robot_radius-(new_gamma*new_alpha*dt**2*vmax/(4*wheels_distance))/2 - 0.015, r'$\mathcal{B}_{\overline{\gamma}}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
ax[1].text(-robot_radius + 0.05, robot_radius+(new_beta*new_alpha*dt**2*vmax/(4*wheels_distance))/2 + 0.015, r'$\mathcal{B}_{\overline{\beta}}$', verticalalignment='center', horizontalalignment='center', color='black', zorder=8, fontsize=35)
# Plot specs
ax[0].grid(zorder=1)
ax[0].set_title(r"Stage 1: Reduce $\alpha$", weight="bold")
ax[0].set_aspect('equal',adjustable='box')
ax[0].set_xlim(-robot_radius - 0.05, vmax * dt + robot_radius + 0.3)
ax[0].set_ylim(-0.55, 0.55)
ax[0].set_xlabel("$x$ (m)")
ax[0].set_ylabel("$y$ (m)")
ax[1].grid(zorder=1)
ax[1].set_title(r"Stage 2: Reduce $\beta$ and $\gamma$", weight="bold")
ax[1].set_aspect('equal',adjustable='box')
ax[1].set_xlim(-robot_radius - 0.15 - 0.2, vmax * dt + robot_radius)
ax[1].set_ylim(-0.55, 0.55)
ax[1].set_xlabel("$x$ (m)")
ax[1].yaxis.tick_right()
h, l = ax[0].get_legend_handles_labels()
h1, l1 = ax[1].get_legend_handles_labels()
h.append(h1[0])
l.append(l1[0])
h.append(mpatches.Patch(color='black', label='Obstacles'))
l.append('Obstacles')
from matplotlib.lines import Line2D
figure.legend(
    h,
    l,
    loc='center',
    bbox_to_anchor=(0.5, 0.5),
)
figure.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_algorithm.eps"), format='eps')
bbox = figure.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
bbox.x1 = (bbox.x0 + bbox.x1) / 2
plt.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_algorithm_1.eps"), format='eps', bbox_inches=bbox)
bbox = figure.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
mid_x = (bbox.x0 + bbox.x1) / 2
bbox.x0 = mid_x  # move left edge to the middle
plt.savefig(os.path.join(os.path.dirname(__file__),"action_space_bounding_algorithm_2.eps"), format='eps', bbox_inches=bbox)