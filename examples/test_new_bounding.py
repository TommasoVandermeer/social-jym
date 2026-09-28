from functools import partial
from jax import lax,jit
import jax.numpy as jnp
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

EPSILON = 1e-5
robot_radius = 0.3
v_max = 1.
wheels_distance = 0.7
w_max = 2*v_max/wheels_distance
w_min = -w_max
dt = 0.25
lidar_point_cloud = jnp.array([
    [0.5, 0.],
    [0., 0.305],
    [0., -0.35],
])

@jit
def old_bound_action_space(lidar_point_cloud, eps=1e-6):
    """
    Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
    WARNING: Assumes LiDAR orientation is align with robot frame.
    """
    # Lower ALPHA
    is_inside_frontal_rect = (
        (lidar_point_cloud[:,0] >=  0 + eps) & # xmin
        (lidar_point_cloud[:,0] <= v_max * dt + robot_radius - eps) & # xmax
        (lidar_point_cloud[:,1] >= -robot_radius + eps) &  # ymin
        (lidar_point_cloud[:,1] <= robot_radius - eps) # ymax
    )
    intersection_points = jnp.where(
        is_inside_frontal_rect[:, None],
        lidar_point_cloud,
        jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
    )
    min_x = jnp.nanmin(intersection_points[:,0])
    new_alpha = lax.cond(
        ~jnp.isnan(min_x),
        lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (v_max * dt),
        lambda _: 1.,
        None,
    )
    @jit
    def _lower_beta_and_gamma(tup:tuple):
        lidar_point_cloud, new_alpha, vmax, wheels_distance, dt = tup
        # Lower BETA
        is_inside_left_rect = (
            (lidar_point_cloud[:,0] >= -robot_radius + eps) & # xmin
            (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= robot_radius + eps) &  # ymin
            (lidar_point_cloud[:,1] <= robot_radius + (new_alpha*dt**2*vmax**2/(4*wheels_distance)) - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_left_rect[:, None],
            lidar_point_cloud,
            jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
        )
        min_y = jnp.nanmin(intersection_points[:,1])
        new_beta = lax.cond(
            ~jnp.isnan(min_y),
            lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
            lambda _: 1.,
            None,
        )
        # Lower GAMMA
        is_inside_right_rect = (
            (lidar_point_cloud[:,0] >=  -robot_radius + eps) & # xmin
            (lidar_point_cloud[:,0] <= new_alpha * vmax * dt + robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= -robot_radius - (new_alpha*dt**2*vmax**2/(4*wheels_distance)) + eps) & # ymin
            (lidar_point_cloud[:,1] <= -robot_radius - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_right_rect[:, None],
            lidar_point_cloud,
            jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
        )
        max_y = jnp.nanmax(intersection_points[:,1])
        new_gamma = lax.cond(
            ~jnp.isnan(max_y),
            lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2 * new_alpha),
            lambda _: 1.,
            None,
        )
        return new_beta, new_gamma
    new_beta, new_gamma = lax.cond(
        new_alpha == 0.,
        lambda _: (1., 1.),
        _lower_beta_and_gamma,
        (lidar_point_cloud, new_alpha, v_max, wheels_distance, dt)
    )
    # Apply lower bound to new_alpha, new_beta, new_gamma
    new_alpha = jnp.max(jnp.array([EPSILON, new_alpha]))
    new_beta = jnp.max(jnp.array([EPSILON, new_beta]))
    new_gamma = jnp.max(jnp.array([EPSILON, new_gamma]))
    return jnp.array([new_alpha, new_beta, new_gamma])

@jit
def new_bound_action_space(lidar_point_cloud, eps=1e-6):
    """
    Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
    WARNING: Assumes LiDAR orientation is align with robot frame.
    """
    # Lower ALPHA
    is_inside_frontal_rect = (
        (lidar_point_cloud[:,0] >=  0 + eps) & # xmin
        (lidar_point_cloud[:,0] <= v_max * dt + robot_radius - eps) & # xmax
        (lidar_point_cloud[:,1] >= -robot_radius + eps) &  # ymin
        (lidar_point_cloud[:,1] <= robot_radius - eps) # ymax
    )
    intersection_points = jnp.where(
        is_inside_frontal_rect[:, None],
        lidar_point_cloud,
        jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
    )
    min_x = jnp.nanmin(intersection_points[:,0])
    new_alpha = lax.cond(
        ~jnp.isnan(min_x),
        lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (v_max * dt),
        lambda _: 1.,
        None,
    )
    @jit
    def _lower_beta_and_gamma(tup:tuple):
        lidar_point_cloud, vmax, wheels_distance, dt = tup
        # Lower BETA
        is_inside_left_rect = (
            (lidar_point_cloud[:,0] >= -robot_radius + eps) & # xmin
            (lidar_point_cloud[:,0] <= vmax * dt + robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= robot_radius + eps) &  # ymin
            (lidar_point_cloud[:,1] <= robot_radius + (dt**2*vmax**2/(4*wheels_distance)) - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_left_rect[:, None],
            lidar_point_cloud,
            jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
        )
        min_y = jnp.nanmin(intersection_points[:,1])
        new_beta = lax.cond(
            ~jnp.isnan(min_y),
            lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2),
            lambda _: 1.,
            None,
        )
        # Lower GAMMA
        is_inside_right_rect = (
            (lidar_point_cloud[:,0] >= -robot_radius + eps) & # xmin
            (lidar_point_cloud[:,0] <= vmax * dt + robot_radius - eps) & # xmax
            (lidar_point_cloud[:,1] >= -robot_radius - (dt**2*vmax**2/(4*wheels_distance)) + eps) & # ymin
            (lidar_point_cloud[:,1] <= -robot_radius - eps) # ymax
        )
        intersection_points = jnp.where(
            is_inside_right_rect[:, None],
            lidar_point_cloud,
            jnp.full_like(lidar_point_cloud, fill_value=jnp.nan)
        )
        max_y = jnp.nanmax(intersection_points[:,1])
        new_gamma = lax.cond(
            ~jnp.isnan(max_y),
            lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax**2 * dt**2),
            lambda _: 1.,
            None,
        )
        return new_beta, new_gamma
    new_beta, new_gamma = lax.cond(
        new_alpha == 0.,
        lambda _: (1., 1.),
        _lower_beta_and_gamma,
        (lidar_point_cloud, v_max, wheels_distance, dt)
    )
    # Apply lower bound to new_alpha, new_beta, new_gamma
    new_alpha = jnp.max(jnp.array([EPSILON, new_alpha]))
    new_beta = jnp.max(jnp.array([EPSILON, new_beta]))
    new_gamma = jnp.max(jnp.array([EPSILON, new_gamma]))
    return jnp.array([new_alpha, new_beta, new_gamma])

old_action_space_params = old_bound_action_space(lidar_point_cloud)
params_labels = ["alpha", "beta", "gamma"]
print("Action space parameters: ", [params_labels[i] + ": " + str(p) for i, p in enumerate(old_action_space_params)])
new_action_space_params = new_bound_action_space(lidar_point_cloud)
params_labels = ["alpha", "beta", "gamma"]
print("Action space parameters: ", [params_labels[i] + ": " + str(p) for i, p in enumerate(new_action_space_params)])

@jit
def old_action_space_vertices(action_space_params):
    return jnp.array([
        [action_space_params[0]*v_max, 0.],
        [0.,action_space_params[1]*2*v_max/wheels_distance],
        [0.,-action_space_params[2]*2*v_max/wheels_distance],
    ])

@jit
def new_action_space_vertices(action_space_params):
    """
    Computes the vertices of the feasible action space polygon.
    The shape is the intersection of the kinematic constraints (triangle)
    and the environment constraints (box scaled by alpha, beta, gamma).
    
    Returns a fixed shape array (6, 2). Duplicate vertices are possible if
    fewer than 6 points are needed to define the shape.
    """
    alpha, beta, gamma = action_space_params
    # 1. Define the Box Limits based on parameters
    V = alpha * v_max
    W_up = beta * w_max          # w_max is positive
    W_down = gamma * w_min       # w_min is negative (e.g. -2.85), so W_down is negative
    # 2. Top-Right Logic
    # The triangle boundary is line: w = w_max * (1 - v/v_max)
    # We check if the box corner (V, W_up) violates this line.
    # Violation condition: W_up > w_max * (1 - V/v_max)
    #                      beta * w_max > w_max * (1 - alpha)
    #                      beta > 1 - alpha  --> alpha + beta > 1
    cut_top = (alpha + beta) > 1.0
    # If cut, we introduce two vertices on the cut line. 
    # If not cut, both vertices collapse to the box corner (V, W_up).
    # Vertex on Top edge (w = W_up) intersected with diagonal
    # w_max * (1 - v/v_max) = W_up  --> v = v_max * (1 - beta)
    v2_x = jnp.where(cut_top, v_max * (1.0 - beta), V)
    v2_y = W_up
    # Vertex on Right edge (v = V) intersected with diagonal
    # w = w_max * (1 - V/v_max) --> w = w_max * (1 - alpha)
    v3_x = V
    v3_y = jnp.where(cut_top, w_max * (1.0 - alpha), W_up)
    # 3. Bottom-Right Logic
    # The triangle boundary is line: w = w_min * (1 - v/v_max)
    # Violation condition (since w is negative): W_down < w_min * (1 - V/v_max)
    #                                            gamma * w_min < w_min * (1 - alpha)
    #                                            gamma > 1 - alpha --> alpha + gamma > 1
    cut_bottom = (alpha + gamma) > 1.0
    # Vertex on Right edge (v = V) intersected with diagonal
    # w = w_min * (1 - V/v_max) --> w = w_min * (1 - alpha)
    v4_x = V
    v4_y = jnp.where(cut_bottom, w_min * (1.0 - alpha), W_down)
    # Vertex on Bottom edge (w = W_down) intersected with diagonal
    # w_min * (1 - v/v_max) = W_down --> v = v_max * (1 - gamma)
    v5_x = jnp.where(cut_bottom, v_max * (1.0 - gamma), V)
    v5_y = W_down
    # 4. Assemble vertices counter-clockwise
    vertices = jnp.array([
        [0.0, W_up],    # V1: Top-Left (always on axis)
        [v2_x, v2_y],   # V2: Top transition
        [v3_x, v3_y],   # V3: Top-Right transition
        [v4_x, v4_y],   # V4: Bottom-Right transition
        [v5_x, v5_y],   # V5: Bottom transition
        [0.0, W_down],  # V6: Bottom-Left (always on axis)
    ])
    vertices = lax.fori_loop(
        1,
        6,
        lambda i, v: lax.cond(
            jnp.all(v[i] == v[i-1]),  # Check if current vertex is the same as the previous one
            lambda: v.at[i].set(jnp.array([jnp.nan, jnp.nan])),
            lambda: v,          
        ),
        vertices
    )
    return vertices
        

# AX :,2: Feasible and bounded action space + action space distribution and action taken
fig, axs = plt.subplots(1,2)
axs[0].set_title("Old Action space")
axs[1].set_title("New Action space")
for ax in axs:
    ax.set_xlabel("$v$ (m/s)")
    ax.set_ylabel("$\omega$ (rad/s)", labelpad=-15)
    ax.set_xlim(-0.1, v_max + 0.1)
    ax.set_ylim(-2*v_max/wheels_distance - 0.3, 2*v_max/wheels_distance + 0.3)
    ax.set_xticks(jnp.arange(0, v_max+0.2, 0.2))
    ax.set_xticklabels([round(i,1) for i in jnp.arange(0, v_max, 0.2)] + [r"$\overline{v}$"])
    ax.set_yticks(jnp.arange(-2,3,1).tolist() + [2*v_max/wheels_distance,-2*v_max/wheels_distance])
    ax.set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
    ax.grid()
    ax.add_patch(
        plt.Polygon(
            [   
                [0,w_max],
                [0,w_min],
                [v_max,0],
            ],
            closed=True,
            fill=True,
            edgecolor='red',
            facecolor='lightcoral',
            linewidth=2,
            zorder=2,
        ),
    )
# OLD ACTION SPACE
old_action_space = old_action_space_vertices(old_action_space_params)
axs[0].add_patch(
    plt.Polygon(
        old_action_space,
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=3,
    ),
)
# NEW ACTION SPACE
new_action_space = new_action_space_vertices(new_action_space_params)
print("\nNew action space vertices:\n", new_action_space)
axs[1].add_patch(
    plt.Polygon(
        new_action_space[jnp.where(~jnp.isnan(new_action_space[:,0]))],  # Filter out NaN vertices
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=3,
    ),
)
plt.show()