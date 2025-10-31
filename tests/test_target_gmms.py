from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
import os
import pickle

from jhsfm.hsfm import vectorized_compute_obstacle_closest_point
from socialjym.utils.distributions.gaussian_mixture_model import GMM
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import animate_trajectory
from socialjym.policies.dir_safe import DIRSAFE

### Parameters
random_seed = 0
grid_resolution = 10  # Number of grid cells per dimension
n_samples_per_dim = 50 # Number of GMM sample points per dimension to show in the animation
p_visualization_threshold = 0.01 # Probability density threshold for visualization in GMM plots
save = False # Whether to save the animation as a .mp4 file
## Environment
robot_radius = 0.3
robot_dt = 0.25
robot_vmax = 1.0
robot_visible = True
kinematics = "unicycle"
scenario = "circular_crossing"
n_humans = 5
n_obstacles = 5
humans_policy = 'hsfm'
env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'n_obstacles': n_obstacles,
            'robot_dt': robot_dt,
            'robot_radius': robot_radius, 
            'humans_dt': 0.01,
            'robot_visible': robot_visible,
            'scenario': scenario,
            'humans_policy': humans_policy,
            'reward_function': DummyReward(kinematics=kinematics),
            'kinematics': kinematics,
        }
env = SocialNav(**env_params)
## Robot policy
policy = DIRSAFE(env.reward_function, v_max=robot_vmax, dt=env_params['robot_dt'])
with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']
## Build local grid over which the GMM is defined
dists = jnp.concatenate([-jnp.arange(0, 5, grid_resolution/10)[::-1][:-1],jnp.arange(0, 5, grid_resolution/10)])
grid_cell_coords = jnp.meshgrid(dists, dists)
grid_cells = jnp.array(jnp.vstack((grid_cell_coords[0].flatten(), grid_cell_coords[1].flatten())).T)
cell_size = (grid_cells[1,0] - grid_cells[0,0], grid_cells[grid_resolution,1] - grid_cells[0,1])  # Assuming uniform grid
min_coord = jnp.min(grid_cells) - cell_size[0] / 2
max_coord = jnp.max(grid_cells) + cell_size[1] / 2
gmm = GMM(n_dimensions=grid_cells.shape[1], n_components=grid_cells.shape[0])
## Generate samples from GMM domain space (for visualization purposes)
sample_dists = jnp.linspace(min_coord, max_coord, n_samples_per_dim)
sample_coords = jnp.meshgrid(sample_dists, sample_dists)
gmm_sample_points = jnp.array(jnp.vstack((sample_coords[0].flatten(), sample_coords[1].flatten())).T)

### Simulate one episode and collect data
state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
all_states = jnp.array([state])
all_robot_goals = jnp.array([info['robot_goal']])
while outcome['nothing']:
    action, policy_key, _, _, distr = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
    state, obs, info, _, outcome, _ = env.step(
            state,
            info,
            action,
            test=True,
            reset_if_done=False,
        )
    all_states = jnp.vstack((all_states, jnp.array([state])))
    all_robot_goals = jnp.vstack((all_robot_goals, jnp.array([info['robot_goal']])))
data = {
    'humans_positions': all_states[:, :-1, :2],
    'humans_orientations': all_states[:, :-1, 4],
    'robot_positions': all_states[:, -1, :2],
    'robot_orientations': all_states[:, -1, 4],
    'humans_radii': info['humans_parameters'][:,0],
    'static_obstacles': info['static_obstacles'][-1],
}

### Animate trajectory
# animate_trajectory(
#     all_states, 
#     info['humans_parameters'][:,0], 
#     env.robot_radius, 
#     env_params['humans_policy'],
#     all_robot_goals,
#     info['current_scenario'],
#     robot_dt=env_params['robot_dt'],
#     static_obstacles=info['static_obstacles'][-1], # Obstacles are repeated for each agent, index -1 is enough
#     kinematics='unicycle',
# )

### Roto-translate humans positions to robot-centric frame
@jit
def roto_translate_pose(position, orientation, ref_position, ref_orientation):
    """Roto-translate a 2D pose given a reference pose."""
    c, s = jnp.cos(-ref_orientation), jnp.sin(-ref_orientation)
    R = jnp.array([[c, -s],
                   [s,  c]])
    translated_position = position - ref_position
    rotated_position = R @ translated_position
    rotated_orientation = orientation - ref_orientation
    return rotated_position, rotated_orientation
@jit
def roto_translate_poses(positions, orientations, ref_positions, ref_orientations):
    return vmap(roto_translate_pose, in_axes=(0, 0, None, None))(positions, orientations, ref_positions, ref_orientations)
@jit
def batch_roto_translate_poses(positions, orientations, ref_positions, ref_orientations):
    return vmap(roto_translate_poses, in_axes=(0, 0, 0, 0))(positions, orientations, ref_positions, ref_orientations)
robot_centric_humans_positions, robot_centric_humans_orientations = batch_roto_translate_poses(
    data['humans_positions'],
    data['humans_orientations'],
    data['robot_positions'],
    data['robot_orientations'],
)

### Roto-translate robot goals to robot-centric frame
@jit
def batch_roto_translate_goals(goals, goals_orientations, ref_positions, ref_orientations):
    return vmap(roto_translate_pose, in_axes=(0, 0, 0, 0))(goals, goals_orientations, ref_positions, ref_orientations)[0]
robot_centric_robot_goals = batch_roto_translate_goals(
    all_robot_goals,
    jnp.zeros((all_robot_goals.shape[0],)),  # Goals have no orientation
    data['robot_positions'],
    data['robot_orientations'],
)

### Roto-translate static obstacles to robot-centric frame
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
    return vmap(roto_translate_obstacles, in_axes=(None, 0, 0))(obstacles, ref_positions, ref_orientations)
robot_centric_static_obstacles = batch_roto_translate_obstacles(
    data['static_obstacles'],
    data['robot_positions'],
    data['robot_orientations'],
)

### Assess object visibility (static and dynamic obstacles) from robot perspective and mask out invisible ones (on a per-frame basis)
@jit
def per_frame_object_visibility(humans_positions, humans_radii, static_obstacles, epsilon=1e-5):
    """
    Assess which humans and static obstacles are visible from the robot's perspective.

    params:
    - humans_positions: (n_humans, 2) array of humans positions in robot-centric frame
    - humans_radii: (n_humans,) array of humans radii
    - static_obstacles: (n_obstacles, 2, 2) array of static obstacle segments in robot-centric frame

    returns:
    - visible_humans_mask: (n_humans,) boolean array indicating which humans are visible
    - visible_static_obstacles_mask: (n_obstacles,n_segments) boolean array indicating which static obstacle segments are visible
    """
    # TODO: Solve bug: humans are marked visible even when occluded by obstacles sometimes

    ### Compute ordered array of all object endpoint angles
    ## Humans
    humans_versors = humans_positions / jnp.linalg.norm(humans_positions, axis=1, keepdims=True)  # Shape: (n_humans, 2)
    left_versors = humans_versors @ jnp.array([[0, 1], [-1, 0]])  # Rotate by +90 degrees
    humans_left_edge_points = humans_positions + (humans_radii[:, None] - epsilon) * left_versors  # Shape: (n_humans, 2)
    humans_right_edge_points = humans_positions - (humans_radii[:, None] - epsilon) * left_versors  # Shape: (n_humans, 2)
    humans_left_angles = jnp.arctan2(humans_left_edge_points[:,1], humans_left_edge_points[:,0]) # Shape: (n_humans,)
    humans_right_angles = jnp.arctan2(humans_right_edge_points[:,1], humans_right_edge_points[:,0]) # Shape: (n_humans,)
    humans_edge_angles = jnp.concatenate((humans_left_angles, humans_right_angles))  # Shape: (2*n_humans,)
    ## Obstacles
    obstacle_segments = static_obstacles.reshape((n_obstacles*env.static_obstacles_per_scenario.shape[2], 2, 2))  # Shape: (n_obstacles*n_segments, 2, 2)
    obstacle_first_edge_points = obstacle_segments[:,0,:]  # Shape: (n_obstacles*n_segments, 2)
    obstacle_second_edge_points = obstacle_segments[:,1,:]  # Shape: (n_obstacles*n_segments, 2)
    first_to_second_versors = obstacle_second_edge_points - obstacle_first_edge_points / jnp.linalg.norm(obstacle_second_edge_points - obstacle_first_edge_points, axis=1, keepdims=True)  # Shape: (n_obstacles*n_segments, 2)
    obstacle_first_edge_points = obstacle_first_edge_points + (epsilon * first_to_second_versors)  # Shape: (n_obstacles*n_segments, 2)
    obstacle_second_edge_points = obstacle_second_edge_points - (epsilon * first_to_second_versors)  # Shape: (n_obstacles*n_segments, 2)
    obstacle_first_edge_angles = jnp.arctan2(obstacle_first_edge_points[:,1], obstacle_first_edge_points[:,0])  # Shape: (n_obstacles*n_segments,)
    obstacle_second_edge_angles = jnp.arctan2(obstacle_second_edge_points[:,1], obstacle_second_edge_points[:,0])  # Shape: (n_obstacles*n_segments,)
    obstacle_edge_angles = jnp.append(obstacle_first_edge_angles, obstacle_second_edge_angles)  # Shape: (2*n_obstacles*n_segments,)
    ## Merge and sort all edge angles
    all_edge_angles = jnp.concatenate((humans_edge_angles, obstacle_edge_angles))  # Shape: (2*n_humans + 2*n_obstacles*n_segments,)
    sorted_all_edge_angles = jnp.sort(all_edge_angles)
    # Wrap around for midpoint computation
    sorted_all_edge_angles = jnp.append(sorted_all_edge_angles, sorted_all_edge_angles[0])  # Shape: (2*n_humans + 2*n_obstacles*n_segments + 1,)
    ### Compute midpoint angles between consecutive object endpoints
    sorted_all_verors = jnp.array([jnp.cos(sorted_all_edge_angles), jnp.sin(sorted_all_edge_angles)]).T  # Shape: (2*n_humans + 2*n_obstacles*n_segments + 1, 2)
    midpoint_verors = (sorted_all_verors[:-1] + sorted_all_verors[1:])  # Shape: (2*n_humans + 2*n_obstacles*n_segments, 2)
    midpoint_verors = midpoint_verors / jnp.linalg.norm(midpoint_verors, axis=1, keepdims=True)  # Normalize
    midpoint_angles = jnp.arctan2(midpoint_verors[:,1], midpoint_verors[:,0])  # Shape: (2*n_humans + 2*n_obstacles*n_segments,)
    all_angles = jnp.concatenate((all_edge_angles, midpoint_angles)) # Shape: (4*n_humans + 4*n_obstacles*n_segments,)
    ### Ray-cast all computed angles and assess visibility of all objects (human_collision_idxs shape: (n_rays,), obstacle_collision_idxs shape: (n_rays, 2))
    distances, human_collision_idxs, obstacle_collision_idxs = env.batch_ray_cast(
        all_angles,
        jnp.array([0., 0.]),
        humans_positions,
        humans_radii,
        static_obstacles
    )
    humans_visibility_mask = vmap(lambda idx: jnp.any(human_collision_idxs == idx))(jnp.arange(n_humans))  # Shape: (n_humans,)
    @jit
    def segment_visibility(obstacle_idx, segment_idx, obstacle_collision_idxs):
        return jnp.any(jnp.all(obstacle_collision_idxs == jnp.array([obstacle_idx, segment_idx]), axis=1))
    @jit
    def obstacle_segments_visibility(obstacle_idx, segment_idxs, obstacle_collision_idxs):
        return vmap(segment_visibility, in_axes=(None, 0, None))(obstacle_idx, segment_idxs, obstacle_collision_idxs)
    obstacles_visibility_mask = vmap(obstacle_segments_visibility, in_axes=(0, None, None))(
        jnp.arange(n_obstacles), 
        jnp.arange(env.static_obstacles_per_scenario.shape[2]), 
        obstacle_collision_idxs
    ) # Shape: (n_obstacles, n_segments)
    return humans_visibility_mask, obstacles_visibility_mask, all_angles, distances
@jit
def batch_object_visibility(batch_humans_positions, humans_radii, batch_static_obstacles):
    return vmap(per_frame_object_visibility, in_axes=(0, None, 0))(batch_humans_positions, humans_radii, batch_static_obstacles)
visible_humans_mask, visible_obstacles_mask, visibility_angles, visibility_distances = batch_object_visibility(
    robot_centric_humans_positions,
    data['humans_radii'],
    robot_centric_static_obstacles,
)

### Fit GMMs to humans positions at each timestep
@jit
def fit_gmm_to_humans_positions(humans_position, humans_visibility, humans_radii, grid_cells, tau=0.2):
    @jit
    def softweight_human_cell(human_pos, human_visibility, human_radius, cell):
        @jit
        def _human_visible(data):
            human_pos, human_radius, cell = data
            dist = jnp.linalg.norm(cell - human_pos) - human_radius
            dist = lax.cond(dist < 0,lambda _: 0.0,lambda d: d,dist)
            return jnp.exp(-dist**2/(2*tau**2))
        return lax.cond(
            human_visibility,
            _human_visible,
            lambda _: 0.0,
            operand=(human_pos, human_radius, cell)
        )
    softweight_human_cells = jit(vmap(softweight_human_cell, in_axes=(None, None, None, 0)))
    batch_softweight_human_cells = jit(vmap(softweight_human_cells, in_axes=(0, 0, 0, None)))
    humans_weights_per_cell = batch_softweight_human_cells(humans_position, humans_visibility, humans_radii, grid_cells) # Shape: (n_humans, n_grid_cells, 2)
    weights_per_cell = jnp.sum(humans_weights_per_cell, axis=0)
    norm_cell_weights = weights_per_cell / (jnp.sum(weights_per_cell) + 1e-8)
    variances_per_cell = jnp.sum(humans_weights_per_cell * humans_radii[:, None]**2, axis=0) / (weights_per_cell + 1e-8) + 1e-8
    fitted_distribution = {
        "means": grid_cells,
        "logsigmas": jnp.stack((jnp.log(variances_per_cell), jnp.log(variances_per_cell)), axis=1),
        "weights": norm_cell_weights,
    }
    return fitted_distribution
@jit
def batch_fit_gmm_to_humans_positions(batch_humans_positions, batch_humans_visibility, humans_radii, grid_cells, tau=0.2):
    return vmap(fit_gmm_to_humans_positions, in_axes=(0, 0, None, None, None))(batch_humans_positions, batch_humans_visibility, humans_radii, grid_cells, tau)
dynamic_gmms = batch_fit_gmm_to_humans_positions(robot_centric_humans_positions, visible_humans_mask, data["humans_radii"], grid_cells)
dynamic_ps = vmap(gmm.batch_p, in_axes=(0, None))(dynamic_gmms, gmm_sample_points)

### Fit GMMs to static obstacles at each timestep
@jit
def fit_gmm_to_obstacles(obstacles, obstacles_visibility, grid_cells, scaling=0.15):
    # Substitute invisible edges with nan to avoid computing closest points on them
    obstacles = jnp.where(
        ~obstacles_visibility[:,:,None,None],
        jnp.nan,
        obstacles
    )
    # Compute closest points to each grid cell
    closest_points = jit(vmap(vectorized_compute_obstacle_closest_point, in_axes=(0, None)))( # Shape: (n_grid_cells, n_obstacles, 2)
        grid_cells,
        obstacles
    )
    # Compute the target per-cell weights for each grid cell
    @jit
    def softweight_obstacle_cell(obstacle_pos, cell):
        """Compute the soft weight of an obstacle for a grid cell based on a Gaussian distribution."""
        @jit
        def _not_nan(x):
            obstacle_pos, cell = x
            diff = cell - obstacle_pos
            diff = lax.cond(
                jnp.linalg.norm(diff) > 0.3,
                lambda d: d - 0.3 * d / jnp.linalg.norm(d),
                lambda d: jnp.zeros_like(d),
                diff
            )
            obstacle_cov = jnp.eye(2) * scaling
            exponent = -0.5 * jnp.dot(diff, jnp.linalg.solve(obstacle_cov, diff))
            norm_const = jnp.sqrt((2 * jnp.pi) ** len(obstacle_pos) * jnp.linalg.det(obstacle_cov))
            return jnp.exp(exponent) / norm_const
        return lax.cond(
            jnp.any(jnp.isnan(obstacle_pos)),
            lambda _: 0.0,
            _not_nan,
            operand=(obstacle_pos, cell)
        )
    softweight_obstacle_cells = jit(vmap(softweight_obstacle_cell, in_axes=(0, 0)))
    batch_softweight_obstacle_cells = jit(vmap(softweight_obstacle_cells, in_axes=(1, None)))
    obstacles_weights_per_cell = batch_softweight_obstacle_cells(
        closest_points,
        grid_cells
    )
    cell_weights = jnp.sum(obstacles_weights_per_cell, axis=0)
    norm_cell_weights = cell_weights / (jnp.sum(cell_weights) + 1e-8)
    # Compute the target per-cell covariance
    norm_obstacles_weights_per_cell = obstacles_weights_per_cell / (jnp.sum(obstacles_weights_per_cell, axis=1, keepdims=True) + 1e-8)
    @jit
    def obstacles_weighted_variance(obstacle_pos, cell, weight):
        @jit
        def _not_nan(x):
            obstacle_pos, cell, weight = x
            diff = cell - obstacle_pos
            obstacle_cov = jnp.eye(2) * scaling
            outer_prod = jnp.outer(diff, diff)
            return jnp.diag(weight * (obstacle_cov + outer_prod))
        return lax.cond(
            jnp.any(jnp.isnan(obstacle_pos)),
            lambda _: jnp.zeros((2,)),
            _not_nan,
            operand=(obstacle_pos, cell, weight)
        )
    batch_obstacles_weighted_variances = jit(vmap(obstacles_weighted_variance, in_axes=(0, None, 0)))
    batch_cells_obstacles_weighted_variances = jit(vmap(lambda op, gc, ow: jnp.sum(batch_obstacles_weighted_variances(op, gc, ow), axis=0), in_axes=(0, 0, 0)))
    obstacles_weighted_variances_per_cell = batch_cells_obstacles_weighted_variances(
        closest_points,
        grid_cells,
        norm_obstacles_weights_per_cell.T
    )
    # Initialize fitted distribution
    fitted_distribution = {
        "means": grid_cells,
        "logsigmas": jnp.log(obstacles_weighted_variances_per_cell),
        "weights": norm_cell_weights,
    }
    return fitted_distribution
def batch_fit_gmm_to_obstacles(batch_obstacles, batch_obstacle_visibility, grid_cells, scaling=0.01):
    return vmap(fit_gmm_to_obstacles, in_axes=(0, 0, None, None))(batch_obstacles, batch_obstacle_visibility, grid_cells, scaling)
static_gmms = batch_fit_gmm_to_obstacles(robot_centric_static_obstacles, visible_obstacles_mask, grid_cells)
static_ps = vmap(gmm.batch_p, in_axes=(0, None))(static_gmms, gmm_sample_points)

### Animation with 3 views: robot-centred simulation view (with grid, humans, obstacles), dynamic obstacles GMM view, static obstacles GMM view
# Matplotlib setting
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 20
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
colors = list(mcolors.TABLEAU_COLORS.values())
scenarios_titles = [
    "Circular crossing", 
    "Parallel traffic", 
    "Perpendicular traffic", 
    "Robot crowding", 
    "Delayed circular crossing",
    "Circular crossing with static obstacles",
    "Crowd navigation",
    "Corner traffic",
]
# Fit GMM to dynamic obstacles (humans) positions
fig, axs = plt.subplots(1, 3, figsize=(19.21,11.22))
fig.suptitle(f"{scenarios_titles[info['current_scenario']]}\n{n_humans} {humans_policy}-driven humans - {n_obstacles} obstacles", fontsize=35)
fig.subplots_adjust(right=0.99, left=0.04, top=0.85, bottom=0.15, wspace=0.)
ax_titles = ["Robot-centric simulation", "Dynamic obstacles GMM", "Static obstacles GMM"]
for i, ax in enumerate(axs):
    ax.set_title(ax_titles[i])
    ax.set(xlim=[-7,7],ylim=[-7,7])
    ax.set_xlabel('X')
    if i == 0: ax.set_ylabel('Y', labelpad=-13)
    else: ax.set_yticks([], labels=[])
    ax.set_aspect('equal', adjustable='box')
def animate(frame):
    # Reset axes
    for i, ax in enumerate(axs):
        ax.clear()
        ax.set_title(ax_titles[i])
        ax.set(xlim=[-7,7],ylim=[-7,7])
        ax.set_xlabel('X')
        if i == 0: ax.set_ylabel('Y', labelpad=-13)
        else: ax.set_yticks([], labels=[])
        ax.set_aspect('equal', adjustable='box')
        for cell_center in grid_cells:
            rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
    ## AXS[0] - Plot robot-centric simulation
    # Plot robot goal
    axs[0].scatter(robot_centric_robot_goals[frame,0], robot_centric_robot_goals[frame,1], marker="*", color="red", zorder=2)
    # Plot humans
    for h in range(len(robot_centric_humans_positions[frame])):
        if humans_policy == 'hsfm':
            head = plt.Circle((robot_centric_humans_positions[frame][h,0] + jnp.cos(robot_centric_humans_orientations[frame][h]) * data['humans_radii'][h], robot_centric_humans_positions[frame][h,1] + jnp.sin(robot_centric_humans_orientations[frame][h]) * data['humans_radii'][h]), 0.1, color=colors[h%len(colors)], zorder=1)
            axs[0].add_patch(head)
        circle = plt.Circle((robot_centric_humans_positions[frame][h,0], robot_centric_humans_positions[frame][h,1]), data['humans_radii'][h], edgecolor=colors[h%len(colors)], facecolor="white", fill=True, zorder=1)
        axs[0].add_patch(circle)
    # Plot robot
    if kinematics == 'unicycle':
        head = plt.Circle((robot_radius, 0.), 0.1, color='black', zorder=1)
        axs[0].add_patch(head)
    circle = plt.Circle((0.,0.), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
    axs[0].add_patch(circle)
    # Plot static obstacles
    if robot_centric_static_obstacles[frame].shape[1] > 1: # Polygon obstacles
        for o in robot_centric_static_obstacles[frame]: axs[0].fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
    else: # One segment obstacles
        for o in robot_centric_static_obstacles[frame]: axs[0].plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)
    # Plot visibility rays
    for angle, distance in zip(visibility_angles[frame], visibility_distances[frame]):
        axs[0].plot(
            [0, distance * jnp.cos(angle)],
            [0, distance * jnp.sin(angle)],
            color='blue',
            linewidth=0.5,
            alpha=0.3,
            zorder=0,
        )
    # Legend
    robot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Robot', markerfacecolor='red', markeredgecolor='black', markersize=15)
    human_legend = plt.Line2D([0], [0], marker='o', color='w', label='Humans', markerfacecolor='white', markeredgecolor='black', markersize=15)
    goal_legend = plt.Line2D([0], [0], marker='*', color='w', label='Goal', markerfacecolor='red', markeredgecolor='red', markersize=10)
    obstacle_legend = plt.Line2D([0], [0], color='black', lw=2, label='Obstacles')
    visibility_lines_legend = plt.Line2D([0], [0], color='blue', lw=1, label='Visibility rays')
    axs[0].legend(handles=[robot_legend, human_legend, goal_legend, obstacle_legend, visibility_lines_legend], loc='upper center', bbox_to_anchor=(0.5, -0.125), fontsize=15)
    ## AXS[1] - Plot dynamic obstacles GMM
    # Plot humans positions
    for h in range(len(robot_centric_humans_positions[frame])):
        color = "red" if visible_humans_mask[frame][h] else "grey"
        alpha = 0.5 if visible_humans_mask[frame][h] else 0.3
        circle = plt.Circle((robot_centric_humans_positions[frame][h,0], robot_centric_humans_positions[frame][h,1]), data['humans_radii'][h], color=color, fill=True, zorder=11, alpha=alpha)
        axs[1].add_patch(circle)
    # Plot color-coded GMM samples
    points_high_p = gmm_sample_points[dynamic_ps[frame] > p_visualization_threshold]
    corresponding_colors = dynamic_ps[frame][dynamic_ps[frame] > p_visualization_threshold]
    axs[1].scatter(
        points_high_p[:,0], 
        points_high_p[:,1],
        s=8,
        c=corresponding_colors,
        cmap='viridis',
        zorder=10,
    )
    # Legend
    visible_human_legend = plt.Line2D([0], [0], marker='o', color='w', label='Visible humans', markerfacecolor='red', markeredgecolor='red', markersize=15, alpha=0.5)
    invisible_human_legend = plt.Line2D([0], [0], marker='o', color='w', label='Invisible humans', markerfacecolor='grey', markeredgecolor='grey', markersize=15, alpha=0.3)
    axs[1].legend(handles=[visible_human_legend, invisible_human_legend], loc='upper center', bbox_to_anchor=(0.5, -0.125), fontsize=15)
    ## AXS[2] - Plot static obstacles GMM
    for i, o in enumerate(robot_centric_static_obstacles[frame]): 
        for j, s in enumerate(o):
            color = 'black' if visible_obstacles_mask[frame][i,j] else 'grey'
            linestyle = 'solid' if visible_obstacles_mask[frame][i,j] else 'dashed'
            alpha = 0.5 if visible_obstacles_mask[frame][i,j] else 0.3
            axs[2].plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
    # Plot color-coded GMM samples
    points_high_p = gmm_sample_points[static_ps[frame] > p_visualization_threshold]
    corresponding_colors = static_ps[frame][static_ps[frame] > p_visualization_threshold]
    axs[2].scatter(
        points_high_p[:,0], 
        points_high_p[:,1],
        s=8,
        c=corresponding_colors,
        cmap='viridis',
        zorder=10,
    )
    # Legend
    visible_obstacle_legend = plt.Line2D([0], [0], color='black', lw=2, label='Visible obstacles', alpha=0.5)
    invisible_obstacle_legend = plt.Line2D([0], [0], color='grey', lw=2, label='Invisible obstacles', alpha=0.3, linestyle='dashed')
    axs[2].legend(handles=[visible_obstacle_legend, invisible_obstacle_legend], loc='upper center', bbox_to_anchor=(0.5, -0.125), fontsize=15)
anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=len(all_states))
if save:
    save_path = os.path.join(os.path.dirname(__file__), f'target_gmms_{scenario}_{n_humans}humans_{n_obstacles}obstacles.mp4')
    writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
    anim.save(save_path, writer=writer_video, dpi=300)
anim.paused = False
def toggle_pause(self, *args, **kwargs):
    if anim.paused: anim.resume()
    else: anim.pause()
    anim.paused = not anim.paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)
plt.show()