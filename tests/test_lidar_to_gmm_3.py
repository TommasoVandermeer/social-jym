from jax import random, jit, vmap, lax, nn, value_and_grad, debug
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import haiku as hk
import optax
from functools import partial
import time

from socialjym.policies.dir_safe import DIRSAFE
from socialjym.utils.distributions.gaussian_mixture_model import GMM
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import plot_lidar_measurements

### Parameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 10_000  # Number of labeled examples to train Lidar to GMM network
grid_resolution = 11  # Number of grid cells per dimension
grid_limit = 4  # Distance limit of the grid in each dimension (from -limit to +limit)
visibility_threshold_from_grid = 0.5  # Distance from grid limit to consider an object inside the grid
n_loss_samples = 1000  # Number of samples to estimate the loss
prediction_horizon = 2  # Number of steps ahead to predict next GMM (in seconds it is prediction_horizon * robot_dt)
learning_rate = 1e-3
batch_size = 200
n_epochs = 20
# Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_visible = True
robot_vmax = 1.0
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 100
scenario = "hybrid_scenario"
n_humans = 5
n_obstacles = 3
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
            'reward_function': DummyReward(kinematics=kinematics), # Don't care about reward using a trained policy
            'kinematics': kinematics,
            'lidar_angular_range':lidar_angular_range,
            'lidar_max_dist':lidar_max_dist,
            'lidar_num_rays':lidar_num_rays,
        }
env = SocialNav(**env_params)
# Robot policy
policy = DIRSAFE(env.reward_function, v_max=robot_vmax, dt=env_params['robot_dt'])
with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']
# Build local grid over which the GMM is defined
visibility_threshold = visibility_threshold_from_grid + grid_limit
ax_lim = visibility_threshold+2
size = (grid_limit * 2 / grid_resolution)
cell_size = jnp.array([size, size])
grid_edges = jnp.linspace(-grid_limit, grid_limit, grid_resolution + 1, endpoint=True)
dists = grid_edges[:-1] + size / 2  # Cell center distances
grid_cell_coords = jnp.meshgrid(dists, dists)
grid_cells = jnp.array(jnp.vstack((grid_cell_coords[0].flatten(), grid_cell_coords[1].flatten())).T)
gmm = GMM(n_dimensions=grid_cells.shape[1], n_components=grid_cells.shape[0])

### Parameters validation
assert n_loss_samples % (n_humans + n_obstacles) == 0, "n_loss_samples must be divisible by (n_humans + n_obstacles)"
assert (n_loss_samples / (n_humans + n_obstacles)) % env.static_obstacles_per_scenario.shape[2] == 0, "n_loss_samples per obstacle must be divisible by number of segments per obstacle" 

def simulate_n_steps(env, n_steps):
    @loop_tqdm(n_steps, desc="Simulating steps")
    @jit
    def _simulate_steps_with_lidar(i:int, for_val:tuple):
        ## Retrieve data from the tuple
        data, state, obs, info, outcome, reset_key = for_val
        ## Compute robot action
        action, _, _, _, _ = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
        ## Get lidar measurements and save output data
        lidar_measurements = env.get_lidar_measurements(obs[-1,:2], obs[-1,5], obs[:-1,:2], info["humans_parameters"][:,0], info["static_obstacles"][-1])
        step_out_data = {
            "episode_starts": ~outcome["nothing"],
            "lidar_measurements": lidar_measurements,
            "humans_positions": obs[:-1,:2],
            "humans_velocities": obs[:-1,2:4],
            "humans_radii": info["humans_parameters"][:,0],
            "humans_orientations": state[:-1,4],
            "robot_positions": obs[-1,:2],
            "robot_orientations": obs[-1,5],
            "robot_actions": action,
            "robot_goals": info["robot_goal"],
            "static_obstacles": info["static_obstacles"][-1],
        }
        data = tree_map(lambda x, y: x.at[i].set(y), data, step_out_data)
        ## Simulate one step
        final_state, final_obs, final_info, _, final_outcome, final_reset_key = env.step(
            state,
            info,
            action, 
            test=False,
            reset_if_done=True,
            reset_key=reset_key
        )
        return data, final_state, final_obs, final_info, final_outcome, final_reset_key
    # Initialize first episode
    state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
    # Initialize setting data
    data = {
        "episode_starts": jnp.zeros((n_steps,), dtype=bool),
        "lidar_measurements": jnp.zeros((n_steps,lidar_num_rays,2)),
        "humans_positions": jnp.zeros((n_steps,n_humans,2)),
        "humans_velocities": jnp.zeros((n_steps,n_humans,2)),
        "humans_orientations": jnp.zeros((n_steps,n_humans)),
        "humans_radii": jnp.zeros((n_steps,n_humans)),
        "robot_positions": jnp.zeros((n_steps,2)),
        "robot_orientations": jnp.zeros((n_steps,)),
        "robot_actions": jnp.zeros((n_steps,2)),
        "robot_goals": jnp.zeros((n_steps,2)),
        "static_obstacles": jnp.zeros((n_steps,n_obstacles,1,2,2)),
    }
    # Step loop
    data, _, _, _, _, _ = lax.fori_loop(
        0,
        n_steps,
        _simulate_steps_with_lidar,
        (data, state, obs, info, outcome, reset_key)
    )
    data["episode_starts"] = data["episode_starts"].at[0].set(True)  # First step is always episode start
    return data

### GENERATE DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl')):
    ## GENERATE RAW DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl')):
        # Generate raw data
        raw_data = simulate_n_steps(env, n_steps)
        # Save raw data dataset
        with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'wb') as f:
            pickle.dump(raw_data, f)
    else:
        # Load raw data dataset
        with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'rb') as f:
            raw_data = pickle.load(f)
    ## GENERATE ROBOT-CENTERED DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl')):
        dataset = {
            "episode_starts": raw_data["episode_starts"],
            "rc_lidar_measurements": jnp.zeros((n_steps, lidar_num_rays, 2)),
            "rc_humans_positions": jnp.zeros((n_steps, n_humans, 2)),
            "rc_humans_orientations": jnp.zeros((n_steps, n_humans)),
            "rc_humans_velocities": jnp.zeros((n_steps, n_humans, 2)),
            "humans_radii": raw_data["humans_radii"],
            "robot_actions": raw_data["robot_actions"],
            "robot_positions": raw_data["robot_positions"],
            "robot_orientations": raw_data["robot_orientations"],
            "rc_robot_goals": jnp.zeros((n_steps, 2)),
            "rc_obstacles": jnp.zeros((n_steps, n_obstacles, 1, 2, 2)),
            "humans_visibility": jnp.zeros((n_steps, n_humans)),
            "obstacles_visibility": jnp.zeros((n_steps, n_obstacles, 1)),
        }
        # Compute robot-centered LiDAR measurements
        dataset["rc_lidar_measurements"] = dataset["rc_lidar_measurements"].at[:,:,0].set(raw_data["lidar_measurements"][:,:,0])  # Ranges remain the same
        dataset["rc_lidar_measurements"] = dataset["rc_lidar_measurements"].at[:,:,1].set(raw_data["lidar_measurements"][:,:,1] - raw_data["robot_orientations"][:,None])  # Angles are rotated to be in the robot frame
        # Compute robot-centered humans positions
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
        def roto_translate_poses_and_vels(positions, orientations, velocities, ref_positions, ref_orientations):
            return vmap(roto_translate_pose_and_vel, in_axes=(0, 0, 0, None, None))(positions, orientations, velocities, ref_positions, ref_orientations)
        @jit
        def batch_roto_translate_poses_and_vels(positions, orientations, velocities, ref_positions, ref_orientations):
            return vmap(roto_translate_poses_and_vels, in_axes=(0, 0, 0, 0, 0))(positions, orientations, velocities, ref_positions, ref_orientations)
        dataset["rc_humans_positions"], dataset["rc_humans_orientations"], dataset["rc_humans_velocities"] = batch_roto_translate_poses_and_vels(
            raw_data['humans_positions'],
            raw_data['humans_orientations'],
            raw_data['humans_velocities'],
            raw_data['robot_positions'],
            raw_data['robot_orientations'],
        )
        # Compute robot-centered robot goals
        @jit
        def batch_roto_translate_goals(goals, ref_positions, ref_orientations):
            return vmap(roto_translate_pose_and_vel, in_axes=(0, None, None, 0, 0))(goals, jnp.array([0.]), jnp.array([0.,0.]), ref_positions, ref_orientations)[0]
        dataset["rc_robot_goals"] = batch_roto_translate_goals(
            raw_data["robot_goals"],
            raw_data['robot_positions'],
            raw_data['robot_orientations'],
        )
        # Compute robot-centered static obstacles
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
        dataset["rc_obstacles"] = batch_roto_translate_obstacles(
            raw_data['static_obstacles'],
            raw_data['robot_positions'],
            raw_data['robot_orientations'],
        )
        # Compute humans and obstacles visibility
        @jit
        def _object_visibility(humans_positions, humans_radii, static_obstacles, epsilon=1e-5):
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
            _, human_collision_idxs, obstacle_collision_idxs = env.batch_ray_cast(
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
            return humans_visibility_mask, obstacles_visibility_mask
        @jit
        def batch_object_visibility(batch_humans_positions, humans_radii, batch_static_obstacles):
            return vmap(_object_visibility, in_axes=(0, 0, 0))(batch_humans_positions, humans_radii, batch_static_obstacles)
        dataset["humans_visibility"], dataset["obstacles_visibility"] = batch_object_visibility(
            dataset['rc_humans_positions'],
            dataset['humans_radii'],
            dataset['rc_obstacles'],
        )
        # Assert if humans and obstacles are closer than visibility_threshold_from_grid to the robot
        @jit
        def _object_is_inside_grid(humans_positions, humans_radii, static_obstacles):
            # Humans
            @jit
            def is_human_inside_grid(position, radius):
                return jnp.all(jnp.abs(position) < (visibility_threshold + radius))
            humans_inside_mask = vmap(is_human_inside_grid, in_axes=(0,0))(humans_positions, humans_radii)
            # Obstacles
            @jit
            def batch_obstacles_is_inside_grid(obstacles):
                return vmap(policy._batch_segment_rectangle_intersection, in_axes=(0,0,0,0,None,None,None,None))(
                    obstacles[:,:,0,0], 
                    obstacles[:,:,0,1], 
                    obstacles[:,:,1,0], 
                    obstacles[:,:,1,1], 
                    -visibility_threshold, 
                    visibility_threshold, 
                    -visibility_threshold, 
                    visibility_threshold,
                )[0]
            obstacles_inside_mask = batch_obstacles_is_inside_grid(static_obstacles)
            return humans_inside_mask, obstacles_inside_mask
        @jit
        def batch_object_is_inside_grid(batch_humans_positions, humans_radii, batch_static_obstacles):
            return vmap(_object_is_inside_grid, in_axes=(0, 0, 0))(batch_humans_positions, humans_radii, batch_static_obstacles)
        humans_inside_mask, obstacles_inside_mask = batch_object_is_inside_grid(
            dataset['rc_humans_positions'],
            dataset['humans_radii'],
            dataset['rc_obstacles'],
        )
        dataset["humans_visibility"] = dataset["humans_visibility"] & humans_inside_mask
        dataset["obstacles_visibility"] = dataset["obstacles_visibility"] & obstacles_inside_mask
        ## DEBUG: Plot frames stream for visual inspection
        from matplotlib import rc, rcParams
        from matplotlib.animation import FuncAnimation
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Plot robot-centric simulation
        fig, ax = plt.subplots(figsize=(8,8))
        def animate(frame):
            ax.clear()
            ax.set_title('Robot-Centric Frame Inspection')
            ax.set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
            ax.set_xlabel('X')
            ax.set_ylabel('Y', labelpad=-13)
            ax.set_aspect('equal', adjustable='box')
            # Plot grid
            for cell_center in grid_cells:
                rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
                ax.add_patch(rect)
            # Plot visibility grid threshold
            rect = plt.Rectangle((-visibility_threshold,-visibility_threshold), visibility_threshold * 2, visibility_threshold * 2, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
            # Plot robot goal
            ax.scatter(dataset["rc_robot_goals"][frame,0], dataset["rc_robot_goals"][frame,1], marker="*", color="red", zorder=2)
            # Plot humans
            for h in range(len(dataset["rc_humans_positions"][frame])):
                color = "green" if dataset["humans_visibility"][frame][h] else "grey"
                alpha = 1 if dataset["humans_visibility"][frame][h] else 0.3
                if humans_policy == 'hsfm':
                    head = plt.Circle((dataset["rc_humans_positions"][frame][h,0] + jnp.cos(dataset["rc_humans_orientations"][frame][h]) * dataset['humans_radii'][frame][h], dataset["rc_humans_positions"][frame][h,1] + jnp.sin(dataset["rc_humans_orientations"][frame][h]) * dataset['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                    ax.add_patch(head)
                circle = plt.Circle((dataset["rc_humans_positions"][frame][h,0], dataset["rc_humans_positions"][frame][h,1]), dataset['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
                ax.add_patch(circle)
            # Plot robot
            if kinematics == 'unicycle':
                head = plt.Circle((robot_radius, 0.), 0.1, color='black', zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((0.,0.), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
            ax.add_patch(circle)
            # Plot static obstacles
            for i, o in enumerate(dataset["rc_obstacles"][frame]):
                for j, s in enumerate(o):
                    color = 'black' if dataset["obstacles_visibility"][frame][i,j] else 'grey'
                    linestyle = 'solid' if dataset["obstacles_visibility"][frame][i,j] else 'dashed'
                    alpha = 1 if dataset["obstacles_visibility"][frame][i,j] else 0.3
                    ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
            # Plot lidar scans
            for distance, angle in dataset["rc_lidar_measurements"][frame]:
                ax.plot(
                    [0, distance * jnp.cos(angle)],
                    [0, distance * jnp.sin(angle)],
                    color='blue',
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=0,
                )
        anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()
        # Save robot-centered dataset
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        data = dataset
    else:
        # Load robot-centered dataset
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
            data = pickle.load(f)
    ### GENERATE LIDAR TO GMM SAMPLES DATASET
    # Initialize final dataset
    dataset = {
        "inputs": jnp.zeros((n_steps, n_stack, 2 * lidar_num_rays + 2)),  # Network input: n_stack * [2 * lidar_rays + action]
        "target_samples": jnp.full((n_steps, n_loss_samples, 2), jnp.nan), # Network target: samples from the rc humans and obstacles positions
        "next_target_samples": jnp.full((n_steps, n_loss_samples, 2), jnp.nan), # Network target: samples from the next step rc humans and obstacles positions
    }
    # Build inputs
    @jit
    def build_input_i(
        i:int, 
        rc_lidar_measurements:jnp.ndarray, 
        robot_actions:jnp.ndarray, 
        robot_positions:jnp.ndarray, 
        robot_orientations:jnp.ndarray,
    ) -> jnp.ndarray:
        lidars = lax.dynamic_slice_in_dim(rc_lidar_measurements, i - n_stack + 1, n_stack, axis=0)  # Shape: (n_stack, lidar_num_rays, 2)
        positions = lax.dynamic_slice_in_dim(robot_positions, i - n_stack + 1, n_stack, axis=0)  # Shape: (n_stack, 2)
        orientations = lax.dynamic_slice_in_dim(robot_orientations, i - n_stack + 1, n_stack, axis=0)  # Shape: (n_stack,)
        actions = lax.dynamic_slice_in_dim(robot_actions, i - n_stack + 1, n_stack, axis=0)  # Shape: (n_stack, 2)
        actions = actions.at[-1,:].set(jnp.zeros((2,))) # Last action is zero (we want to predict next action)
        # Align temporal LiDAR scans to i-th robot frame
        ref_position = robot_positions[i]
        ref_orientation = robot_orientations[i]
        @jit
        def align_lidar_scan(lidar_scan, position, orientation, ref_position, ref_orientation):
            # Compute cartesian coordinates of LiDAR points in world frame
            xs = lidar_scan[:,0] * jnp.cos(lidar_scan[:,1] + orientation) + position[0]
            ys = lidar_scan[:,0] * jnp.sin(lidar_scan[:,1] + orientation) + position[1]
            points_world = jnp.stack((xs, ys), axis=-1)  # Shape: (lidar_num_rays, 2)
            # Roto-translate points to robot frame
            c, s = jnp.cos(ref_orientation), jnp.sin(ref_orientation)
            R = jnp.array([
                [c, -s],
                [s,  c]
            ])
            points_robot = jnp.dot(points_world - ref_position, R)
            return points_robot
        point_cloud = vmap(align_lidar_scan, in_axes=(0, 0, 0, None, None))(
            lidars, 
            positions, 
            orientations, 
            ref_position, 
            ref_orientation
        )  # Shape: (n_stack, lidar_num_rays, 2)
        point_cloud = point_cloud.reshape((n_stack, lidar_num_rays * 2))  # Shape: (n_stack, 2 * lidar_num_rays)
        return jnp.concatenate([point_cloud, actions], axis=-1) # Shape: (n_stack, 2 * lidar_num_rays + 2)
    @jit
    def build_inputs(
        idxs:jnp.ndarray,
        padded_rc_lidar_measurements:jnp.ndarray, 
        padded_robot_actions:jnp.ndarray, 
        padded_robot_positions:jnp.ndarray, 
        padded_robot_orientations:jnp.ndarray,
    ) -> jnp.ndarray:
        return vmap(build_input_i, in_axes=(0, None, None, None, None))(
            idxs, 
            padded_rc_lidar_measurements, 
            padded_robot_actions, 
            padded_robot_positions, 
            padded_robot_orientations
        )
    num_episodes = jnp.sum(data["episode_starts"])
    start_idxs = jnp.append(jnp.where(data["episode_starts"], size=num_episodes)[0], n_steps)
    for i in range(num_episodes):
        length = start_idxs[i+1] - start_idxs[i]
        dataset["inputs"] = dataset["inputs"].at[start_idxs[i]:start_idxs[i+1]].set(
            build_inputs(
                jnp.arange(n_stack-1, length+n_stack-1),
                jnp.concatenate((jnp.tile(data["rc_lidar_measurements"][start_idxs[i],:,:], (n_stack-1, 1, 1)), data["rc_lidar_measurements"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(jnp.zeros((2,)), (n_stack-1, 1)), data["robot_actions"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(data["robot_positions"][start_idxs[i],:], (n_stack-1, 1)), data["robot_positions"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(data["robot_orientations"][start_idxs[i]], (n_stack-1,)), data["robot_orientations"][start_idxs[i]:start_idxs[i+1]]), axis=0),
            )
        )
    ## DEBUG: Inspect inputs
    from matplotlib import rc, rcParams
    from matplotlib.animation import FuncAnimation
    rc('font', weight='regular', size=20)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Network Inputs Inspection')
        ax.set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot grid
        for cell_center in grid_cells:
            rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
        # Plot visibility grid threshold
        rect = plt.Rectangle((-visibility_threshold,-visibility_threshold), visibility_threshold * 2, visibility_threshold * 2, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(data["rc_humans_positions"][frame])):
            color = "green" if data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((data["rc_humans_positions"][frame][h,0] + jnp.cos(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h], data["rc_humans_positions"][frame][h,1] + jnp.sin(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((data["rc_humans_positions"][frame][h,0], data["rc_humans_positions"][frame][h,1]), data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot lidar scans
        point_clouds = dataset["inputs"][frame][:, :-2].reshape((n_stack, lidar_num_rays, 2))
        for i, cloud in enumerate(point_clouds):
            # color/alpha fade with i (smaller i -> more faded)
            t = i / (n_stack - 1)  # 0..1
            ax.scatter(
                cloud[:,0],
                cloud[:,1],
                c=0.3 + 0.7 * jnp.ones((lidar_num_rays,)) * t,
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                alpha=0.3 + 0.7 * t,
                zorder=20 + i,
            )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    # Build target samples
    @partial(jit, static_argnames=['n_samples', 'n_humans', 'n_obstacles'])
    def build_frame_target_samples(
        humans_positions:jnp.ndarray,
        humans_radii:jnp.ndarray,
        humans_visibility:jnp.ndarray,
        static_obstacles:jnp.ndarray,
        obstacles_visibility:jnp.ndarray,
        key:random.PRNGKey,
        n_samples:int=n_loss_samples,
        n_humans:int=n_humans,
        n_obstacles:int=n_obstacles,
    ) -> jnp.ndarray:
        # TODO: Remove nan samples 
        ### Mask invisible humans and obstacles with NaNs
        humans = jnp.where(
            humans_visibility[:, None],
            jnp.concatenate((humans_positions, humans_radii[:, None]), axis=-1),
            jnp.array([jnp.nan, jnp.nan, jnp.nan])
        )  # Shape: (n_humans, 3)
        obstacles = jnp.where(
            obstacles_visibility[:,:, None, None],
            static_obstacles,
            jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])
        )  # Shape: (n_obstacles, 1, 2, 2)
        ### Split n_samples evenly with respect to n_humans + n_obstacles
        total_objects = n_humans + n_obstacles
        samples_per_object = n_samples // total_objects
        # Humans samples
        @partial(jit, static_argnames=['n_hum_samples'])
        def sample_from_human(human: jnp.ndarray, key: random.PRNGKey, n_hum_samples: int) -> jnp.ndarray:
            @jit
            def _not_nan_human(human):
                position, radius = human[0:2], human[2]
                angle_key, radius_key = random.split(key)
                angles = random.uniform(angle_key, (n_hum_samples,), minval=0.0, maxval=2*jnp.pi)
                rs = radius * jnp.sqrt(random.uniform(radius_key, (n_hum_samples,)))
                xs = position[0] + rs * jnp.cos(angles)
                ys = position[1] + rs * jnp.sin(angles)
                return jnp.stack((xs, ys), axis=-1)  # Shape: (n_hum_samples, 2)

            return lax.cond(
                jnp.any(jnp.isnan(human)),
                lambda _: jnp.full((n_hum_samples, 2), jnp.nan),
                _not_nan_human,
                human
            )
        humans_key, obstacles_key = random.split(key)
        humans_keys = random.split(humans_key, n_humans)
        humans_samples = vmap(sample_from_human, in_axes=(0, 0, None))(humans, humans_keys, samples_per_object)  # Shape: (n_humans, samples_per_object, 2)
        humans_samples = humans_samples.reshape((n_humans * samples_per_object, 2))
        # Obstacles samples
        @partial(jit, static_argnames=['n_obs_samples'])
        def sample_from_obstacle(obstacle: jnp.ndarray, key: random.PRNGKey, n_obs_samples: int) -> jnp.ndarray:
            n_seg_samples = n_obs_samples // obstacle.shape[0]
            @jit
            def _segment_samples(segment):
                @jit
                def _not_nan_segment(segment):
                    p1, p2 = segment[0], segment[1]
                    # Sample uniformly on the segment
                    ts = random.uniform(key, (n_seg_samples,), minval=0.0, maxval=1.0)
                    xs = p1[0] + ts * (p2[0] - p1[0])
                    ys = p1[1] + ts * (p2[1] - p1[1])
                    return jnp.stack((xs, ys), axis=-1)  # Shape: (n_samples, 2)
                return lax.cond(
                    jnp.any(jnp.isnan(segment)),
                    lambda _: jnp.full((n_seg_samples, 2), jnp.nan),
                    _not_nan_segment,
                    segment
                )
            return jnp.concatenate(vmap(_segment_samples, in_axes=(0,))(obstacle), axis=0)  # Shape: (n_obs_samples, 2)
        obstacles_keys = random.split(obstacles_key, n_obstacles)
        obstacles_samples = vmap(sample_from_obstacle, in_axes=(0, 0, None))(obstacles, obstacles_keys, samples_per_object)  # Shape: (n_obstacles, samples_per_object, 2)
        obstacles_samples = obstacles_samples.reshape((n_obstacles * samples_per_object, 2))
        # Combine humans and obstacles samples
        samples = jnp.concatenate((humans_samples, obstacles_samples), axis=0)  # Shape: (n_samples, 2)
        # Randomly fill nan samples with already sampled points
        nan_mask = jnp.isnan(samples).any(axis=1)
        total_nans = jnp.sum(nan_mask)
        aux_samples = jnp.nan_to_num(samples, nan=jnp.inf)  # Temporary replace NaNs with large negative number for sorting
        idxs = jnp.argsort(aux_samples, axis=0)
        samples = samples[idxs[:,0]]  # Sort samples so that NaNs are at the end
        keys = random.split(key, n_samples)
        @jit
        def fill_nan_samples(idx:int, total_nans:int, samples: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
            return lax.cond(
                idx < n_samples - total_nans,
                lambda data: data[1][idx],
                lambda data: data[1][random.randint(key, (), 0, n_samples - total_nans)],
                (total_nans, samples, key)
            )
        return lax.cond(
            total_nans < n_samples,
            lambda _: vmap(fill_nan_samples, in_axes=(0, None, None, 0))(
                    jnp.arange(n_samples),
                    total_nans,
                    samples,
                    keys
                ),
            lambda _: samples,
            None
        )
    dataset["target_samples"] = vmap(build_frame_target_samples, in_axes=(0, 0, 0, 0, 0, 0))(
        data["rc_humans_positions"],
        data["humans_radii"],
        data["humans_visibility"],
        data["rc_obstacles"],
        data["obstacles_visibility"],
        random.split(random.PRNGKey(random_seed), n_steps),
    )
    ## DEBUG: Inspect targets
    from matplotlib import rc, rcParams
    from matplotlib.animation import FuncAnimation
    rc('font', weight='regular', size=20)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Target Samples Inspection')
        ax.set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot grid
        for cell_center in grid_cells:
            rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
        # Plot visibility grid threshold
        rect = plt.Rectangle((-visibility_threshold,-visibility_threshold), visibility_threshold * 2, visibility_threshold * 2, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(data["rc_humans_positions"][frame])):
            color = "green" if data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((data["rc_humans_positions"][frame][h,0] + jnp.cos(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h], data["rc_humans_positions"][frame][h,1] + jnp.sin(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((data["rc_humans_positions"][frame][h,0], data["rc_humans_positions"][frame][h,1]), data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot target samples
        ax.scatter(
            dataset["target_samples"][frame][:,0],
            dataset["target_samples"][frame][:,1],
            c='blue',
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    # Build next target samples
    dataset["next_target_samples"] = vmap(build_frame_target_samples, in_axes=(0, 0, 0, 0, 0, 0))(
        data["rc_humans_positions"] + data["rc_humans_velocities"] * (robot_dt * prediction_horizon),
        data["humans_radii"],
        data["humans_visibility"],
        data["rc_obstacles"],
        data["obstacles_visibility"],
        random.split(random.PRNGKey(random_seed), n_steps),
    )
    ## DEBUG: Inspect next targets
    from matplotlib import rc, rcParams
    from matplotlib.animation import FuncAnimation
    rc('font', weight='regular', size=20)
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    ax_lim = visibility_threshold+2
    def animate(frame):
        ax.clear()
        ax.set_title('Next Target Samples Inspection')
        ax.set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot grid
        for cell_center in grid_cells:
            rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
        # Plot visibility grid threshold
        rect = plt.Rectangle((-visibility_threshold,-visibility_threshold), visibility_threshold * 2, visibility_threshold * 2, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(data["rc_humans_positions"][frame])):
            color = "green" if data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((data["rc_humans_positions"][frame][h,0] + jnp.cos(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h], data["rc_humans_positions"][frame][h,1] + jnp.sin(data["rc_humans_orientations"][frame][h]) * data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((data["rc_humans_positions"][frame][h,0], data["rc_humans_positions"][frame][h,1]), data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot human velocities
        for h in range(len(data["rc_humans_positions"][frame])):
            color = "green" if data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if data["humans_visibility"][frame][h] else 0.3
            if data["humans_visibility"][frame][h]:
                ax.arrow(
                    data["rc_humans_positions"][frame][h,0],
                    data["rc_humans_positions"][frame][h,1],
                    data["rc_humans_velocities"][frame][h,0],
                    data["rc_humans_velocities"][frame][h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    zorder=30,
                )
        # Plot static obstacles
        for i, o in enumerate(data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot target samples
        ax.scatter(
            dataset["next_target_samples"][frame][:,0],
            dataset["next_target_samples"][frame][:,1],
            c='blue',
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    # Save dataset
    with open(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
else:
    # Load datasets
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'rb') as f:
        raw_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
        robot_centric_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)

### DEFINE NEURAL NETWORK
mlp_params = {
    "activation": nn.relu,
    "activate_final": False,
    "w_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
    "b_init": hk.initializers.VarianceScaling(1/3, mode="fan_in", distribution="uniform"),
}
class LidarNetwork(hk.Module):
    def __init__(
            self,
            grid_cell_positions:jnp.ndarray,
            lidar_num_rays:int,
            n_stack:int,
            mlp_params:dict=mlp_params,
        ) -> None:
        super().__init__()  
        self.gmm_means = grid_cell_positions  # Fixed means
        self.n_gmm_cells = grid_cell_positions.shape[0]
        self.lidar_rays = lidar_num_rays
        self.n_stack = n_stack
        self.n_inputs = n_stack * (2 * lidar_num_rays + 2)
        self.n_outputs = self.n_gmm_cells * 3 * 2  # 3 outputs per GMM cell (var_x, var_y, weight) times  2 GMMs (current and next)
        self.mlp = hk.nets.MLP(
            **mlp_params, 
            output_sizes=[self.n_inputs * 3, self.n_inputs, self.n_inputs // 2, self.n_outputs], 
            name="mlp"
        )

    def __call__(
            self, 
            x: jnp.ndarray
        ) -> jnp.ndarray:
        """
        Maps Lidar scan to GMM parameters
        """
        mlp_output = self.mlp(x)
        ### Separate outputs
        x_vars = nn.softplus(mlp_output[:, :self.n_gmm_cells]) + 1e-3  # Variance in x
        y_vars = nn.softplus(mlp_output[:, self.n_gmm_cells:2*self.n_gmm_cells]) + 1e-3  # Variance in y
        weights = nn.softmax(mlp_output[:, 2*self.n_gmm_cells:3*self.n_gmm_cells], axis=-1)  # Weights
        next_x_vars = nn.softplus(mlp_output[:, 3*self.n_gmm_cells:4*self.n_gmm_cells]) + 1e-3  # Next variance in x
        next_y_vars = nn.softplus(mlp_output[:, 4*self.n_gmm_cells:5*self.n_gmm_cells]) + 1e-3  # Next variance in y
        next_weights = nn.softmax(mlp_output[:, 5*self.n_gmm_cells:], axis=-1)  # Next weights
        ### Construct current GMM parameters
        distr = {
            "means": jnp.tile(self.gmm_means, (x.shape[0], 1, 1)),  # Fixed means
            "variances": jnp.stack((x_vars, y_vars), axis=-1),  # Shape (batch_size, n_gmm_cells, 2)
            "weights": weights,  # Shape (batch_size, n_gmm_cells)
        }
        ### Construct next GMM parameters
        next_distr = {
            "means": jnp.tile(self.gmm_means, (x.shape[0], 1, 1)),  # Fixed means
            "variances": jnp.stack((next_x_vars, next_y_vars), axis=-1),  # Shape (batch_size, n_gmm_cells, 2)
            "weights": next_weights,  # Shape (batch_size, n_gmm_cells)
        }
        return distr, next_distr
@hk.transform
def lidar_to_gmm_network(x):
    net = LidarNetwork(grid_cells, lidar_num_rays, n_stack)
    return net(x)
# Initialize network
sample_input = jnp.zeros((1, n_stack * (2 * lidar_num_rays + 2)))
network = lidar_to_gmm_network
params = network.init(random.PRNGKey(random_seed), sample_input)
# Count network parameters
def count_params(params):
    return sum(jnp.prod(jnp.array(p.shape)) for layer in params.values() for p in layer.values())
n_params = count_params(params)
print(f"# Lidar network parameters: {n_params}")
# Compute test samples to visualize GMMs
s = jnp.linspace(-grid_limit, grid_limit, num=60, endpoint=True)
test_samples_x, test_samples_y = jnp.meshgrid(s, s)
test_samples = jnp.stack((test_samples_x.flatten(), test_samples_y.flatten()), axis=-1)

# ### TEST INITIAL NETWORK
# # Forward pass
# output_distr, output_next_distr = network.apply(
#     params, 
#     None, 
#     sample_input, 
# )
# distr = {k: jnp.squeeze(v) for k, v in output_distr.items()}
# next_distr = {k: jnp.squeeze(v) for k, v in output_next_distr.items()}
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))
# # Plot output current distribution
# test_p = gmm.batch_p(distr, test_samples)
# ax[0].set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
# ax[0].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[0].set_title("Random LiDAR network Output")
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# for cell_center in grid_cells:
#     rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
#     ax[0].add_patch(rect)
# ax[0].set_aspect('equal', adjustable='box')
# # Plot output next distribution
# test_p = gmm.batch_p(next_distr, test_samples)
# ax[1].set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
# ax[1].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[1].set_title("Random LiDAR network Next Output")
# ax[1].set_xlabel("X")
# ax[1].set_ylabel("Y")
# for cell_center in grid_cells:
#     rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
#     ax[1].add_patch(rect)
# ax[1].set_aspect('equal', adjustable='box')
# plt.show()

### DEFINE LOSS FUNCTION, UPDATE FUNCTIONS, AND OPTIMIZER
@jit
def _compute_loss_and_gradients(
    current_params:dict,  
    experiences:dict[str:jnp.ndarray],
    # Experiences: {"inputs":jnp.ndarray, "target_samples":jnp.ndarray, "next_target_samples":jnp.ndarray}
) -> tuple:
    @jit
    def _batch_loss_function(
        current_params:dict,
        inputs:jnp.ndarray,
        targets:jnp.ndarray,  
        next_targets:jnp.ndarray,
        ) -> jnp.ndarray:
        
        @partial(vmap, in_axes=(None, 0, 0, 0))
        def _loss_function(
            current_params:dict,
            input:jnp.ndarray,
            target:jnp.ndarray,
            next_target:jnp.ndarray
            ) -> jnp.ndarray:
            # Compute the prediction
            input = jnp.reshape(input, (1, n_stack * (2 * lidar_num_rays + 2)))
            prediction, next_prediction = network.apply(current_params, None, input)
            prediction = {k: jnp.squeeze(v) for k, v in prediction.items()}
            next_prediction = {k: jnp.squeeze(v) for k, v in next_prediction.items()}
            # Compute the loss
            loss1 = jnp.mean(gmm.batch_neglogp(prediction, target))
            loss2 = jnp.mean(gmm.batch_neglogp(next_prediction, next_target))
            loss = 0.5 * loss1 + 0.5 * loss2
            return loss
        
        return jnp.mean(_loss_function(
                current_params,
                inputs,
                targets,
                next_targets
            ))

    inputs = experiences["inputs"]
    targets = experiences["target_samples"]
    next_targets = experiences["next_target_samples"]
    # Compute the loss and gradients
    loss, grads = value_and_grad(_batch_loss_function)(
        current_params, 
        inputs,
        targets,
        next_targets
    )
    return loss, grads
@partial(jit, static_argnames=("optimizer"))
def update(
    current_params:dict, 
    optimizer:optax.GradientTransformation, 
    optimizer_state: jnp.ndarray,
    experiences:dict[str:jnp.ndarray],
    # Experiences: {"inputs":jnp.ndarray, "target_samples":jnp.ndarray, "next_target_samples":jnp.ndarray}
) -> tuple:
    # Compute loss and gradients
    loss, grads = _compute_loss_and_gradients(current_params, experiences)
    # Compute parameter updates
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    # Apply updates
    updated_params = optax.apply_updates(current_params, updates)
    return updated_params, optimizer_state, loss
# Initialize optimizer and its state
optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
optimizer_state = optimizer.init(params)

### TRAINING LOOP
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl')):
    ## DEBUG: Inspect training data
    full_nan_experiences = jnp.logical_and(jnp.isnan(dataset["target_samples"]).all(axis=(1,2)),jnp.isnan(dataset["next_target_samples"]).all(axis=(1,2)))
    print(f"# Number of full NaN experiences in training dataset: {jnp.sum(full_nan_experiences)} / {n_steps}")
    # Filter out full NaN experiences from training dataset
    dataset = tree_map(lambda x: x[~full_nan_experiences], dataset)
    n_data = dataset["inputs"].shape[0]
    print(f"# Training dataset size after filtering: {dataset['inputs'].shape[0]} experiences")
    @loop_tqdm(n_epochs, desc="Training Lidar->GMM network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        dataset, params, optimizer_state, losses = epoch_for_val
        # Shuffle dataset at the beginning of the epoch
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_data)
        shuffled_indexes = random.permutation(shuffle_key, indexes)
        epoch_data = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(shuffled_indexes, dataset)
        # Batch loop
        @jit
        def _batch_loop(
            j:int,
            batch_for_val:tuple
        ) -> tuple:
            epoch_data, params, optimizer_state, losses = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(batch_size) + j * batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, epoch_data)
            # Update parameters
            params, optimizer_state, loss = update(
                params, 
                optimizer, 
                optimizer_state,
                batch,
            )
            # Save loss
            losses = losses.at[i,j].set(loss)
            return epoch_data, params, optimizer_state, losses
        n_batches = n_data // batch_size
        _, params, optimizer_state, losses = lax.fori_loop(
            0,
            n_batches,
            _batch_loop,
            (epoch_data, params, optimizer_state, losses)
        )
        return dataset, params, optimizer_state, losses
    # Epoch loop
    _, params, optimizer_state, losses = lax.fori_loop(
        0,
        n_epochs,
        _epoch_loop,
        (dataset, params, optimizer_state, jnp.zeros((n_epochs, int(n_data // batch_size))))
    )
    # Save trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl'), 'wb') as f:
        pickle.dump(params, f)
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(jnp.arange(n_epochs), avg_losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Lidar to GMM Network Training Loss")
    fig.savefig(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_training_loss.eps'), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl'), 'rb') as f:
        params = pickle.load(f)

### TEST TRAINED NETWORK
example_idx = 57
# Forward pass
output_distr, output_next_distr = network.apply(
    params, 
    None, 
    jnp.reshape(dataset["inputs"][example_idx], (1, n_stack * (2 * lidar_num_rays + 2))), 
)
distr = {k: jnp.squeeze(v) for k, v in output_distr.items()}
next_distr = {k: jnp.squeeze(v) for k, v in output_next_distr.items()}
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
# Plot output current distribution
test_p = gmm.batch_p(distr, test_samples)
ax[0].set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
ax[0].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7, zorder=50)
ax[0].set_title("Trained LiDAR network Output")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
# Plot visibility grid threshold
rect = plt.Rectangle((-visibility_threshold,-visibility_threshold), visibility_threshold * 2, visibility_threshold * 2, edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
ax[0].add_patch(rect)
for h in range(len(robot_centric_data["rc_humans_positions"][example_idx])):
    color = "green" if robot_centric_data["humans_visibility"][example_idx][h] else "grey"
    alpha = 0.6 if robot_centric_data["humans_visibility"][example_idx][h] else 0.3
    if humans_policy == 'hsfm':
        head = plt.Circle((robot_centric_data["rc_humans_positions"][example_idx][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][example_idx][h]) * robot_centric_data['humans_radii'][example_idx][h], robot_centric_data["rc_humans_positions"][example_idx][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][example_idx][h]) * robot_centric_data['humans_radii'][example_idx][h]), 0.1, color='black', alpha=alpha, zorder=1)
        ax[0].add_patch(head)
    circle = plt.Circle((robot_centric_data["rc_humans_positions"][example_idx][h,0], robot_centric_data["rc_humans_positions"][example_idx][h,1]), robot_centric_data['humans_radii'][example_idx][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
    ax[0].add_patch(circle)
for i, o in enumerate(robot_centric_data["rc_obstacles"][example_idx]):
    for j, s in enumerate(o):
        color = 'black' if robot_centric_data["obstacles_visibility"][example_idx][i,j] else 'grey'
        linestyle = 'solid' if robot_centric_data["obstacles_visibility"][example_idx][i,j] else 'dashed'
        alpha = 0.6 if robot_centric_data["obstacles_visibility"][example_idx][i,j] else 0.3
        ax[0].plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
for cell_center in grid_cells:
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    ax[0].add_patch(rect)
ax[0].set_aspect('equal', adjustable='box')
# Plot output next distribution
test_p = gmm.batch_p(next_distr, test_samples)
ax[1].set(xlim=[-ax_lim, ax_lim], ylim=[-ax_lim, ax_lim])
ax[1].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7, zorder=50)
ax[1].set_title("Trained LiDAR network Next Output")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
for cell_center in grid_cells:
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    ax[1].add_patch(rect)
ax[1].set_aspect('equal', adjustable='box')
plt.show()