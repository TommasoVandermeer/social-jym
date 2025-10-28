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
learning_rate = 1e-3
batch_size = 200
n_epochs = 10
n_iterations_fit_gmm = 20
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
size = (grid_limit * 2 / grid_resolution)
cell_size = jnp.array([size, size])
grid_edges = jnp.linspace(-grid_limit, grid_limit, grid_resolution + 1, endpoint=True)
dists = grid_edges[:-1] + size / 2  # Cell center distances
grid_cell_coords = jnp.meshgrid(dists, dists)
grid_cells = jnp.array(jnp.vstack((grid_cell_coords[0].flatten(), grid_cell_coords[1].flatten())).T)
gmm = GMM(n_dimensions=grid_cells.shape[1], n_components=grid_cells.shape[0])

def simulate_n_steps(env, n_steps):
    @loop_tqdm(n_steps, desc="Simulating steps")
    @jit
    def _simulate_steps_with_lidar(i:int, for_val:tuple):
        ## Retrieve data from the tuple
        data, state, obs, info, reset_key = for_val
        ## Compute robot action
        action, _, _, _, _ = policy.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
        ## Get lidar measurements and save output data
        lidar_measurements = env.get_lidar_measurements(obs[-1,:2], obs[-1,5], obs[:-1,:2], info["humans_parameters"][:,0], info["static_obstacles"][-1])
        step_out_data = {
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
        final_state, final_obs, final_info, _, _, final_reset_key = env.step(
            state,
            info,
            action, 
            test=False,
            reset_if_done=True,
            reset_key=reset_key
        )
        return data, final_state, final_obs, final_info, final_reset_key
    # Initialize first episode
    state, reset_key, obs, info, _ = env.reset(random.PRNGKey(random_seed))
    # Initialize setting data
    data = {
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
    data, _, _, _, _ = lax.fori_loop(
        0,
        n_steps,
        _simulate_steps_with_lidar,
        (data, state, obs, info, reset_key)
    )
    return data

### GENERATE DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_samples_dataset.pkl')):
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
        ax_lim = visibility_threshold+2
        def animate(frame):
            ax.clear()
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
    else:
        # Load robot-centered dataset
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
            data = pickle.load(f)
    ### GENERATE LIDAR TO GMM SAMPLES DATASET
    # Initialize final dataset
    dataset = {
        "inputs": jnp.zeros((n_steps, n_stack, 2 * lidar_num_rays + 2)),  # Network input: n_stack * [2 * lidar_rays + action]
        "target_samples": jnp.zeros((n_steps, n_loss_samples, 2)),
        "next_target_samples": jnp.zeros((n_steps, n_loss_samples, 2))
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
        # Retrieve stacked data
        lidars = lax.dynamic_slice_in_dim(rc_lidar_measurements, i - n_stack + 1, n_stack, 0)  # Shape: (n_stack, lidar_num_rays, 2)
        positions = lax.dynamic_slice_in_dim(robot_positions, i - n_stack + 1, n_stack, 0)  # Shape: (n_stack, 2)
        orientations = lax.dynamic_slice_in_dim(robot_orientations, i - n_stack + 1, n_stack, 0)  # Shape: (n_stack,)
        actions = lax.dynamic_slice_in_dim(robot_actions, i - n_stack + 1, n_stack, 0)  # Shape: (n_stack, 2)
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
            c, s = jnp.cos(-ref_orientation), jnp.sin(-ref_orientation)
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
        rc_lidar_measurements:jnp.ndarray, 
        robot_actions:jnp.ndarray, 
        robot_positions:jnp.ndarray, 
        robot_orientations:jnp.ndarray,
    ) -> jnp.ndarray:
        return vmap(build_input_i, in_axes=(0, None, None, None, None))(
            jnp.arange(n_steps), 
            rc_lidar_measurements, 
            robot_actions, 
            robot_positions, 
            robot_orientations
        )
    dataset["inputs"] = build_inputs(
        data["rc_lidar_measurements"],
        data["robot_actions"],
        data["robot_positions"],
        data["robot_orientations"],
    )
    # TODO: Debug input computation

    # Save dataset
    # with open(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_samples_dataset.pkl'), 'wb') as f:
    #     pickle.dump(dataset, f)
else:
    # Load datasets
    with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'rb') as f:
        raw_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
        robot_centric_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_samples_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)