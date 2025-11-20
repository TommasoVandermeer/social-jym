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
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.jessi import JESSI
from socialjym.utils.distributions.gaussian_mixture_model import BivariateGMM
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from jhsfm.hsfm import vectorized_compute_obstacle_closest_point

save_videos = False  # Whether to save videos of the debug inspections
### Parameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 30_000  # Number of labeled examples to train Lidar to GMM network
n_gaussian_mixture_components = 10  # Number of GMM components
box_limits = jnp.array([[-2,4], [-3,3]])  # Grid limits in meters [[x_min,x_max],[y_min,y_max]]
visibility_threshold_from_grid = 0.5  # Distance from grid limit to consider an object inside the grid
n_loss_samples = 1000  # Number of samples to estimate the loss
prediction_horizon = 4  # Number of steps ahead to predict next GMM (in seconds it is prediction_horizon * robot_dt)
max_humans_velocity = 1.5  # Maximum humans velocity (m/s) used to compute the maximum displacement in the prediction horizon
negative_samples_threshold = 0.2 # Distance threshold from objects to consider a sample as negative (in meters)
learning_rate = 1e-3
batch_size = 200
n_epochs = 1000
p_visualization_threshold = 0.05  # Minimum probability threshold to visualize GMM components
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
# DIR-SAFE policy
dir_safe = DIRSAFE(env.reward_function, v_max=robot_vmax, dt=env_params['robot_dt'])
# JESSI policy
jessi = JESSI(
    v_max=robot_vmax, 
    dt=env_params['robot_dt'], 
    lidar_num_rays=lidar_num_rays, 
    lidar_max_dist=lidar_max_dist,
    lidar_angular_range=lidar_angular_range,
    n_stack=n_stack, 
    gmm_means_limits=box_limits, 
    n_gmm_components=n_gaussian_mixture_components, 
    prediction_horizon=prediction_horizon, 
    max_humans_velocity=max_humans_velocity
)
with open(os.path.join(os.path.dirname(__file__), 'best_dir_safe.pkl'), 'rb') as f:
    actor_params = pickle.load(f)['actor_params']
# Build local grid over which the GMM is defined
visibility_thresholds = jnp.array([
    [box_limits[0,0] - visibility_threshold_from_grid, box_limits[0,1] + visibility_threshold_from_grid],
    [box_limits[1,0] - visibility_threshold_from_grid, box_limits[1,1] + visibility_threshold_from_grid]
])
ax_lims = jnp.array([
    [visibility_thresholds[0,0]-2, visibility_thresholds[0,1]+2],
    [visibility_thresholds[1,0]-2, visibility_thresholds[1,1]+2]
])
gmm = BivariateGMM(n_components=n_gaussian_mixture_components)

### Parameters validation
cond1 = n_loss_samples % n_humans == 0
cond2 = n_loss_samples % n_obstacles == 0
cond3 = (n_loss_samples / n_obstacles) % env.static_obstacles_per_scenario.shape[2] == 0
if not (cond1 and cond2 and cond3):
    print("\nWarning: n_loss_samples must be divisible by (n_humans) and (n_obstacles), and n_loss_samples per obstacle must be divisible by number of segments per obstacle")
    print(f"Finding suitable n_loss_samples...")
    while not (cond1 and cond2 and cond3):
        n_loss_samples += 1
        cond1 = n_loss_samples % n_humans == 0
        cond2 = n_loss_samples % n_obstacles == 0
        cond3 = (n_loss_samples / n_obstacles) % env.static_obstacles_per_scenario.shape[2] == 0
    print(f"Found! Using n_loss_samples = {n_loss_samples}\n")

def simulate_n_steps(env, n_steps):
    @loop_tqdm(n_steps, desc="Simulating steps")
    @jit
    def _simulate_steps_with_lidar(i:int, for_val:tuple):
        ## Retrieve data from the tuple
        data, state, obs, info, outcome, reset_key = for_val
        ## Compute robot action
        action, _, _, _, _ = dir_safe.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
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
        robot_centric_data = {
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
        robot_centric_data["rc_lidar_measurements"] = robot_centric_data["rc_lidar_measurements"].at[:,:,0].set(raw_data["lidar_measurements"][:,:,0])  # Ranges remain the same
        robot_centric_data["rc_lidar_measurements"] = robot_centric_data["rc_lidar_measurements"].at[:,:,1].set(raw_data["lidar_measurements"][:,:,1] - raw_data["robot_orientations"][:,None])  # Angles are rotated to be in the robot frame
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
        robot_centric_data["rc_humans_positions"], robot_centric_data["rc_humans_orientations"], robot_centric_data["rc_humans_velocities"] = batch_roto_translate_poses_and_vels(
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
        robot_centric_data["rc_robot_goals"] = batch_roto_translate_goals(
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
        robot_centric_data["rc_obstacles"] = batch_roto_translate_obstacles(
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
        robot_centric_data["humans_visibility"], robot_centric_data["obstacles_visibility"] = batch_object_visibility(
            robot_centric_data['rc_humans_positions'],
            robot_centric_data['humans_radii'],
            robot_centric_data['rc_obstacles'],
        )
        # Assert if humans and obstacles are closer than visibility_threshold_from_grid to the robot
        @jit
        def _object_is_inside_grid(humans_positions, humans_radii, static_obstacles):
            # Humans
            @jit
            def is_human_inside_grid(position, radius):
                return (
                    (position[0] >= visibility_thresholds[0,0] - radius) &
                    (position[0] <= visibility_thresholds[0,1] + radius) &
                    (position[1] >= visibility_thresholds[1,0] - radius) &
                    (position[1] <= visibility_thresholds[1,1] + radius)
                )
            humans_inside_mask = vmap(is_human_inside_grid, in_axes=(0,0))(humans_positions, humans_radii)
            # Obstacles
            @jit
            def batch_obstacles_is_inside_grid(obstacles):
                return vmap(dir_safe._batch_segment_rectangle_intersection, in_axes=(0,0,0,0,None,None,None,None))(
                    obstacles[:,:,0,0], 
                    obstacles[:,:,0,1], 
                    obstacles[:,:,1,0], 
                    obstacles[:,:,1,1], 
                    visibility_thresholds[0,0], 
                    visibility_thresholds[0,1], 
                    visibility_thresholds[1,0], 
                    visibility_thresholds[1,1],
                )[0]
            obstacles_inside_mask = batch_obstacles_is_inside_grid(static_obstacles)
            return humans_inside_mask, obstacles_inside_mask
        @jit
        def batch_object_is_inside_grid(batch_humans_positions, humans_radii, batch_static_obstacles):
            return vmap(_object_is_inside_grid, in_axes=(0, 0, 0))(batch_humans_positions, humans_radii, batch_static_obstacles)
        humans_inside_mask, obstacles_inside_mask = batch_object_is_inside_grid(
            robot_centric_data['rc_humans_positions'],
            robot_centric_data['humans_radii'],
            robot_centric_data['rc_obstacles'],
        )
        robot_centric_data["humans_visibility"] = robot_centric_data["humans_visibility"] & humans_inside_mask
        robot_centric_data["obstacles_visibility"] = robot_centric_data["obstacles_visibility"] & obstacles_inside_mask
        ## DEBUG: Plot frames stream for visual inspection
        # Plot robot-centric simulation
        fig, ax = plt.subplots(figsize=(8,8))
        def animate(frame):
            ax.clear()
            ax.set_title('Robot-Centric Frame Inspection')
            ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y', labelpad=-13)
            ax.set_aspect('equal', adjustable='box')
            # Plot box limits
            rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
            # Plot visibility threshold
            rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
            ax.add_patch(rect)
            # Plot robot goal
            ax.scatter(robot_centric_data["rc_robot_goals"][frame,0], robot_centric_data["rc_robot_goals"][frame,1], marker="*", color="red", zorder=2)
            # Plot humans
            for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
                color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
                alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
                if humans_policy == 'hsfm':
                    head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                    ax.add_patch(head)
                circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
                ax.add_patch(circle)
            # Plot robot
            if kinematics == 'unicycle':
                head = plt.Circle((robot_radius, 0.), 0.1, color='black', zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((0.,0.), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
            ax.add_patch(circle)
            # Plot static obstacles
            for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
                for j, s in enumerate(o):
                    color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                    linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                    alpha = 1 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                    ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
            # Plot lidar scans
            for distance, angle in robot_centric_data["rc_lidar_measurements"][frame]:
                ax.plot(
                    [0, distance * jnp.cos(angle)],
                    [0, distance * jnp.sin(angle)],
                    color='blue',
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=0,
                )
        anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
        if save_videos:
            save_path = os.path.join(os.path.dirname(__file__), f'robot_centric_simulation.mp4')
            writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
            anim.save(save_path, writer=writer_video, dpi=300)
        anim.paused = False
        def toggle_pause(self, *args, **kwargs):
            if anim.paused: anim.resume()
            else: anim.pause()
            anim.paused = not anim.paused
        fig.canvas.mpl_connect('button_press_event', toggle_pause)
        plt.show()
        # Save robot-centered robot_centric_data
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'wb') as f:
            pickle.dump(robot_centric_data, f)
    else:
        # Load robot-centered dataset
        with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
            robot_centric_data = pickle.load(f)
    ### GENERATE LIDAR TO GMM SAMPLES DATASET
    # Initialize final dataset
    dataset = {
        "inputs": jnp.zeros((n_steps, n_stack, 2 * lidar_num_rays + 2)),  # Network input: n_stack * [2 * lidar_rays + action]
        "humans_samples": {  # Network target: samples from the rc humans positions
            "position": jnp.full((n_steps, n_loss_samples, 2), jnp.nan),
            "is_positive": jnp.zeros((n_steps, n_loss_samples), dtype=bool),
        },
        "obstacles_samples": {  # Network target: samples from the rc obstacles positions
            "position": jnp.full((n_steps, n_loss_samples, 2), jnp.nan),
            "is_positive": jnp.zeros((n_steps, n_loss_samples), dtype=bool),
        },
        "next_humans_samples": { # Network target: samples from the next step rc humans positions
            "position": jnp.full((n_steps, n_loss_samples, 2), jnp.nan),
            "is_positive": jnp.zeros((n_steps, n_loss_samples), dtype=bool),
        },
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
    num_episodes = jnp.sum(robot_centric_data["episode_starts"])
    start_idxs = jnp.append(jnp.where(robot_centric_data["episode_starts"], size=num_episodes)[0], n_steps)
    for i in range(num_episodes):
        length = start_idxs[i+1] - start_idxs[i]
        dataset["inputs"] = dataset["inputs"].at[start_idxs[i]:start_idxs[i+1]].set(
            build_inputs(
                jnp.arange(n_stack-1, length+n_stack-1),
                jnp.concatenate((jnp.tile(robot_centric_data["rc_lidar_measurements"][start_idxs[i],:,:], (n_stack-1, 1, 1)), robot_centric_data["rc_lidar_measurements"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(jnp.zeros((2,)), (n_stack-1, 1)), robot_centric_data["robot_actions"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(robot_centric_data["robot_positions"][start_idxs[i],:], (n_stack-1, 1)), robot_centric_data["robot_positions"][start_idxs[i]:start_idxs[i+1]]), axis=0),
                jnp.concatenate((jnp.tile(robot_centric_data["robot_orientations"][start_idxs[i]], (n_stack-1,)), robot_centric_data["robot_orientations"][start_idxs[i]:start_idxs[i+1]]), axis=0),
            )
        )
    ## DEBUG: Inspect inputs
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Network Inputs Inspection')
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot visibility threshold
        rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
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
    if save_videos:
        save_path = os.path.join(os.path.dirname(__file__), f'network_inputs.mp4')
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    ## Build humans samples
    @partial(jit, static_argnames=['n_samples', 'n_humans'])
    def build_frame_humans_samples(
        humans_positions:jnp.ndarray,
        humans_radii:jnp.ndarray,
        humans_visibility:jnp.ndarray,
        key:random.PRNGKey,
        n_samples:int=n_loss_samples,
        n_humans:int=n_humans,
    ) -> jnp.ndarray:
        # TODO: Remove nan samples 
        ### Mask invisible humans and obstacles with NaNs
        humans = jnp.where(
            humans_visibility[:, None],
            jnp.concatenate((humans_positions, humans_radii[:, None]), axis=-1),
            jnp.array([jnp.nan, jnp.nan, jnp.nan])
        )  # Shape: (n_humans, 3)
        ### Split n_samples evenly with respect to humans
        samples_per_object = n_samples // n_humans
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
        humans_keys = random.split(key, n_humans)
        humans_samples = vmap(sample_from_human, in_axes=(0, 0, None))(humans, humans_keys, samples_per_object)  # Shape: (n_humans, samples_per_object, 2)
        humans_samples = humans_samples.reshape((n_humans * samples_per_object, 2))
        # Randomly fill nan samples with negative samples
        nan_mask = jnp.isnan(humans_samples).any(axis=1)
        total_nans = jnp.sum(nan_mask)
        aux_samples = jnp.nan_to_num(humans_samples, nan=jnp.inf)  # Temporary replace NaNs with large negative number for sorting
        idxs = jnp.argsort(aux_samples, axis=0)
        positive = vmap(lambda x: x < n_loss_samples - total_nans)(jnp.arange(n_samples))
        humans_samples = { 
            "position": humans_samples[idxs[:,0]],  # Sort samples so that NaNs are at the end
            "is_positive": positive,
        }
        keys = random.split(key, n_samples)
        @jit
        def fill_nan_samples_with_negatives(sample: jnp.ndarray, key: random.PRNGKey, humans: jnp.ndarray) -> jnp.ndarray:
            @jit
            def find_negative_sample(val:tuple) -> jnp.ndarray:
                _, key, humans = val
                def _while_body(state):
                    key, _, is_positive, humans = state
                    key, subkey = random.split(key)
                    x_key, y_key = random.split(subkey)
                    x = random.uniform(x_key, minval=box_limits[0,0], maxval=box_limits[0,1])
                    y = random.uniform(y_key, minval=box_limits[1,0], maxval=box_limits[1,1])
                    pos = jnp.array([x, y])
                    is_positive = jnp.any(jnp.linalg.norm(pos - humans[:,:2], axis=1) < humans[:,2] + negative_samples_threshold)
                    return key, pos, is_positive, humans
                _, sample_position, _, _ = lax.while_loop(
                    lambda state: state[2],
                    _while_body,
                    (key, jnp.array([0.,0.]), True, humans)
                )
                return {"position": sample_position, "is_positive": False}
            return lax.cond(
                sample["is_positive"],
                lambda x: {"position": x[0]["position"], "is_positive": x[0]["is_positive"]},
                find_negative_sample,
                (sample, key, humans)
            )
        return vmap(fill_nan_samples_with_negatives, in_axes=(0, 0, None))(
            humans_samples,
            keys,
            humans,
        )
    dataset["humans_samples"] = vmap(build_frame_humans_samples, in_axes=(0, 0, 0, 0))(
        robot_centric_data["rc_humans_positions"],
        robot_centric_data["humans_radii"],
        robot_centric_data["humans_visibility"],
        random.split(random.PRNGKey(random_seed), n_steps),
    )
    ## DEBUG: Inspect targets
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Humans Samples Inspection')
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot visibility threshold
        rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'grey'
                linestyle = 'dashed'
                alpha = 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot target samples
        col = ['red', 'blue']
        ax.scatter(
            dataset["humans_samples"]["position"][frame][:,0],
            dataset["humans_samples"]["position"][frame][:,1],
            c=[col[int(is_pos)] for is_pos in dataset["humans_samples"]["is_positive"][frame]],
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    if save_videos:
        save_path = os.path.join(os.path.dirname(__file__), f'humans_samples.mp4')
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    ## Build next humans samples
    dataset["next_humans_samples"] = vmap(build_frame_humans_samples, in_axes=(0, 0, 0, 0))(
        robot_centric_data["rc_humans_positions"] + robot_centric_data["rc_humans_velocities"] * (robot_dt * prediction_horizon),
        robot_centric_data["humans_radii"],
        robot_centric_data["humans_visibility"],
        random.split(random.PRNGKey(random_seed), n_steps),
    )
    ## DEBUG: Inspect next targets
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Next Humans Samples Inspection')
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot visibility threshold
        rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot human velocities
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 1 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if robot_centric_data["humans_visibility"][frame][h]:
                ax.arrow(
                    robot_centric_data["rc_humans_positions"][frame][h,0],
                    robot_centric_data["rc_humans_positions"][frame][h,1],
                    robot_centric_data["rc_humans_velocities"][frame][h,0],
                    robot_centric_data["rc_humans_velocities"][frame][h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    zorder=30,
                )
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'grey'
                linestyle = 'dashed'
                alpha = 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot target samples
        col = ['red', 'blue']
        ax.scatter(
            dataset["next_humans_samples"]["position"][frame][:,0],
            dataset["next_humans_samples"]["position"][frame][:,1],
            c=[col[int(is_pos)] for is_pos in dataset["next_humans_samples"]["is_positive"][frame]],
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    if save_videos:
        save_path = os.path.join(os.path.dirname(__file__), f'next_humans_samples.mp4')
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()
    ## Build obstacles samples
    @partial(jit, static_argnames=['n_samples', 'n_obstacles'])
    def build_frame_obstacles_samples(
        static_obstacles:jnp.ndarray,
        obstacles_visibility:jnp.ndarray,
        key:random.PRNGKey,
        n_samples:int=n_loss_samples,
        n_obstacles:int=n_obstacles,
    ) -> jnp.ndarray:
        ### WARNING: CURRENT IMPLEMENTATION CONSIDERS THE INSIDE OF POLIGONAL OBSTACLES AS FREE SPACE
        obstacles = jnp.where(
            obstacles_visibility[:,:, None, None],
            static_obstacles,
            jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])
        )  # Shape: (n_obstacles, 1, 2, 2)
        ### Split n_samples evenly with respect to n_obstacles
        samples_per_object = n_samples // n_obstacles
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
        obstacles_keys = random.split(key, n_obstacles)
        obstacles_samples = vmap(sample_from_obstacle, in_axes=(0, 0, None))(obstacles, obstacles_keys, samples_per_object)  # Shape: (n_obstacles, samples_per_object, 2)
        obstacles_samples = obstacles_samples.reshape((n_obstacles * samples_per_object, 2))
        # Randomly fill nan samples with negative samples
        nan_mask = jnp.isnan(obstacles_samples).any(axis=1)
        total_nans = jnp.sum(nan_mask)
        aux_samples = jnp.nan_to_num(obstacles_samples, nan=jnp.inf)  # Temporary replace NaNs with large negative number for sorting
        idxs = jnp.argsort(aux_samples, axis=0)
        positive = vmap(lambda x: x < n_loss_samples - total_nans)(jnp.arange(n_samples))
        obstacles_samples = { 
            "position": obstacles_samples[idxs[:,0]],  # Sort samples so that NaNs are at the end
            "is_positive": positive,
        }
        keys = random.split(key, n_samples)
        @jit
        def fill_nan_samples_with_negatives(sample: jnp.ndarray, key: random.PRNGKey, obstacles: jnp.ndarray) -> jnp.ndarray:
            @jit
            def find_negative_sample(val:tuple) -> jnp.ndarray:
                _, key, obstacles = val
                def _while_body(state):
                    key, _, is_positive, obstacles = state
                    key, subkey = random.split(key)
                    x_key, y_key = random.split(subkey)
                    x = random.uniform(x_key, minval=box_limits[0,0], maxval=box_limits[0,1])
                    y = random.uniform(y_key, minval=box_limits[1,0], maxval=box_limits[1,1])
                    pos = jnp.array([x, y])
                    closest_points = vectorized_compute_obstacle_closest_point(pos, obstacles)
                    is_positive = jnp.any(jnp.linalg.norm(pos - closest_points, axis=1) < negative_samples_threshold)
                    return key, pos, is_positive, obstacles
                _, sample_position, _, _ = lax.while_loop(
                    lambda state: state[2],
                    _while_body,
                    (key, jnp.array([0.,0.]), True, obstacles)
                )
                return {"position": sample_position, "is_positive": False}
            return lax.cond(
                sample["is_positive"],
                lambda x: {"position": x[0]["position"], "is_positive": x[0]["is_positive"]},
                find_negative_sample,
                (sample, key, obstacles)
            )
        return vmap(fill_nan_samples_with_negatives, in_axes=(0, 0, None))(
            obstacles_samples,
            keys,
            obstacles,
        )
    dataset["obstacles_samples"] = vmap(build_frame_obstacles_samples, in_axes=(0, 0, 0))(
        robot_centric_data["rc_obstacles"],
        robot_centric_data["obstacles_visibility"],
        random.split(random.PRNGKey(random_seed), n_steps),
    )
    ## DEBUG: Inspect obstacles targets
    # Plot robot-centric simulation
    fig, ax = plt.subplots(figsize=(8,8))
    def animate(frame):
        ax.clear()
        ax.set_title('Obstacles Samples Inspection')
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot visibility threshold
        rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "grey"
            alpha = 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 1 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
        # Plot target samples
        col = ['red', 'blue']
        ax.scatter(
            dataset["obstacles_samples"]["position"][frame][:,0],
            dataset["obstacles_samples"]["position"][frame][:,1],
            c=[col[int(is_pos)] for is_pos in dataset["obstacles_samples"]["is_positive"][frame]],
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
    if save_videos:
        save_path = os.path.join(os.path.dirname(__file__), f'obstacles_samples.mp4')
        writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
        anim.save(save_path, writer=writer_video, dpi=300)
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
# Initialize network
sample_input = jnp.zeros((1, n_stack * (2 * lidar_num_rays + 2)))
params = jessi.encoder.init(random.PRNGKey(random_seed), sample_input)
# Count network parameters
def count_params(params):
    return sum(jnp.prod(jnp.array(p.shape)) for layer in params.values() for p in layer.values())
n_params = count_params(params)
print(f"# Lidar network parameters: {n_params}")
# Compute test samples to visualize GMMs
sx = jnp.linspace(box_limits[0, 0], box_limits[0, 1], num=60, endpoint=True)
sy = jnp.linspace(box_limits[1, 0], box_limits[1, 1], num=60, endpoint=True)
test_samples_x, test_samples_y = jnp.meshgrid(sx, sy)
test_samples = jnp.stack((test_samples_x.flatten(), test_samples_y.flatten()), axis=-1)

# ### TEST INITIAL NETWORK
# # Forward pass
# obs_distr, hum_distr, next_hum_distr = jessi.encoder.apply(
#     params, 
#     None, 
#     sample_input, 
# )
# obs_distr = {k: jnp.squeeze(v) for k, v in obs_distr.items()}
# hum_distr = {k: jnp.squeeze(v) for k, v in hum_distr.items()}
# next_hum_distr = {k: jnp.squeeze(v) for k, v in next_hum_distr.items()}
# fig, ax = plt.subplots(1, 3, figsize=(24, 8))
# fig.subplots_adjust(left=0.03, right=0.99, wspace=0.1)
# # Plot output obstacles distribution
# test_p = gmm.batch_p(obs_distr, test_samples)
# ax[0].set(xlim=[box_limits[0,0]-1, box_limits[0,1]+1], ylim=[box_limits[1,0]-1, box_limits[1,1]+1])
# ax[0].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[0].set_title("Random Obstacles GMM")
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
# ax[0].add_patch(rect)
# ax[0].set_aspect('equal', adjustable='box')
# # Plot output humans distribution
# test_p = gmm.batch_p(hum_distr, test_samples)
# ax[1].set(xlim=[box_limits[0,0]-1, box_limits[0,1]+1], ylim=[box_limits[1,0]-1, box_limits[1,1]+1])
# ax[1].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[1].set_title("Random Humans GMM")
# ax[1].set_xlabel("X")
# ax[1].set_ylabel("Y")
# rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
# ax[1].add_patch(rect)
# ax[1].set_aspect('equal', adjustable='box')
# # Plot output next humans distribution
# test_p = gmm.batch_p(next_hum_distr, test_samples)
# ax[2].set(xlim=[box_limits[0,0]-1, box_limits[0,1]+1], ylim=[box_limits[1,0]-1, box_limits[1,1]+1])
# ax[2].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[2].set_title("Random Next Humans GMM")
# ax[2].set_xlabel("X")
# ax[2].set_ylabel("Y")
# rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
# ax[2].add_patch(rect)
# ax[2].set_aspect('equal', adjustable='box')
# plt.show()

### TRAINING LOOP
# Initialize optimizer and its state
optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
optimizer_state = optimizer.init(params)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl')):
    n_data = dataset["inputs"].shape[0]
    print(f"# Training dataset size: {dataset['inputs'].shape[0]} experiences")
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
            params, optimizer_state, loss = jessi.update_encoder(
                params, 
                optimizer, 
                optimizer_state,
                batch,
            )
            # debug.print("Epoch {x}, Batch {y}, Loss: {l}", x=i, y=j, l=loss)
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

### CHECK TRAINED NETWORK PREDICTIONS
fig, axs = plt.subplots(1,3,figsize=(24,8))
fig.subplots_adjust(left=0.05, right=0.99, wspace=0.13)
def animate(frame):
    for ax in axs:
        ax.clear()
        ax.set(xlim=[ax_lims[0,0], ax_lims[0,1]], ylim=[ax_lims[1,0], ax_lims[1,1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot visibility threshold
        rect = plt.Rectangle((visibility_thresholds[0,0], visibility_thresholds[1,0]), visibility_thresholds[0,1] - visibility_thresholds[0,0], visibility_thresholds[1,1] - visibility_thresholds[1,0], edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(rect)
        # Plot humans
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0] + jnp.cos(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h], robot_centric_data["rc_humans_positions"][frame][h,1] + jnp.sin(robot_centric_data["rc_humans_orientations"][frame][h]) * robot_centric_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((robot_centric_data["rc_humans_positions"][frame][h,0], robot_centric_data["rc_humans_positions"][frame][h,1]), robot_centric_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot human velocities
        for h in range(len(robot_centric_data["rc_humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if robot_centric_data["humans_visibility"][frame][h]:
                ax.arrow(
                    robot_centric_data["rc_humans_positions"][frame][h,0],
                    robot_centric_data["rc_humans_positions"][frame][h,1],
                    robot_centric_data["rc_humans_velocities"][frame][h,0],
                    robot_centric_data["rc_humans_velocities"][frame][h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    zorder=30,
                )
        # Plot static obstacles
        for i, o in enumerate(robot_centric_data["rc_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 0.6 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
    # Plot predicted GMM samples
    obs_distr, hum_distr, next_hum_distr = jessi.encoder.apply(
        params, 
        None, 
        jnp.reshape(dataset["inputs"][frame], (1, n_stack * (2 * lidar_num_rays + 2))), 
    )
    obs_distr = {k: jnp.squeeze(v) for k, v in obs_distr.items()}
    hum_distr = {k: jnp.squeeze(v) for k, v in hum_distr.items()}
    next_hum_distr = {k: jnp.squeeze(v) for k, v in next_hum_distr.items()}
    # print(f"Obstacles distribution covariance (frame {frame}): {gmm.covariances(obs_distr)}")
    # print(f"Humans distribution covariance (frame {frame}): {gmm.covariances(hum_distr)}")
    # print(f"Next humans distribution covariance (frame {frame}): {gmm.covariances(next_hum_distr)}")
    test_p = gmm.batch_p(obs_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    axs[0].scatter(obs_distr["means"][:,0], obs_distr["means"][:,1], c='red', s=10, marker='x', zorder=100)
    axs[0].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[0].set_title("Obstacles Predicted GMM")
    test_p = gmm.batch_p(hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    axs[1].scatter(hum_distr["means"][:,0], hum_distr["means"][:,1], c='red', s=10, marker='x', zorder=100)
    axs[1].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[1].set_title("Humans Predicted GMM")
    test_p = gmm.batch_p(next_hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    axs[2].scatter(next_hum_distr["means"][:,0], next_hum_distr["means"][:,1], c='red', s=10, marker='x', zorder=100)
    axs[2].scatter(points_high_p[:, 0], points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[2].set_title("Next Humans Predicted GMM")
anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=n_steps)
if save_videos:
    save_path = os.path.join(os.path.dirname(__file__), f'trained_network.mp4')
    writer_video = FFMpegWriter(fps=int(1/robot_dt), bitrate=1800)
    anim.save(save_path, writer=writer_video, dpi=300)
anim.paused = False
def toggle_pause(self, *args, **kwargs):
    if anim.paused: anim.resume()
    else: anim.pause()
    anim.paused = not anim.paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)
plt.show()
