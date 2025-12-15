from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import optax
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.dir_safe import DIRSAFE
from socialjym.policies.jessi import JESSI
from socialjym.utils.distributions.gaussian_mixture_model import BivariateGMM
from socialjym.envs.socialnav import SocialNav
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward as SocialNavDummyReward
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward as LaserNavDummyReward

save_videos = False  # Whether to save videos of the debug inspections
### Parameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 1_000  # Number of labeled examples to train Lidar to GMM network
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
    'kinematics': kinematics,
    'lidar_angular_range':lidar_angular_range,
    'lidar_max_dist':lidar_max_dist,
    'lidar_num_rays':lidar_num_rays,
}
env = SocialNav(**env_params, humans_policy=humans_policy, reward_function=SocialNavDummyReward(kinematics=kinematics))
laser_env = LaserNav(**env_params, n_stack=n_stack, reward_function=LaserNavDummyReward(robot_radius=robot_radius))
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
ax_visibility = 7
ax_lims = jnp.array([
    [visibility_thresholds[0,0]-ax_visibility, visibility_thresholds[0,1]+ax_visibility],
    [visibility_thresholds[1,0]-ax_visibility, visibility_thresholds[1,1]+ax_visibility]
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

def simulate_n_steps(n_steps):
    @loop_tqdm(n_steps, desc="Simulating steps")
    @jit
    def _simulate_steps_with_lidar(i:int, for_val:tuple):
        ## Retrieve data from the tuple
        data, state, obs, info, outcome, reset_key, lasernav_obs, lasernav_info = for_val
        ## Compute robot action
        action, _, _, _, _ = dir_safe.act(random.PRNGKey(0), obs, info, actor_params, sample=False)
        ## Save output data
        step_out_data = {
            "episode_starts": ~outcome["nothing"],
            "lasernav_observations": lasernav_obs,
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
        ## Simulate one step SOCIALNAV
        final_state, final_obs, final_info, _, final_outcome, final_reset_key = env.step(
            state,
            info,
            action, 
            test=False,
            reset_if_done=True,
            reset_key=reset_key
        )
        ## Simulate one step LASERNAV to update stacked observations
        final_lasernav_state, final_lasernav_obs, final_lasernav_info, _, final_lasernav_outcome, _ = laser_env.step(
            state,
            lasernav_info,
            action, 
            test=False,
            reset_if_done=True,
            reset_key=reset_key
        )
        # debug.print("Equal states: {x}", x=jnp.allclose(final_state, final_lasernav_state))
        # debug.print("Equal outcomes: {out}", out=(final_lasernav_outcome['nothing'] == final_outcome['nothing']))
        return data, final_state, final_obs, final_info, final_outcome, final_reset_key, final_lasernav_obs, final_lasernav_info
    # Initialize first episode
    state, reset_key, obs, info, outcome = env.reset(random.PRNGKey(random_seed))
    _, _, lasernav_obs, lasernav_info, _ = laser_env.reset(random.PRNGKey(random_seed))
    # Initialize setting data
    data = {
        "episode_starts": jnp.zeros((n_steps,), dtype=bool),
        "lasernav_observations": jnp.zeros((n_steps,n_stack,lidar_num_rays+6)),
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
    data, _, _, _, _, _, _, _ = lax.fori_loop(
        0,
        n_steps,
        _simulate_steps_with_lidar,
        (data, state, obs, info, outcome, reset_key, lasernav_obs, lasernav_info)
    )
    data["episode_starts"] = data["episode_starts"].at[0].set(True)  # First step is always episode start
    return data

### GENERATE DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl')):
    ## GENERATE RAW DATASET
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl')):
        # Generate raw data
        raw_data = simulate_n_steps(n_steps)
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
            "lasernav_observations": raw_data["lasernav_observations"],
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
        # Compute robot-centered simulation
        robot_centric_data["rc_humans_positions"], robot_centric_data["rc_humans_orientations"], \
        robot_centric_data["rc_humans_velocities"], robot_centric_data["rc_obstacles"], \
        robot_centric_data["rc_robot_goals"] = laser_env.batch_robot_centric_transform(
            raw_data['humans_positions'],
            raw_data['humans_orientations'],
            raw_data['humans_velocities'],
            raw_data['static_obstacles'],
            raw_data['robot_positions'],
            raw_data['robot_orientations'],
            raw_data["robot_goals"],
        )
        # Compute humans and obstacles visibility
        robot_centric_data["humans_visibility"], robot_centric_data["obstacles_visibility"] = laser_env.batch_object_visibility(
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
            for distance, angle in zip(robot_centric_data["lasernav_observations"][frame,0,6:], jessi.lidar_angles_robot_frame):
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
        "observations": robot_centric_data["lasernav_observations"],
        "rc_humans_positions": robot_centric_data["rc_humans_positions"],
        "rc_humans_velocities": robot_centric_data["rc_humans_velocities"],
        "humans_radii": robot_centric_data["humans_radii"],
        "humans_visibility": robot_centric_data["humans_visibility"],
        "rc_obstacles": robot_centric_data["rc_obstacles"],
        "obstacles_visibility": robot_centric_data["obstacles_visibility"],
    }
    ## DEBUG: Inspect inputs
    debugging_steps = 100
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
        point_clouds = jessi.process_lidar(dataset["observations"][frame])[0]  # (n_stack, lidar_num_rays, 2)
        for i, cloud in enumerate(point_clouds):
            # color/alpha fade with i (smaller i -> less faded)
            t = (1 - i / (n_stack - 1))  # in [0,1]
            ax.scatter(
                cloud[:,0],
                cloud[:,1],
                c=0.3 + 0.7 * jnp.ones((lidar_num_rays,)) * t,
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                alpha=0.3 + 0.7 * t,
                zorder=20 + n_stack - i,
            )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=debugging_steps)
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
    ## Build humans samples (just the first hundred steps for debugging)
    human_samples = jessi.batch_build_frame_humans_samples(
        robot_centric_data["rc_humans_positions"][:debugging_steps],
        robot_centric_data["humans_radii"][:debugging_steps],
        robot_centric_data["humans_visibility"][:debugging_steps],
        random.split(random.PRNGKey(random_seed), debugging_steps),
        n_samples = n_loss_samples,
        n_humans = n_humans,
        negative_samples_threshold=negative_samples_threshold,
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
            human_samples["position"][frame][:,0],
            human_samples["position"][frame][:,1],
            c=[col[int(is_pos)] for is_pos in human_samples["is_positive"][frame]],
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
    next_human_samples = jessi.batch_build_frame_humans_samples(
        robot_centric_data["rc_humans_positions"][:debugging_steps] + robot_centric_data["rc_humans_velocities"][:debugging_steps] * (robot_dt * prediction_horizon),
        robot_centric_data["humans_radii"][:debugging_steps],
        robot_centric_data["humans_visibility"][:debugging_steps],
        random.split(random.PRNGKey(random_seed), debugging_steps),
        n_samples = n_loss_samples,
        n_humans = n_humans,
        negative_samples_threshold=negative_samples_threshold,
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
            next_human_samples["position"][frame][:,0],
            next_human_samples["position"][frame][:,1],
            c=[col[int(is_pos)] for is_pos in next_human_samples["is_positive"][frame]],
            s=5,
            alpha=0.5,
            zorder=20,
        )
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=debugging_steps)
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
    dataset["obstacles_samples"] = jessi.batch_build_frame_obstacles_samples(
        robot_centric_data["rc_obstacles"][:debugging_steps],
        robot_centric_data["obstacles_visibility"][:debugging_steps],
        random.split(random.PRNGKey(random_seed), debugging_steps),
        n_samples = n_loss_samples,
        n_obstacles = n_obstacles,
        negative_samples_threshold=negative_samples_threshold,
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
    anim = FuncAnimation(fig, animate, interval=robot_dt*1000, frames=debugging_steps)
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
    # Delete robot_centric_data and raw_data to save memory
    del robot_centric_data
    del raw_data
else:
    # Load dataset
    with open(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)

### DEFINE NEURAL NETWORK
# Initialize network
sample_input = jnp.zeros((n_stack * lidar_num_rays, 12))
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
# encoder_distr = jessi.encoder.apply(
#     params, 
#     None, 
#     sample_input, 
# )
# fig, ax = plt.subplots(1, 3, figsize=(24, 8))
# fig.subplots_adjust(left=0.03, right=0.99, wspace=0.1)
# # Plot output obstacles distribution
# test_p = gmm.batch_p(encoder_distr["obs_distr"], test_samples)
# ax[0].set(xlim=[box_limits[0,0]-1, box_limits[0,1]+1], ylim=[box_limits[1,0]-1, box_limits[1,1]+1])
# ax[0].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[0].set_title("Random Obstacles GMM")
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
# ax[0].add_patch(rect)
# ax[0].set_aspect('equal', adjustable='box')
# # Plot output humans distribution
# test_p = gmm.batch_p(encoder_distr["hum_distr"], test_samples)
# ax[1].set(xlim=[box_limits[0,0]-1, box_limits[0,1]+1], ylim=[box_limits[1,0]-1, box_limits[1,1]+1])
# ax[1].scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=7)
# ax[1].set_title("Random Humans GMM")
# ax[1].set_xlabel("X")
# ax[1].set_ylabel("Y")
# rect = plt.Rectangle((box_limits[0,0], box_limits[1,0]), box_limits[0,1] - box_limits[0,0], box_limits[1,1] - box_limits[1,0], facecolor='none', edgecolor='grey', linewidth=1, alpha=0.5, zorder=1)
# ax[1].add_patch(rect)
# ax[1].set_aspect('equal', adjustable='box')
# # Plot output next humans distribution
# test_p = gmm.batch_p(encoder_distr["next_hum_distr"], test_samples)
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
    n_data = dataset["observations"].shape[0]
    print(f"# Training dataset size: {dataset['observations'].shape[0]} experiences")
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
            ## Tranform batch into training_batch data (it is done during training loop to save memory)
            # training_batch = {
            #     "inputs": jnp.zeros((batch_size, n_stack * lidar_num_rays, 12)),
            #     "humans_samples": {  # Network target: samples from the rc humans positions
            #         "position": jnp.full((batch_size, n_loss_samples, 2), jnp.nan),
            #         "is_positive": jnp.zeros((batch_size, n_loss_samples), dtype=bool),
            #     },
            #     "obstacles_samples": {  # Network target: samples from the rc obstacles positions
            #         "position": jnp.full((batch_size, n_loss_samples, 2), jnp.nan),
            #         "is_positive": jnp.zeros((batch_size, n_loss_samples), dtype=bool),
            #     },
            #     "next_humans_samples": { # Network target: samples from the next step rc humans positions
            #         "position": jnp.full((batch_size, n_loss_samples, 2), jnp.nan),
            #         "is_positive": jnp.zeros((batch_size, n_loss_samples), dtype=bool),
            #     },
            # }
            training_batch = {}
            training_batch["inputs"] = vmap(jessi.compute_encoder_input, in_axes=(0))(batch["observations"])[0]
            training_batch["humans_samples"] = jessi.batch_build_frame_humans_samples(
                batch["rc_humans_positions"],
                batch["humans_radii"],
                batch["humans_visibility"],
                random.split(random.PRNGKey(random_seed + i * n_data // batch_size + j), batch_size),
                n_samples = n_loss_samples,
                n_humans = n_humans,
                negative_samples_threshold=negative_samples_threshold,
            )
            training_batch["obstacles_samples"] = jessi.batch_build_frame_obstacles_samples(
                batch["rc_obstacles"],
                batch["obstacles_visibility"],
                random.split(random.PRNGKey(random_seed + i * n_data // batch_size + j), batch_size),
                n_samples = n_loss_samples,
                n_obstacles = n_obstacles,
                negative_samples_threshold=negative_samples_threshold,
            )
            training_batch["next_humans_samples"] = jessi.batch_build_frame_humans_samples(
                batch["rc_humans_positions"] + batch["rc_humans_velocities"] * (robot_dt * prediction_horizon),
                batch["humans_radii"],
                batch["humans_visibility"],
                random.split(random.PRNGKey(random_seed + i * n_data // batch_size + j), batch_size),
                n_samples = n_loss_samples,
                n_humans = n_humans,
                negative_samples_threshold=negative_samples_threshold,
            )
            # Update parameters
            params, optimizer_state, loss = jessi.update_encoder(
                params, 
                optimizer, 
                optimizer_state,
                training_batch,
            )
            debug.print("Epoch {x}, Batch {y}, Loss: {l}", x=i, y=j, l=loss)
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
with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
    robot_centric_data = pickle.load(f)
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
        jessi.compute_encoder_input(dataset["observations"][frame]), 
    )
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
