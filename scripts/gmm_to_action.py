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

from socialjym.policies.jessi import JESSI
from socialjym.utils.distributions.gaussian_mixture_model import BivariateGMM
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward

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
n_epochs = 100
p_visualization_threshold = 0.05
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
    'reward_function': DummyReward(kinematics=kinematics), # Don't care about reward for now
    'kinematics': kinematics,
    'lidar_angular_range':lidar_angular_range,
    'lidar_max_dist':lidar_max_dist,
    'lidar_num_rays':lidar_num_rays,
}
env = SocialNav(**env_params)
# Robot jessi
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
# Build local grid over which the GMM is defined
box_points = jnp.array([
    [box_limits[0,0], box_limits[1,0]],
    [box_limits[0,1], box_limits[1,0]],
    [box_limits[0,1], box_limits[1,1]],
    [box_limits[0,0], box_limits[1,1]],
])
visibility_thresholds = jnp.array([
    [box_limits[0,0] - visibility_threshold_from_grid, box_limits[0,1] + visibility_threshold_from_grid],
    [box_limits[1,0] - visibility_threshold_from_grid, box_limits[1,1] + visibility_threshold_from_grid]
])
visibility_threshold_points = jnp.array([
    [visibility_thresholds[0,0], visibility_thresholds[1,0]],
    [visibility_thresholds[0,1], visibility_thresholds[1,0]],
    [visibility_thresholds[0,1], visibility_thresholds[1,1]],
    [visibility_thresholds[0,0], visibility_thresholds[1,1]],
])
ax_lims = jnp.array([
    [visibility_thresholds[0,0]-2, visibility_thresholds[0,1]+2],
    [visibility_thresholds[1,0]-2, visibility_thresholds[1,1]+2]
])
gmm = BivariateGMM(n_components=n_gaussian_mixture_components)
# Compute test samples to visualize GMMs
sx = jnp.linspace(box_limits[0, 0], box_limits[0, 1], num=60, endpoint=True)
sy = jnp.linspace(box_limits[1, 0], box_limits[1, 1], num=60, endpoint=True)
test_samples_x, test_samples_y = jnp.meshgrid(sx, sy)
test_samples = jnp.stack((test_samples_x.flatten(), test_samples_y.flatten()), axis=-1)

### Parameters validation
cond1 = n_loss_samples % n_humans == 0
cond2 = n_loss_samples % n_obstacles == 0
cond3 = (n_loss_samples / n_obstacles) % env.static_obstacles_per_scenario.shape[2] == 0
if not (cond1 and cond2 and cond3):
    while not (cond1 and cond2 and cond3):
        n_loss_samples += 1
        cond1 = n_loss_samples % n_humans == 0
        cond2 = n_loss_samples % n_obstacles == 0
        cond3 = (n_loss_samples / n_obstacles) % env.static_obstacles_per_scenario.shape[2] == 0

### LOAD DATASETs
with open(os.path.join(os.path.dirname(__file__), 'dir_safe_experiences_dataset.pkl'), 'rb') as f:
    raw_data = pickle.load(f)
    # raw_data = {
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
with open(os.path.join(os.path.dirname(__file__), 'robot_centric_dir_safe_experiences_dataset.pkl'), 'rb') as f:
    robot_centric_data = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), 'final_gmm_training_dataset.pkl'), 'rb') as f:
    dataset = pickle.load(f)

### LOAD ENCODER PARAMETERS
# Load trained parameters
with open(os.path.join(os.path.dirname(__file__), 'gmm_network.pkl'), 'rb') as f:
    encoder_params = pickle.load(f)

### CHECK TRAINED NETWORK PREDICTIONS WITH NON ROBO-CENTRIC ANIMATION
fig, axs = plt.subplots(1,3,figsize=(24,8))
fig.subplots_adjust(left=0.05, right=0.99, wspace=0.13)
def animate(frame):
    for ax in axs:
        ax.clear()
        ax.set(xlim=[-10,10], ylim=[-10,10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        c, s = jnp.cos(raw_data["robot_orientations"][frame]), jnp.sin(raw_data["robot_orientations"][frame])
        rot = jnp.array([[c, -s], [s, c]])
        rotated_box_points = jnp.einsum('ij,jk->ik', rot, box_points.T).T + raw_data["robot_positions"][frame]
        to_plot = jnp.vstack((rotated_box_points, rotated_box_points[0:1,:]))
        ax.plot(to_plot[:,0], to_plot[:,1], color='grey', linewidth=2, alpha=0.5, zorder=1)
        # Plot visibility threshold
        rotated_visibility_threshold_points = jnp.einsum('ij,jk->ik', rot, visibility_threshold_points.T).T + raw_data["robot_positions"][frame]
        to_plot = jnp.vstack((rotated_visibility_threshold_points, rotated_visibility_threshold_points[0:1,:]))
        ax.plot(to_plot[:,0], to_plot[:,1], color='red', linewidth=1, alpha=0.5, zorder=1, linestyle='dashed')
        # Plot humans
        for h in range(len(raw_data["humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((raw_data["humans_positions"][frame][h,0] + jnp.cos(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h], raw_data["humans_positions"][frame][h,1] + jnp.sin(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((raw_data["humans_positions"][frame][h,0], raw_data["humans_positions"][frame][h,1]), raw_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot human velocities
        for h in range(len(raw_data["humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if robot_centric_data["humans_visibility"][frame][h]:
                ax.arrow(
                    raw_data["humans_positions"][frame][h,0],
                    raw_data["humans_positions"][frame][h,1],
                    raw_data["humans_velocities"][frame][h,0],
                    raw_data["humans_velocities"][frame][h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    zorder=30,
                )
        # Plot robot
        robot_position = raw_data['robot_positions'][frame]
        if kinematics == 'unicycle':
            head = plt.Circle((robot_position[0] + robot_radius * jnp.cos(raw_data["robot_orientations"][frame]), robot_position[1] + robot_radius * jnp.sin(raw_data["robot_orientations"][frame])), 0.1, color='black', zorder=1)
            ax.add_patch(head)
        circle = plt.Circle((robot_position[0], robot_position[1]), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
        ax.add_patch(circle)
        # Plot robot goal
        ax.plot(
            raw_data['robot_goals'][frame][0],
            raw_data['robot_goals'][frame][1],
            marker='*',
            markersize=7,
            color='red',
            zorder=5,
        )
        # Plot static obstacles
        for i, o in enumerate(raw_data["static_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 0.6 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
    # Plot predicted GMM samples
    obs_distr, hum_distr, next_hum_distr = jessi.encoder.apply(
        encoder_params, 
        None, 
        jnp.reshape(dataset["inputs"][frame], (1, n_stack * (2 * lidar_num_rays + 2))), 
    )
    obs_distr = {k: jnp.squeeze(v) for k, v in obs_distr.items()}
    hum_distr = {k: jnp.squeeze(v) for k, v in hum_distr.items()}
    next_hum_distr = {k: jnp.squeeze(v) for k, v in next_hum_distr.items()}
    test_p = gmm.batch_p(obs_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, obs_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[0].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[0].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[0].set_title("Obstacles Predicted GMM")
    test_p = gmm.batch_p(hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, hum_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[1].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[1].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[1].set_title("Humans Predicted GMM")
    test_p = gmm.batch_p(next_hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, next_hum_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[2].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[2].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
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

### CREATE ACTOR INPUTS DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl')):
    # Compute actor-critic inputs for the entire dataset
    ins = jnp.reshape(dataset["inputs"], (n_steps, dataset["inputs"].shape[1] * dataset["inputs"].shape[2]))
    obs_distrs, hum_distrs, next_hum_distrs = jessi.encoder.apply(
        encoder_params, 
        None, 
        ins
    )
    actor_actions = raw_data["robot_actions"]
    controller_dataset = {
        "inputs": {
            "obs_distrs": obs_distrs,
            "hum_distrs": hum_distrs,
            "next_hum_distrs": next_hum_distrs,
        },
        "actor_actions": actor_actions,
    }
    # TODO: Add also critic targets (need to save the reward in the first place in lidar_to_gmm raw_data generation)
    # Save actor inputs
    with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'wb') as f:
        pickle.dump(controller_dataset, f)
else:
    # Load actor inputs
    with open(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl'), 'rb') as f:
        controller_dataset = pickle.load(f)
# print(obs_distrs["means"].shape, obs_distrs["logsigmas"].shape, obs_distrs["correlations"].shape, obs_distrs["weights"].shape)

### FREE UNUSED MEMORY
del dataset
del robot_centric_data
del raw_data

### INITIALIZE ACTOR NETWORK
# Initialize actor network
sample_input = jnp.zeros((1, 3 * 6 * n_gaussian_mixture_components))
actor_params = jessi.actor.init(random.PRNGKey(random_seed), sample_input)
# Count network parameters
def count_params(actor_params):
    return sum(jnp.prod(jnp.array(p.shape)) for layer in actor_params.values() for p in layer.values())
n_params = count_params(actor_params)
print(f"# Controller network parameters: {n_params}")

### TRAINING LOOP
# Initialize optimizer and its state
optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
optimizer_state = optimizer.init(actor_params)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_network.pkl')):
    n_data = controller_dataset["inputs"]["obs_distrs"]["means"].shape[0]
    print(f"# Training dataset size: {controller_dataset['inputs']['obs_distrs']['means'].shape[0]} experiences")
    @loop_tqdm(n_epochs, desc="Training Lidar->GMM network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        dataset, actor_params, optimizer_state, losses = epoch_for_val
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
            epoch_data, actor_params, optimizer_state, losses = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(batch_size) + j * batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, epoch_data)
            # Update parameters
            actor_params, optimizer_state, loss = jessi.update_il_only_actor(
                actor_params, 
                optimizer, 
                optimizer_state,
                batch,
            )
            # Save loss
            losses = losses.at[i,j].set(loss)
            return epoch_data, actor_params, optimizer_state, losses
        n_batches = n_data // batch_size
        _, actor_params, optimizer_state, losses = lax.fori_loop(
            0,
            n_batches,
            _batch_loop,
            (epoch_data, actor_params, optimizer_state, losses)
        )
        return dataset, actor_params, optimizer_state, losses
    # Epoch loop
    _, actor_params, optimizer_state, losses = lax.fori_loop(
        0,
        n_epochs,
        _epoch_loop,
        (controller_dataset, actor_params, optimizer_state, jnp.zeros((n_epochs, int(n_data // batch_size))))
    )
    # Save trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'wb') as f:
        pickle.dump(actor_params, f)
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(jnp.arange(n_epochs), avg_losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Controller Network Training Loss")
    fig.savefig(os.path.join(os.path.dirname(__file__), 'controller_network_training_loss.eps'), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'rb') as f:
        actor_params = pickle.load(f)

### TEST TRAINED CONTROLLER NETWORK
#TODO: implement LaserNav for testing (at least the _get_obs())
state, reset_key, obs, info, _ = env.reset(random.PRNGKey(random_seed))
fig, axs = plt.subplots(1,3,figsize=(24,8))
fig.subplots_adjust(left=0.05, right=0.99, wspace=0.13)
def animate(frame):
    ## Compute action from trained JESSI
    action, _, _, _, _ = jessi.act(random.PRNGKey(0), obs, info, encoder_params, actor_params, sample=False)
    ## Simulate one step
    state, obs, info, _, _, reset_key = env.step(
        state,
        info,
        action, 
        test=False,
        reset_if_done=True,
        reset_key=reset_key
    )
    for ax in axs:
        ax.clear()
        ax.set(xlim=[-10,10], ylim=[-10,10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y', labelpad=-13)
        ax.set_aspect('equal', adjustable='box')
        # Plot box limits
        c, s = jnp.cos(raw_data["robot_orientations"][frame]), jnp.sin(raw_data["robot_orientations"][frame])
        rot = jnp.array([[c, -s], [s, c]])
        rotated_box_points = jnp.einsum('ij,jk->ik', rot, box_points.T).T + raw_data["robot_positions"][frame]
        to_plot = jnp.vstack((rotated_box_points, rotated_box_points[0:1,:]))
        ax.plot(to_plot[:,0], to_plot[:,1], color='grey', linewidth=2, alpha=0.5, zorder=1)
        # Plot visibility threshold
        rotated_visibility_threshold_points = jnp.einsum('ij,jk->ik', rot, visibility_threshold_points.T).T + raw_data["robot_positions"][frame]
        to_plot = jnp.vstack((rotated_visibility_threshold_points, rotated_visibility_threshold_points[0:1,:]))
        ax.plot(to_plot[:,0], to_plot[:,1], color='red', linewidth=1, alpha=0.5, zorder=1, linestyle='dashed')
        # Plot humans
        for h in range(len(raw_data["humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if humans_policy == 'hsfm':
                head = plt.Circle((raw_data["humans_positions"][frame][h,0] + jnp.cos(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h], raw_data["humans_positions"][frame][h,1] + jnp.sin(raw_data["humans_orientations"][frame][h]) * raw_data['humans_radii'][frame][h]), 0.1, color='black', alpha=alpha, zorder=1)
                ax.add_patch(head)
            circle = plt.Circle((raw_data["humans_positions"][frame][h,0], raw_data["humans_positions"][frame][h,1]), raw_data['humans_radii'][frame][h], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
            ax.add_patch(circle)
        # Plot human velocities
        for h in range(len(raw_data["humans_positions"][frame])):
            color = "green" if robot_centric_data["humans_visibility"][frame][h] else "grey"
            alpha = 0.6 if robot_centric_data["humans_visibility"][frame][h] else 0.3
            if robot_centric_data["humans_visibility"][frame][h]:
                ax.arrow(
                    raw_data["humans_positions"][frame][h,0],
                    raw_data["humans_positions"][frame][h,1],
                    raw_data["humans_velocities"][frame][h,0],
                    raw_data["humans_velocities"][frame][h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    zorder=30,
                )
        # Plot robot
        robot_position = raw_data['robot_positions'][frame]
        if kinematics == 'unicycle':
            head = plt.Circle((robot_position[0] + robot_radius * jnp.cos(raw_data["robot_orientations"][frame]), robot_position[1] + robot_radius * jnp.sin(raw_data["robot_orientations"][frame])), 0.1, color='black', zorder=1)
            ax.add_patch(head)
        circle = plt.Circle((robot_position[0], robot_position[1]), robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
        ax.add_patch(circle)
        # Plot robot goal
        ax.plot(
            raw_data['robot_goals'][frame][0],
            raw_data['robot_goals'][frame][1],
            marker='*',
            markersize=7,
            color='red',
            zorder=5,
        )
        # Plot static obstacles
        for i, o in enumerate(raw_data["static_obstacles"][frame]):
            for j, s in enumerate(o):
                color = 'black' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'grey'
                linestyle = 'solid' if robot_centric_data["obstacles_visibility"][frame][i,j] else 'dashed'
                alpha = 0.6 if robot_centric_data["obstacles_visibility"][frame][i,j] else 0.3
                ax.plot(s[:,0],s[:,1], color=color, linewidth=2, zorder=11, alpha=alpha, linestyle=linestyle)
    # Plot predicted GMM samples
    obs_distr, hum_distr, next_hum_distr = jessi.encoder.apply(
        encoder_params, 
        None, 
        jnp.reshape(dataset["inputs"][frame], (1, n_stack * (2 * lidar_num_rays + 2))), 
    )
    obs_distr = {k: jnp.squeeze(v) for k, v in obs_distr.items()}
    hum_distr = {k: jnp.squeeze(v) for k, v in hum_distr.items()}
    next_hum_distr = {k: jnp.squeeze(v) for k, v in next_hum_distr.items()}
    test_p = gmm.batch_p(obs_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, obs_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[0].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[0].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[0].set_title("Obstacles Predicted GMM")
    test_p = gmm.batch_p(hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, hum_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[1].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[1].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
    axs[1].set_title("Humans Predicted GMM")
    test_p = gmm.batch_p(next_hum_distr, test_samples)
    points_high_p = test_samples[test_p > p_visualization_threshold]
    corresponding_colors = test_p[test_p > p_visualization_threshold]
    rotated_means = jnp.einsum('ij,jk->ik', rot, next_hum_distr["means"].T).T + raw_data["robot_positions"][frame]
    rotated_points_high_p = jnp.einsum('ij,jk->ik', rot, points_high_p.T).T + raw_data["robot_positions"][frame]
    axs[2].scatter(rotated_means[:,0], rotated_means[:,1], c='red', s=10, marker='x', zorder=100)
    axs[2].scatter(rotated_points_high_p[:, 0], rotated_points_high_p[:, 1], c=corresponding_colors, cmap='viridis', s=7, zorder=50)
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