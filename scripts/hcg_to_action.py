from jax import random, jit, vmap, lax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt
import os
import pickle
import optax
from matplotlib import rc, rcParams
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.policies.jessi import JESSI
from socialjym.envs.lasernav import LaserNav
from socialjym.utils.rewards.lasernav_rewards.dummy_reward import DummyReward

save_videos = False  # Whether to save videos of the debug inspections
### Parameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_steps = 30_000  # Number of labeled examples to train Lidar to GMM network
n_detectable_humans = 10  # Number of HCGs that can be detected by the policy
max_humans_velocity = 1.5  # Maximum humans velocity (m/s) used to compute the maximum displacement in the prediction horizon
negative_samples_threshold = 0.2 # Distance threshold from objects to consider a sample as negative (in meters)
learning_rate = 0.005
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
env_params = {
    'robot_radius': 0.3,
    'n_humans': n_humans,
    'n_obstacles': n_obstacles,
    'robot_dt': robot_dt,
    'robot_radius': robot_radius, 
    'humans_dt': 0.01,
    'robot_visible': robot_visible,
    'scenario': scenario,
    'reward_function': DummyReward(0.3), # Don't care about reward for now
    'kinematics': kinematics,
    'lidar_angular_range':lidar_angular_range,
    'lidar_max_dist':lidar_max_dist,
    'lidar_num_rays':lidar_num_rays,
}
env = LaserNav(**env_params)
# Robot jessi
jessi = JESSI(
    v_max=robot_vmax, 
    dt=env_params['robot_dt'], 
    lidar_num_rays=lidar_num_rays, 
    lidar_max_dist=lidar_max_dist,
    lidar_angular_range=lidar_angular_range,
    n_stack=n_stack, 
    n_detectable_humans=n_detectable_humans, 
    max_humans_velocity=max_humans_velocity
)
# Build local grid over which the GMM is defined
ax_visibility = 2
ax_lims = jnp.array([
    [-lidar_max_dist-ax_visibility,lidar_max_dist+ax_visibility],
    [-lidar_max_dist-ax_visibility, lidar_max_dist+ax_visibility]
])

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
    # robot_centric_data = {
    #     "episode_starts": raw_data["episode_starts"],
    #     "rc_lidar_measurements": jnp.zeros((n_steps, lidar_num_rays, 2)),
    #     "rc_humans_positions": jnp.zeros((n_steps, n_humans, 2)),
    #     "rc_humans_orientations": jnp.zeros((n_steps, n_humans)),
    #     "rc_humans_velocities": jnp.zeros((n_steps, n_humans, 2)),
    #     "humans_radii": raw_data["humans_radii"],
    #     "robot_actions": raw_data["robot_actions"],
    #     "robot_positions": raw_data["robot_positions"],
    #     "robot_orientations": raw_data["robot_orientations"],
    #     "rc_robot_goals": jnp.zeros((n_steps, 2)),
    #     "rc_obstacles": jnp.zeros((n_steps, n_obstacles, 1, 2, 2)),
    #     "humans_visibility": jnp.zeros((n_steps, n_humans)),
    #     "obstacles_visibility": jnp.zeros((n_steps, n_obstacles, 1)),
    # }
with open(os.path.join(os.path.dirname(__file__), 'final_hcg_training_dataset.pkl'), 'rb') as f:
    dataset = pickle.load(f)

### LOAD ENCODER PARAMETERS
# Load trained parameters
with open(os.path.join(os.path.dirname(__file__), 'perception_network.pkl'), 'rb') as f:
    encoder_params = pickle.load(f)

### CREATE ACTOR INPUTS DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_training_dataset.pkl')):
    # Compute actor-critic inputs for the entire dataset
    actor_actions = raw_data["robot_actions"]
    controller_dataset = {
        "observations": dataset['observations'],
        "rc_robot_goals": robot_centric_data["rc_robot_goals"],
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
_, actor_params, _ = jessi.init_nns(random.PRNGKey(random_seed))
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
    n_data = controller_dataset["observations"].shape[0]
    print(f"# Training dataset size: {controller_dataset['observations'].shape[0]} experiences")
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
            # Compute training batch
            perception_input, last_lidar_point_clouds = vmap(jessi.compute_encoder_input)(batch["observations"])
            hcgs, scan_embeddings = jessi.perception.apply(
                encoder_params, 
                None, 
                perception_input,
            )
            bounding_parameters = vmap(jessi.bound_action_space)(last_lidar_point_clouds)
            actor_input = vmap(jessi.compute_actor_input)(
                hcgs,
                bounding_parameters,
                batch["rc_robot_goals"],
            )
            train_batch = {
                "actor_inputs": actor_input,
                "scan_embeddings": scan_embeddings,
                "actor_actions": batch["actor_actions"],
            }
            # Update parameters
            actor_params, optimizer_state, loss = jessi.update_il_only_actor(
                actor_params, 
                optimizer, 
                optimizer_state,
                train_batch,
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