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

save_videos = False  # Whether to save videos of the debug inspections
### Parameters
random_seed = 0
n_stack = 5  # Number of stacked LiDAR scans as input
n_detectable_humans = 10  # Number of HCGs that can be detected by the policy
max_humans_velocity = 1.5  # Maximum humans velocity (m/s) used to compute the maximum displacement in the prediction horizon
learning_rate = 0.005
batch_size = 200
n_epochs = 10
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
# Robot jessi
jessi = JESSI(
    v_max=robot_vmax, 
    dt=robot_dt, 
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
    controller_dataset = {
        "observations": dataset['observations'],
        "rc_robot_goals": robot_centric_data["rc_robot_goals"],
        "actor_actions": raw_data["robot_actions"],
        "returns": raw_data["returns"],
    }
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
_, actor_critic_params, _ = jessi.init_nns(random.PRNGKey(random_seed))
# Count network parameters
def count_params(actor_critic_params):
    return sum(jnp.prod(jnp.array(p.shape)) for layer in actor_critic_params.values() for p in layer.values())
n_params = count_params(actor_critic_params)
print(f"# Controller network parameters: {n_params}")

### TRAINING LOOP
# Initialize optimizer and its state
optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
optimizer_state = optimizer.init(actor_critic_params)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'controller_network.pkl')):
    n_data = controller_dataset["observations"].shape[0]
    n_train_batches = n_data // batch_size
    print(f"# Training dataset size: {controller_dataset['observations'].shape[0]} experiences")
    @jit
    def batch_augment_data(
        batch:dict,
        keys:random.PRNGKey,
        base_lidar_noise_std:float = 0.01, # 1cm base noise
        proportional_lidar_noise_std:float = 0.01, # 1% proportional noise
        beam_dropout_prob:float = 0.03, # 3% beams dropout
    ) -> dict:
        return vmap(augment_data, in_axes=(0, 0, None, None, None))(batch, keys, base_lidar_noise_std, proportional_lidar_noise_std, beam_dropout_prob)
    @jit 
    def augment_data(
        data:dict,
        key:random.PRNGKey,
        base_lidar_noise_std:float,
        proportional_lidar_noise_std:float,
        beam_dropout_prob:float,
    ) -> dict:
        # data = {
        #     "inputs": shape (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
        #               each token: [norm_dist, hit, x, y, sin_fixed_theta, cos_fixed_theta, delta_t]
        #     "targets": {
        #         "gt_mask": shape (n_humans,),
        #         "gt_poses": shape (n_humans, 2),
        #         "gt_vels": shape (n_humans, 2),
        #     }
        # }
        input_key, beam_dropout_key = random.split(key, 2)
        ## Gaussian noise to LiDAR scans + Beam dropout
        raw_distances = data['inputs'][:,:,0] * jessi.max_beam_range  # (n_stack, lidar_num_rays)
        sigma = base_lidar_noise_std + proportional_lidar_noise_std * raw_distances  # (n_stack, lidar_num_rays)
        noise = random.normal(input_key, shape=raw_distances.shape) * sigma * data['inputs'][:,:,1]  # (n_stack, lidar_num_rays)
        noisy_distances = jnp.clip(raw_distances + noise, 0., jessi.max_beam_range) # (n_stack, lidar_num_rays)
        is_dropout = random.bernoulli(beam_dropout_key, p=beam_dropout_prob, shape=raw_distances.shape)
        noisy_distances = jnp.where(is_dropout, jessi.max_beam_range, noisy_distances)  # (n_stack, lidar_num_rays)
        new_hit = jnp.where(noisy_distances < jessi.max_beam_range, 1.0, 0.0) * (1.0 - is_dropout)  # (n_stack, lidar_num_rays)
        cos = data['inputs'][:,:,2] / (raw_distances + 1e-6)  # (n_stack, lidar_num_rays)
        sin = data['inputs'][:,:,3] / (raw_distances + 1e-6)  # (n_stack, lidar_num_rays)
        x = noisy_distances * cos  # (n_stack, lidar_num_rays)
        y = noisy_distances * sin  # (n_stack, lidar_num_rays)
        data['inputs'] = data['inputs'].at[:,:,0].set(noisy_distances / jessi.max_beam_range)
        data['inputs'] = data['inputs'].at[:,:,1].set(new_hit)
        data['inputs'] = data['inputs'].at[:,:,2].set(x)
        data['inputs'] = data['inputs'].at[:,:,3].set(y)
        return data
    @loop_tqdm(n_epochs, desc="Training Lidar->GMM network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        dataset, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = epoch_for_val
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
            epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = batch_for_val
            # Retrieve batch experiences
            indexes = (jnp.arange(batch_size) + j * batch_size).astype(jnp.int32)
            batch = vmap(lambda idxs, data: tree_map(lambda x: x[idxs], data), in_axes=(0, None))(indexes, epoch_data)
            # Compute training batch
            perception_input, last_lidar_point_clouds = vmap(jessi.compute_encoder_input)(batch["observations"])
            ## DATA AUGMENTATION
            noisy_inputs = batch_augment_data(
                {
                    "inputs": perception_input,
                },
                random.split(random.PRNGKey(i * n_train_batches + j), batch_size),
            )
            hcgs, scan_embeddings = jessi.perception.apply(
                encoder_params, 
                None, 
                noisy_inputs["inputs"],
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
                "returns": batch["returns"],
            }
            # Update parameters
            actor_critic_params, optimizer_state, loss, actor_loss, critic_loss = jessi.update_il(
                actor_critic_params, 
                optimizer, 
                optimizer_state,
                train_batch,
            )
            # Save loss
            losses = losses.at[i,j].set(loss)
            actor_losses = actor_losses.at[i,j].set(actor_loss)
            critic_losses = critic_losses.at[i,j].set(critic_loss)
            return epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses
        n_batches = n_data // batch_size
        _, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
            0,
            n_batches,
            _batch_loop,
            (epoch_data, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses)
        )
        return dataset, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses
    # Epoch loop
    _, actor_critic_params, optimizer_state, losses, actor_losses, critic_losses = lax.fori_loop(
        0,
        n_epochs,
        _epoch_loop,
        (controller_dataset, actor_critic_params, optimizer_state, jnp.zeros((n_epochs, int(n_data // batch_size))), jnp.zeros((n_epochs, int(n_data // batch_size))), jnp.zeros((n_epochs, int(n_data // batch_size))))
    )
    # Save trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'wb') as f:
        pickle.dump(actor_critic_params, f)
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    avg_actor_losses = jnp.mean(actor_losses, axis=1)
    avg_critic_losses = jnp.mean(critic_losses, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(jnp.arange(n_epochs), avg_losses, c='red')
    ax[1].plot(jnp.arange(n_epochs), avg_actor_losses, c='green')
    ax[2].plot(jnp.arange(n_epochs), avg_critic_losses, c='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss (with regularization)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Actor Loss (weighted)")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Critic Loss (weighted)")
    fig.savefig(os.path.join(os.path.dirname(__file__), 'controller_network_training_loss.eps'), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'controller_network.pkl'), 'rb') as f:
        actor_critic_params = pickle.load(f)