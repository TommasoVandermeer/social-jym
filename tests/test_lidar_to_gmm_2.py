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

from socialjym.utils.distributions.gaussian_mixture_model import GMM
from socialjym.envs.socialnav import SocialNav
from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward
from socialjym.utils.aux_functions import plot_lidar_measurements

### Parameters
random_seed = 0
n_steps = 100_000  # Number of labeled examples to train Lidar to GMM network
grid_resolution = 10  # Number of grid cells per dimension
n_loss_samples = 100  # Number of samples to estimate the loss
learning_rate = 1e-3
batch_size = 200
n_epochs = 10
n_iterations_fit_gmm = 20
# Environment parameters
robot_radius = 0.3
robot_dt = 0.25
robot_visible = False
kinematics = "unicycle"
lidar_angular_range = 2*jnp.pi
lidar_max_dist = 10.
lidar_num_rays = 180
scenario = "hybrid_scenario"
n_humans = 5
humans_policy = 'hsfm'
env_params = {
            'robot_radius': 0.3,
            'n_humans': n_humans,
            'robot_dt': robot_dt,
            'robot_radius': robot_radius, 
            'humans_dt': 0.01,
            'robot_visible': robot_visible,
            'scenario': scenario,
            'humans_policy': humans_policy,
            'reward_function': DummyReward(kinematics=kinematics),
            'kinematics': kinematics,
            'lidar_angular_range':lidar_angular_range,
            'lidar_max_dist':lidar_max_dist,
            'lidar_num_rays':lidar_num_rays,
        }
env = SocialNav(**env_params)
# Build local grid over which the GMM is defined
dists = jnp.concatenate([-jnp.arange(0, 5, grid_resolution/10)[::-1][:-1],jnp.arange(0, 5, grid_resolution/10)])
grid_cell_coords = jnp.meshgrid(dists, dists)
grid_cells = jnp.array(jnp.vstack((grid_cell_coords[0].flatten(), grid_cell_coords[1].flatten())).T)
cell_size = (grid_cells[1,0] - grid_cells[0,0], grid_cells[grid_resolution,1] - grid_cells[0,1])  # Assuming uniform grid
gmm = GMM(n_dimensions=grid_cells.shape[1], n_components=grid_cells.shape[0])

@jit
def fit_gmm_to_humans_positions(humans_position, humans_radii, grid_cells, scaling=0.01):
    humans_covariances = vmap(lambda r: (1 / r)**2 * jnp.eye(2))(humans_radii) * scaling
    # Compute the target per-cell weights for each grid cell
    @jit
    def softweight_human_cell(human_pos, human_radius, human_cov, cell):
        """Compute the soft weight of a human for a grid cell based on a Gaussian distribution."""
        diff = cell - human_pos
        diff = lax.cond(
            jnp.linalg.norm(diff) > human_radius,
            lambda d: d - human_radius * d / jnp.linalg.norm(d),
            lambda d: jnp.zeros_like(d),
            diff
        )
        exponent = -0.5 * jnp.dot(diff, jnp.linalg.solve(human_cov, diff))
        norm_const = jnp.sqrt((2 * jnp.pi) ** len(human_pos) * jnp.linalg.det(human_cov))
        return jnp.exp(exponent) / norm_const
    softweight_human_cells = jit(vmap(softweight_human_cell, in_axes=(None, None, None, 0)))
    batch_softweight_human_cells = jit(vmap(softweight_human_cells, in_axes=(0, 0, 0, None)))
    humans_weights_per_cell = batch_softweight_human_cells(humans_position, humans_radii, humans_covariances, grid_cells)
    cell_weights = jnp.sum(humans_weights_per_cell, axis=0)
    norm_cell_weights = cell_weights / (jnp.sum(cell_weights) + 1e-8)
    # Compute the target per-cell covariance
    norm_humans_weights_per_cell = humans_weights_per_cell / (jnp.sum(humans_weights_per_cell, axis=1, keepdims=True) + 1e-8)
    @jit
    def human_weighted_covariance(human_pos, human_cov, cell, weight):
        diff = cell - human_pos
        outer_prod = jnp.outer(diff, diff)
        return jnp.diag(weight * (human_cov + outer_prod))
    batch_human_weighted_covariances = jit(vmap(human_weighted_covariance, in_axes=(0, 0, None, 0)))
    batch_cells_human_weighted_covariances = jit(vmap(lambda hp, hc, gc, hw: jnp.sum(batch_human_weighted_covariances(hp, hc, gc, hw), axis=0), in_axes=(None, None, 0, 0)))
    human_weighted_covariances_per_cell = batch_cells_human_weighted_covariances(
        humans_position, 
        humans_covariances, 
        grid_cells, 
        norm_humans_weights_per_cell.T
    )
    # Initialize fitted distribution
    fitted_distribution = {
        "means": grid_cells,
        "variances": human_weighted_covariances_per_cell,
        "weights": norm_cell_weights,
    }
    return fitted_distribution
@jit
def batch_fit_gmm_to_humans_positions(batch_humans_position, batch_humans_radii, grid_cells, scaling=0.01):
    return vmap(fit_gmm_to_humans_positions, in_axes=(0, 0, None, None))(batch_humans_position, batch_humans_radii, grid_cells, scaling)
def simulate_n_steps(env, n_steps):
    @loop_tqdm(n_steps, desc="Simulating steps")
    @jit
    def _simulate_steps_with_lidar(i:int, for_val:tuple):
        ## Retrieve data from the tuple
        data, state, info, reset_key = for_val
        ## Simulate one step
        state = state.at[-1,:2].set(jnp.array([0.,0.]))  # Put robot in [0,0]
        state = state.at[-1,4].set(jnp.pi/2)  # Set robot orientation to pi/2
        final_state, final_obs, final_info, _, _, final_reset_key = env.step(
            state,
            info,
            jnp.array([0.,0.]), # No action
            test=False,
            reset_if_done=True,
            reset_key=reset_key
        )
        lidar_measurements = env.get_lidar_measurements(final_obs[-1,:2], final_obs[-1,5], final_obs[:-1,:2], info["humans_parameters"][:,0])
        # Save output data
        step_out_data = {
            "lidar_measurements": lidar_measurements,
            "humans_positions": final_obs[:-1,:2],
            "humans_velocities": final_obs[:-1,2:4],
            "humans_radii": info["humans_parameters"][:,0],
            "robot_positions": final_obs[-1,:2],
            "robot_orientations": final_obs[-1,5],
        }
        data = tree_map(lambda x, y: x.at[i].set(y), data, step_out_data)
        return data, final_state, final_info, final_reset_key
    # Initialize first episode
    state, reset_key, _, info, _ = env.reset(random.PRNGKey(random_seed))
    # Initialize setting data
    data = {
        "lidar_measurements": jnp.zeros((n_steps,lidar_num_rays,2)),
        "humans_positions": jnp.zeros((n_steps,n_humans,2)),
        "humans_velocities": jnp.zeros((n_steps,n_humans,2)),
        "humans_radii": jnp.zeros((n_steps,n_humans)),
        "robot_positions": jnp.zeros((n_steps,2)),
        "robot_orientations": jnp.zeros((n_steps,1)),
    }
    # Step loop
    data, _, _, _ = lax.fori_loop(
        0,
        n_steps,
        _simulate_steps_with_lidar,
        (data, state, info, reset_key)
    )
    return data

### GENERATE DATASET
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_dataset.pkl')):
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'lidar_to_humans_state_dataset.pkl')):
        # Generate raw data
        raw_data = simulate_n_steps(env, n_steps)
        # Save raw data dataset
        with open(os.path.join(os.path.dirname(__file__), 'lidar_to_humans_state_dataset.pkl'), 'wb') as f:
            pickle.dump(raw_data, f)
    else:
        # Load raw data dataset
        with open(os.path.join(os.path.dirname(__file__), 'lidar_to_humans_state_dataset.pkl'), 'rb') as f:
            raw_data = pickle.load(f)
    # Initialize final dataset
    dataset = {
        "lidar_measurements": raw_data["lidar_measurements"],
        "distributions": {
            "means": jnp.zeros((n_steps, grid_cells.shape[0], grid_cells.shape[1])),
            "variances": jnp.zeros((n_steps, grid_cells.shape[0], grid_cells.shape[1])),
            "weights": jnp.zeros((n_steps, grid_cells.shape[0])),
        }
    }
    # Generate target GMMs
    dataset["distributions"] = batch_fit_gmm_to_humans_positions(raw_data["humans_positions"], raw_data["humans_radii"], grid_cells)
    # Save dataset
    with open(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
else:
    # Load datasets
    with open(os.path.join(os.path.dirname(__file__), 'lidar_to_humans_state_dataset.pkl'), 'rb') as f:
        raw_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'lidar_to_gmm_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    ## DEBUG: Visualize a target GMM
    # n_samples = 10_000
    # n_distribution = 25
    # distr = {
    #     "means": dataset["distributions"]["means"][n_distribution],
    #     "variances": dataset["distributions"]["variances"][n_distribution],
    #     "weights": dataset["distributions"]["weights"][n_distribution],
    # }
    # samples = gmm.batch_sample(distr, random.split(random.PRNGKey(random_seed), n_samples))
    # p = gmm.batch_p(distr, samples)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.set_xlim(-7, 7)
    # ax.set_ylim(-7, 7)
    # for pos, rad, vel in zip(raw_data["humans_positions"][n_distribution], raw_data["humans_radii"][n_distribution], raw_data["humans_velocities"][n_distribution]):
    #     circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
    #     ax.add_artist(circle)
    #     ax.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    # ax.scatter(samples[:, 0], samples[:, 1], c=p, cmap='viridis', s=5, alpha=0.5)
    # ax.set_title("Humans and Fitted GMM Top View")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # for cell_center in grid_cells:
    #     rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    #     ax.add_patch(rect)
    # plot_lidar_measurements(ax, raw_data["lidar_measurements"][n_distribution], raw_data["robot_positions"][n_distribution], 0.3)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

### DEFINE NEURAL NETWORK
class LidarNetwork(hk.Module):
    def __init__(
        self,
        grid_cell_positions: jnp.ndarray,  # (n_cells, 2)
        lidar_num_rays: int,
        lidar_local_angles: jnp.ndarray = raw_data["lidar_measurements"][0,:,1],  # (n_rays,)
        d_model: int = 256,
        hidden_dim: int = 256,
        pos_enc_dim: int = 32,  # angular encoding size
        name: str = "LidarToGMM",
    ):
        super().__init__(name=name)
        self.gmm_means = grid_cell_positions
        self.lidar_angles = lidar_local_angles
        self.n_cells = grid_cell_positions.shape[0]
        self.n_rays = lidar_num_rays
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc = self.angular_positional_encoding()

        # --- Encoder: 1D CNN stack
        self.encoder = hk.Sequential([
            hk.Conv1D(output_channels=64, kernel_shape=9, stride=1, padding="SAME"),
            nn.relu,
            hk.Conv1D(output_channels=128, kernel_shape=5, stride=1, padding="SAME"),
            nn.relu,
            hk.Conv1D(output_channels=d_model, kernel_shape=3, stride=1, padding="SAME"),
            nn.relu,
        ])

        # --- Positional embeddings for fixed GMM anchors
        self.pos_emb = hk.get_parameter(
            "pos_emb",
            shape=(self.n_cells, d_model),
            init=hk.initializers.RandomNormal(0.02),
        )

        # --- Anchor-wise shared MLP decoder
        self.anchor_mlp = hk.nets.MLP([hidden_dim, hidden_dim, 3])

    def angular_positional_encoding(self) -> jnp.ndarray:
        """
        Sinusoidal encoding for LiDAR beam angles.
        Returns: (n_rays, pos_enc_dim)
        """
        dims = jnp.arange(self.pos_enc_dim // 2)
        freqs = 1.0 / (10000 ** (dims / (self.pos_enc_dim / 2)))
        angles = self.lidar_angles[:, None] * freqs[None, :]  # (n_rays, pos_enc_dim//2)
        pos_enc = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (n_rays, pos_enc_dim)
        return pos_enc

    def __call__(self, lidar_scan: jnp.ndarray) -> dict:
        """
        Args:
            lidar_scan: (B, n_rays)
        Returns:
            dict with means, variances, weights
        """
        B = lidar_scan.shape[0]

        # --- Angular positional encoding
        pos_enc = jnp.broadcast_to(self.pos_enc[None, :, :], (B, self.n_rays, self.pos_enc_dim))  # (B, n_rays, pos_enc_dim)

        # --- Combine LiDAR ranges + positional encoding
        x = jnp.concatenate([lidar_scan[..., None], pos_enc], axis=-1)  # (B, n_rays, 1+pos_enc_dim)

        # --- Encode LiDAR scan
        feat = self.encoder(x)  # (B, n_rays, d_model)
        global_feat = jnp.mean(feat, axis=1)  # (B, d_model)

        # --- Broadcast anchor embeddings + global context
        pos_emb = jnp.broadcast_to(self.pos_emb, (B, self.n_cells, self.d_model))  # (B, n_cells, d_model)
        global_feat_expanded = jnp.repeat(global_feat[:, None, :], self.n_cells, axis=1)  # (B, n_cells, d_model)
        fused = jnp.concatenate([pos_emb, global_feat_expanded], axis=-1)  # (B, n_cells, 2*d_model)

        # --- Vectorized decoding
        BNC = fused.shape
        flat = fused.reshape(B * self.n_cells, BNC[-1])  # (B*n_cells, 2*d_model)
        dec = self.anchor_mlp(flat)  # (B*n_cells, 3)
        dec = dec.reshape(B, self.n_cells, 3)  # (B, n_cells, 3)

        log_sigma_x = dec[..., 0]
        log_sigma_y = dec[..., 1]
        raw_weight = dec[..., 2]

        sigmas = jnp.stack(
            [nn.softplus(log_sigma_x) + 1e-3, nn.softplus(log_sigma_y) + 1e-3],
            axis=-1,
        )  # (B, n_cells, 2)
        weights = nn.softmax(raw_weight, axis=-1)  # (B, n_cells)

        return {
            "means": jnp.broadcast_to(self.gmm_means[None, :, :], (B, self.n_cells, 2)),
            "variances": sigmas,
            "weights": weights,
        }
@hk.transform
def lidar_to_gmm_network(x):
    net = LidarNetwork(grid_cells, lidar_num_rays)
    return net(x)
# Initialize network
sample_input = jnp.zeros((1, lidar_num_rays))
network = lidar_to_gmm_network
params = network.init(random.PRNGKey(random_seed), sample_input)
# Count network parameters
def count_params(params):
    return sum(jnp.prod(jnp.array(p.shape)) for layer in params.values() for p in layer.values())
n_params = count_params(params)
print(f"# Lidar network parameters: {n_params}")

### TEST INITIAL NETWORK
# # Forward pass
# output_distr = network.apply(
#     params, 
#     None, 
#     sample_input, 
# )
# print("Output means shape:", output_distr["means"].shape)
# print("Output variances shape:", output_distr["variances"].shape)
# print("Output weights shape:", output_distr["weights"].shape)
# distr = {k: jnp.squeeze(v) for k, v in output_distr.items()}
# # Plot output distribution
# n_test_samples = 10_000
# test_samples = gmm.batch_sample(distr, random.split(random.PRNGKey(0), n_test_samples))
# test_p = gmm.batch_p(distr, test_samples)
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.set_xlim(-7, 7)
# ax.set_ylim(-7, 7)
# ax.scatter(test_samples[:, 0], test_samples[:, 1], c=test_p, cmap='viridis', s=5, alpha=0.5)
# ax.set_title("Random LiDAR network Output")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# for cell_center in grid_cells:
#     rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
#     ax.add_patch(rect)
# ax.set_aspect('equal', adjustable='box')
# plt.show()

### DEFINE LOSS FUNCTION, UPDATE FUNCTIONS, AND OPTIMIZER
@partial(jit, static_argnames=("n_samples"))
def _compute_loss_and_gradients(
    current_params:dict,  
    experiences:dict[str:jnp.ndarray],
    # Experiences: {"inputs":jnp.ndarray, "targets":{"means":jnp.ndarray, "variances":jnp.ndarray, "weights":jnp.ndarray}}
    sample_key: random.PRNGKey,
    n_samples:int=n_loss_samples,
) -> tuple:
    @jit
    def _batch_loss_function(
        current_params:dict,
        inputs:jnp.ndarray,
        targets:jnp.ndarray,  
        ) -> jnp.ndarray:
        
        @partial(vmap, in_axes=(None, 0, 0))
        def _loss_function(
            current_params:dict,
            input:jnp.ndarray,
            target:jnp.ndarray, 
            ) -> jnp.ndarray:
            # Compute the prediction
            prediction = network.apply(current_params, None, jnp.expand_dims(input, axis=0))
            prediction = {k: jnp.squeeze(v) for k, v in prediction.items()}
            # Sample from target distribution
            target = {k: jnp.squeeze(v) for k, v in target.items()}
            target_samples = gmm.batch_sample(target, random.split(sample_key, n_samples))
            # Compute the Negative Log-Likelihood (NLL) loss
            neglogp_prediction_distr = gmm.batch_neglogp(prediction, target_samples)
            nll_loss = jnp.mean(neglogp_prediction_distr)
            # # Compute the KL divergence loss (from target to prediction)
            # logp_target_distr = gmm.batch_logp(target, target_samples)
            # kl_loss = jnp.mean(logp_target_distr + neglogp_prediction_distr)
            # # Compute weights misalignment loss (L2 norm between weights)
            # weights_loss = jnp.linalg.norm(prediction["weights"] - target["weights"])
            # # Compute entropy loss (regularization on weights to avoid collapsing)
            # weights_entropy = -jnp.sum(prediction["weights"] * jnp.log(prediction["weights"] + 1e-10))
            return  nll_loss

        return jnp.mean(_loss_function(
                current_params,
                inputs,
                targets))

    inputs = experiences["lidar_measurements"]
    targets = experiences["distributions"]
    # Compute the loss and gradients
    loss, grads = value_and_grad(_batch_loss_function)(
        current_params, 
        inputs,
        targets)
    return loss, grads
@partial(jit, static_argnames=("optimizer"))
def update(
    current_params:dict, 
    optimizer:optax.GradientTransformation, 
    optimizer_state: jnp.ndarray,
    experiences:dict[str:jnp.ndarray],
    # Experiences: {"lidar_measurements":jnp.ndarray, "distributions":{"means":jnp.ndarray, "variances":jnp.ndarray, "weights":jnp.ndarray}}
    sample_key:random.PRNGKey,
) -> tuple:
    # Use only LiDAR distances as input
    experiences["lidar_measurements"] = experiences["lidar_measurements"][:,:,0]
    # Compute loss and gradients
    loss, grads = _compute_loss_and_gradients(current_params, experiences, sample_key)
    # Compute parameter updates
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    # Apply updates
    updated_params = optax.apply_updates(current_params, updates)
    return updated_params, optimizer_state, loss
# Initialize optimizer and its state
optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
optimizer_state = optimizer.init(params)

### TRAINING LOOP
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'lidar_cnn_network_params.pkl')):
    @loop_tqdm(n_epochs, desc="Training Lidar->GMM network")
    @jit 
    def _epoch_loop(
        i:int,
        epoch_for_val:tuple,
    ) -> tuple:
        dataset, params, optimizer_state, losses = epoch_for_val
        # Shuffle dataset at the beginning of the epoch
        shuffle_key = random.PRNGKey(random_seed + i)
        indexes = jnp.arange(n_steps)
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
                random.PRNGKey(random_seed + i * n_batches + j)
            )
            # Save loss
            losses = losses.at[i,j].set(loss)
            return epoch_data, params, optimizer_state, losses
        n_batches = n_steps // batch_size
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
        (dataset, params, optimizer_state, jnp.zeros((n_epochs, int(n_steps // batch_size))))
    )
    # Save trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'lidar_cnn_network_params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    # Plot training loss
    avg_losses = jnp.mean(losses, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(jnp.arange(n_epochs), avg_losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Lidar to GMM Network Training Loss")
    fig.savefig(os.path.join(os.path.dirname(__file__), 'lidar_cnn_network_training_loss.eps'), format='eps')
else:
    # Load trained parameters
    with open(os.path.join(os.path.dirname(__file__), 'lidar_cnn_network_params.pkl'), 'rb') as f:
        params = pickle.load(f)

### BENCHMARKING: Visualize target vs predicted GMM vs Lidar Fit2GMM
example = 24 # Choose an example to visualize between 0 and n_steps-1
n_samples = 1_000
target_distr = {
    "means": dataset["distributions"]["means"][example],
    "variances": dataset["distributions"]["variances"][example],
    "weights": dataset["distributions"]["weights"][example],
}
## Sample from target GMM
target_samples = gmm.batch_sample(target_distr, random.split(random.PRNGKey(random_seed), n_samples))
target_p = gmm.batch_p(target_distr, target_samples)
## Sample from predicted GMM
input = jnp.expand_dims(raw_data["lidar_measurements"][example,:,0], axis=0)
predicted_distr = network.apply(
    params, 
    None,
    input
)
predicted_distr = {k: jnp.squeeze(v) for k, v in predicted_distr.items()}
start_time = time.time()
for _ in range(10): _ = network.apply(params, None, input)
predict_time = time.time() - start_time
predicted_samples = gmm.batch_sample(predicted_distr, random.split(random.PRNGKey(random_seed), n_samples))
predicted_p = gmm.batch_p(predicted_distr, predicted_samples)
## Fit LiDAR measurements to GMM with Expected Maximization algorithm
# Convert LiDAR measurements to Cartesian coordinates
lidar_measurements = raw_data["lidar_measurements"][example,:,0]
lidar_angles = raw_data["lidar_measurements"][example,:,1]
points = vmap(lambda r, a: jnp.array([r * jnp.cos(a), r * jnp.sin(a)]))(lidar_measurements, lidar_angles)
points = points[lidar_measurements < env.lidar_max_dist]  
@partial(jit, static_argnames=("n_iterations_fit_gmm"))
def fit_gmm_to_points(points:jnp.ndarray, grid_cells:jnp.ndarray, n_iterations_fit_gmm:int=10, epsilon:float=1e-5) -> dict:
    """
    Fit a Gaussian Mixture Model (GMM) to the given points with the EM algorithm using the specified grid cells as fixed means.

    params:
    - points: jnp.ndarray of shape (n_points, n_dimensions), the data points to fit the GMM to.
    - grid_cells: jnp.ndarray of shape (n_components, n_dimensions), the fixed means of the GMM components.
    - n_iterations_fit_gmm: int, the number of EM iterations to perform.
    - epsilon: float, a small value to prevent division by zero.

    returns:
    - distr: dict, the fitted GMM parameters with keys "means", "variances", and "weights".

    """
    # Initialize
    distr = {
        "means": grid_cells,
        "variances": jnp.ones((grid_cells.shape[0], 2)) * 0.2,
        "weights": jnp.ones((grid_cells.shape[0],)) / grid_cells.shape[0],
    }
    # EM loop
    def em_step(carry, _):
        distr = carry
        # E-step: compute responsibilities
        weighted_ps = gmm.batch_samples_batch_p_single_component(distr, points) * jnp.expand_dims(distr["weights"], axis=0)  # Shape (n_points, n_components)
        responsibilities = weighted_ps / (jnp.sum(weighted_ps, axis=1, keepdims=True) + epsilon)  # Shape (n_points, n_components)
        # debug.print("Responsibilities sum axis 1 (should be 1): {x}", x=jnp.sum(responsibilities, axis=1))
        # M-step: update parameters
        @jit
        def compute_variance(k, responsabilities_per_k):
            diff = points - grid_cells[k] # Shape (n_points, n_dimensions)
            weighted_diff_squared = responsabilities_per_k[:, None] * (diff ** 2) # Shape (n_points, n_dimensions)
            variance_k = jnp.sum(weighted_diff_squared, axis=0) / (jnp.sum(responsabilities_per_k) + epsilon)
            return variance_k  
        variances = vmap(compute_variance, in_axes=(0, 1))(jnp.arange(grid_cells.shape[0]), responsibilities) # Shape (n_components, n_dimensions)
        weights = jnp.sum(responsibilities, axis=0) / jnp.sum(responsibilities)  # Shape (n_components,)
        updated_distr = {
            "means": grid_cells,  # Keep means fixed to grid cells
            "variances": variances + epsilon,
            "weights": weights,
        }
        return updated_distr, None
    distr, _ = lax.scan(em_step, distr, None, length=n_iterations_fit_gmm)
    return distr
fitted_distr = fit_gmm_to_points(points, grid_cells, n_iterations_fit_gmm=n_iterations_fit_gmm)
start_time = time.time()
for _ in range(10): _ = fit_gmm_to_points(points, grid_cells, n_iterations_fit_gmm=n_iterations_fit_gmm)
fit_time = time.time() - start_time
fitted_samples = gmm.batch_sample(fitted_distr, random.split(random.PRNGKey(random_seed), n_samples))
fitted_p = gmm.batch_p(fitted_distr, fitted_samples)
## Plotting
if os.path.exists(os.path.join(os.path.dirname(__file__), 'lidar_network_params.pkl')):
    ## Compute predicted GMM with MLP network for comparison
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
                mlp_params:dict=mlp_params,
            ) -> None:
            super().__init__()  
            self.gmm_means = grid_cell_positions  # Fixed means
            self.n_gmm_cells = grid_cell_positions.shape[0]
            self.n_inputs = lidar_num_rays  # Each ray has distance
            self.n_outputs = self.n_gmm_cells * 3  # 3 outputs per GMM cell (var_x, var_y, weight)
            self.mlp = hk.nets.MLP(
                **mlp_params, 
                output_sizes=[self.n_inputs * 5, self.n_inputs * 3, self.n_inputs * 3, self.n_outputs], 
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
            weights = nn.softmax(mlp_output[:, 2*self.n_gmm_cells:], axis=-1)  # Weights
            ### Construct GMM parameters
            distr = {
                "means": jnp.tile(self.gmm_means, (x.shape[0], 1, 1)),  # Fixed means
                "variances": jnp.stack((x_vars, y_vars), axis=-1),  # Shape (batch_size, n_gmm_cells, 2)
                "weights": weights,  # Shape (batch_size, n_gmm_cells)
            }
            return distr
    @hk.transform
    def lidar_to_gmm_network_mlp(x):
        net = LidarNetwork(grid_cells, lidar_num_rays)
        return net(x)
    network_mlp = lidar_to_gmm_network_mlp
    with open(os.path.join(os.path.dirname(__file__), 'lidar_network_params.pkl'), 'rb') as f:
        params_mlp = pickle.load(f)
    predicted_distr_mlp = network_mlp.apply(
        params_mlp,
        None,
        input
    )
    predicted_distr_mlp = {k: jnp.squeeze(v) for k, v in predicted_distr_mlp.items()}
    start_time = time.time()
    for _ in range(10): _ = network_mlp.apply(params_mlp, None, input)
    predict_mlp_time = time.time() - start_time
    predicted_samples_mlp = gmm.batch_sample(predicted_distr_mlp, random.split(random.PRNGKey(random_seed), n_samples))
    predicted_p_mlp = gmm.batch_p(predicted_distr_mlp, predicted_samples_mlp)
    ## Plot
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    fig.subplots_adjust(right=0.99, left=0.03, wspace=0.05, hspace=0.15, top=0.95, bottom=0.05)
    # Plot target GMM
    ax[0,0].set_xlim(-7, 7)
    ax[0,0].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[0,0].add_artist(circle)
        ax[0,0].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[0,0].scatter(target_samples[:, 0], target_samples[:, 1], c=target_p, cmap='viridis', s=5, alpha=0.5)
    ax[0,0].set_title("Target GMM")
    ax[0,0].set_xlabel("X")
    ax[0,0].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[0,0].add_patch(rect)
    plot_lidar_measurements(ax[0,0], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[0,0].set_aspect('equal', adjustable='box')
    # Plot fitted GMM
    ax[0,1].set_xlim(-7, 7)
    ax[0,1].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[0,1].add_artist(circle)
        ax[0,1].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[0,1].scatter(fitted_samples[:, 0], fitted_samples[:, 1], c=fitted_p, cmap='viridis', s=5, alpha=0.5)
    ax[0,1].scatter(points[:,0], points[:,1], c='brown', s=15, label='LiDAR Points', zorder=12)
    ax[0,1].set_title(f"Fitted GMM (EM - {n_iterations_fit_gmm} iterations) - Time: {fit_time/10:.6f}s")
    ax[0,1].set_xlabel("X")
    ax[0,1].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[0,1].add_patch(rect)
    plot_lidar_measurements(ax[0,1], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[0,1].set_aspect('equal', adjustable='box')
    # Plot predicted GMM with MLP network
    ax[1,0].set_xlim(-7, 7)
    ax[1,0].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[1,0].add_artist(circle)
        ax[1,0].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[1,0].scatter(predicted_samples_mlp[:, 0], predicted_samples_mlp[:, 1], c=predicted_p_mlp, cmap='viridis', s=5, alpha=0.5)
    ax[1,0].set_title(f"Predicted GMM (MLP) - Time: {predict_mlp_time/10:.6f}s")
    ax[1,0].set_xlabel("X")
    ax[1,0].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[1,0].add_patch(rect)
    plot_lidar_measurements(ax[1,0], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[1,0].set_aspect('equal', adjustable='box')
    # Plot predicted GMM with CNN network
    ax[1,1].set_xlim(-7, 7)
    ax[1,1].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[1,1].add_artist(circle)
        ax[1,1].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[1,1].scatter(predicted_samples[:, 0], predicted_samples[:, 1], c=predicted_p, cmap='viridis', s=5, alpha=0.5)
    ax[1,1].set_title(f"Predicted GMM (CNN) - Time: {predict_time/10:.6f}s")
    ax[1,1].set_xlabel("X")
    ax[1,1].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[1,1].add_patch(rect)
    plot_lidar_measurements(ax[1,1], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[1,1].set_aspect('equal', adjustable='box')
    fig.savefig(os.path.join(os.path.dirname(__file__), 'gmm_benchmark.eps'), format='eps')
else:
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    fig.subplots_adjust(right=0.99, left=0.03, wspace=0.1)
    # Plot target GMM
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[0].add_artist(circle)
        ax[0].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[0].scatter(target_samples[:, 0], target_samples[:, 1], c=target_p, cmap='viridis', s=5, alpha=0.5)
    ax[0].set_title("Target GMM")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[0].add_patch(rect)
    plot_lidar_measurements(ax[0], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[0].set_aspect('equal', adjustable='box')
    # Plot predicted GMM
    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[1].add_artist(circle)
        ax[1].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[1].scatter(predicted_samples[:, 0], predicted_samples[:, 1], c=predicted_p, cmap='viridis', s=5, alpha=0.5)
    ax[1].set_title(f"Predicted GMM - Time: {predict_time/10:.6f}s")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[1].add_patch(rect)
    plot_lidar_measurements(ax[1], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[1].set_aspect('equal', adjustable='box')
    # Plot fitted GMM
    ax[2].set_xlim(-7, 7)
    ax[2].set_ylim(-7, 7)
    for pos, rad, vel in zip(raw_data["humans_positions"][example], raw_data["humans_radii"][example], raw_data["humans_velocities"][example]):
        circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
        ax[2].add_artist(circle)
        ax[2].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
    ax[2].scatter(fitted_samples[:, 0], fitted_samples[:, 1], c=fitted_p, cmap='viridis', s=5, alpha=0.5)
    ax[2].scatter(points[:,0], points[:,1], c='brown', s=15, label='LiDAR Points', zorder=12)
    ax[2].set_title(f"Fitted GMM (EM - {n_iterations_fit_gmm} iterations) - Time: {fit_time/10:.6f}s")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")
    for cell_center in grid_cells:
        rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
        ax[2].add_patch(rect)
    plot_lidar_measurements(ax[2], raw_data["lidar_measurements"][example], raw_data["robot_positions"][example], 0.3)
    ax[2].set_aspect('equal', adjustable='box')
    plt.show()

