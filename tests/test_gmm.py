from jax import random, jit, vmap, lax, debug
import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt

from socialjym.utils.distributions.gaussian_mixture_model import GMM

# Parameters
random_seed = 2
n_samples = 10_000
grid_resolution = 10  # Number of grid cells per dimension
scaling = 0.01  # Scaling factor for the covariance based on human radius
sampling_time = 2 # Time step for propagating humans' positions
# Humans position, radii and velocities
humans_position = random.uniform(random.PRNGKey(random_seed), shape=(5, 2), minval=-5., maxval=5.)
humans_radii = random.uniform(random.PRNGKey(random_seed), shape=(5,), minval=0.25, maxval=0.4)
humans_velocity = jnp.array([[0.5, 0.5], [0.7, 0.5], [0.9, 0.0], [0.0, 0.9], [0.3, -0.7]])
# Local grid over which the GMM is defined
dists = jnp.concatenate([-jnp.arange(0, 5, grid_resolution/10)[::-1][:-1],jnp.arange(0, 5, grid_resolution/10)])
grid_cell_coords = jnp.meshgrid(dists, dists)
grid_cells = jnp.array(jnp.vstack((grid_cell_coords[0].flatten(), grid_cell_coords[1].flatten())).T)
cell_size = (grid_cells[1,0] - grid_cells[0,0], grid_cells[grid_resolution,1] - grid_cells[0,1])  # Assuming uniform grid
# Initialize GMM
gmm = GMM(n_dimensions=grid_cells.shape[1], n_components=grid_cells.shape[0])

# ### Initialize a random GMM and visualize it
# # Initialize a GMM with means at the grid cell centers, random variances and uniform weights.
# random_weights = random.uniform(random.PRNGKey(random_seed), shape=(len(grid_cells),), minval=0., maxval=10.)
# random_weights = softmax(random_weights)
# distribution = {
#     "means": grid_cells,
#     "logvariances": random.uniform(random.PRNGKey(random_seed), shape=(len(grid_cells), len(grid_cells[0])), minval=-100, maxval=10),
#     "weights": random_weights,
# }
# gmm = GMM(n_dimensions=len(distribution["means"][0]), n_components=len(distribution["means"]))
# print("Mean:", gmm.mean(distribution))
# # Sample from the GMM and compute the probability density of each sample
# samples = gmm.batch_sample(distribution, random.split(random.PRNGKey(random_seed), n_samples))
# p = gmm.batch_p(distribution, samples)
# # Plot the samples in 2D and color them by their probability density
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(samples[:, 0], samples[:, 1], p, c=p, cmap='viridis', s=5, alpha=0.7)
# ax.set_title("Random Grid Gaussian Mixture Model")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Probability Density")
# fig.colorbar(ax.collections[0], ax=ax, label='Probability Density')
# plt.show()

### Encode the humans' positions in the GMM (Fit the GMM to the humans' positions)
@jit
def fit_gmm_to_humans_positions(humans_position, humans_radii, grid_cells, scaling):
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
        "logvariances": jnp.log(human_weighted_covariances_per_cell),
        "weights": norm_cell_weights,
    }
    return fitted_distribution
fitted_distribution = fit_gmm_to_humans_positions(humans_position, humans_radii, grid_cells, scaling)
samples = gmm.batch_sample(fitted_distribution, random.split(random.PRNGKey(random_seed), n_samples))
p = gmm.batch_p(fitted_distribution, samples)
# Plot fitted distribution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], p, c=p, cmap='viridis', s=5, alpha=0.7)
for pos in humans_position:
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, ax.get_zlim()[1]], color='red', linewidth=2)
ax.set_title("Humans' position fitted GMM")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlabel("Probability Density")
ax.set_xticks(jnp.append(dists - cell_size[0]/2, dists[-1] + cell_size[0]/2))
ax.set_yticks(jnp.append(dists - cell_size[1]/2, dists[-1] + cell_size[1]/2))
fig.colorbar(ax.collections[0], ax=ax, label='Probability Density')

### Encode next humans' position in the GMM (Fit the GMM to the humans' velocities)
# Propagate humans' positions according to their velocities
future_humans_position = humans_position + humans_velocity * sampling_time
future_fitted_distribution = fit_gmm_to_humans_positions(
    future_humans_position,
    humans_radii,
    grid_cells,
    scaling
)
future_samples = gmm.batch_sample(future_fitted_distribution, random.split(random.PRNGKey(random_seed), n_samples))
future_p = gmm.batch_p(future_fitted_distribution, future_samples)
# Plot fitted distribution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(future_samples[:, 0], future_samples[:, 1], future_p, c=future_p, cmap='viridis', s=5, alpha=0.7)
for pos in future_humans_position:
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, ax.get_zlim()[1]], color='red', linewidth=2)
ax.set_title("Next Humans' position fitted GMM")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlabel("Probability Density")
ax.set_xticks(jnp.append(dists - cell_size[0]/2, dists[-1] + cell_size[0]/2))
ax.set_yticks(jnp.append(dists - cell_size[1]/2, dists[-1] + cell_size[1]/2))
fig.colorbar(ax.collections[0], ax=ax, label='Probability Density')

### Plot top views of the fitted distributions
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].set_xlim(-7, 7)
ax[0].set_ylim(-7, 7)
for pos, rad, vel in zip(humans_position, humans_radii, humans_velocity):
    circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
    ax[0].add_artist(circle)
    ax[0].arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=11)
ax[0].scatter(samples[:, 0], samples[:, 1], c=p, cmap='viridis', s=5, alpha=0.5)
ax[0].set_title("Humans and Fitted GMM Top View")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
for cell_center in grid_cells:
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    ax[0].add_patch(rect)
ax[0].set_aspect('equal', adjustable='box')
# Plot top view of the fitted distribution
ax[1].set_xlim(-7, 7)
ax[1].set_ylim(-7, 7)
for pos, rad in zip(future_humans_position, humans_radii):
    circle = plt.Circle(pos, rad, color='red', alpha=1, zorder=10)
    ax[1].add_artist(circle)
ax[1].scatter(future_samples[:, 0], future_samples[:, 1], c=future_p, cmap='viridis', s=5, alpha=0.5)
ax[1].set_title(f"Next Humans and Fitted GMM Top View - $\Delta t$={sampling_time}s")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
for cell_center in grid_cells:
    rect = plt.Rectangle((cell_center[0]-cell_size[0]/2, cell_center[1]-cell_size[1]/2), cell_size[0], cell_size[1], facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=1)
    ax[1].add_patch(rect)
ax[1].set_aspect('equal', adjustable='box')

plt.show()