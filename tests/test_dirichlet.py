import jax.numpy as jnp
from jax import random, jit
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
font = {
    'weight' : 'regular',
    'size'   : 20
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.utils.distributions.dirichlet import Dirichlet

random_seed = 0
n_samples = 5_000
vmax = 1
wheels_distance = 0.7
alpha = jnp.array([1.2,1.2,1.])
concentration = jnp.sum(alpha)
epsilon = 1e-3

# Generate distribution
distribution = Dirichlet(epsilon)

# Normalize alpha
@jit
def _reparametrize_alphas(alpha:jnp.ndarray, concentration:float) -> jnp.ndarray:
    return alpha / jnp.sum(alpha) * concentration   
# alpha = _reparametrize_alphas(alpha, concentration)
print("Normalized alpha: ", alpha)
distr = {"alphas": alpha, "vertices": jnp.array([[0,2*vmax/wheels_distance],[0,-2*vmax/wheels_distance],[vmax,0]])}

# Get samples
key = random.PRNGKey(random_seed)
keys = random.split(key, n_samples)
samples = distribution.batch_sample(distr, keys)

# Compute entropy of distribution
entropy = distribution.entropy(distr)
print("Entropy: {:.2f}".format(entropy))
print("Std: ", distribution.std(distr))

# Compute mean action and its probability
mean_v, mean_w = distribution.mean(distr)
print("Mean action: [{:.2f}, {:.2f}]".format(mean_v, mean_w), " - PDF value: ", "{:.2f}".format(distribution.p(distr, jnp.array([mean_v, mean_w]))))
# Compute probability of several actions
actions_on_the_border = jnp.array([
    [0, 0],
    [vmax, 0],
    [0, 1],
    [0, -1],
    [0, 2*vmax/wheels_distance],
    [0, -2*vmax/wheels_distance],
    [0.70365906, 0.8466885],
])
print("\nActions on the border of the feasible region:")
for action in actions_on_the_border:
    print("PDF value of action : [{:.2f}, {:.2f}]".format(action[0], action[1]), \
          " - PDF value: ", "{:.2f}".format(distribution.p(distr, action)), \
          " - Log PDF: ", distribution.logp(distr, action))
actions_outside = jnp.array([
    [2*vmax, 0],
    [-vmax, 0],
    [-2*vmax, 0],
    [vmax+0.01, 0.01],
    [vmax+0.01, -0.01],
])
print("\nActions outside the feasible region:")
for action in actions_outside:
    print("PDF value of action : [{:.2f}, {:.2f}]".format(action[0], action[1]), \
          " - PDF value: ", "{:.2f}".format(distribution.p(distr, action)), \
          " - Log PDF: ", distribution.logp(distr, action))
actions_inside = jnp.array([
    [vmax/3, 1],
    [vmax/3, -1],
    [vmax/5, 1],
    [vmax/5, -1],
    [vmax/7, 1],
    [vmax/7, -1],
    [0.2, 2.],
    [0.2, -2],
    [0.5, 0.5],
    [0.5, -0.5],
])
print("\nActions inside the feasible region:")
for action in actions_inside:
    print("PDF value of action : [{:.2f}, {:.2f}]".format(action[0], action[1]), \
          " - PDF value: ", "{:.2f}".format(distribution.p(distr, action)), \
          " - Log PDF: ", distribution.logp(distr, action))

# Plot samples and mean action
figure, ax = plt.subplots(1, 1, figsize=(8, 8))
figure.suptitle(f'{n_samples} samples from action space - Alpha = {alpha}')
ax.scatter(samples[:,0], samples[:,1], label='Samples')
ax.scatter(mean_v, mean_w, c='r', label='Mean action')
ax.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
actions_space_bound = Polygon(
    jnp.array([[vmax,0.],[0.,vmax*2/wheels_distance],[0.,-vmax*2/wheels_distance]]), 
    closed=True, 
    fill=None, 
    edgecolor='black',
    linewidth=2,
    zorder=1,
    label="Action space bounds"
)
ax.add_patch(actions_space_bound)
ax.legend()
plt.show()

# Compute PDF value of samples and plot (unbounded) distribution
pdf_values = distribution.batch_p(distr, samples)
fig = plt.figure(figsize=(16, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, wspace=0.08)
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
ax = fig.add_subplot(gs[0, 0], projection='3d')
ax.scatter(samples[:,0], samples[:,1], pdf_values, label='Feasible actions', zorder=5)
ax.plot([vmax, 0, 0, vmax], [0, -2*vmax/wheels_distance, 2*vmax/wheels_distance,0], [0,0,0,0], c='black', linewidth=2, zorder=3, label="Action space bounds")
ax.set(xlim=[-vmax-0.3,vmax+0.3], ylim=[-vmax*2/wheels_distance-0.5,+vmax*2/wheels_distance+0.5])
ax.set_xlabel('$v$', labelpad=15)
ax.set_ylabel('$\\omega$', labelpad=15)
ax.set_zlabel(r'$f_{V \Omega}(v, \omega)$', labelpad=15)
ax.legend()
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(samples[:,0], samples[:,1], c=pdf_values, cmap='viridis', label='Samples')
ax2.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
# # Add horizontal colorbar for pdf values on top of the figure
# mappable = ax2.collections[-1]  # the scatter we just drew
# cbar = fig.colorbar(mappable, ax=ax2, orientation='horizontal', pad=0.12, fraction=0.04)
# cbar.ax.xaxis.set_ticks_position('top')
# cbar.ax.xaxis.set_label_position('top')
# cbar.set_label('PDF value')
# ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('$v$')
ax2.set_ylabel('$\\omega$', labelpad=-5)
actions_space_bound = Polygon(
    jnp.array([[vmax,0.],[0.,vmax*2/wheels_distance],[0.,-vmax*2/wheels_distance]]), 
    closed=True, 
    fill=None, 
    edgecolor='black',
    linewidth=4,
    zorder=1,
    label="Action space bounds"
)
ax2.add_patch(actions_space_bound)
fig.savefig(os.path.join(os.path.dirname(__file__), 'dirichlet_unbounded_distribution.png'), dpi=300)
plt.show()