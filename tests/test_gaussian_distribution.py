import jax.numpy as jnp
from jax import random, jit, vmap
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 20
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from socialjym.utils.distributions.gaussian import Gaussian

random_seed = 0
n_samples = 5_000
vmax = 1.
wheels_distance = 0.7
means = jnp.array([0.29, 0.]) #jnp.array([0., 0.])
sigmas = jnp.array([0.21722028, 1.144384]) #jnp.array([vmax/2, (2*vmax/wheels_distance)/2])
linear_angular = True

# Initialize Gaussian distribution
gaussian = Gaussian()
distr = {"means": means, "logsigmas": jnp.log(sigmas)}
print("Std: ", gaussian.std(distr))

@jit
def _bound_action(action:jnp.ndarray, vmax:float, wheels_distance:float) -> jnp.ndarray:
    if linear_angular:
        v = jnp.clip(action[0], 0., vmax)
        w_max = (2 * (vmax - v)) / wheels_distance # Real feasible actions
        # w_max = vmax * 2 / wheels_distance # Box bounded actions
        w = jnp.clip(action[1], -w_max, w_max)
        return jnp.array([v, w])
    else:
        v_left = jnp.clip(action[0], -vmax, vmax)
        v_right = jnp.clip(action[1], -vmax, vmax)
        v = (v_left + v_right) / 2
        v = jnp.abs(v)
        w = (v_right - v_left) / wheels_distance
        return jnp.array([v, w])
    
@jit
def compute_neg_log_pdf_value(
    mu1:jnp.ndarray, 
    mu2:jnp.ndarray,
    logsigma:jnp.ndarray,
    action:jnp.ndarray
) -> jnp.ndarray:
    return .5 * jnp.sum(jnp.square((action - jnp.array([mu1, mu2])) / jnp.exp(logsigma))) + jnp.log(2 * jnp.pi) + logsigma

# Get samples
key = random.PRNGKey(random_seed)
keys = random.split(key, n_samples)
samples = gaussian.batch_sample(distr, keys)

# Compute bounded samples
# bounded_samples = vmap(_bound_action, in_axes=(0, None, None))(samples, vmax, wheels_distance)
bounded_samples = vmap(gaussian.bound_action, in_axes=(0, None, None, None))(samples, 1, vmax, wheels_distance)
samples_actually_bounded = samples[jnp.any(bounded_samples != samples, axis=1)]
samples_not_bounded = samples[jnp.all(bounded_samples == samples, axis=1)]

# Compute entropy of distribution
entropy = gaussian.entropy(distr)
print("Entropy: {:.2f}".format(entropy))

# Compute PDF values
actions_to_test = jnp.array([
    [means[0], means[1]],
    [0, 0],
    [vmax, 0],
    [-vmax, 0],
    [2*vmax, 0],
    [0, 1],
    [0, -1],
    [vmax/3, 1],
    [vmax/3, -1],
    [0, 2*vmax/wheels_distance],
    [0, -2*vmax/wheels_distance],
])
for action in actions_to_test:
    print("PDF value of action : [{:.2f}, {:.2f}]".format(action[0], action[1]), \
          " - PDF value: ", "{:.2f}".format(gaussian.p(distr, action)), \
          " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(distr, action)), \
          " - Neg log PDF value 2: ", "{:.2f}".format(compute_neg_log_pdf_value(means[0], means[1], jnp.log(sigmas[0]), action)))

# Plot samples and mean action
figure, ax = plt.subplots(1, 1, figsize=(8, 8))
figure.suptitle(f'{n_samples} samples from action space - Means = {means} - Sigmas = {sigmas}')
ax.scatter(bounded_samples[:,0], bounded_samples[:,1], label='Samples')
ax.scatter(means[0], means[1], c='r', label='Mean action')
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
pdf_values = gaussian.batch_p(distr, samples)
fig = plt.figure(figsize=(16, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, wspace=0.08)
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
ax = fig.add_subplot(gs[0, 0], projection='3d')
ax.scatter(samples_not_bounded[:,0], samples_not_bounded[:,1], gaussian.batch_p(distr, samples_not_bounded), label='Feasible actions', zorder=5)
ax.scatter(samples_actually_bounded[:,0], samples_actually_bounded[:,1], gaussian.batch_p(distr, samples_actually_bounded), c='r', alpha=0.15, zorder=2)
ax.plot([vmax, 0, 0, vmax], [0, -2*vmax/wheels_distance, 2*vmax/wheels_distance,0], [0,0,0,0], c='black', linewidth=2, zorder=3, label="Action space bounds")
ax.set(xlim=[-vmax-0.3,vmax+0.3], ylim=[-vmax*2/wheels_distance-0.5,+vmax*2/wheels_distance+0.5])
ax.set_xlabel('$v$', labelpad=15)
ax.set_ylabel('$\\omega$', labelpad=15)
ax.set_zlabel(r'$f_{V \Omega}(v, \omega)$', labelpad=15)
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, plt.Line2D([0], [0], marker='o', color='w', label='Infeasible actions', markerfacecolor='r', alpha=.8, markersize=6))
labels.insert(0, 'Infeasible actions')
ax.legend(handles, labels)
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
fig.savefig(os.path.join(os.path.dirname(__file__), 'gaussian_unbounded_distribution.png'), dpi=300)
plt.show()

# Compute PDF value of samples and plot (unbounded) distribution
pdf_values = gaussian.batch_p(distr, samples)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(bounded_samples[:,0], bounded_samples[:,1], pdf_values, label='PDF values')
ax.scatter(0, 0, 0, c='g', label='Origin')
ax.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
ax.set_xlabel('$v$')
ax.set_ylabel('$\\omega$')
ax.set_zlabel(r'$f_{V \Omega}(v, \omega)$')
plt.show()

# Plot real (bounded) distribution
hist, xedges, yedges = jnp.histogram2d(bounded_samples[:,0], bounded_samples[:,1], bins=70, range=[[0, vmax], [-vmax*2/wheels_distance, vmax*2/wheels_distance]])
xpos, ypos = jnp.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = 1/50 * jnp.ones_like(zpos)
dy = 1/10 * jnp.ones_like(zpos)
dz = hist.ravel()
mask_dz = dz == 0
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
ax.bar3d(xpos[~mask_dz], ypos[~mask_dz], zpos, dx, dy, dz[~mask_dz], zsort='average')
plt.show()