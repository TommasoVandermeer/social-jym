import jax.numpy as jnp
from jax import random, jit, vmap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from socialjym.utils.distributions.gaussian import Gaussian

random_seed = 0
n_samples = 1_000
vmax = 1.
wheels_distance = 0.7
means = jnp.array([0., 0.])
sigmas = jnp.array([vmax/2, (2*vmax/wheels_distance)/2])
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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(bounded_samples[:,0], bounded_samples[:,1], pdf_values, label='PDF values')
ax.scatter(0, 0, 0, c='g', label='Origin')
ax.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
ax.legend()
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