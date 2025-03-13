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
sigmas = jnp.array([1., 1.])
linear_angular = False

# Initialize Gaussian distribution
gaussian = Gaussian()

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
    

# Get samples
key = random.PRNGKey(random_seed)
keys = random.split(key, n_samples)
samples = gaussian.batch_sample(means, sigmas, keys)

# Compute bounded samples
bounded_samples = vmap(_bound_action, in_axes=(0, None, None))(samples, vmax, wheels_distance)

# Compute entropy of distribution
entropy = gaussian.entropy(jnp.log(sigmas))
print("Entropy: {:.2f}".format(entropy))

# Compute PDF values
print("Mean action: [{:.2f}, {:.2f}]".format(means[0], means[1]), " - PDF value: ", "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), means)), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), means)))
print("PDF value of action [0, 0]: ", "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([0, 0]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([0, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(vmax), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([vmax, 0]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([vmax, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(-vmax), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([-vmax, 0]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([-vmax, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(2*vmax), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([2*vmax, 0]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([2*vmax, 0]))))
print("PDF value of action [0, 1]: ", "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([0, 1]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([0, 1]))))
print("PDF value of action [0, -1]: ", "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([0, -1]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([0, -1]))))
print("PDF value of action [{:.2f}, 1]: ".format(vmax/3), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([vmax/3, 1]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([vmax/3, 1]))))
print("PDF value of action [{:.2f}, -1]: ".format(vmax/3), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([vmax/3, -1]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([vmax/3, -1]))))
print("PDF value of action [0, {:.2f}]: ".format(2*vmax/wheels_distance), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([0, 2*vmax/wheels_distance]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([0, 2*vmax/wheels_distance]))))
print("PDF value of action [0, {:.2f}]: ".format(-2*vmax/wheels_distance), "{:.2f}".format(gaussian.p(means, jnp.log(sigmas), jnp.array([0, -2*vmax/wheels_distance]))), " - Neg log PDF value: ", "{:.2f}".format(gaussian.neglogp(means, jnp.log(sigmas), jnp.array([0, -2*vmax/wheels_distance]))))

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
pdf_values = gaussian.batch_p(means, jnp.log(sigmas), samples)
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