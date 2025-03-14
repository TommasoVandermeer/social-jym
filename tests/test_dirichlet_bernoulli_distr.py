import jax.numpy as jnp
from jax import random, jit, vmap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from socialjym.utils.distributions.dirichlet_bernoulli import DirichletBernoulli

random_seed = 0
n_samples = 5_000
vmax = 1
wheels_distance = 0.7
alpha = jnp.array([1, 3, 1])
concentration = jnp.sum(alpha)
p = .5
epsilon = 1e-6

# Generate distribution
distribution = DirichletBernoulli(vmax, wheels_distance, epsilon)

# Normalize alpha and p
@jit
def _normalize_alphas(alpha:jnp.ndarray, concentration:float) -> jnp.ndarray:
    return alpha / jnp.sum(alpha) * concentration   
alpha = _normalize_alphas(alpha, concentration)
print("Normalized alpha: ", alpha)
p = jnp.clip(p, 0., 1.)
distr = {"alphas": alpha, "p": p}

# Get samples
key = random.PRNGKey(random_seed)
keys = random.split(key, n_samples)
samples = distribution.batch_sample(distr, keys)

# Compute entropy of distribution
entropy = distribution.entropy(distr)
print("Entropy: {:.2f}".format(entropy))

# Compute mean action and its probability
mean_v, mean_w = distribution.mean(distr)
print("Mean action: [{:.2f}, {:.2f}]".format(mean_v, mean_w), " - PDF value: ", "{:.2f}".format(distribution.p(distr, jnp.array([mean_v, mean_w]))))
# Compute probability of several actions
print("PDF value of action [0, 0]: ", "{:.2f}".format(distribution.p(distr, jnp.array([0, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(vmax), "{:.2f}".format(distribution.p(distr, jnp.array([vmax, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(-vmax), "{:.2f}".format(distribution.p(distr, jnp.array([-vmax, 0]))))
print("PDF value of action [{:.2f}, 0]: ".format(2*vmax), "{:.2f}".format(distribution.p(distr, jnp.array([2*vmax, 0]))))
print("PDF value of action [0, 1]: ", "{:.2f}".format(distribution.p(distr, jnp.array([0, 1]))))
print("PDF value of action [0, -1]: ", "{:.2f}".format(distribution.p(distr, jnp.array([0, -1]))))
print("PDF value of action [{:.2f}, 1]: ".format(vmax/3), "{:.2f}".format(distribution.p(distr, jnp.array([vmax/3, 1]))))
print("PDF value of action [{:.2f}, -1]: ".format(vmax/3), "{:.2f}".format(distribution.p(distr, jnp.array([vmax/3, -1]))))
print("PDF value of action [0, {:.2f}]: ".format(2*vmax/wheels_distance), "{:.2f}".format(distribution.p(distr, jnp.array([0, 2*vmax/wheels_distance]))))
print("PDF value of action [0, {:.2f}]: ".format(-2*vmax/wheels_distance), "{:.2f}".format(distribution.p(distr, jnp.array([0, -2*vmax/wheels_distance]))))

# Plot samples and mean action
figure, ax = plt.subplots(1, 1, figsize=(8, 8))
figure.suptitle(f'{n_samples} samples from action space - Alpha = {alpha} - P = {p}')
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

# Compute PDF value of samples
more_samples = jnp.append(samples, jnp.column_stack((jnp.arange(0,1,0.01),jnp.zeros(100))), axis=0)
pdf_values = distribution.batch_p(distr, more_samples)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(more_samples[:,0], more_samples[:,1], pdf_values, label='PDF values')
ax.scatter(0, 0, 0, c='g', label='Origin')
ax.set(xlim=[-0.1,vmax+0.1], ylim=[-vmax*2/wheels_distance-0.2,+vmax*2/wheels_distance+0.2])
ax.legend()
plt.show()