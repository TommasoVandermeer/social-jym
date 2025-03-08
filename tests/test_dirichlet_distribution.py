import jax.numpy as jnp
from jax import random, jit, vmap, nn
import matplotlib.pyplot as plt

random_seed = 0
n_samples = 10_000
vmax = 1.0
wheels_distance = 0.7

@jit
def _dirichlet_sample(key:random.PRNGKey, alpha:jnp.ndarray, vmax:float, wheels_distance:float) -> jnp.ndarray:
    sample = random.dirichlet(key, alpha)
    return sample

@jit
def _batch_dirichlet_sample(keys:random.PRNGKey, alpha:jnp.ndarray, vmax:float, wheels_distance:float) -> jnp.ndarray:
    return vmap(_dirichlet_sample, in_axes=(0,None, None, None))(keys, alpha, vmax, wheels_distance)

key = random.PRNGKey(random_seed)
keys = random.split(key, n_samples)
alpha = jnp.array([1, 1, 1])
samples = _batch_dirichlet_sample(keys, alpha, vmax, wheels_distance)

figure, ax = plt.subplots(1, 1, figsize=(8, 8))
figure.suptitle(f'{n_samples} samples from a Dirichlet distribution with alpha {alpha}')
ax.scatter(samples[:,0], samples[:,1])
plt.show()