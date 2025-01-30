import jax.numpy as jnp
from jax import random, jit, vmap, nn
import matplotlib.pyplot as plt

key = random.PRNGKey(0)
n_samples = 1_000
mean = 0.
sigma = 1.
max_speed = 1.
wheel_distance = 0.7

@jit
def f_soft(key, mean, sigma, max_speed, wheel_distance=0.7):
    key1, key2 = random.split(key)
    vleft = mean + sigma * random.normal(key1) 
    vright = mean + sigma * random.normal(key2) 
    ## Bound the final action with SMOOTH CLIPPING (ensures gradient continuity)
    vleft = max_speed * jnp.tanh(vleft / max_speed)
    vright = max_speed * jnp.tanh(vright / max_speed)
    v = (vleft + vright) / 2
    #  v = v * jnp.tanh(v / 0.1) # Robot can only go forward
    v = nn.leaky_relu(v) # Robot can only go forward
    action = jnp.array([v, (vright - vleft) / wheel_distance])
    return action

@jit
def f_hard(key, mean, sigma, max_speed, wheel_distance):
    key1, key2 = random.split(key)
    vleft = mean + sigma * random.normal(key1) 
    vright = mean + sigma * random.normal(key2) 
    ## Bouind the final action with HARD CLIPPING (gradients discontinuity)
    vleft = jnp.clip(vleft, -max_speed, max_speed)
    vright = jnp.clip(vright, -max_speed, max_speed)
    v = jnp.abs((vleft + vright) / 2) # Robot can only go forward
    action = jnp.array([v, (vright - vleft) / wheel_distance])
    return action

@jit
def f2_soft(key, mean, sigma, max_speed):
    key1, key2 = random.split(key)
    vx = mean + sigma * random.normal(key1)
    vy = mean + sigma * random.normal(key2)
    norm = jnp.linalg.norm(jnp.array([vx, vy]))
    ## Bound the norm of the velocity with SMOOTH CLIPPING (ensures gradients continuity)
    scaling_factor = jnp.tanh(norm / max_speed) / (norm + 1e-5)
    vx = vx * scaling_factor
    vy = vy * scaling_factor
    ## Build final action
    action = jnp.array([vx, vy])
    return action

@jit
def f2_hard(key, mean, sigma, max_speed):
    key1, key2 = random.split(key)
    vx = mean + sigma * random.normal(key1)
    vy = mean + sigma * random.normal(key2)
    norm = jnp.linalg.norm(jnp.array([vx, vy]))
    ## Bound the norm of the velocity with HARD CLIPPING (gradients discontinuity)
    scaling_factor = jnp.clip(norm, 0., max_speed) / (norm + 1e-5)
    vx = vx * scaling_factor
    vy = vy * scaling_factor
    ## Build final action
    action = jnp.array([vx, vy])
    return action

soft_actions = vmap(f_soft, in_axes=(0, None, None, None, None))(random.split(key, n_samples), mean, sigma, max_speed, wheel_distance)
hard_actions = vmap(f_hard, in_axes=(0, None, None, None, None))(random.split(key, n_samples), mean, sigma, max_speed, wheel_distance)
figure, ax = plt.subplots(1,2, figsize=(10,10))
figure.suptitle(f"Smooth vs Hard actions clipping - Samples {n_samples} - Mean: {mean} - Sigma: {sigma} - Max speed: {max_speed} - Wheel distance: {wheel_distance}\nUNICYCLE KINEMATICS")
ax[0].plot(soft_actions[:,0], soft_actions[:,1], 'o', markersize=1)
ax[0].set_title('Smooth clipping')
ax[0].set_xlabel('v ($m/s$)')
ax[0].set_ylabel('$\omega$ $(rad/s)$')
ax[1].plot(hard_actions[:,0], hard_actions[:,1], 'o', markersize=1)
ax[1].set_title('Hard clipping')
ax[1].set_xlabel('v ($m/s$)')
ax[1].set_ylabel('$\omega$ $(rad/s)$')
plt.show()

soft_actions = vmap(f2_soft, in_axes=(0, None, None, None))(random.split(key, n_samples), mean, sigma, max_speed)
hard_actions = vmap(f2_hard, in_axes=(0, None, None, None))(random.split(key, n_samples), mean, sigma, max_speed)
figure, ax = plt.subplots(1,2, figsize=(10,10))
figure.suptitle(f"Smooth vs Hard actions clipping - Samples {n_samples} - Mean: {mean} - Sigma: {sigma} - Max speed: {max_speed}\nHOLONOMIC KINEMATICS")
ax[0].plot(soft_actions[:,0], soft_actions[:,1], 'o', markersize=1)
ax[0].set_title('Smooth clipping')
ax[0].set_xlabel('vx ($m/s$)')
ax[0].set_ylabel('vy $(m/s)$')
ax[1].plot(hard_actions[:,0], hard_actions[:,1], 'o', markersize=1)
ax[1].set_title('Hard clipping')
ax[1].set_xlabel('v ($m/s$)')
ax[1].set_ylabel('vy $(m/s)$')
plt.show()