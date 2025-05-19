import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax, jit, vmap

vmax = 1.0
wheels_distance = 0.7
samples = 205
dt = 0.25

def exact_integration_of_action_space(x:jnp.ndarray, action:jnp.ndarray) -> jnp.ndarray:
    @jit
    def exact_integration_with_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + action[0] * jnp.cos(x[2]) * dt)
        x = x.at[1].set(x[1] + action[0] * jnp.sin(x[2]) * dt)
        return x
    @jit
    def exact_integration_with_non_zero_omega(x:jnp.ndarray) -> jnp.ndarray:
        x = x.at[0].set(x[0] + (action[0]/action[1]) * (jnp.sin(x[2] + action[1] * dt) - jnp.sin(x[2])))
        x = x.at[1].set(x[1] + (action[0]/action[1]) * (jnp.cos(x[2]) - jnp.cos(x[2] + action[1] * dt)))
        x = x.at[2].set(x[2] + action[1] * dt)
        return x
    x = lax.cond(
        action[1] != 0,
        exact_integration_with_non_zero_omega,
        exact_integration_with_zero_omega,
        x)
    return x
angular_speeds = jnp.linspace(-vmax/(wheels_distance/2), vmax/(wheels_distance/2), 2*samples-1)
speeds = jnp.linspace(0, vmax, samples)
unconstrained_action_space = jnp.empty((len(angular_speeds)*len(speeds),2))
unconstrained_action_space = lax.fori_loop(
    0,
    len(angular_speeds),
    lambda i, x: lax.fori_loop(
        0,
        len(speeds),
        lambda j, y: lax.cond(
            jnp.all(jnp.array([i<len(angular_speeds)-j, i>=j])),
            lambda z: z.at[i*len(speeds)+j].set(jnp.array([speeds[j],angular_speeds[i]])),
            lambda z: z.at[i*len(speeds)+j].set(jnp.array([jnp.nan,jnp.nan])),
            y),
        x),
    unconstrained_action_space)
action_space = unconstrained_action_space[~jnp.isnan(unconstrained_action_space).any(axis=1)]
pxy_theta = vmap(exact_integration_of_action_space, in_axes=(0, 0))(jnp.zeros((len(action_space),3)), action_space) / dt
plt.scatter(pxy_theta[:,0], pxy_theta[:,1])
# ## [vmax/2, vmax/(wheels_distance/2)]
# plt.scatter((wheels_distance/2) * (jnp.sin(dt/wheels_distance)) / dt, (wheels_distance/2) * (1-jnp.cos(dt/wheels_distance)) / dt, color='red')
# ## [vmax/2, -vmax/(wheels_distance/2)]
# plt.scatter(-(wheels_distance/2) * (jnp.sin(-dt/wheels_distance)) / dt, -(wheels_distance/2) * (1-jnp.cos(-dt/wheels_distance)) / dt, color='red')
## Plot restrictive linear contraints of (v,w) in (vx, vy) space
v_bound = 1.
w_bound = (vmax/(wheels_distance/2)) / 2
v_w_bound = jnp.zeros((samples, 2))
v_w_bound = v_w_bound.at[:,0].set(jnp.linspace(0, v_bound, samples, endpoint=True))
v_w_bound = v_w_bound.at[:,1].set(((v_bound - v_w_bound[:,0]) * w_bound) / v_bound)
vx_vy_bound = vmap(exact_integration_of_action_space, in_axes=(0, 0))(jnp.zeros((len(v_w_bound),3)), v_w_bound) / dt
# plt.scatter(vx_vy_bound[:,0], vx_vy_bound[:,1], color='red')
plt.scatter((wheels_distance/(2*dt)) * jnp.sin(vmax * dt / wheels_distance), vmax / jnp.pi, color='green')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Vx")
plt.ylabel("Vy")
plt.show()

# Plot Vy as a function of v (w lies on the outer positive boundary of the action space (v,w))
v = jnp.linspace(0, vmax, samples, endpoint=True)
vy = (v * wheels_distance / (2 * (vmax - v) * dt)) * (1 - jnp.cos(2*(vmax-v)*dt/ wheels_distance))
plt.scatter(v, vy)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("v")
plt.ylabel("Vy")
plt.show()

# Plot Vx as a function of w (v lies on the outer positive boundary of the action space (v,w))
w = jnp.linspace(0, vmax/(wheels_distance/2), samples, endpoint=True)
vy = ((2 * vmax - wheels_distance * w) / (2 * w * dt)) * (1 - jnp.cos(w * dt))
approx_vy = ((2 * vmax - wheels_distance * w)) / jnp.pi
approx_vy2 = wheels_distance * w / jnp.pi
# derivative_in_w0 = vmax * dt * w
# derivative_in_wmax = (wheels_distance * (jnp.cos((2*vmax*dt/wheels_distance) -1))) / (4 * vmax * dt) * w
plt.scatter(w, vy)
plt.scatter(w, approx_vy, color='red')
plt.scatter(w, approx_vy2, color='green')
plt.scatter(vmax/wheels_distance, vmax / jnp.pi, color='blue')
plt.xlabel("w")
plt.ylabel("Vy")
plt.show()

# Plot cos function from 0 to 2*vmax/R * dt
w = jnp.linspace(0, 2*vmax/(wheels_distance) / dt, samples, endpoint=True)
cos = jnp.cos(w * dt)
approx_cos = (jnp.pi - 2 * w * dt) / jnp.pi
plt.scatter(w, cos)
plt.scatter(w, approx_cos, color='red')
plt.xlabel("w")
plt.ylabel("cos(w * dt)")
plt.show()