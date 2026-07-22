import jax.numpy as jnp
from jax import vmap
import os
import matplotlib.pyplot as plt

from socialjym.utils.distributions.gaussian import Gaussian

distr = Gaussian()
L = 0.7 # Distance between the wheels of the robot
v_max = 1. # Maximum linear velocity of the robot
dt = 0.25
radius = 0.3

### UTILS
w_max = 2*v_max/L # Maximum angular velocity of the robot, given the maximum linear velocity and the distance between the wheels
vertices = jnp.array([
    [0.,w_max],
    [0.,-w_max],
    [v_max, 0],
])

## Compute bounded action
unbounded_actions = jnp.array([
    [2.0,0.0],
    [-1.0,0.0],
    [0.9, 2.0],
    [0.9, -2.0],
    [0.0, 4.0],
    [0.0,-4.0],
    [3.3,4.4],
    [-2.0,4.0],
])
bounded_actions = vmap(distr.bound_action_safety, in_axes=(0,None))(unbounded_actions, vertices)

### FIG1: action_space.eps
figure, ax = plt.subplots(1,1, figsize=(10, 3))
figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25)
ax.add_patch(
    plt.Polygon(
        [   
            [w_max,0],
            [-w_max,0],
            [0.,v_max],
        ],
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=2,
    ),
)
ax.set_xticks([w_max, 0., -w_max])
ax.set_xticklabels([r"$\overline{\omega}$", "0", r"$-\overline{\omega}$"])
ax.set_yticks([0.,v_max])
ax.set_yticklabels(["0", r"$\overline{v}$"])
ax.set_ylim(-2, v_max + 2)
ax.set_xlim(-w_max - 2, w_max + 2)
ax.set_ylabel("$v$ (m/s)", labelpad=-15)
ax.set_xlabel("$\omega$ (rad/s)", labelpad=-5)
ax.plot([-10, 10], [0, 0], color='black', linewidth=3, zorder=5)
ax.text(0, 0.1, "$v \geq 0$", zorder=5, verticalalignment='bottom', horizontalalignment='center')
ax.plot([-(w_max)*2, w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
# ax.text(-2.5 , L, r"$\omega \leq \frac{\overline{\omega}}{\overline{v}}v - \overline{\omega}$", zorder=5, verticalalignment='center', horizontalalignment='left')
ax.text(-2.5 , L, r"$\omega \geq \frac{2(v-\overline{v})}{L}$", zorder=5, verticalalignment='center', horizontalalignment='left')
ax.plot([(w_max)*2, -w_max], [-v_max, 2], color='black', linewidth=3, zorder=5)
# ax.text(2.5 , L, r"$\omega \leq \overline{\omega} - \frac{\overline{\omega}}{\overline{v}}v$", zorder=5, verticalalignment='center', horizontalalignment='right')
ax.text(2.5 , L, r"$\omega \leq \frac{2(\overline{v}-v)}{L}$", zorder=5, verticalalignment='center', horizontalalignment='right')
ax.scatter(unbounded_actions[:,1], unbounded_actions[:,0], color='red', zorder=100)
ax.scatter(bounded_actions[:,1], bounded_actions[:,0], color='blue', zorder=100)

ax.grid()
plt.show()
