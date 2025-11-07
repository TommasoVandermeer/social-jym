from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import time

from jhsfm.hsfm import step
from jhsfm.utils import *

# Hyperparameters
leg_radius = 0.1
step_length = 0.5
n_humans = 15
circle_radius = 7
dt = 0.01
end_time = 15

# Initial conditions
humans_parameters = get_standard_humans_parameters(n_humans)
humans_state = np.zeros((n_humans, 6))
humans_leg_state = np.zeros((n_humans, 6)) # (x1,y1,phase1,x2,y2,phase2)
humans_goal = np.zeros((n_humans, 2))
angle_width = (2 * jnp.pi) / (n_humans)
for i in range(n_humans):
    # State: (px, py, bvx, bvy, theta, omega)
    humans_state[i,0] = circle_radius * jnp.cos(i * angle_width)
    humans_state[i,1] = circle_radius * jnp.sin(i * angle_width)
    humans_state[i,2] = 0
    humans_state[i,3] = 0
    humans_state[i,4] = -jnp.pi + i * angle_width
    humans_state[i,5] = 0
    # Legs state: (x1, y1, phase1, x2, y2, phase2)
    humans_leg_state[i,0] = humans_state[i,0] + 0.6 * humans_parameters[i,0] * jnp.cos(humans_state[i,4] + jnp.pi/2) + (step_length / 2) * jnp.cos(humans_state[i,4])
    humans_leg_state[i,1] = humans_state[i,1] + 0.6 * humans_parameters[i,0] * jnp.sin(humans_state[i,4] + jnp.pi/2) + (step_length / 2) * jnp.sin(humans_state[i,4])
    humans_leg_state[i,2] = 0
    humans_leg_state[i,3] = humans_state[i,0] + 0.6 * humans_parameters[i,0] * jnp.cos(humans_state[i,4] - jnp.pi/2) - (step_length / 2) * jnp.cos(humans_state[i,4])
    humans_leg_state[i,4] = humans_state[i,1] + 0.6 * humans_parameters[i,0] * jnp.sin(humans_state[i,4] - jnp.pi/2) - (step_length / 2) * jnp.sin(humans_state[i,4])
    humans_leg_state[i,5] = 0.5
    # Goal: (gx, gy)
    humans_goal[i,0] = -humans_state[i,0]
    humans_goal[i,1] = -humans_state[i,1]
humans_state = jnp.array(humans_state)
humans_goal = jnp.array(humans_goal)
# Obstacles
static_obstacles = jnp.array([[[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]]]) # dummy obstacles
static_obstacles_per_human = jnp.stack([static_obstacles for _ in range(len(humans_state))])

# Legs update function
@jit
def update_human_legs(human_state, human_leg_state, human_parameters, dt):
    position = human_state[0:2]
    radius = human_parameters[0]
    orientation = human_state[4]
    com_left_shifts = jnp.array([
        0.6 * radius * jnp.cos(orientation + jnp.pi/2),
        0.6 * radius * jnp.sin(orientation + jnp.pi/2)
    ])
    com_right_shifts = jnp.array([
        0.6 * radius * jnp.cos(orientation - jnp.pi/2),
        0.6 * radius * jnp.sin(orientation - jnp.pi/2)
    ])
    @jit
    def stance_leg(leg_x, leg_y):
        leg_x_new = leg_x
        leg_y_new = leg_y
        return leg_x_new, leg_y_new
    @jit
    def swing_leg(shifted_com, leg_x, leg_y, phase):
        alpha = (phase - 0.5) * 2 # Normalized phase from 0 to 1
        target_x = shifted_com[0] + (step_length / 2) * jnp.cos(orientation)
        target_y = shifted_com[1] + (step_length / 2) * jnp.sin(orientation)
        leg_x_new = (1 - alpha) * leg_x + alpha * target_x
        leg_y_new = (1 - alpha) * leg_y + alpha * target_y
        return leg_x_new, leg_y_new
    leftx, lefty = lax.cond(
        human_leg_state[2] < 0.5,
        lambda: stance_leg(human_leg_state[0], human_leg_state[1]),
        lambda: swing_leg(position + com_left_shifts, human_leg_state[0], human_leg_state[1], human_leg_state[2])
    )
    rightx, righty = lax.cond(
        human_leg_state[5] < 0.5,
        lambda: stance_leg(human_leg_state[3], human_leg_state[4]),
        lambda: swing_leg(position + com_right_shifts, human_leg_state[3], human_leg_state[4], human_leg_state[5])
    )
    phase_left_new = (human_leg_state[2] + dt) % 1.0
    phase_right_new = (human_leg_state[5] + dt) % 1.0
    new_human_leg_state = jnp.array([leftx, lefty, phase_left_new, rightx, righty, phase_right_new])
    # debug.print("Left leg velocity: {x}", x=jnp.linalg.norm(new_human_leg_state[:2]-human_leg_state[:2]) / dt)
    # debug.print("Right leg velocity: {x}", x=jnp.linalg.norm(new_human_leg_state[3:5]-human_leg_state[3:5]) / dt)
    return new_human_leg_state

@jit
def update_all_humans_legs(humans_state, humans_leg_state, humans_parameters, dt):
    return jax.vmap(update_human_legs, in_axes=(0,0,0,None))(humans_state, humans_leg_state, humans_parameters, dt)

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
_ = step(humans_state, humans_goal, humans_parameters, static_obstacles_per_human, dt)
_ = update_all_humans_legs(humans_state, humans_leg_state, humans_parameters, dt)

# Simulation 
steps = int(end_time/dt)
print(f"\nAvailable devices: {jax.devices()}\n")
print(f"Starting simulation... - Simulation time: {steps*dt} seconds\n")
start_time = time.time()
all_states = np.empty((steps+1, n_humans, 6), np.float32)
all_leg_states = np.empty((steps+1, n_humans, 6), np.float32)
all_states[0] = humans_state
all_leg_states[0] = humans_leg_state
for i in range(steps):
    # Update humans center of mass position and velocity
    humans_state = step(humans_state, humans_goal, humans_parameters, static_obstacles_per_human, dt)
    # Update legs phase and position
    humans_leg_state = update_all_humans_legs(humans_state, humans_leg_state, humans_parameters, dt)
    # Store state
    all_states[i+1] = humans_state
    all_leg_states[i+1] = humans_leg_state
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

### ANIMATION
animation_dt = 0.05
animation_ratio = int(animation_dt / dt)
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation
rc('font', weight='regular', size=20)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
fig, ax = plt.subplots(figsize=(8,8))
def animate(frame):
    ax.clear()
    ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
    ax.set_aspect('equal', adjustable='box')
    # Plot humans
    for h in range(len(all_states[int(frame*animation_ratio)])):
        color = "green"
        alpha = 0.3
        head = plt.Circle((all_states[int(frame*animation_ratio)][h,0] + jnp.cos(all_states[int(frame*animation_ratio)][h,4]) * humans_parameters[h,0], all_states[int(frame*animation_ratio)][h,1] + jnp.sin(all_states[int(frame*animation_ratio)][h,4]) * humans_parameters[h,0]), 0.1, color='black', alpha=alpha, zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((all_states[int(frame*animation_ratio)][h,0], all_states[int(frame*animation_ratio)][h,1]), humans_parameters[h,0], edgecolor='black', facecolor=color, alpha=alpha, fill=True, zorder=1)
        ax.add_patch(circle)
    # Plot legs
        leftx = all_leg_states[int(frame*animation_ratio)][h,0]
        lefty = all_leg_states[int(frame*animation_ratio)][h,1]
        rightx = all_leg_states[int(frame*animation_ratio)][h,3]
        righty = all_leg_states[int(frame*animation_ratio)][h,4]
        leg_circle = plt.Circle((leftx, lefty), leg_radius, edgecolor='black', facecolor='black', zorder=2)
        ax.add_patch(leg_circle)
        leg_circle = plt.Circle((rightx, righty), leg_radius, edgecolor='black', facecolor='black', zorder=2)
        ax.add_patch(leg_circle)
anim = FuncAnimation(fig, animate, interval=animation_dt*1000, frames=steps//animation_ratio)
anim.paused = False
def toggle_pause(self, *args, **kwargs):
    if anim.paused: anim.resume()
    else: anim.pause()
    anim.paused = not anim.paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)
plt.show()
