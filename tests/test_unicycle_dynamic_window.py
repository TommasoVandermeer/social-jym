import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from shapely.geometry import Polygon

v_max = 0.45
wheels_distance = 0.235
w_max = (2 * v_max) / wheels_distance

init_v = 0.3
init_w = 0.0
init_amax = 0.9
init_dt = 0.25
init_alpha = 1.0
init_beta = 1.0
init_gamma = 1.0

fig, ax = plt.subplots(figsize=(8, 10))
plt.subplots_adjust(bottom=0.45)
ax.set_xlim(-0.2, v_max + 0.2)
ax.set_ylim(-w_max - 1, w_max + 1)
ax.set_xlabel('v [m/s]')
ax.set_ylabel('omega [rad/s]')
ax.grid(True, linestyle='--', alpha=0.6)

ax.plot([v_max, 0.], [0., w_max], color='black', linewidth=2)
ax.plot([v_max, 0.], [0., -w_max], color='black', linewidth=2)
ax.plot([0., 0.], [w_max, -w_max], color='black', linewidth=2)

scaled_poly_patch = ax.fill([], [], color='darkgreen', alpha=0.3, zorder=2)[0]
fill_patch = ax.fill([], [], color='green', alpha=0.6, zorder=3)[0]
state, = ax.plot([], [], 'ro', markersize=8, zorder=5)
line_blue1, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue2, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue3, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue4, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)

def update_plot(v, w, a_max, dt, alpha, beta, gamma):
    dv = a_max * dt
    dw = (2 * dt * a_max) / wheels_distance
    
    state.set_data([v], [w])
    line_blue1.set_data([v, v + dv], [w - dw, w])
    line_blue2.set_data([v, v + dv], [w + dw, w])
    line_blue3.set_data([v, v - dv], [w - dw, w])
    line_blue4.set_data([v, v - dv], [w + dw, w])
    
    scaled_v_max = v_max * alpha
    scaled_w_max_pos = w_max * beta
    scaled_w_max_neg = w_max * gamma
    
    scaled_poly = Polygon([(0., scaled_w_max_pos), (scaled_v_max, 0.), (0., -scaled_w_max_neg)])
    
    x_scaled, y_scaled = scaled_poly.exterior.xy
    scaled_poly_patch.set_xy(np.column_stack((x_scaled, y_scaled)))

    dynamic_poly = Polygon([
        (v, w + dw),
        (v + dv, w),
        (v, w - dw),
        (v - dv, w)
    ])
    
    feasible_poly = scaled_poly.intersection(dynamic_poly)
    
    if not feasible_poly.is_empty and feasible_poly.geom_type == 'Polygon':
        x, y = feasible_poly.exterior.xy
        fill_patch.set_xy(np.column_stack((x, y)))
    else:
        fill_patch.set_xy(np.empty((0, 2)))
                
    fig.canvas.draw_idle()

update_plot(init_v, init_w, init_amax, init_dt, init_alpha, init_beta, init_gamma)

ax_v = plt.axes([0.15, 0.35, 0.65, 0.02])
ax_w = plt.axes([0.15, 0.30, 0.65, 0.02])
ax_a = plt.axes([0.15, 0.25, 0.65, 0.02])
ax_dt = plt.axes([0.15, 0.20, 0.65, 0.02])
ax_alpha = plt.axes([0.15, 0.15, 0.65, 0.02])
ax_beta = plt.axes([0.15, 0.10, 0.65, 0.02])
ax_gamma = plt.axes([0.15, 0.05, 0.65, 0.02])

slider_v = Slider(ax_v, 'current_v', -0.2, v_max + 0.2, valinit=init_v)
slider_w = Slider(ax_w, 'current_w', -w_max, w_max, valinit=init_w)
slider_a = Slider(ax_a, 'a_max', 0.1, 2.0, valinit=init_amax)
slider_dt = Slider(ax_dt, 'dt', 0.05, 1.0, valinit=init_dt)
slider_alpha = Slider(ax_alpha, 'alpha', 0.0, 1.0, valinit=init_alpha)
slider_beta = Slider(ax_beta, 'beta', 0.0, 1.0, valinit=init_beta)
slider_gamma = Slider(ax_gamma, 'gamma', 0.0, 1.0, valinit=init_gamma)

def on_change(val):
    update_plot(slider_v.val, slider_w.val, slider_a.val, slider_dt.val, slider_alpha.val, slider_beta.val, slider_gamma.val)

slider_v.on_changed(on_change)
slider_w.on_changed(on_change)
slider_a.on_changed(on_change)
slider_dt.on_changed(on_change)
slider_alpha.on_changed(on_change)
slider_beta.on_changed(on_change)
slider_gamma.on_changed(on_change)

plt.show()

### SAFE VELOCITY CONSTRAINT

init_A_max = 0.9
init_r_robot = 0.2

fig, (ax_lin, ax_ang) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.3)

d_front = np.linspace(0, 2.0, 500)
d_lat = np.linspace(0, 1.0, 500)

ax_lin.set_xlim(0, 2.0)
ax_lin.set_ylim(0, v_max + 0.1)
ax_lin.set_xlabel('d_front (from center) [m]')
ax_lin.set_ylabel('v [m/s]')
ax_lin.grid(True, linestyle='--', alpha=0.6)
ax_lin.axhline(v_max, color='black', linestyle='--', linewidth=1.5)

ax_ang.set_xlim(0, 1.0)
ax_ang.set_ylim(0, w_max + 0.5)
ax_ang.set_xlabel('d_lat (from center) [m]')
ax_ang.set_ylabel('omega [rad/s]')
ax_ang.grid(True, linestyle='--', alpha=0.6)
ax_ang.axhline(w_max, color='black', linestyle='--', linewidth=1.5)

line_lin, = ax_lin.plot([], [], color='green', linewidth=2)
line_ang, = ax_ang.plot([], [], color='green', linewidth=2)

def update(A_max, r_robot):
    clearance_front = np.maximum(0, d_front - r_robot)
    v_safe = np.minimum(v_max, np.sqrt(2 * A_max * clearance_front))
    
    clearance_lat = np.maximum(0, d_lat - r_robot)
    alpha_max = (2 * A_max) / wheels_distance
    w_safe = np.minimum(w_max, np.sqrt(2 * alpha_max * (clearance_lat / r_robot)))
    
    line_lin.set_data(d_front, v_safe)
    line_ang.set_data(d_lat, w_safe)
    fig.canvas.draw_idle()

update(init_A_max, init_r_robot)

ax_slider_A = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_slider_r = plt.axes([0.15, 0.08, 0.7, 0.03])

slider_A = Slider(ax_slider_A, 'A_max (wheels)', 0.1, 2.0, valinit=init_A_max)
slider_r = Slider(ax_slider_r, 'r_robot', 0.1, 0.5, valinit=init_r_robot)

def on_change(val):
    update(slider_A.val, slider_r.val)

slider_A.on_changed(on_change)
slider_r.on_changed(on_change)

plt.show()