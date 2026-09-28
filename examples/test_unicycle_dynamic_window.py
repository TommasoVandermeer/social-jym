import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from shapely.geometry import Polygon

v_max = 0.45
wheels_distance = 0.235
w_max = (2 * v_max) / wheels_distance
w_limit_val = 1.9
r_robot = 0.2

init_v = 0.3
init_w = 0.0
init_amax = 0.9
init_dt = 0.25
init_alpha = 1.0
init_beta = 1.0
init_gamma = 1.0
init_limit = True

fig, ax = plt.subplots(figsize=(9, 10))
plt.subplots_adjust(bottom=0.45, right=0.8)
ax.set_xlim(-0.2, v_max + 0.2)
ax.set_ylim(-w_max - 1, w_max + 1)
ax.set_xlabel('v [m/s]')
ax.set_ylabel('omega [rad/s]')
ax.grid(True, linestyle='--', alpha=0.6)

ax.plot([v_max, 0.], [0., w_max], color='black', linewidth=2)
ax.plot([v_max, 0.], [0., -w_max], color='black', linewidth=2)
ax.plot([0., 0.], [w_max, -w_max], color='black', linewidth=2)

line_w_pos = ax.axhline(w_limit_val, color='red', linestyle='--', linewidth=1.5, zorder=1)
line_w_neg = ax.axhline(-w_limit_val, color='red', linestyle='--', linewidth=1.5, zorder=1)

scaled_poly_patch = ax.fill([], [], color='darkgreen', alpha=0.3, zorder=2)[0]
fill_patch = ax.fill([], [], color='green', alpha=0.6, zorder=3)[0]
state, = ax.plot([], [], 'ro', markersize=8, zorder=5)
line_blue1, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue2, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue3, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)
line_blue4, = ax.plot([], [], color='blue', linewidth=1.5, zorder=4)

def update_plot(v, w, a_max, dt, alpha, beta, gamma, apply_limit):
    dv = a_max * dt
    dw = (2 * dt * a_max) / wheels_distance
    state.set_data([v], [w])
    line_blue1.set_data([v, v + dv], [w - dw, w])
    line_blue2.set_data([v, v + dv], [w + dw, w])
    line_blue3.set_data([v, v - dv], [w - dw, w])
    line_blue4.set_data([v, v - dv], [w + dw, w])
    line_w_pos.set_visible(apply_limit)
    line_w_neg.set_visible(apply_limit)
    scaled_v_max = v_max * alpha
    scaled_w_max_pos = w_max * beta
    scaled_w_max_neg = w_max * gamma
    scaled_poly = Polygon([(0., scaled_w_max_pos), (scaled_v_max, 0.), (0., -scaled_w_max_neg)])
    if apply_limit:
        limit_poly = Polygon([(-1.0, w_limit_val), (2.0, w_limit_val), (2.0, -w_limit_val), (-1.0, -w_limit_val)])
        scaled_poly = scaled_poly.intersection(limit_poly)
    if not scaled_poly.is_empty and scaled_poly.geom_type == 'Polygon':
        x_scaled, y_scaled = scaled_poly.exterior.xy
        scaled_poly_patch.set_xy(np.column_stack((x_scaled, y_scaled)))
    else:
        scaled_poly_patch.set_xy(np.empty((0, 2)))
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

ax_v = plt.axes([0.15, 0.35, 0.60, 0.02])
ax_w = plt.axes([0.15, 0.30, 0.60, 0.02])
ax_a = plt.axes([0.15, 0.25, 0.60, 0.02])
ax_dt = plt.axes([0.15, 0.20, 0.60, 0.02])
ax_alpha = plt.axes([0.15, 0.15, 0.60, 0.02])
ax_beta = plt.axes([0.15, 0.10, 0.60, 0.02])
ax_gamma = plt.axes([0.15, 0.05, 0.60, 0.02])
ax_check = plt.axes([0.80, 0.15, 0.15, 0.1])

slider_v = Slider(ax_v, 'current_v', -0.2, v_max + 0.2, valinit=init_v)
slider_w = Slider(ax_w, 'current_w', -w_max, w_max, valinit=init_w)
slider_a = Slider(ax_a, 'a_max', 0.1, 2.0, valinit=init_amax)
slider_dt = Slider(ax_dt, 'dt', 0.05, 1.0, valinit=init_dt)
slider_alpha = Slider(ax_alpha, 'alpha', 0.0, 1.0, valinit=init_alpha)
slider_beta = Slider(ax_beta, 'beta', 0.0, 1.0, valinit=init_beta)
slider_gamma = Slider(ax_gamma, 'gamma', 0.0, 1.0, valinit=init_gamma)
check = CheckButtons(ax_check, ['w limit (1.9)'], [init_limit])

def on_change(val):
    update_plot(slider_v.val, slider_w.val, slider_a.val, slider_dt.val, slider_alpha.val, slider_beta.val, slider_gamma.val, check.get_status()[0])

slider_v.on_changed(on_change)
slider_w.on_changed(on_change)
slider_a.on_changed(on_change)
slider_dt.on_changed(on_change)
slider_alpha.on_changed(on_change)
slider_beta.on_changed(on_change)
slider_gamma.on_changed(on_change)
check.on_clicked(on_change)

update_plot(init_v, init_w, init_amax, init_dt, init_alpha, init_beta, init_gamma, init_limit)

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

### UNICYCLE SAFE CONSTRAINED ACTION SPACE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from shapely.geometry import Polygon

v_max = 0.45
wheels_distance = 0.235
w_max = (2 * v_max) / wheels_distance
w_limit_val = 1.9
r_robot = 0.17

init_v = 0.3
init_w = 0.0
init_amax = 0.9
init_dt = 0.25
init_d_front = 1.0 
init_d_left = 1.0  
init_d_right = 1.0 
init_limit = True

fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.45, right=0.75)
ax.set_xlim(-0.2, v_max + 0.2)
ax.set_ylim(-w_max - 1, w_max + 1)
ax.set_xlabel('v [m/s]')
ax.set_ylabel('omega [rad/s]')
ax.grid(True, linestyle='--', alpha=0.6)

# Limiti Statici base
ax.plot([v_max, 0.], [0., w_max], color='black', linewidth=2)
ax.plot([v_max, 0.], [0., -w_max], color='black', linewidth=2)
ax.plot([0., 0.], [w_max, -w_max], color='black', linewidth=2)

# Limite 1.9 Turtlebot
line_w_pos = ax.axhline(w_limit_val, color='red', linestyle='--', linewidth=1.5, zorder=1)
line_w_neg = ax.axhline(-w_limit_val, color='red', linestyle='--', linewidth=1.5, zorder=1)

# Limiti Fisici di Frenata (Arancioni)
line_v_safe = ax.axvline(v_max, color='orange', linestyle='-.', linewidth=2, zorder=6, label='Kinematic Safe Limit')
line_w_safe_pos = ax.axhline(w_max, color='orange', linestyle='-.', linewidth=2, zorder=6)
line_w_safe_neg = ax.axhline(-w_max, color='orange', linestyle='-.', linewidth=2, zorder=6)

scaled_poly_patch = ax.fill([], [], color='darkgreen', alpha=0.2, zorder=2)[0]
fill_patch = ax.fill([], [], color='green', alpha=0.7, zorder=4)[0]
state, = ax.plot([], [], 'ro', markersize=8, zorder=7)
line_blue1, = ax.plot([], [], color='blue', linewidth=1.5, zorder=5)
line_blue2, = ax.plot([], [], color='blue', linewidth=1.5, zorder=5)
line_blue3, = ax.plot([], [], color='blue', linewidth=1.5, zorder=5)
line_blue4, = ax.plot([], [], color='blue', linewidth=1.5, zorder=5)

def update_plot(v, w, a_max, dt, d_front, d_left, d_right, apply_limit):
    dv = a_max * dt
    dw = (2 * dt * a_max) / wheels_distance
    
    alpha = np.clip(d_front / (v_max * dt), 0.0, 1.0)
    alpha_safe = max(alpha, 1e-6) 
    
    beta = np.clip(4 * ((d_left * wheels_distance) / (alpha_safe * (v_max**2) * (dt**2))), 0.0, 1.0)
    gamma = np.clip(4 * ((d_right * wheels_distance) / (alpha_safe * (v_max**2) * (dt**2))), 0.0, 1.0)
    
    scaled_v_max = v_max * alpha
    scaled_w_max_pos = w_max * beta
    scaled_w_max_neg = w_max * gamma
    poly_scaled = Polygon([(0., scaled_w_max_pos), (scaled_v_max, 0.), (0., -scaled_w_max_neg)])
    
    v_safe = np.sqrt(2 * a_max * max(0.0, d_front))
    alpha_max_ang = (2 * a_max) / wheels_distance
    w_safe_pos = np.sqrt(2 * alpha_max_ang * (max(0.0, d_left) / r_robot))
    w_safe_neg = np.sqrt(2 * alpha_max_ang * (max(0.0, d_right) / r_robot))
    
    line_v_safe.set_xdata([v_safe, v_safe])
    line_w_safe_pos.set_ydata([w_safe_pos, w_safe_pos])
    line_w_safe_neg.set_ydata([-w_safe_neg, -w_safe_neg])
    
    poly_safe = Polygon([
        (0.0, w_safe_pos), 
        (v_safe, w_safe_pos), 
        (v_safe, -w_safe_neg), 
        (0.0, -w_safe_neg)
    ])
    
    state.set_data([v], [w])
    line_blue1.set_data([v, v + dv], [w - dw, w])
    line_blue2.set_data([v, v + dv], [w + dw, w])
    line_blue3.set_data([v, v - dv], [w - dw, w])
    line_blue4.set_data([v, v - dv], [w + dw, w])
    line_w_pos.set_visible(apply_limit)
    line_w_neg.set_visible(apply_limit)
    
    poly_static_valid = poly_scaled.intersection(poly_safe)
    
    if apply_limit:
        limit_poly = Polygon([(-2.0, w_limit_val), (2.0, w_limit_val), (2.0, -w_limit_val), (-2.0, -w_limit_val)])
        poly_static_valid = poly_static_valid.intersection(limit_poly)
        
    if not poly_static_valid.is_empty and poly_static_valid.geom_type == 'Polygon':
        x_s, y_s = poly_static_valid.exterior.xy
        scaled_poly_patch.set_xy(np.column_stack((x_s, y_s)))
    else:
        scaled_poly_patch.set_xy(np.empty((0, 2)))
        
    poly_dynamic = Polygon([
        (v, w + dw),
        (v + dv, w),
        (v, w - dw),
        (v - dv, w)
    ])

    feasible_poly = poly_static_valid.intersection(poly_dynamic)
    
    if not feasible_poly.is_empty and feasible_poly.geom_type == 'Polygon':
        x, y = feasible_poly.exterior.xy
        fill_patch.set_xy(np.column_stack((x, y)))
    else:
        fill_patch.set_xy(np.empty((0, 2)))
        
    fig.canvas.draw_idle()

ax_v = plt.axes([0.10, 0.35, 0.60, 0.02])
ax_w = plt.axes([0.10, 0.31, 0.60, 0.02])
ax_a = plt.axes([0.10, 0.27, 0.60, 0.02])
ax_dt = plt.axes([0.10, 0.23, 0.60, 0.02])
ax_d_front = plt.axes([0.10, 0.19, 0.60, 0.02])
ax_d_left = plt.axes([0.10, 0.15, 0.60, 0.02])
ax_d_right = plt.axes([0.10, 0.11, 0.60, 0.02])
ax_check = plt.axes([0.75, 0.20, 0.20, 0.1])

slider_v = Slider(ax_v, 'v', -0.2, v_max + 0.2, valinit=init_v)
slider_w = Slider(ax_w, 'w', -w_max, w_max, valinit=init_w)
slider_a = Slider(ax_a, 'a_max', 0.1, 2.0, valinit=init_amax)
slider_dt = Slider(ax_dt, 'dt', 0.05, 1.0, valinit=init_dt)
slider_d_front = Slider(ax_d_front, 'd_front', 0.0, 2.0, valinit=init_d_front)
slider_d_left = Slider(ax_d_left, 'd_left', 0.0, 2.0, valinit=init_d_left)
slider_d_right = Slider(ax_d_right, 'd_right', 0.0, 2.0, valinit=init_d_right)
check = CheckButtons(ax_check, ['w limit (1.9)'], [init_limit])

def on_change(val):
    update_plot(slider_v.val, slider_w.val, slider_a.val, slider_dt.val, 
                slider_d_front.val, slider_d_left.val, slider_d_right.val, 
                check.get_status()[0])

slider_v.on_changed(on_change)
slider_w.on_changed(on_change)
slider_a.on_changed(on_change)
slider_dt.on_changed(on_change)
slider_d_front.on_changed(on_change)
slider_d_left.on_changed(on_change)
slider_d_right.on_changed(on_change)
check.on_clicked(on_change)

update_plot(init_v, init_w, init_amax, init_dt, init_d_front, init_d_left, init_d_right, init_limit)

plt.show()