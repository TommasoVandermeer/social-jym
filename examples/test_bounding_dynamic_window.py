import jax
import jax.numpy as jnp

def get_feasible_action_space(v, w, x_min, y_min_left, y_min_right, a_max=0.9, dt=0.25):
    """
    Computes the vertices of the Feasible Action Space Polygon.
    x_min: Minimum clearance directly in front of the robot (from pointcloud)
    y_min_left: Minimum lateral clearance on the left side
    y_min_right: Minimum lateral clearance on the right side
    
    Returns an array of shape (16, 2). Unused vertices are padded with jnp.nan.
    """
    # 1. Costanti fisiche del TurtleBot4
    v_max = 0.45
    wheels_distance = 0.235
    w_max = (2 * v_max) / wheels_distance
    w_limit_val = 1.9
    r_robot = 0.17
    MAX_VERTICES = 16

    v = jnp.asarray(v, dtype=jnp.float32)
    w = jnp.asarray(w, dtype=jnp.float32)
    
    # 2. INIZIALIZZAZIONE: Il Rombo Dinamico (Accelerazione Accoppiata)
    dv = a_max * dt
    dw = (2 * a_max * dt) / wheels_distance

    init_verts = jnp.array([
        [v + dv, w],
        [v, w + dw],
        [v - dv, w],
        [v, w - dw]
    ])
    
    vertices = jnp.pad(init_verts, ((0, MAX_VERTICES - 4), (0, 0)), constant_values=jnp.nan)
    num_vertices = jnp.int32(4)

    # 3. CALCOLO DEI VINCOLI (Cinematica vs Dinamica)
    
    # --- A. Vincoli Cinematici (Intenzione: Heuristic Scaling DIR-SAFE) ---
    alpha = jnp.clip(x_min / (v_max * dt), 0.0, 1.0)
    alpha_safe = jnp.maximum(alpha, 1e-6)
    beta = jnp.clip(4 * ((y_min_left * wheels_distance) / (alpha_safe * v_max**2 * dt**2)), 0.0, 1.0)
    gamma = jnp.clip(4 * ((y_min_right * wheels_distance) / (alpha_safe * v_max**2 * dt**2)), 0.0, 1.0)

    scaled_v_max = v_max * alpha
    scaled_w_pos = w_max * beta
    scaled_w_neg = w_max * gamma

    # --- B. Vincoli Dinamici (Sopravvivenza: Torricelli) ---
    # Torricelli Lineare
    v_safe = jnp.sqrt(2 * a_max * jnp.maximum(0.0, x_min))
    
    # Torricelli Rotazionale (basato sulle distanze laterali y)
    alpha_max_ang = (2 * a_max) / wheels_distance
    w_safe_pos = jnp.sqrt(2 * alpha_max_ang * (jnp.maximum(0.0, y_min_left) / r_robot))
    w_safe_neg = jnp.sqrt(2 * alpha_max_ang * (jnp.maximum(0.0, y_min_right) / r_robot))

    # Clipping assoluto al limite hardware di omega (1.9)
    w_pos_limit = jnp.minimum(w_safe_pos, w_limit_val)
    w_neg_limit = jnp.minimum(w_safe_neg, w_limit_val)

    # 4. DEFINIZIONE DELLE LAME (6 Semipiani: a*v + b*w + c <= 0)
    A = jnp.array([
        -1.0,                             # [0] v >= 0
        1.0,                              # [1] v <= v_safe (Gabbia Dinamica Frontale)
        0.0,                              # [2] w <= w_pos_limit (Gabbia Dinamica Sx)
        0.0,                              # [3] w >= -w_neg_limit (Gabbia Dinamica Dx)
        scaled_w_pos,                     # [4] Diagonale Heuristica Alpha/Beta
        scaled_w_neg                      # [5] Diagonale Heuristica Alpha/Gamma
    ])

    B = jnp.array([
        0.0,                              # [0]
        0.0,                              # [1]
        1.0,                              # [2]
        -1.0,                             # [3]
        scaled_v_max,                     # [4]
        -scaled_v_max                     # [5]
    ])

    C = jnp.array([
        0.0,                              # [0]
        -v_safe,                          # [1]
        -w_pos_limit,                     # [2]
        -w_neg_limit,                     # [3]
        -scaled_v_max * scaled_w_pos,     # [4]
        -scaled_v_max * scaled_w_neg      # [5]
    ])
    
    planes = jnp.stack([A, B, C], axis=1)

    # 5. JITTABLE SUTHERLAND-HODGMAN POLYGON CLIPPING
    # Prende il rombo dinamico e lo affetta contro i 6 semipiani
    def clip_against_plane(carry, plane):
        verts, num_verts = carry
        a, b, c = plane

        indices = jnp.arange(MAX_VERTICES)
        next_indices = (indices + 1) % jnp.maximum(num_verts, 1)

        p1 = verts
        p2 = verts[next_indices]

        d1 = a * p1[:, 0] + b * p1[:, 1] + c
        d2 = a * p2[:, 0] + b * p2[:, 1] + c

        in1 = d1 <= 1e-5
        in2 = d2 <= 1e-5

        denominator = d1 - d2
        t = jnp.where(jnp.abs(denominator) > 1e-7, d1 / denominator, 0.0)
        p_int = p1 + t[:, None] * (p2 - p1)

        out = jnp.zeros((MAX_VERTICES, 2, 2))
        counts = jnp.zeros(MAX_VERTICES, dtype=jnp.int32)

        mask1 = in1 & in2
        out = jnp.where(mask1[:, None, None], jnp.stack([p2, jnp.zeros_like(p2)], axis=1), out)
        counts = jnp.where(mask1, 1, counts)

        mask2 = in1 & ~in2
        out = jnp.where(mask2[:, None, None], jnp.stack([p_int, jnp.zeros_like(p_int)], axis=1), out)
        counts = jnp.where(mask2, 1, counts)

        mask4 = ~in1 & in2
        out = jnp.where(mask4[:, None, None], jnp.stack([p_int, p2], axis=1), out)
        counts = jnp.where(mask4, 2, counts)

        valid_edges = indices < num_verts
        counts = jnp.where(valid_edges, counts, 0)

        flat_out = out.reshape(-1, 2)
        j_idx = jnp.tile(jnp.arange(2), MAX_VERTICES)
        rep_counts = jnp.repeat(counts, 2)
        valid_mask = j_idx < rep_counts

        target_indices = jnp.cumsum(valid_mask) - 1
        new_count = jnp.sum(valid_mask)

        def get_vertex(i):
            match = valid_mask & (target_indices == i)
            idx = jnp.argmax(match)
            val = jnp.where(i < new_count, flat_out[idx], jnp.array([jnp.nan, jnp.nan]))
            return val

        new_verts = jax.vmap(get_vertex)(jnp.arange(MAX_VERTICES))
        return (new_verts, new_count), None

    final_carry, _ = jax.lax.scan(clip_against_plane, (vertices, num_vertices), planes)
    final_vertices, _ = final_carry
    
    return final_vertices

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # --- 1. Define Parameters ---
    v = 0.15
    w = 0.8
    d_front = 0.1
    d_left = 0.01
    d_right = 1.0
    a_max = 0.88
    dt = 0.05

    # Robot Physical Constants (must match the ones in the jax function)
    v_max = 0.45
    wheels_distance = 0.235
    w_max = (2 * v_max) / wheels_distance

    # --- 2. Execute JAX Function ---
    # The first execution will trigger the JIT compilation
    vertices = get_feasible_action_space(v, w, d_front, d_left, d_right, a_max, dt)
    print("Feasible Action Space Vertices (JAX Output):")
    print(np.array2string(np.array(vertices), formatter={'float_kind': lambda x: f'{x:.2f}'}))
    
    # --- 3. Process JAX Output for Plotting ---
    np_verts = np.array(vertices)
    # Remove NaN rows
    valid_verts = np_verts[~np.isnan(np_verts).any(axis=1)]

    # Sort vertices counter-clockwise so matplotlib fills it correctly
    if len(valid_verts) > 0:
        centroid = valid_verts.mean(axis=0)
        angles = np.arctan2(valid_verts[:, 1] - centroid[1], valid_verts[:, 0] - centroid[0])
        valid_verts = valid_verts[np.argsort(angles)]

    # --- 4. Reconstruct the Starting Heuristic Triangle ---
    alpha = np.clip(d_front / (v_max * dt), 0.0, 1.0)
    alpha_safe = max(alpha, 1e-6)
    beta = np.clip(4 * ((d_left * wheels_distance) / (alpha_safe * v_max**2 * dt**2)), 0.0, 1.0)
    gamma = np.clip(4 * ((d_right * wheels_distance) / (alpha_safe * v_max**2 * dt**2)), 0.0, 1.0)
    
    scaled_v = v_max * alpha
    scaled_w_pos = w_max * beta
    scaled_w_neg = w_max * gamma
    
    triangle_verts = np.array([
        [0.0, scaled_w_pos],
        [scaled_v, 0.0],
        [0.0, -scaled_w_neg]
    ])

    # --- 5. Plotting ---
    fig, ax = plt.subplots(figsize=(9, 9))
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-0.2, v_max + 0.2)
    ax.set_ylim(-w_max - 1, w_max + 1)
    ax.set_xlabel('Linear Velocity (v) [m/s]')
    ax.set_ylabel('Angular Velocity (omega) [rad/s]')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot A: Heuristic Triangle (Starting Constraints)
    tri_patch = Polygon(triangle_verts, closed=True, fill=True, color='darkgreen', alpha=0.15, label='Heuristic Triangle')
    ax.add_patch(tri_patch)
    tri_plot = np.vstack([triangle_verts, triangle_verts[0]])
    ax.plot(tri_plot[:, 0], tri_plot[:, 1], 'g--', linewidth=2)

    # Plot B: Dynamic Rhombus (Acceleration Constraints)
    dv = a_max * dt
    dw = (2 * a_max * dt) / wheels_distance
    rhombus_verts = np.array([
        [v + dv, w],
        [v, w + dw],
        [v - dv, w],
        [v, w - dw]
    ])
    rhombus_patch = Polygon(rhombus_verts, closed=True, fill=False, edgecolor='blue', linestyle='-.', linewidth=1.5, label='Dynamic Rhombus')
    ax.add_patch(rhombus_patch)

    # Plot C: Feasible Action Space (Final JAX Output)
    if len(valid_verts) > 0:
        feasible_patch = Polygon(valid_verts, closed=True, fill=True, color='lime', alpha=0.7, label='Feasible Space (JAX)')
        ax.add_patch(feasible_patch)
        valid_plot = np.vstack([valid_verts, valid_verts[0]])
        ax.plot(valid_plot[:, 0], valid_plot[:, 1], 'k-', linewidth=2)

    # Plot D: Current State
    ax.plot(v, w, 'ro', markersize=8, label='Current State')

    ax.legend(loc='upper right')
    ax.set_title("Feasible Action Space Verification (JAX Output)")
    plt.show()