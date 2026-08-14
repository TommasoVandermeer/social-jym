import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from collections import deque
import math
import sys
import os

class Track:
    def __init__(self, track_id, initial_pos, timestamp):
        self.track_id = track_id
        self.active = True
        self.missed_frames = 0
        self.timestamps = [timestamp]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        self.P = np.eye(4) * 0.5 
        self.history_x_pred = []
        self.history_P_pred = []
        self.history_x_upd = [self.x.copy()]
        self.history_P_upd = [self.P.copy()]
        self.history_t = [timestamp]
        self.smoothed_states = []
    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.diag([0.01, 0.01, 0.1, 0.1]) * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.history_x_pred.append(self.x.copy())
        self.history_P_pred.append(self.P.copy())
    def update(self, measurement, timestamp):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.diag([0.02, 0.02])
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.history_x_upd.append(self.x.copy())
        self.history_P_upd.append(self.P.copy())
        self.history_t.append(timestamp)
        self.timestamps.append(timestamp)
        self.missed_frames = 0
    def apply_rts_smoother(self):
        if len(self.history_x_upd) < 2:
            self.smoothed_states = self.history_x_upd
            return
        x_smooth = [np.zeros(4) for _ in range(len(self.history_x_upd))]
        P_smooth = [np.zeros((4,4)) for _ in range(len(self.history_P_upd))]
        x_smooth[-1] = self.history_x_upd[-1]
        P_smooth[-1] = self.history_P_upd[-1]
        for k in range(len(self.history_x_upd) - 2, -1, -1):
            dt = self.history_t[k+1] - self.history_t[k]
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            P_upd = self.history_P_upd[k]
            P_pred = self.history_P_pred[k]
            C_k = P_upd @ F.T @ np.linalg.inv(P_pred)
            x_smooth[k] = self.history_x_upd[k] + C_k @ (x_smooth[k+1] - self.history_x_pred[k])
            P_smooth[k] = P_upd + C_k @ (P_smooth[k+1] - P_pred) @ C_k.T
        self.smoothed_states = x_smooth

def get_yaw_from_q(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t

def icp_2d(src_points, tgt_points, max_iterations=30, distance_threshold=0.4):
    src = np.copy(src_points)
    tgt_tree = KDTree(tgt_points)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    for i in range(max_iterations):
        distances, indices = tgt_tree.query(src)
        valid = distances < distance_threshold
        if np.sum(valid) < 10: break
        src_valid = src[valid]
        tgt_valid = tgt_points[indices[valid]]
        R, t = best_fit_transform(src_valid, tgt_valid)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        if np.linalg.norm(t) < 1e-4 and np.abs(np.arccos(np.clip((np.trace(R)-1)/1, -1.0, 1.0))) < 1e-4: break
    return R_total, t_total

def get_dynamic_points(current_scan, map_points, threshold=0.3):
    if len(map_points) == 0: return current_scan, np.empty((0, 2))
    tree = KDTree(map_points)
    dists, _ = tree.query(current_scan)
    dyn_mask = dists > threshold
    return current_scan[dyn_mask], current_scan[~dyn_mask]
    
def main():
    filename = "lists_jessi_recorded_obs.pkl"
    print(f"Loading {filename}...")
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        sys.exit(1)
    scans = data['scan']
    odoms = data['odom']
    odom_t = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in odoms])
    odom_x = np.array([m.pose.pose.position.x for m in odoms])
    odom_y = np.array([m.pose.pose.position.y for m in odoms])
    odom_th = np.array([get_yaw_from_q(m.pose.pose.orientation) for m in odoms])
    odom_th_unwrapped = np.unwrap(odom_th)
    render_frames = []
    tracks = []
    next_track_id = 1
    completed_tracks = []
    previous_t = None
    laser_rx, laser_ry, laser_rth = None, None, None
    prev_odom_x, prev_odom_y, prev_odom_th = None, None, None
    local_map_buffer = deque(maxlen=20)
    for i, scan in enumerate(scans):
        t_scan = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        rx = np.interp(t_scan, odom_t, odom_x)
        ry = np.interp(t_scan, odom_t, odom_y)
        r_th = math.atan2(math.sin(np.interp(t_scan, odom_t, odom_th_unwrapped)), math.cos(np.interp(t_scan, odom_t, odom_th_unwrapped)))
        dt = t_scan - previous_t if previous_t is not None else 0.1
        if dt > 0.5:
            for trk in tracks:
                if trk.active:
                    trk.active = False
                    completed_tracks.append(trk)
            tracks = []
            local_map_buffer.clear()
            laser_rx, laser_ry, laser_rth = None, None, None
            previous_t = t_scan
            prev_odom_x, prev_odom_y, prev_odom_th = rx, ry, r_th
            continue
        previous_t = t_scan
        ranges = np.array(scan.ranges)
        ranges = np.nan_to_num(ranges, nan=30., posinf=30., neginf=30.)
        ranges[ranges < 0.15] = 10.0
        ranges = np.clip(ranges, 0.0, 10.0)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges)) + (np.pi / 2)
        valid_idx = (ranges < 9.9)
        local_x = ranges[valid_idx] * np.cos(angles[valid_idx])
        local_y = ranges[valid_idx] * np.sin(angles[valid_idx])
        local_points = np.column_stack((local_x, local_y))
        if laser_rx is None:
            laser_rx, laser_ry, laser_rth = rx, ry, r_th
            prev_odom_x, prev_odom_y, prev_odom_th = rx, ry, r_th
            c, s = np.cos(laser_rth), np.sin(laser_rth)
            global_points = (np.array([[c, -s], [s, c]]) @ local_points.T).T + np.array([laser_rx, laser_ry])
            local_map_buffer.append(global_points)
            continue
        dx_global = rx - prev_odom_x
        dy_global = ry - prev_odom_y
        dth = math.atan2(math.sin(r_th - prev_odom_th), math.cos(r_th - prev_odom_th))
        c_prev, s_prev = np.cos(-prev_odom_th), np.sin(-prev_odom_th)
        local_dx, local_dy = np.array([[c_prev, -s_prev], [s_prev, c_prev]]) @ np.array([dx_global, dy_global])
        prev_odom_x, prev_odom_y, prev_odom_th = rx, ry, r_th
        c_las, s_las = np.cos(laser_rth), np.sin(laser_rth)
        guess_dx, guess_dy = np.array([[c_las, -s_las], [s_las, c_las]]) @ np.array([local_dx, local_dy])
        guess_x, guess_y, guess_th = laser_rx + guess_dx, laser_ry + guess_dy, laser_rth + dth
        c, s = np.cos(guess_th), np.sin(guess_th)
        guessed_global_points = (np.array([[c, -s], [s, c]]) @ local_points.T).T + np.array([guess_x, guess_y])
        map_points = np.vstack(local_map_buffer)
        R_icp, t_icp = icp_2d(guessed_global_points, map_points, max_iterations=30, distance_threshold=0.5)
        global_points = (R_icp @ guessed_global_points.T).T + t_icp
        laser_rx, laser_ry = guess_x + t_icp[0], guess_y + t_icp[1]
        laser_rth = math.atan2(math.sin(guess_th + math.atan2(R_icp[1,0], R_icp[0,0])), math.cos(guess_th + math.atan2(R_icp[1,0], R_icp[0,0])))
        human_points_global, static_points = get_dynamic_points(global_points, map_points, threshold=0.25)
        if len(static_points) > 0: 
            local_map_buffer.append(static_points[::4])
        centroids = []
        if len(human_points_global) > 0:
            clustering = DBSCAN(eps=0.25, min_samples=8).fit(human_points_global)
            for label in set(clustering.labels_):
                if label != -1:
                    cluster_pts = human_points_global[clustering.labels_ == label]
                    width = np.max(cluster_pts[:, 0]) - np.min(cluster_pts[:, 0])
                    depth = np.max(cluster_pts[:, 1]) - np.min(cluster_pts[:, 1])
                    if width < 1.0 and depth < 1.0:
                        centroids.append(cluster_pts.mean(axis=0))
        for trk in tracks:
            if trk.active: trk.predict(dt)
        if len(centroids) > 0 and len(tracks) > 0:
            active_tracks = [t for t in tracks if t.active]
            cost_matrix = np.zeros((len(active_tracks), len(centroids)))
            for r, trk in enumerate(active_tracks):
                for c_idx, cent in enumerate(centroids):
                    dist = np.linalg.norm(trk.x[:2] - cent)
                    cost_matrix[r, c_idx] = dist if dist < 1.0 else 1e5
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_centroids = set()
            for r, c_idx in zip(row_ind, col_ind):
                if cost_matrix[r, c_idx] < 1e5:
                    active_tracks[r].update(centroids[c_idx], t_scan)
                    assigned_centroids.add(c_idx)
                else: active_tracks[r].missed_frames += 1
            for c_idx, cent in enumerate(centroids):
                if c_idx not in assigned_centroids:
                    tracks.append(Track(next_track_id, cent, t_scan))
                    next_track_id += 1
        elif len(centroids) > 0:
            for cent in centroids:
                tracks.append(Track(next_track_id, cent, t_scan))
                next_track_id += 1
        else:
            for trk in tracks:
                if trk.active: trk.missed_frames += 1
        for trk in tracks:
            if trk.active and trk.missed_frames > 15:
                trk.active = False
                completed_tracks.append(trk)
        render_frames.append({'t': t_scan, 'robot_pose': (laser_rx, laser_ry, laser_rth), 'global_scan': global_points})
    for trk in tracks:
        if trk.active: completed_tracks.append(trk)
    valid_tracks = []
    for trk in completed_tracks:
        if len(trk.history_x_upd) > 5:
            trk.apply_rts_smoother()
            valid_tracks.append(trk)
    for i, frame in enumerate(render_frames):
        frame_t = frame['t']
        frame['humans'] = []
        for trk in valid_tracks:
            if frame_t in trk.history_t:
                frame['humans'].append(trk.smoothed_states[trk.history_t.index(frame_t)])
    fig, ax = plt.subplots(figsize=(8, 8))
    scan_plot, = ax.plot([], [], 'k.', markersize=2, alpha=0.3, label='LiDAR')
    robot_plot, = ax.plot([], [], 'bo', markersize=8, label='Robot')
    robot_heading, = ax.plot([], [], 'b-', linewidth=2)
    MAX_HUMANS = 50
    humans_scatter = ax.scatter(np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), c='red', s=100, edgecolors='black', label='Humans', zorder=5)
    velocity_quiver = ax.quiver(np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), color='red', scale=5, width=0.005, zorder=6)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_title("ICP Laser Odometry & Dynamic Obstacle Tracking")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return scan_plot, robot_plot, robot_heading, humans_scatter, velocity_quiver
    def update(frame_idx):
        frame = render_frames[frame_idx]
        rx, ry, r_th = frame['robot_pose']
        window_size = 8.0
        ax.set_xlim(rx - window_size, rx + window_size)
        ax.set_ylim(ry - window_size, ry + window_size)
        scans_pts = frame['global_scan']
        scan_plot.set_data(scans_pts[:, 0], scans_pts[:, 1])
        robot_plot.set_data([rx], [ry])
        robot_heading.set_data([rx, rx + np.cos(r_th)], [ry, ry + np.sin(r_th)])
        humans_data = frame['humans']
        x_pad, y_pad, u_pad, v_pad = np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan), np.full(MAX_HUMANS, np.nan)
        if len(humans_data) > 0:
            h_pts = np.array(humans_data)
            n = min(len(h_pts), MAX_HUMANS)
            x_pad[:n], y_pad[:n], u_pad[:n], v_pad[:n] = h_pts[:n, 0], h_pts[:n, 1], h_pts[:n, 2], h_pts[:n, 3] 
        humans_scatter.set_offsets(np.column_stack((x_pad, y_pad)))
        velocity_quiver.set_offsets(np.column_stack((x_pad, y_pad)))
        velocity_quiver.set_UVC(u_pad, v_pad)
        return scan_plot, robot_plot, robot_heading, humans_scatter, velocity_quiver
    ani = animation.FuncAnimation(fig, update, frames=len(render_frames), init_func=init, blit=False, interval=20, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()