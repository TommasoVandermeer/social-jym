"""Extract and render manually initialised pedestrians from recorded LiDAR scans.

Only tracks explicitly selected by the user are maintained. This prevents map
artifacts and unrelated moving objects from creating false human tracks.
"""

import argparse
import math
import os
import pickle
import sys
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


MAX_SCAN_GAP = 0.5
HUMAN_EXCLUSION_RADIUS = 0.65
PEDESTRIAN_MAX_SPEED_M_S = 2.5
PEDESTRIAN_MAX_ACCEL_M_S2 = 3.0


class Track:
    """Constant-velocity Kalman track with history for RTS smoothing."""

    def __init__(self, track_id, initial_pos, timestamp):
        self.track_id = track_id
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([0.12, 0.12, 1.0, 1.0])
        self.missed_frames = 0
        self.timestamps = [timestamp]
        self.filtered_x = [self.x.copy()]
        self.filtered_P = [self.P.copy()]
        # Entry k is the prediction/transition from filtered frame k-1 to k.
        self.predicted_x = [None]
        self.predicted_P = [None]
        self.transitions = [None]
        self.measurement_used = [True]
        self.manual_correction_used = [True]
        self.smoothed_states = []
        self.last_manual_position = self.x[:2].copy()
        self.last_manual_timestamp = timestamp
        self.last_prediction_dt = None

    def predict(self, dt):
        dt = max(float(dt), 1e-3)
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        sigma_a = 1.2
        Q = sigma_a**2 * np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_prediction_dt = dt
        return F, self.x.copy(), self.P.copy()

    def innovation_cost(self, measurement):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.diag([0.06**2, 0.06**2])
        residual = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        return (
            float(residual @ np.linalg.solve(S, residual)),
            float(np.linalg.norm(residual)),
        )

    def update(self, measurement):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.diag([0.06**2, 0.06**2])
        residual = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        predicted_velocity = self.x[2:].copy()
        self.x = self.x + K @ residual
        # Joseph form keeps the covariance symmetric and positive semi-definite.
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        if self.last_prediction_dt is not None:
            velocity_change = self.x[2:] - predicted_velocity
            change = np.linalg.norm(velocity_change)
            maximum_change = PEDESTRIAN_MAX_ACCEL_M_S2 * self.last_prediction_dt
            if change > maximum_change:
                self.x[2:] = predicted_velocity + velocity_change * maximum_change / change
        speed = np.linalg.norm(self.x[2:])
        if speed > PEDESTRIAN_MAX_SPEED_M_S:
            self.x[2:] *= PEDESTRIAN_MAX_SPEED_M_S / speed
        self.missed_frames = 0

    def apply_manual_correction(self, position, timestamp):
        """Re-anchor this identity without carrying a bad association forward."""
        position = np.asarray(position, dtype=float)
        elapsed = float(timestamp - self.last_manual_timestamp)
        if elapsed > 1e-3:
            velocity = (position - self.last_manual_position) / elapsed
            speed = np.linalg.norm(velocity)
            if speed > PEDESTRIAN_MAX_SPEED_M_S:
                velocity *= PEDESTRIAN_MAX_SPEED_M_S / speed
            self.x[2:] = velocity
        else:
            self.x[2:] = 0.0
        self.x[:2] = position
        self.P = np.diag([0.015**2, 0.015**2, 0.35**2, 0.35**2])
        self.missed_frames = 0
        self.last_manual_position = position.copy()
        self.last_manual_timestamp = timestamp

    def commit_frame(
        self, timestamp, transition, prediction, used_measurement, manual_correction=False
    ):
        self.timestamps.append(timestamp)
        self.filtered_x.append(self.x.copy())
        self.filtered_P.append(self.P.copy())
        self.transitions.append(transition)
        self.predicted_x.append(prediction[0])
        self.predicted_P.append(prediction[1])
        self.measurement_used.append(used_measurement)
        self.manual_correction_used.append(manual_correction)

    def apply_rts_smoother(self):
        self.smoothed_states = [state.copy() for state in self.filtered_x]
        smoothed_covariances = [cov.copy() for cov in self.filtered_P]
        for k in range(len(self.filtered_x) - 2, -1, -1):
            F = self.transitions[k + 1]
            P_pred = self.predicted_P[k + 1]
            gain = np.linalg.solve(P_pred, F @ self.filtered_P[k]).T
            self.smoothed_states[k] = self.filtered_x[k] + gain @ (
                self.smoothed_states[k + 1] - self.predicted_x[k + 1]
            )
            smoothed_covariances[k] = self.filtered_P[k] + gain @ (
                smoothed_covariances[k + 1] - P_pred
            ) @ gain.T
        # RTS is unconstrained and can slightly overshoot the physical limits
        # even when every forward Kalman update obeys them. Project only the
        # velocity components back onto the pedestrian motion envelope.
        for state in self.smoothed_states:
            speed = np.linalg.norm(state[2:])
            if speed > PEDESTRIAN_MAX_SPEED_M_S:
                state[2:] *= PEDESTRIAN_MAX_SPEED_M_S / speed
        for index in range(1, len(self.smoothed_states)):
            dt = max(float(self.timestamps[index] - self.timestamps[index - 1]), 1e-3)
            previous_velocity = self.smoothed_states[index - 1][2:]
            velocity_change = self.smoothed_states[index][2:] - previous_velocity
            change = np.linalg.norm(velocity_change)
            maximum_change = PEDESTRIAN_MAX_ACCEL_M_S2 * dt
            if change > maximum_change:
                self.smoothed_states[index][2:] = (
                    previous_velocity + velocity_change * maximum_change / change
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Manually initialise and track humans in recorded LiDAR scans."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="lists_jessi_recorded_obs.pkl",
        help="pickle recording (relative paths are resolved beside this script)",
    )
    parser.add_argument(
        "--initial-human",
        metavar=("X", "Y"),
        type=float,
        nargs=2,
        action="append",
        default=[],
        help="seed one human in global metres; repeat for multiple humans",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="do not open selection/animation windows (useful for automated tests)",
    )
    parser.add_argument("--save", metavar="PATH", help="save the animation (for example out.gif)")
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="render online Kalman estimates instead of RTS-smoothed tracks",
    )
    parser.add_argument(
        "--correction-interval",
        metavar="N",
        type=int,
        help="pause every N displayed frames for optional manual track corrections",
    )
    return parser.parse_args(argv)


def get_yaw_from_q(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    U, _, Vt = np.linalg.svd((A - centroid_A).T @ (B - centroid_B))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    return R, centroid_B - R @ centroid_A


def icp_2d(src_points, tgt_points, max_iterations=30, distance_threshold=0.4):
    if len(src_points) < 10 or len(tgt_points) < 10:
        return np.eye(2), np.zeros(2)
    src = np.copy(src_points)
    tgt_tree = KDTree(tgt_points)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    for _ in range(max_iterations):
        distances, indices = tgt_tree.query(src)
        valid = distances < distance_threshold
        if np.sum(valid) < 10:
            break
        R, t = best_fit_transform(src[valid], tgt_points[indices[valid]])
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        if np.linalg.norm(t) < 1e-4 and abs(math.atan2(R[1, 0], R[0, 0])) < 1e-4:
            break
    return R_total, t_total


def get_dynamic_points(current_scan, map_points, threshold=0.25):
    if len(map_points) == 0:
        return current_scan, np.empty((0, 2))
    dists, _ = KDTree(map_points).query(current_scan)
    dynamic_mask = dists > threshold
    return current_scan[dynamic_mask], current_scan[~dynamic_mask]


def scan_to_local_points(scan):
    ranges = np.asarray(scan.ranges, dtype=float)
    valid = (
        np.isfinite(ranges)
        & (ranges >= max(0.15, float(scan.range_min)))
        & (ranges < 9.9)
    )
    # Preserve the recording's original laser-to-plot convention.
    angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges)) + np.pi / 2
    return np.column_stack(
        (ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid]))
    )


def transform_points(points, x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return (np.array([[c, -s], [s, c]]) @ points.T).T + np.array([x, y])


def longest_continuous_segment(scans, max_gap=MAX_SCAN_GAP):
    if not scans:
        return 0, 0
    times = np.array(
        [s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for s in scans]
    )
    boundaries = np.flatnonzero(np.diff(times) > max_gap) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(scans)]
    selected = int(np.argmax(ends - starts))
    return int(starts[selected]), int(ends[selected])


def exclude_near_tracks(points, tracks, radius=HUMAN_EXCLUSION_RADIUS):
    if len(points) == 0 or not tracks:
        return points
    keep = np.ones(len(points), dtype=bool)
    for track in tracks:
        keep &= np.linalg.norm(points - track.x[:2], axis=1) > radius
    return points[keep]


def _cluster_candidates(points, dynamic, eps=0.28, min_samples=4):
    if len(points) < min_samples:
        return []
    # DBSCAN's connected-core behavior for this small 2-D point cloud, without
    # requiring scikit-learn (the project Docker image only provides SciPy).
    tree = KDTree(points)
    neighbours = tree.query_ball_point(points, eps)
    core = np.array([len(indices) >= min_samples for indices in neighbours])
    labels = np.full(len(points), -1, dtype=int)
    next_label = 0
    for seed in np.flatnonzero(core):
        if labels[seed] != -1:
            continue
        labels[seed] = next_label
        stack = [int(seed)]
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                if labels[neighbour] == -1:
                    labels[neighbour] = next_label
                    if core[neighbour]:
                        stack.append(neighbour)
        next_label += 1
    candidates = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        extent = np.ptp(cluster, axis=0)
        if (
            len(cluster) <= 80
            and np.max(extent) <= 0.9
            and np.linalg.norm(extent) <= 1.1
        ):
            candidates.append(
                {
                    "position": np.median(cluster, axis=0),
                    "dynamic": dynamic,
                    "points": len(cluster),
                }
            )
    return candidates


def find_human_candidates(global_points, dynamic_points):
    # Full-scan candidates make stationary people trackable; dynamic candidates
    # receive priority during association.
    candidates = _cluster_candidates(dynamic_points, dynamic=True)
    candidates.extend(_cluster_candidates(global_points, dynamic=False))
    deduplicated = []
    for candidate in sorted(candidates, key=lambda c: (not c["dynamic"], -c["points"])):
        if all(
            np.linalg.norm(candidate["position"] - other["position"]) > 0.22
            for other in deduplicated
        ):
            deduplicated.append(candidate)
    return deduplicated


def show_fullscreen(fig, block=True):
    """Show a Matplotlib figure fullscreen when supported by its GUI backend."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.full_screen_toggle()
        except (AttributeError, NotImplementedError):
            # Some uncommon backends only expose a native maximise operation.
            window = getattr(manager, "window", None)
            if window is not None and hasattr(window, "showMaximized"):
                window.showMaximized()
    plt.show(block=block)


def select_initial_humans(global_points, robot_pose):
    """Open the first-frame GUI and return selected global coordinates."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(global_points[:, 0], global_points[:, 1], c="black", s=5, alpha=0.55)
    rx, ry, rth = robot_pose
    ax.plot(rx, ry, "bo", markersize=9)
    ax.plot([rx, rx + np.cos(rth)], [ry, ry + np.sin(rth)], "b-", linewidth=2)
    selected, labels = [], []
    markers = ax.scatter([], [], c="red", s=120, edgecolors="black", zorder=5)

    def redraw():
        nonlocal labels
        markers.set_offsets(np.asarray(selected) if selected else np.empty((0, 2)))
        for label in labels:
            label.remove()
        labels = [
            ax.text(p[0], p[1], f"  {i + 1}", color="red", weight="bold")
            for i, p in enumerate(selected)
        ]
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        clicked = np.array([event.xdata, event.ydata])
        if event.button == 1:
            selected.append(clicked)
        elif event.button == 3 and selected:
            nearest = int(np.argmin([np.linalg.norm(p - clicked) for p in selected]))
            selected.pop(nearest)
        redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("backspace", "delete") and selected:
            selected.pop()
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Global X [m]")
    ax.set_ylabel("Global Y [m]")
    ax.set_title(
        "Select humans: left-click add, right-click/Delete undo, Enter start"
    )
    ax.margins(0.08)
    plt.tight_layout()
    show_fullscreen(fig, block=True)
    return [np.asarray(point, dtype=float) for point in selected]


def select_checkpoint_corrections(global_points, robot_pose, tracks, frame_number):
    """Collect optional identity-preserving corrections at a periodic checkpoint."""
    if not tracks:
        return {}
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(global_points[:, 0], global_points[:, 1], c="black", s=5, alpha=0.55)
    rx, ry, rth = robot_pose
    ax.plot(rx, ry, "bo", markersize=9, label="Robot")
    ax.plot([rx, rx + np.cos(rth)], [ry, ry + np.sin(rth)], "b-", linewidth=2)

    predicted = np.array([track.x[:2] for track in tracks])
    ax.scatter(
        predicted[:, 0],
        predicted[:, 1],
        c="orange",
        marker="x",
        s=140,
        linewidths=3,
        label="Current prediction",
        zorder=5,
    )
    predicted_labels = [
        ax.text(
            point[0],
            point[1],
            f"  #{track.track_id}",
            color="darkorange",
            weight="bold",
        )
        for track, point in zip(tracks, predicted)
    ]
    corrections = {}
    active_index = 0
    correction_markers = ax.scatter(
        [], [], c="red", s=120, edgecolors="black", label="Manual correction", zorder=6
    )
    correction_labels = []

    def redraw():
        nonlocal correction_labels
        points = np.array(list(corrections.values())) if corrections else np.empty((0, 2))
        correction_markers.set_offsets(points)
        for label in correction_labels:
            label.remove()
        correction_labels = [
            ax.text(
                position[0],
                position[1],
                f"  #{track_id}",
                color="red",
                weight="bold",
            )
            for track_id, position in corrections.items()
        ]
        active_id = tracks[active_index].track_id
        ax.set_title(
            f"Checkpoint frame {frame_number} — active track #{active_id}\n"
            "Left-click its real position; number/Tab selects a track; Enter continues"
        )
        for index, label in enumerate(predicted_labels):
            label.set_color("orangered" if index == active_index else "darkorange")
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal active_index
        if event.inaxes is not ax or event.xdata is None or event.button != 1:
            return
        track_id = tracks[active_index].track_id
        corrections[track_id] = np.array([event.xdata, event.ydata])
        if len(tracks) > 1:
            active_index = (active_index + 1) % len(tracks)
        redraw()

    def on_key(event):
        nonlocal active_index
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key == "tab":
            active_index = (active_index + 1) % len(tracks)
            redraw()
        elif event.key in ("backspace", "delete"):
            corrections.pop(tracks[active_index].track_id, None)
            redraw()
        elif event.key and event.key.isdigit():
            requested_id = int(event.key)
            for index, track in enumerate(tracks):
                if track.track_id == requested_id:
                    active_index = index
                    redraw()
                    break

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Global X [m]")
    ax.set_ylabel("Global Y [m]")
    ax.legend(loc="upper right")
    ax.margins(0.08)
    redraw()
    plt.tight_layout()
    show_fullscreen(fig, block=True)
    return corrections


def associate_and_update(tracks, candidates, locked_track_indices=None):
    locked_track_indices = set(locked_track_indices or ())
    if not candidates:
        for index, track in enumerate(tracks):
            if index not in locked_track_indices:
                track.missed_frames += 1
        return locked_track_indices
    large_cost = 1e6
    costs = np.full((len(tracks), len(candidates)), large_cost)
    for row, track in enumerate(tracks):
        if row in locked_track_indices:
            continue
        distance_gate = min(1.8, 0.8 + 0.08 * track.missed_frames)
        for col, candidate in enumerate(candidates):
            mahalanobis_sq, distance = track.innovation_cost(candidate["position"])
            if distance <= distance_gate and mahalanobis_sq <= 16.0:
                costs[row, col] = mahalanobis_sq + (0.0 if candidate["dynamic"] else 1.5)
    rows, cols = linear_sum_assignment(costs)
    updated = set(locked_track_indices)
    for row, col in zip(rows, cols):
        if costs[row, col] < large_cost:
            tracks[row].update(candidates[col]["position"])
            updated.add(row)
    for row, track in enumerate(tracks):
        if row not in updated:
            track.missed_frames += 1
    return updated


def load_recording(path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    try:
        with open(path, "rb") as stream:
            return pickle.load(stream), path
    except FileNotFoundError:
        sys.exit(f"Recording not found: {path}")


def process_recording(
    data,
    initial_positions=None,
    use_gui=True,
    smooth=True,
    correction_interval=None,
    checkpoint_callback=None,
):
    scans, odoms = data["scan"], data["odom"]
    if not scans or not odoms:
        raise ValueError("The recording must contain non-empty 'scan' and 'odom' lists")
    segment_start, segment_end = longest_continuous_segment(scans)
    scans = scans[segment_start:segment_end]
    print(f"Using scan segment [{segment_start}:{segment_end}] ({len(scans)} frames).")

    odom_t = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in odoms])
    odom_x = np.array([m.pose.pose.position.x for m in odoms])
    odom_y = np.array([m.pose.pose.position.y for m in odoms])
    odom_th = np.unwrap(np.array([get_yaw_from_q(m.pose.pose.orientation) for m in odoms]))

    def pose(timestamp):
        theta = np.interp(timestamp, odom_t, odom_th)
        return (
            np.interp(timestamp, odom_t, odom_x),
            np.interp(timestamp, odom_t, odom_y),
            math.atan2(math.sin(theta), math.cos(theta)),
        )

    first_scan = scans[0]
    first_t = first_scan.header.stamp.sec + first_scan.header.stamp.nanosec * 1e-9
    laser_rx, laser_ry, laser_rth = pose(first_t)
    prev_odom_x, prev_odom_y, prev_odom_th = laser_rx, laser_ry, laser_rth
    first_points = transform_points(scan_to_local_points(first_scan), laser_rx, laser_ry, laser_rth)

    seeds = [np.asarray(p, dtype=float) for p in (initial_positions or [])]
    if use_gui:
        seeds.extend(select_initial_humans(first_points, (laser_rx, laser_ry, laser_rth)))
    tracks = [Track(i + 1, point, first_t) for i, point in enumerate(seeds)]
    print(f"Initialised {len(tracks)} human track(s).")

    local_map_buffer = deque(maxlen=20)
    initial_static = exclude_near_tracks(first_points, tracks)
    if len(initial_static):
        local_map_buffer.append(initial_static[::4])
    render_frames = [{"t": first_t, "robot_pose": (laser_rx, laser_ry, laser_rth), "global_scan": first_points}]
    previous_t = first_t

    for frame_number, scan in enumerate(scans[1:], start=2):
        t_scan = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        rx, ry, r_th = pose(t_scan)
        dt, previous_t = t_scan - previous_t, t_scan
        local_points = scan_to_local_points(scan)

        dx_global, dy_global = rx - prev_odom_x, ry - prev_odom_y
        dth = math.atan2(math.sin(r_th - prev_odom_th), math.cos(r_th - prev_odom_th))
        c_prev, s_prev = np.cos(-prev_odom_th), np.sin(-prev_odom_th)
        local_dx, local_dy = np.array([[c_prev, -s_prev], [s_prev, c_prev]]) @ np.array([dx_global, dy_global])
        prev_odom_x, prev_odom_y, prev_odom_th = rx, ry, r_th
        c_las, s_las = np.cos(laser_rth), np.sin(laser_rth)
        guess_dx, guess_dy = np.array([[c_las, -s_las], [s_las, c_las]]) @ np.array([local_dx, local_dy])
        guess_x, guess_y, guess_th = laser_rx + guess_dx, laser_ry + guess_dy, laser_rth + dth
        guessed_points = transform_points(local_points, guess_x, guess_y, guess_th)
        map_points = np.vstack(local_map_buffer) if local_map_buffer else np.empty((0, 2))
        R_icp, t_icp = icp_2d(guessed_points, map_points, 30, 0.5)
        global_points = (R_icp @ guessed_points.T).T + t_icp
        correction_theta = math.atan2(R_icp[1, 0], R_icp[0, 0])
        laser_rx, laser_ry = guess_x + t_icp[0], guess_y + t_icp[1]
        laser_rth = math.atan2(math.sin(guess_th + correction_theta), math.cos(guess_th + correction_theta))

        transitions, predictions = [], []
        for track in tracks:
            F, x_pred, P_pred = track.predict(dt)
            transitions.append(F)
            predictions.append((x_pred, P_pred))

        corrections = {}
        if correction_interval and frame_number % correction_interval == 0:
            callback = checkpoint_callback
            if callback is None and use_gui:
                callback = select_checkpoint_corrections
            if callback is not None:
                corrections = callback(
                    global_points,
                    (laser_rx, laser_ry, laser_rth),
                    tracks,
                    frame_number,
                )
                if corrections:
                    corrected_ids = ", ".join(f"#{track_id}" for track_id in corrections)
                    print(f"Checkpoint frame {frame_number}: corrected {corrected_ids}.")
                else:
                    print(f"Checkpoint frame {frame_number}: predictions accepted.")
        corrected_indices = set()
        for index, track in enumerate(tracks):
            if track.track_id in corrections:
                track.apply_manual_correction(corrections[track.track_id], t_scan)
                corrected_indices.add(index)

        dynamic_points, static_points = get_dynamic_points(global_points, map_points)
        updated = associate_and_update(
            tracks,
            find_human_candidates(global_points, dynamic_points),
            locked_track_indices=corrected_indices,
        )
        for i, track in enumerate(tracks):
            track.commit_frame(
                t_scan,
                transitions[i],
                predictions[i],
                i in updated,
                manual_correction=i in corrected_indices,
            )

        safe_static = exclude_near_tracks(static_points, tracks)
        if len(safe_static):
            local_map_buffer.append(safe_static[::4])
        render_frames.append({"t": t_scan, "robot_pose": (laser_rx, laser_ry, laser_rth), "global_scan": global_points})

    for track in tracks:
        if smooth:
            track.apply_rts_smoother()
        else:
            track.smoothed_states = [state.copy() for state in track.filtered_x]
    for frame_index, frame in enumerate(render_frames):
        frame["humans"] = [track.smoothed_states[frame_index] for track in tracks]
        frame["measured"] = [track.measurement_used[frame_index] for track in tracks]
    return render_frames, tracks


def render_animation(render_frames, tracks, show=True, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    (scan_plot,) = ax.plot([], [], "k.", markersize=2, alpha=0.3, label="LiDAR")
    (robot_plot,) = ax.plot([], [], "bo", markersize=8, label="Robot")
    (robot_heading,) = ax.plot([], [], "b-", linewidth=2)
    humans_scatter = ax.scatter([], [], c="red", s=100, edgecolors="black", label="Seeded humans", zorder=5)
    initial_arrows = np.zeros((len(tracks), 2))
    velocity_quiver = ax.quiver(
        initial_arrows[:, 0],
        initial_arrows[:, 1],
        initial_arrows[:, 0],
        initial_arrows[:, 1],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        label="Velocity (m/s; one-second displacement)",
        zorder=6,
    )
    labels = [ax.text(0, 0, str(t.track_id), color="darkred", weight="bold", zorder=7) for t in tracks]
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        humans_scatter.set_offsets(np.empty((0, 2)))
        return scan_plot, robot_plot, robot_heading, humans_scatter, velocity_quiver

    def update(frame_idx):
        frame = render_frames[frame_idx]
        rx, ry, r_th = frame["robot_pose"]
        ax.set_xlim(rx - 8, rx + 8)
        ax.set_ylim(ry - 8, ry + 8)
        points = frame["global_scan"]
        scan_plot.set_data(points[:, 0], points[:, 1])
        robot_plot.set_data([rx], [ry])
        robot_heading.set_data([rx, rx + np.cos(r_th)], [ry, ry + np.sin(r_th)])
        humans = np.asarray(frame["humans"])
        if len(humans):
            humans_scatter.set_offsets(humans[:, :2])
            velocity_quiver.set_offsets(humans[:, :2])
            velocity_quiver.set_UVC(humans[:, 2], humans[:, 3])
            for label, track, state in zip(labels, tracks, humans):
                label.set_position((state[0] + 0.08, state[1] + 0.08))
                label.set_text(
                    f"#{track.track_id}  {np.linalg.norm(state[2:]):.2f} m/s"
                )
        else:
            humans_scatter.set_offsets(np.empty((0, 2)))
            velocity_quiver.set_offsets(np.empty((0, 2)))
            velocity_quiver.set_UVC(np.array([]), np.array([]))
        ax.set_title(f"Seeded ICP pedestrian tracking — frame {frame_idx + 1}/{len(render_frames)}")
        return scan_plot, robot_plot, robot_heading, humans_scatter, velocity_quiver

    ani = animation.FuncAnimation(fig, update, frames=len(render_frames), init_func=init, blit=False, interval=50, repeat=True)
    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer="pillow" if save_path.lower().endswith(".gif") else None, fps=20)
        print(f"Saved animation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


def main(argv=None):
    args = parse_args(argv)
    if args.correction_interval is not None and args.correction_interval <= 0:
        raise SystemExit("--correction-interval must be a positive integer")
    if args.no_gui and args.correction_interval:
        raise SystemExit("--correction-interval requires the GUI; remove --no-gui")
    if args.no_gui and not args.initial_human:
        print("Warning: --no-gui without --initial-human will track no people.")
    data, filename = load_recording(args.input)
    print(f"Loaded {filename}")
    render_frames, tracks = process_recording(
        data,
        initial_positions=args.initial_human,
        use_gui=not args.no_gui,
        smooth=not args.no_smoothing,
        correction_interval=args.correction_interval,
    )
    if tracks:
        print(
            "Track measurement coverage: "
            + ", ".join(
                f"#{track.track_id}={sum(track.measurement_used)}/{len(render_frames)}"
                for track in tracks
            )
        )
        if args.correction_interval:
            print(
                "Manual checkpoint corrections: "
                + ", ".join(
                    f"#{track.track_id}={sum(track.manual_correction_used) - 1}"
                    for track in tracks
                )
            )
    if not args.no_gui or args.save:
        render_animation(render_frames, tracks, show=not args.no_gui, save_path=args.save)


if __name__ == "__main__":
    main()
