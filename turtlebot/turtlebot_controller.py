#!/usr/bin/env python3
import sys
import argparse
import copy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.srv import ResetPose
import numpy as np
import os
import pickle
from jax import random
from collections import deque
import math
import jax.numpy as jnp
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from pathlib import Path
from matplotlib import pyplot as plt

from socialjym.policies.jessi import JESSI
from socialjym.policies.dwa import DWA
from socialjym.policies.mppi import MPPI
from socialjym.policies.vanilla_e2e import VanillaE2E

PLANNERS = [
    'JESSI',
    'DWA',
    'MPPI',
    'VANILLA-E2E',
    'BOUNDED-VANILLA-E2E',
]

class TB4Controller(Node):
    def __init__(
            self, 
            frequency, 
            lidar_num_rays, 
            planner, 
            rc_goal_list, 
            patrol_mode, 
            interp_mode, 
            network_name, 
            save_file_name, 
            save_lists, 
            align, 
            diagnostics, 
            san_niccolo,
            engineering_filters,
            pure_pursuit,
            experiment_dir=None,
            timeout=None,
            goal_tolerance=None,
            stop_on_goal=False,
        ):
        super().__init__('TB4_controller')

        # Alignement method
        self.alignement = "ransac" # "hough" or "ransac"

        # San Niccolò Waypoints
        if san_niccolo:
            waypoints = jnp.array([
                [17.6, 22.3],       # 0
                [14.956, 22.618],   # 1
                [1.503, 22.633],    # 2
                [-4.893, 23.1],     # 2.5
                [-5.893, 22.991],   # 3
                [-6.493, 21.991],   # 3.5
                [-6.873, 19.245],   # 4
                [-6.873, 8.458],    # 5
                [-6.584, 0.597],    # 6
                [-6.6, -4.7],       # 6.33
                [-6.3, -5.8],       # 6.66
                [-5.107, -6.143],   # 7
                [4.02, -8.118],     # 8
                [11.00, -8.134],    # 9
                [17.511, -2.1],     # 10
                [17.593, 4.008],    # 11
                [17.593, 12.719],   # 12
                [17.617, 20.306],   # 13
                [17.6, 22.3],       # 14, start again
                [14.956, 22.618],   # 15
                [1.503, 22.633],    # 16
                [-5.893, 22.991],   # 17
                [-6.873, 19.245],   # 18
                [-6.873, 8.458],    # 19
                [-7.162, 3.347],    # 20
                [-6.584, 0.597],    # 21
                [-5.107, -6.143],   # 22
                [4.02, -8.118],     # 23
                [11.00, -8.134],    # 24
                [15.7, -6.7],       # 24.33
                [17.5, -5.0],       # 24.66
                [17.511, -2.1],     # 25
                [17.593, 4.008],    # 26
                [17.593, 12.719],   # 27
                [17.617, 20.306],   # 28
            ])
            initial_pose = jnp.array([17.62, 20.306, jnp.pi/2])
            inv_rotation_matrix = jnp.array([
                [jnp.cos(-initial_pose[3]), jnp.sin(-initial_pose[3])],
                [-jnp.sin(-initial_pose[3]), jnp.cos(-initial_pose[3])],
            ])
            waypoints = (waypoints - initial_pose[:2][None,:]) @ inv_rotation_matrix
            # plt.scatter(waypoints[:,0], waypoints[:,1], marker="*", color="red", zorder=2)
            # plt.show()

        self.frequency = frequency
        self.planner = planner
        self.diagnostics = diagnostics
        self.engineering_filters = engineering_filters
        self.experiment_dir = Path(experiment_dir).resolve() if experiment_dir else None
        self.timeout = timeout
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else 0.8
        self.stop_on_goal = stop_on_goal
        self.first_command_timestamp = None
        self.final_event = None
        self.finished = False
        if self.diagnostics:
            self.get_logger().info("📈 Initialize live diagnostics plot...")
            plt.ion() # NON blocking mode for live update
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.scan_plot, = self.ax.plot([], [], 'r.', markersize=2, alpha=0.5, label='Lidar Scan')
            self.goal_plot, = self.ax.plot([], [], 'g*', markersize=12, label='Current Goal')
            self.future_goals_plot, = self.ax.plot([], [], 'c*', markersize=8, alpha=0.5, label='Future Goals')
            self.ax.plot(0, 0, 'ks', markersize=8, label='Robot Center') 
            self.ax.plot([0, 3], [0, 0], 'b-', linewidth=1.5, label='X-Axis (Front)') 
            self.ax.set_xlim(-5.0, 5.0) 
            self.ax.set_ylim(-5.0, 5.0)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend(loc='upper right')
            self.ax.set_title("Live Diagnostics (Robot Frame)")
            self.ax.set_xlabel("X [m]")
            self.ax.set_ylabel("Y [m]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        self.align=align
        self.first_scan_received = False
        # Pure pursuit parameters
        self.pure_pursuit = pure_pursuit
        self.lookahead_distance = 1.0
        self.initial_pos = None # Set on first iteration

        self.init_time = time.time()
        self.previous_control_time = time.time()

        self.v_max = 0.45 # m/s
        self.w_max = 1.9 # rad/s
        self.n_stack = 5 
        self.dt = 1.0 / self.frequency  # Control frequency
        self.radius = 0.3
        if san_niccolo:
            self.patrol = False
        else:
            self.patrol = patrol_mode # Back and forth from initial position to goal
        self.save_file_name = save_file_name

        self.obs_stack = deque(maxlen=self.n_stack)
        self.recorded_data = []

        self.latest_scan = None
        self.latest_odom = None
        self.odom_buffer = deque(maxlen=200)
        self.odom_cmd_time_offset = None
        self.odom_scan_time_offset = None
        self.latest_odom_aligned_time = None
        self.latest_scan_aligned_time = None
        self.latest_scan_odom_aligned_time = None
        if san_niccolo:
            self.robot_goal_list = jnp.append(jnp.array([[0., 0.]]), waypoints, axis=0)
            self.robot_goal_index = 1
        else:
            self.robot_goal_list = jnp.array([[0., 0.]] + rc_goal_list)
            self.robot_goal_index = 1
        self.robot_goal = self.robot_goal_list[self.robot_goal_index]  # Set the first goal as the current goal
        self.robot_goal_forward = True # Direction for patrol mode
        self.goal_reached = False
        self.interp_mode = interp_mode

        self.lidar_num_rays = lidar_num_rays
        self.lidar_min_angle = -jnp.pi
        self.lidar_max_angle = jnp.pi
        self.lidar_max_dist = 10
        self.angular_res = (float(self.lidar_max_angle) - float(self.lidar_min_angle)) / self.lidar_num_rays
        self.previous_scan_time = 0.

        if planner == 'JESSI':
            self.policy = JESSI(
                v_max=self.v_max,
                wheels_distance=2*self.v_max/self.w_max,
                n_stack=self.n_stack,
                robot_radius=self.radius,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1,
                ablation_mode = 6,
            )
        elif planner == 'DWA':
            self.policy = DWA(
                v_max=self.v_max,
                wheels_distance=2*self.v_max/self.w_max,
                dt=self.dt,
                n_stack=self.n_stack,
                robot_radius=self.radius,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                use_box_action_space=True,
                predict_time_horizon=1.,
                heading_cost_coeff=0.2,
                clearance_cost_coeff=0.2,
            )
        elif planner == 'MPPI':
            self.policy = MPPI(
                v_max=self.v_max,
                wheels_distance=2*self.v_max/self.w_max,
                dt=self.dt,
                robot_radius=self.radius,
                n_stack=self.n_stack,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
            )
            self.u_mean = self.policy.init_u_mean()
        elif planner == 'VANILLA-E2E':
            self.policy = VanillaE2E(
                v_max=self.v_max,
                wheels_distance=2*self.v_max/self.w_max,
                robot_radius=self.radius,
                n_stack=self.n_stack,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1
            )
        elif planner == 'BOUNDED-VANILLA-E2E':
            self.policy = VanillaE2E(
                v_max=self.v_max,
                wheels_distance=2*self.v_max/self.w_max,
                robot_radius=self.radius,
                n_stack=self.n_stack,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1,
                action_space_bounding=True,
            )
        self.rng_key = random.PRNGKey(0)
        self.network_params = None
        if planner in ('JESSI', 'VANILLA-E2E', 'BOUNDED-VANILLA-E2E'):
            network_path = network_name if os.path.isabs(network_name) else os.path.join(os.path.dirname(__file__), network_name)
            with open(network_path, 'rb') as f:
                self.network_params, _, _ = pickle.load(f)
        
        # Reset turtlebot odometry
        self.odom_reset_confirmed = False
        self.reset_odom_client = self.create_client(ResetPose, '/turtlebot1/reset_pose')
        if self.reset_odom_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("Sending odometry reset request...")
            req = ResetPose.Request()
            req.pose.position.x = 0.0
            req.pose.position.y = 0.0
            req.pose.position.z = 0.0
            req.pose.orientation.x = 0.0
            req.pose.orientation.y = 0.0
            req.pose.orientation.z = 0.0
            req.pose.orientation.w = 1.0
            future = self.reset_odom_client.call_async(req)
            future.add_done_callback(self.odom_reset_callback)
        else:
            self.get_logger().warn("❌❌❌ ERROR: Odometry reset service /turtlebot1/reset_pose not found.\nControl loop will not start...")
            exit()

        # ROS 2 Subscribers
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(LaserScan, '/turtlebot1/scan', self.scan_callback, qos_scan)
        self.sub_odom = self.create_subscription(Odometry, '/turtlebot1/odom', self.odom_callback, qos_profile_sensor_data)
        self.sub_cmd = self.create_subscription(TwistStamped, '/turtlebot1/cmd_vel_stamped', self.cmd_callback, qos_profile_sensor_data)
        
        # ROS 2 Publisher
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_cmd = self.create_publisher(Twist, '/turtlebot1/cmd_vel', qos_cmd)
        self.pub_cmd_stamped = self.create_publisher(TwistStamped, '/turtlebot1/cmd_vel_stamped', qos_cmd)
        
        # ROS 2 Timer
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.previous_action = jnp.array([0.,0.])

        # Saving Lists
        self.save_lists = save_lists
        self.scan_list = []
        self.odom_list = []
        self.cmd_list = []

        self.get_logger().info(f"{planner} Controller initialized at {self.frequency:.1f}Hz!")

    def get_lookahead_point(self, rx, ry, A, B):
        """
        Calculate the lookahead point for Pure Pursuit on segment A -> B.
        If the robot's distance from the segment exceeds `lookahead_distance`, use 
        the orthogonal projection (Nearest Point) as a fallback.
        """
        A = np.array(A)
        B = np.array(B)
        P = np.array([rx, ry])
        V = B - A
        D = A - P
        a = np.dot(V, V)
        b = 2.0 * np.dot(D, V)
        c = np.dot(D, D) - self.lookahead_distance**2
        if a < 1e-6: # A e B coincide
            return B
        discriminant = b**2 - 4*a*c
        t_proj = -b / (2*a) # t value for orthogonal projection
        if discriminant < 0:
            # No intersection: the robot is too far (fallback strategy)
            t = np.clip(t_proj, 0.0, 1.0)
            return A + t * V
        # Compute the two possible intersections
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        valid_ts = [t for t in (t1, t2) if 0.0 <= t <= 1.0]   
        if len(valid_ts) > 0:
            t = max(valid_ts)
        else:
            if max(t1, t2) > 1.0:
                t = 1.0
            else:
                t = 0.0     
        return A + t * V

    def compute_alignment_angle_hough(self, scan_x, scan_y, theta_res=0.2, rho_res=0.05):
        """
        Applies the Hough Transform to a 2D pointcloud to find the lateral walls' angle 
        relative to the robot's X-axis. Returns the yaw error in radians.
        """
        thetas_deg = np.arange(-90.0, 90.0, theta_res)
        thetas_rad = np.deg2rad(thetas_deg)
        cos_t = np.cos(thetas_rad)
        sin_t = np.sin(thetas_rad)
        rhos = scan_x[:, None] * cos_t + scan_y[:, None] * sin_t
        min_rho = np.min(rhos)
        max_rho = np.max(rhos)
        rho_bins = np.arange(min_rho, max_rho + rho_res, rho_res)
        theta_indices = np.tile(thetas_rad, (len(scan_x), 1))
        H, _, _ = np.histogram2d(
            rhos.flatten(), 
            theta_indices.flatten(), 
            bins=[rho_bins, thetas_rad]
        )
        peak_idx = np.unravel_index(np.argmax(H), H.shape)
        best_theta_rad = thetas_rad[peak_idx[1]]
        alignment_error = best_theta_rad - np.sign(best_theta_rad) * (np.pi / 2.0)
        return alignment_error

    def compute_alignment_angle_ransac(self, scan_x: np.ndarray, scan_y: np.ndarray, distance_threshold: float = 0.05, max_iterations: int = 300) -> float:
        """
        Estimates the heading error relative to corridor walls from a 2D LiDAR scan using RANSAC and SVD.
        """
        points = np.column_stack((scan_x, scan_y))
        points = points[np.isfinite(points).all(axis=1)]
        num_points = len(points)
        if num_points < 2:
            return 0.0

        best_inliers_mask = None
        max_inliers_count = -1

        for _ in range(max_iterations):
            idx = np.random.choice(num_points, size=2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            vec = p2 - p1
            norm = np.linalg.norm(vec)
            if norm < 1e-4:
                continue

            normal = np.array([-vec[1], vec[0]]) / norm
            distances = np.abs(np.dot(points - p1, normal))
            inliers_mask = distances < distance_threshold
            inliers_count = np.sum(inliers_mask)

            if inliers_count > max_inliers_count:
                max_inliers_count = inliers_count
                best_inliers_mask = inliers_mask

        if best_inliers_mask is None or max_inliers_count < 2:
            return 0.0

        inlier_points = points[best_inliers_mask]
        centered_points = inlier_points - np.mean(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(centered_points)
        dx, dy = vh[0]

        wall_angle = np.arctan2(dy, dx)
        yaw_error = (wall_angle + np.pi / 2.0) % np.pi - np.pi / 2.0
        return yaw_error

    def clean_and_shift_scan(self, ranges:np.ndarray):
        cleaned = np.nan_to_num(ranges, nan=30., posinf=30., neginf=30.)
        cleaned[cleaned < 0.15] = self.lidar_max_dist
        cleaned = np.clip(cleaned, 0.0, self.lidar_max_dist)
        tb4_angle_min = self.latest_scan.angle_min
        tb4_angle_max = self.latest_scan.angle_max
        tb4_num_rays = len(cleaned)
        angular_res_tb4 = (tb4_angle_max - tb4_angle_min) / (tb4_num_rays - 1)
        jessi_angle_min = float(self.lidar_min_angle)
        shift_rad = (tb4_angle_min - jessi_angle_min) + jnp.deg2rad(jnp.array([90]))[0]
        shift_bins = int(round(shift_rad / angular_res_tb4))
        shifted_cleaned = np.roll(cleaned, shift_bins)
        return shifted_cleaned

    def odom_reset_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info("OK: Odometry reset on turtlebot4")
            self.odom_reset_confirmed = True
        except Exception as e:
            self.get_logger().error(f"Error during odometry reset: {e}\nControl loop will not start...")

    def scan_callback(self, msg):
        raw_scan_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.latest_odom is None or self.latest_odom_aligned_time is None:
            return
        if self.odom_scan_time_offset is None:
            # Map the scan device clock into the host/control clock without
            # mutating the original ROS message header.
            self.odom_scan_time_offset = self.latest_odom_aligned_time - raw_scan_t
            self.get_logger().info(f"🕒 Software Time-Sync: Applied offset {self.odom_scan_time_offset:+.3f}s to Scan w.r.t. host control time!")
        corrected_t = raw_scan_t + self.odom_scan_time_offset
        self.latest_scan = msg
        self.latest_scan_aligned_time = corrected_t
        # Since odomoetry runs at higher freq. we save the latest odometry at the moment of receiving the scan, to have them synchronized for the control loop
        self.latest_scan_odom = self.latest_odom
        self.latest_scan_odom_aligned_time = self.latest_odom_aligned_time
        ### Diagnostics
        if self.diagnostics:
            ranges = np.array(self.latest_scan.ranges)
            shifted_cleaned = self.clean_and_shift_scan(ranges)
            angles = jnp.linspace( - self.policy.lidar_angular_range/2,  + self.policy.lidar_angular_range/2, len(shifted_cleaned))
            scan_x = shifted_cleaned * jnp.cos(angles)
            scan_y = shifted_cleaned * jnp.sin(angles)
            self.scan_plot.set_data(scan_x, scan_y)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        ### Align
        if self.align and not self.first_scan_received:
            self.get_logger().info("🔧 Aligning waypoints with Hough Transform of LiDAR scan...")
            ranges = np.array(self.latest_scan.ranges)
            shifted_cleaned = self.clean_and_shift_scan(ranges)
            angles = jnp.linspace( - self.policy.lidar_angular_range/2,  + self.policy.lidar_angular_range/2, len(shifted_cleaned))
            scan_x = shifted_cleaned * jnp.cos(angles)
            scan_y = shifted_cleaned * jnp.sin(angles)
            if self.alignement == "hough": alignement_error = self.compute_alignment_angle_hough(scan_x, scan_y)
            elif self.alignement == "ransac": alignement_error = self.compute_alignment_angle_ransac(scan_x, scan_y)
            else: raise NotImplementedError(f"Method {self.alignement} has not been implemented for initial alignement")
            c, s = jnp.cos(alignement_error), jnp.sin(alignement_error)
            R = jnp.array([[c, -s],[s,  c]])
            self.robot_goal_list = (R @ self.robot_goal_list.T).T
            self.robot_goal = self.robot_goal_list[self.robot_goal_index]
            self.get_logger().info(f"Alignment complete!\nAlignment error: {alignement_error:.2f} rad\nFirst goal is now at: {self.robot_goal}")
            self.first_scan_received = True
        ### Collect data
        if self.save_lists:
            saved_msg = msg
            if self.experiment_dir is None:
                saved_msg = copy.deepcopy(msg)
                saved_msg.header.stamp.sec = int(corrected_t)
                saved_msg.header.stamp.nanosec = int((corrected_t - int(corrected_t)) * 1e9)
            self.scan_list.append(saved_msg)
        ### DEBUG
        # t_scan = self.latest_scan.header.stamp
        # scan_time_sec = t_scan.sec + t_scan.nanosec * 1e-9
        # t_odom = self.latest_scan_odom.header.stamp
        # odom_time_sec = t_odom.sec + t_odom.nanosec * 1e-9
        # print(f"Scan received - Scan time delta. {abs(scan_time_sec - self.previous_scan_time):.2f} s | Sync delta: {abs(scan_time_sec - odom_time_sec):.2f} s")
        # self.previous_scan_time = scan_time_sec

    def odom_callback(self, msg):
        raw_odom_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.odom_cmd_time_offset is None:
            local_t = self.get_clock().now().nanoseconds * 1e-9
            self.odom_cmd_time_offset = local_t - raw_odom_t
            self.get_logger().info(f"🕒 Software Time-Sync: Applied offset {self.odom_cmd_time_offset:+.3f}s to Odometry w.r.t. Commands!")
        corrected_t = raw_odom_t + self.odom_cmd_time_offset
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.odom_buffer.append((corrected_t, x, y, theta, vx, wz))
        self.latest_odom = msg
        self.latest_odom_aligned_time = corrected_t
        if self.save_lists:
            saved_msg = msg
            if self.experiment_dir is None:
                saved_msg = copy.deepcopy(msg)
                saved_msg.header.stamp.sec = int(corrected_t)
                saved_msg.header.stamp.nanosec = int((corrected_t - int(corrected_t)) * 1e9)
            self.odom_list.append(saved_msg)

    def cmd_callback(self, msg):
        if self.save_lists:
            self.cmd_list.append(msg)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def interpolate_pose(self, t_target):
        if len(self.odom_buffer) < 2:
            return None
        buffer_list = list(self.odom_buffer)
        if t_target < buffer_list[0][0]:
            self.get_logger().warn("Huge latency! The scan is older than the oldest odometry in memory.\nSsh into the turtlebot's Raspberry Pi and run 'sudo chronyc makestep'")
            return buffer_list[0][1:]
        if t_target > buffer_list[-1][0]: # Scan time is newer than the latest odometry, we return the latest pose (no extrapolation)
            return buffer_list[-1][1:]

        for i in range(len(buffer_list) - 1):
            t0, x0, y0, theta0, vx0, wz0 = buffer_list[i]
            t1, x1, y1, theta1, vx1, wz1 = buffer_list[i+1]
            if t0 <= t_target <= t1:
                ratio = (t_target - t0) / (t1 - t0)
                x_interp = x0 + ratio * (x1 - x0)
                y_interp = y0 + ratio * (y1 - y0)
                diff_theta = math.atan2(math.sin(theta1 - theta0), math.cos(theta1 - theta0))
                theta_interp = theta0 + ratio * diff_theta
                theta_interp = math.atan2(math.sin(theta_interp), math.cos(theta_interp))
                vx_interp = vx0 + ratio * (vx1 - vx0)
                wz_interp = wz0 + ratio * (wz1 - wz0)
                return x_interp, y_interp, theta_interp, vx_interp, wz_interp 
        return None

    def finish_experiment(self, reason, timestamp, pose=None, goal_distance=None):
        if self.final_event is not None:
            return
        self.final_event = {
            'reason': reason,
            'timestamp': float(timestamp),
            'pose': list(map(float, pose)) if pose is not None else None,
            'goal_distance': float(goal_distance) if goal_distance is not None else None,
        }
        self.goal_reached = reason == 'goal_reached'
        if self.stop_on_goal or reason == 'timeout':
            self.finished = True
        stop = Twist()
        self.pub_cmd.publish(stop)
        stop_stamped = TwistStamped()
        stop_stamped.header.stamp = self.get_clock().now().to_msg()
        stop_stamped.twist = stop
        self.pub_cmd_stamped.publish(stop_stamped)
        self.get_logger().info(f"Experiment finished: {reason}")

    def control_loop(self):
        if self.latest_scan is None or self.latest_odom is None or not self.odom_reset_confirmed:
            self.get_logger().warn("Waiting data from sensors...")
            return
        # Timestamp extraction
        raw_scan_time_sec = self.latest_scan.header.stamp.sec + self.latest_scan.header.stamp.nanosec * 1e-9
        raw_odom_time_sec = self.latest_scan_odom.header.stamp.sec + self.latest_scan_odom.header.stamp.nanosec * 1e-9
        scan_time_sec = self.latest_scan_aligned_time
        odom_time_sec = self.latest_scan_odom_aligned_time
        control_time_sec = self.get_clock().now().nanoseconds * 1e-9
        # print(f"Scan time: {scan_time_sec}\nOdom time: {odom_time_sec}\nCmd time: {self.get_clock().now().nanoseconds * 1e-9}")
        # Odometry
        if self.interp_mode:
            pose_interp = self.interpolate_pose(scan_time_sec)
            if pose_interp is None:
                self.get_logger().warn("Impossible to interpolate pose at scan timestamp, skipping this control step...")
                return
            rx, ry, r_theta, vx , wz = pose_interp
        else:
            rx = self.latest_scan_odom.pose.pose.position.x
            ry = self.latest_scan_odom.pose.pose.position.y
            r_theta = self.get_yaw_from_quaternion(self.latest_scan_odom.pose.pose.orientation)
            vx = self.latest_scan_odom.twist.twist.linear.x
            wz = self.latest_scan_odom.twist.twist.angular.z
        print(f"Current pose - x: {rx:.2f}, y: {ry:.2f}, theta: {r_theta:.2f}, delta t: {time.time() - self.previous_control_time:.2f} s")
        self.previous_control_time = time.time()
        # Ranges cleaning and shifting (TB4 ranges start on the right)
        ranges = np.array(self.latest_scan.ranges)
        shifted_cleaned = self.clean_and_shift_scan(ranges)
        # Ranges resampling (from self.original_num_rays to self.lidar_num_rays)
        x_old_indices = np.linspace(0, len(ranges) - 1, self.lidar_num_rays).round().astype(int)
        lidar_scan = shifted_cleaned[x_old_indices]
        lidar_scan = np.clip(lidar_scan, 0, self.lidar_max_dist)

        # Observation
        current_step_obs = np.concatenate(([rx, ry, r_theta, self.radius, vx, wz, self.previous_action[0], self.previous_action[1]], [scan_time_sec], [odom_time_sec], [self.get_clock().now().nanoseconds * 1e-9], lidar_scan))
        self.obs_stack.appendleft(current_step_obs)
        while len(self.obs_stack) < self.n_stack:
            self.obs_stack.appendleft(current_step_obs) 
        obs_matrix = jnp.array(self.obs_stack) # Shape: (n_stack, n_rays + 11)

        # Goal (with or without pure pursuit)
        if self.pure_pursuit and len(self.robot_goal_list) > 1:
            B = self.robot_goal # Current target
            if self.robot_goal_forward:
                A = self.robot_goal_list[self.robot_goal_index - 1] if self.robot_goal_index > 0 else self.initial_pos
            else:
                A = self.robot_goal_list[self.robot_goal_index + 1] if self.robot_goal_index < len(self.robot_goal_list) - 1 else self.initial_pos
            # Lookahead goal
            local_goal = self.get_lookahead_point(rx, ry, A, B)
            info_dict = {
                "robot_goal": jnp.array(local_goal)
            }
        else:
            # No pure pursuit
            info_dict = {
                "robot_goal": jnp.array(self.robot_goal)
            }

        # Check distance to goal
        dist = jnp.linalg.norm(jnp.array([rx, ry]) - self.robot_goal)

        if self.first_command_timestamp is not None and self.timeout is not None:
            if control_time_sec - self.first_command_timestamp >= self.timeout:
                self.finish_experiment('timeout', control_time_sec, (rx, ry, r_theta), dist)
                return

        if self.goal_reached:
            stop = Twist()
            self.pub_cmd.publish(stop)
            return

        # Goal reset logic
        if dist < self.goal_tolerance:
            if self.robot_goal_forward:
                if self.robot_goal_index < len(self.robot_goal_list) - 1:
                    self.robot_goal_index += 1
                    self.robot_goal = self.robot_goal_list[self.robot_goal_index]
                    self.get_logger().info(f"🏆 Goal reached! New goal: {self.robot_goal}")
                    info_dict["robot_goal"] = jnp.array(self.robot_goal)
                else:
                    if self.patrol:
                        self.robot_goal_forward = False
                        self.robot_goal_index -= 1
                        self.robot_goal = self.robot_goal_list[self.robot_goal_index]
                        self.get_logger().info(f"🏆🔄 Goal reached, back to the previous goal {self.robot_goal} (PATROL MODE)")
                        info_dict["robot_goal"] = jnp.array(self.robot_goal)
                    else:
                        self.get_logger().info("🏆 Goal reached. Stopping the robot...")
                        self.finish_experiment('goal_reached', control_time_sec, (rx, ry, r_theta), dist)
            else:
                if self.robot_goal_index > 0:
                    self.robot_goal_index -= 1
                    self.robot_goal = self.robot_goal_list[self.robot_goal_index]
                    self.get_logger().info(f"🏆🔄 Goal reached, back to the previous goal {self.robot_goal} (PATROL MODE)")
                    info_dict["robot_goal"] = jnp.array(self.robot_goal)
                else:
                    self.robot_goal_forward = True
                    self.robot_goal_index += 1
                    self.robot_goal = self.robot_goal_list[self.robot_goal_index]
                    self.get_logger().info(f"🏆 Goal reached, new goal: {self.robot_goal}")
                    info_dict["robot_goal"] = jnp.array(self.robot_goal)
        else:
            # ACTION INFERENCE
            try:
                if self.planner == 'JESSI':
                    action, self.rng_key, _, _, _, _, perception_output, actor_distr, _, spat_attn, temp_attn, human_attn = self.policy.act(
                        key=self.rng_key,
                        obs=obs_matrix,
                        info=info_dict,
                        e2e_network_params=self.network_params,
                        sample=False # Use mean action
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    step_record = {
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'perception_distr': perception_output,
                        'actor_distr': actor_distr,
                        'spatial_attention': spat_attn[0],
                        'temporal_attention': temp_attn[0],
                        'human_attention': human_attn[0],
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    }
                elif self.planner == 'DWA':
                    action, actions_costs = self.policy.act(
                        obs=obs_matrix,
                        info=info_dict,
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    step_record = {
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'action_costs': actions_costs,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    }
                elif self.planner == 'MPPI':
                    action, self.u_mean, trajectories, costs, self.rng_key  = self.policy.act(
                        obs=obs_matrix,
                        info=info_dict,
                        u_mean=self.u_mean,
                        key=self.rng_key,
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    step_record = {
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'trajectories': trajectories,
                        'costs': costs,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    }
                elif self.planner == 'VANILLA-E2E' or self.planner == 'BOUNDED-VANILLA-E2E':
                    action, self.rng_key, _, _, _, actor_distr, _ = self.policy.act(
                        self.rng_key,
                        obs=obs_matrix,
                        info=info_dict,
                        network_params=self.network_params,
                        sample=False # Use mean action
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    step_record = {
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'actor_distr': actor_distr,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    }
                ### ==========================================================
                ### SIM-TO-REAL ENGINEERING FILTERS
                ### ==========================================================
                policy_action = np.array([v_cmd, w_cmd], dtype=float)
                if self.engineering_filters:
                    if self.planner in ['JESSI', 'VANILLA-E2E', 'BOUNDED-VANILLA-E2E']:
                        ## 1. Geometric auxiliary controller (P-Controller)
                        dx = float(self.robot_goal[0]) - rx
                        dy = float(self.robot_goal[1]) - ry
                        angle_to_goal = np.arctan2(dy, dx)
                        heading_error = angle_to_goal - r_theta
                        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                        w_geo = float(np.clip(self.w_max * heading_error, -self.w_max, self.w_max)) 
                        v_max_geo = self.v_max - (abs(w_geo)*self.v_max/self.w_max)
                        v_geo = float(np.clip(self.v_max * dist, 0.0, v_max_geo))
                        ## 2. Action Blending
                        # Analysis of the free frontal space (centrale cone of the scan)
                        num_rays = len(lidar_scan)
                        center_idx = num_rays // 2
                        front_window = lidar_scan[max(0, center_idx - 10):min(num_rays, center_idx + 10)]
                        min_front_dist = np.min(front_window)
                        # Alpha coefficient: 0.0 (RL) with close-by obstacles, 1.0 (Geometric) with free space
                        d_min = 1.5 # Start of RL intervention
                        d_max = 3.0 # Geometric guide only
                        alpha = float(np.clip((min_front_dist - d_min) / (d_max - d_min), 0.0, 1.0))
                        v_cmd = alpha * v_geo + (1.0 - alpha) * v_cmd
                        w_cmd = alpha * w_geo + (1.0 - alpha) * w_cmd
                        ## 3. Recovery Mode
                        # Forced in-place rotation if 60 degrees (1.04 rad) off-axis and the path is clear.
                        if abs(heading_error) > 1.04 and min_front_dist > 1.0:
                            v_cmd = 0.0
                            w_cmd = float(np.sign(heading_error)) * 0.5
                    ## 4. Angular velocity deadband
                    w_deadband = 0.05
                    if abs(w_cmd) < w_deadband:
                        w_cmd = 0.0
                    ## 5. Inertial filter (Low-pass)
                    beta = 0.3 # Weight of the previous step for peak damping
                    v_cmd = beta * float(self.previous_action[0]) + (1.0 - beta) * v_cmd
                    w_cmd = beta * float(self.previous_action[1]) + (1.0 - beta) * w_cmd
                ### ==========================================================
                cmd_msg = Twist()
                cmd_msg.linear.x = v_cmd
                cmd_msg.angular.z = w_cmd
                self.pub_cmd.publish(cmd_msg)
                cmd_stamped = TwistStamped()
                cmd_stamped.header.stamp = self.get_clock().now().to_msg()
                cmd_stamped.twist = cmd_msg
                self.pub_cmd_stamped.publish(cmd_stamped)
                command_timestamp = (
                    cmd_stamped.header.stamp.sec
                    + cmd_stamped.header.stamp.nanosec * 1e-9
                )
                if self.first_command_timestamp is None:
                    self.first_command_timestamp = command_timestamp
                step_record.update({
                    'action': np.array([v_cmd, w_cmd], dtype=float),
                    'policy_action': policy_action,
                    'published_action': np.array([v_cmd, w_cmd], dtype=float),
                    'command_timestamp': command_timestamp,
                    'control_loop_timestamp': control_time_sec,
                    'raw_scan_timestamp': raw_scan_time_sec,
                    'raw_odom_timestamp': raw_odom_time_sec,
                    'scan_timestamp': scan_time_sec,
                    'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    'robot_pose': np.array([rx, ry, r_theta], dtype=float),
                    'measured_twist': np.array([vx, wz], dtype=float),
                    'goal_distance': float(dist),
                    'planner': self.planner,
                    'inference_ok': True,
                })
                self.recorded_data.append(step_record)
                self.previous_action = jnp.array([v_cmd, w_cmd])
            except Exception as e:
                self.get_logger().error(f"Error during {self.planner} inference: {e}")
        
        ## Diagnostics plotting
        if self.diagnostics:
            c, s = jnp.cos(-r_theta), jnp.sin(-r_theta)
            R = jnp.array([[c, -s],[s,  c]])
            translated_position = info_dict["robot_goal"] - jnp.array([rx, ry])
            rc_robot_goal = R @ translated_position
            self.goal_plot.set_data([rc_robot_goal[0]], [rc_robot_goal[1]])
            if self.robot_goal_forward:
                future_goals = self.robot_goal_list[self.robot_goal_index + 1 :]
            else:
                future_goals = self.robot_goal_list[: self.robot_goal_index]
            if len(future_goals) > 0:
                translated_futures = future_goals - jnp.array([rx, ry])
                rc_future_goals = (R @ translated_futures.T).T
                self.future_goals_plot.set_data(rc_future_goals[:, 0], rc_future_goals[:, 1])
            else:
                self.future_goals_plot.set_data([], [])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save_data(self):
        save_path = (
            self.experiment_dir / 'controller.pkl'
            if self.experiment_dir
            else Path(os.path.dirname(__file__)) / self.save_file_name
        )
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            out = {
                'schema_version': 2,
                'trajectory': self.recorded_data,
                'final_event': self.final_event,
                'params': {
                    'planner': self.planner,
                    'v_max': self.policy.v_max,
                    'wheels_distance': self.policy.wheels_distance,
                    'robot_radius': self.radius,
                    'n_stack': self.policy.n_stack,
                    'lidar_num_rays': self.lidar_num_rays,
                    'lidar_angular_range': self.lidar_max_angle-self.lidar_min_angle,
                    'lidar_max_dist': self.lidar_max_dist,
                    'frequency': self.frequency,
                    'goal_tolerance': self.goal_tolerance,
                    'timeout': self.timeout,
                    'engineering_filters': self.engineering_filters,
                },
                'clock_offsets': {
                    'odom_to_host': self.odom_cmd_time_offset,
                    'scan_to_host': self.odom_scan_time_offset,
                },
            }
            if self.planner == 'JESSI' or self.planner == 'BOUNDED-VANILLA-E2E':
                out['params']['n_stack_for_action_space_bounding'] = self.policy.n_stack_for_action_space_bounding
            temporary = save_path.with_suffix(save_path.suffix + '.tmp')
            with open(temporary, 'wb') as f:
                pickle.dump(out, f)
            os.replace(temporary, save_path)
            if self.save_lists:
                lists_save_path = (
                    self.experiment_dir / 'raw_sensor_messages.pkl'
                    if self.experiment_dir
                    else Path(os.path.dirname(__file__)) / f"lists_{self.save_file_name}"
                )
                temporary_lists = lists_save_path.with_suffix(lists_save_path.suffix + '.tmp')
                with open(temporary_lists, 'wb') as f:
                    pickle.dump({
                        'scan': self.scan_list,
                        'odom': self.odom_list,
                        'cmd': self.cmd_list,
                    }, f)
                os.replace(temporary_lists, lists_save_path)
            self.get_logger().info(f"Record saved! {len(self.recorded_data)} frames saved in: {save_path}")
        except Exception as e:
            self.get_logger().error(f"Error during saving procedure: {e}")

def main(args=None):
    parser = argparse.ArgumentParser(description='TB4 Robot Controller - Local Planner')
    parser.add_argument('--planner', type=str, default='JESSI', help='Network weights pickle file name')
    parser.add_argument('-g', '--goals', nargs='+', type=float, default=[2.0, 0.0], help='Sequence of Goal X Y pairs (in meters). Example: -g 2.0 0.0 3.0 1.0 4.0 -0.5')
    parser.add_argument('-p', '--patrol', action='store_true', help='Activate Patrol Mode (back and forth continuously)')
    parser.add_argument('-i', '--interp', action='store_true', help='Activate Interpolation Mode for pose with respect to LiDAR timestamp (instead of using the latest odometry)')
    parser.add_argument('-n', '--network', type=str, default='jessi_finetuned_rl_out_turtlebot.pkl', help='Network weights pickle file name')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parser.add_argument('-d', '--diagnostics', action='store_true', help='Activate diagnostic during control to debug')
    parser.add_argument('-a', '--align', action='store_true', help='Activate alignement of waypoints with Hough Transform of LiDAR scan')
    parser.add_argument('-f', '--frequency', type=float, default=4.0, help='Control frequency in Hz')
    parser.add_argument('-l', '--lidar_rays', type=int, default=300, help='Number of rays used to infer the policy action')
    parser.add_argument('-sn', '--san_niccolo', action='store_true', help='Activare mode San Niccolò experiment with hardcoded waypoints')
    parser.add_argument('-e', '--engineering_filters', action='store_true', help='Activate Engineering Filters for enhanced sim-to-real performance')
    parser.add_argument('-pp', '--pure_pursuit', action='store_true', help='Activate Pure Pursuit for intermediate goal generation')
    parser.add_argument('--experiment-dir', type=str, help='Structured experiment run directory')
    parser.add_argument('--timeout', type=float, help='Automatically stop after this many seconds from the first command')
    parser.add_argument('--goal-tolerance', type=float, default=0.8, help='Goal center-distance threshold in metres')
    parser.add_argument('--stop-on-goal', action='store_true', help='Exit the controller after reaching the final goal')

    parsed_args, ros_args = parser.parse_known_args(sys.argv)

    if len(parsed_args.goals) % 2 != 0:
        print("❌ ERROR: Argument --goals requires an even number of coordinates (X and Y).")
        sys.exit(1)
    assert parsed_args.planner in PLANNERS, f"Planner {parsed_args.planner} not recognized. Available planners: {PLANNERS}"

    rc_goals_list = [[parsed_args.goals[i], parsed_args.goals[i+1]] for i in range(0, len(parsed_args.goals), 2)]

    rclpy.init(args=ros_args)
    
    node = TB4Controller(
        frequency=parsed_args.frequency,
        lidar_num_rays=parsed_args.lidar_rays,
        planner=parsed_args.planner,
        rc_goal_list=rc_goals_list, 
        patrol_mode=parsed_args.patrol,
        interp_mode=parsed_args.interp,
        network_name=parsed_args.network,
        save_file_name=parsed_args.save_file,
        save_lists=True,
        align=parsed_args.align,
        diagnostics=parsed_args.diagnostics,
        san_niccolo=parsed_args.san_niccolo,
        engineering_filters=parsed_args.engineering_filters,
        pure_pursuit=parsed_args.pure_pursuit,
        experiment_dir=parsed_args.experiment_dir,
        timeout=parsed_args.timeout,
        goal_tolerance=parsed_args.goal_tolerance,
        stop_on_goal=parsed_args.stop_on_goal,
    )
    mode_text = "🔄 PATROL MODE" if parsed_args.patrol else "🛑 SINGLE TARGET MODE"
    node.get_logger().info(f"Goals set to: {rc_goals_list} in the robot frame | {mode_text}")
    
    try:
        if parsed_args.stop_on_goal or parsed_args.timeout is not None:
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.pub_cmd.publish(stop)
        node.get_logger().info("🛑 Stopping control...")
        if node.final_event is None:
            node.final_event = {
                'reason': 'interrupted',
                'timestamp': node.get_clock().now().nanoseconds * 1e-9,
                'pose': None,
                'goal_distance': None,
            }
    finally:
        node.save_data()
        if len(node.recorded_data) > 1:
            scan_times = np.array([step['scan_timestamp'] for step in node.recorded_data][5:])
            odom_times = np.array([step['odom_timestamp'] for step in node.recorded_data][5:])
            # Sync jitter (laser-odom)
            sync_diffs = np.abs(scan_times - odom_times)
            mean_sync = np.mean(sync_diffs)
            std_sync = np.std(sync_diffs)
            # Control loop jitter
            step_diffs = np.diff(scan_times)
            mean_step = np.mean(step_diffs)
            std_step = np.std(step_diffs)
            node.get_logger().info("📊 --- TIMING DIAGNOSTICS ---")
            node.get_logger().info(f"   Odom-Scan Sync Delta : {mean_sync:.2f} s ± {std_sync:.2f} s")
            node.get_logger().info(f"   Control Loop dt      : {mean_step:.4f} s ± {std_step:.4f} s (Target: {node.dt} s)")
            node.get_logger().info("-----------------------------")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
