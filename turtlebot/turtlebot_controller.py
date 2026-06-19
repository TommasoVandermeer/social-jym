#!/usr/bin/env python3
import sys
import argparse
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
    def __init__(self, frequency, lidar_num_rays, planner, rc_goal_list, patrol_mode, interp_mode, network_name, save_file_name, save_lists, align, diagnostics, san_niccolo):
        super().__init__('TB4_controller')

        # San Niccolò Waypoints
        if san_niccolo:
            waypoints = jnp.array([
                [17.6, 22.3],       # 0
                [14.956, 22.618],   # 1
                [1.503, 22.633],    # 2
                [-5.893, 22.991],   # 3
                [-6.873, 19.245],   # 4
                [-6.873, 8.458],    # 5
                [-6.584, 0.597],    # 6
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

        self.frequency = frequency
        self.planner = planner
        self.diagnostics = diagnostics
        if self.diagnostics:
            self.get_logger().info("📈 Initialize live diagnostics plot...")
            plt.ion() # NON blocking mode for live update
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.scan_plot, = self.ax.plot([], [], 'r.', markersize=2, alpha=0.5, label='Lidar Scan')
            self.goal_plot, = self.ax.plot([], [], 'g*', markersize=12, label='Current Goal')
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

        self.init_time = time.time()
        self.previous_control_time = time.time()

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
        if san_niccolo:
            self.robot_goal_list = waypoints
            self.robot_goal_index = 0
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
                v_max=0.45,
                wheels_distance=2*0.45/1.9,
                n_stack=self.n_stack,
                robot_radius=self.radius,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1
            )
        elif planner == 'DWA':
            self.policy = DWA(
                v_max=0.45,
                wheels_distance=2*0.45/1.9,
                dt=self.dt,
                n_stack=self.n_stack,
                robot_radius=self.radius,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
            )
        elif planner == 'MPPI':
            self.policy = MPPI(
                v_max=0.45,
                wheels_distance=2*0.45/1.9,
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
                v_max=0.45,
                wheels_distance=2*0.45/1.9,
                robot_radius=self.radius,
                n_stack=self.n_stack,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1
            )
        elif planner == 'BOUNDED-VANILLA-E2E':
            self.policy = VanillaE2E(
                v_max=0.45,
                wheels_distance=2*0.45/1.9,
                robot_radius=self.radius,
                n_stack=self.n_stack,
                lidar_num_rays=self.lidar_num_rays,
                lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
                lidar_max_dist=self.lidar_max_dist,
                n_stack_for_action_space_bounding=1,
                action_space_bounding=True,
            )
        self.rng_key = random.PRNGKey(0)
        with open(os.path.join(os.path.dirname(__file__), network_name), 'rb') as f:
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

    def compute_alignment_angle(self, scan_x, scan_y, theta_res=0.2, rho_res=0.05):
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
        if self.latest_odom is not None and self.odom_scan_time_offset is None:
            raw_odom_t = self.latest_odom.header.stamp.sec + self.latest_odom.header.stamp.nanosec * 1e-9
            self.odom_scan_time_offset = raw_scan_t - raw_odom_t
            self.get_logger().info(f"🕒 Software Time-Sync: Applied offset {self.odom_scan_time_offset:+.3f}s to Scan w.r.t. Odometry!")
        corrected_t = raw_scan_t - self.odom_scan_time_offset
        msg.header.stamp.sec = int(corrected_t)
        msg.header.stamp.nanosec = int((corrected_t - int(corrected_t)) * 1e9)
        self.latest_scan = msg
        # Since odomoetry runs at higher freq. we save the latest odometry at the moment of receiving the scan, to have them synchronized for the control loop
        self.latest_scan_odom = self.latest_odom
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
            alignment_error = self.compute_alignment_angle(scan_x, scan_y)
            c, s = jnp.cos(alignment_error), jnp.sin(alignment_error)
            R = jnp.array([[c, -s],[s,  c]])
            self.robot_goal_list = (R @ self.robot_goal_list.T).T
            self.robot_goal = self.robot_goal_list[self.robot_goal_index]
            self.get_logger().info(f"Alignment complete!\nAlignment error: {alignment_error:.2f} rad\nFirst goal is now at: {self.robot_goal}")
            self.first_scan_received = True
        ### Collect data
        if self.save_lists:
            self.scan_list.append(msg)
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
        msg.header.stamp.sec = int(corrected_t)
        msg.header.stamp.nanosec = int((corrected_t - int(corrected_t)) * 1e9)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_buffer.append((corrected_t, x, y, theta))
        self.latest_odom = msg 
        if self.save_lists:
            self.odom_list.append(msg)

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
            t0, x0, y0, theta0 = buffer_list[i]
            t1, x1, y1, theta1 = buffer_list[i+1]
            if t0 <= t_target <= t1:
                ratio = (t_target - t0) / (t1 - t0)
                x_interp = x0 + ratio * (x1 - x0)
                y_interp = y0 + ratio * (y1 - y0)
                diff_theta = math.atan2(math.sin(theta1 - theta0), math.cos(theta1 - theta0))
                theta_interp = theta0 + ratio * diff_theta
                theta_interp = math.atan2(math.sin(theta_interp), math.cos(theta_interp))
                return x_interp, y_interp, theta_interp   
        return None

    def control_loop(self):
        if self.latest_scan is None or self.latest_odom is None or not self.odom_reset_confirmed:
            self.get_logger().warn("Waiting data from sensors...")
            return
        # Timestamp extraction
        t_scan = self.latest_scan.header.stamp
        scan_time_sec = t_scan.sec + t_scan.nanosec * 1e-9
        t_odom = self.latest_scan_odom.header.stamp
        odom_time_sec = t_odom.sec + t_odom.nanosec * 1e-9
        # print(f"Scan time: {scan_time_sec}\nOdom time: {odom_time_sec}\nCmd time: {self.get_clock().now().nanoseconds * 1e-9}")
        # Odometry
        if self.interp_mode:
            pose_interp = self.interpolate_pose(scan_time_sec)
            if pose_interp is None:
                self.get_logger().warn("Impossible to interpolate pose at scan timestamp, skipping this control step...")
                return
            rx, ry, r_theta = pose_interp
        else:
            rx = self.latest_scan_odom.pose.pose.position.x
            ry = self.latest_scan_odom.pose.pose.position.y
            r_theta = self.get_yaw_from_quaternion(self.latest_scan_odom.pose.pose.orientation)
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
        current_step_obs = np.concatenate(([rx, ry, r_theta, self.radius, self.previous_action[0], self.previous_action[1]], [scan_time_sec], [odom_time_sec], [self.get_clock().now().nanoseconds * 1e-9], lidar_scan))
        self.obs_stack.appendleft(current_step_obs)
        while len(self.obs_stack) < self.n_stack:
            self.obs_stack.appendleft(current_step_obs) 
        obs_matrix = jnp.array(self.obs_stack) # Shape: (n_stack, 326)
        info_dict = {
            "robot_goal": jnp.array(self.robot_goal)
        }

        # Check distance to goal
        dist = jnp.linalg.norm(jnp.array([rx, ry]) - self.robot_goal)

        if self.goal_reached:
            stop = Twist()
            self.pub_cmd.publish(stop)
            return

        # Goal reset logic
        if dist < self.radius + 0.3:
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
                        stop = Twist()
                        self.pub_cmd.publish(stop)
                        self.goal_reached = True
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
                    self.recorded_data.append({
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
                    })
                elif self.planner == 'DWA':
                    action, actions_costs = self.policy.act(
                        obs=obs_matrix,
                        info=info_dict,
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    self.recorded_data.append({
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'action_costs': actions_costs,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    })
                elif self.planner == 'MPPI':
                    action, self.u_mean, trajectories, costs, self.rng_key  = self.policy.act(
                        obs=obs_matrix,
                        info=info_dict,
                        u_mean=self.u_mean,
                        key=self.rng_key,
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    self.recorded_data.append({
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'trajectories': trajectories,
                        'costs': costs,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    })
                elif self.planner == 'VANILLA-E2E' or self.planner == 'BOUNDED-VANILLA-E2E':
                    action, self.rng_key, _, _, _, actor_distr, _ = self.policy.act(
                        self.rng_key,
                        obs=obs_matrix,
                        info=info_dict,
                        network_params=self.network_params,
                        sample=False # Use mean action
                    )
                    v_cmd, w_cmd = float(action[0]), float(action[1])
                    self.recorded_data.append({
                        'observation': np.array(obs_matrix),
                        'robot_goal': np.array(self.robot_goal),
                        'action': np.array([v_cmd, w_cmd]),
                        'actor_distr': actor_distr,
                        'scan_timestamp': scan_time_sec,
                        'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                    })
                cmd_msg = Twist()
                cmd_msg.linear.x = v_cmd
                cmd_msg.angular.z = w_cmd
                self.pub_cmd.publish(cmd_msg)
                cmd_stamped = TwistStamped()
                cmd_stamped.header.stamp = self.get_clock().now().to_msg()
                cmd_stamped.twist = cmd_msg
                self.pub_cmd_stamped.publish(cmd_stamped)
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
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save_data(self):
        if len(self.recorded_data) > 0:
            save_path = os.path.join(os.path.dirname(__file__), self.save_file_name)
            try:
                out = {
                    'trajectory': self.recorded_data,
                    'params': {
                        'v_max':self.policy.v_max,
                        'wheels_distance':self.policy.wheels_distance,
                        'robot_radius':self.radius,
                        'n_stack':self.policy.n_stack,
                        'lidar_num_rays':self.lidar_num_rays,
                        'lidar_angular_range':self.lidar_max_angle-self.lidar_min_angle,
                        'lidar_max_dist':self.lidar_max_dist,
                        'n_stack':self.policy.n_stack
                    },
                }
                if self.planner == 'JESSI' or self.planner == 'BOUNDED-VANILLA-E2E':
                    out['params']['n_stack_for_action_space_bounding'] = self.policy.n_stack_for_action_space_bounding
                with open(save_path, 'wb') as f:
                    pickle.dump(out, f)
                if self.save_lists:
                    lists_save_path = os.path.join(os.path.dirname(__file__), f"lists_{self.save_file_name}")
                    with open(lists_save_path, 'wb') as f:
                        pickle.dump({
                            'scan': self.scan_list,
                            'odom': self.odom_list,
                            'cmd': self.cmd_list
                        }, f)
                self.get_logger().info(f"Record saved! {len(self.recorded_data)} frame saved in: {save_path}")
            except Exception as e:
                self.get_logger().error(f"Error during saving procedure: {e}")
        else:
            self.get_logger().info("NO DATA TO SAVE.")

def main(args=None):
    parser = argparse.ArgumentParser(description='TB4 Robot Controller - Local Planner')
    parser.add_argument('--planner', type=str, default='JESSI', help='Network weights pickle file name')
    parser.add_argument('-g', '--goals', nargs='+', type=float, default=[2.0, 0.0], help='Sequence of Goal X Y pairs (in meters). Example: -g 2.0 0.0 3.0 1.0 4.0 -0.5')
    parser.add_argument('-p', '--patrol', action='store_true', help='Activate Patrol Mode (back and forth continuously)')
    parser.add_argument('-i', '--interp', action='store_true', help='Activate Interpolation Mode for pose with respect to LiDAR timestamp (instead of using the latest odometry)')
    parser.add_argument('-c', '--collect', action='store_true', help='Activate Full Data Collection Mode')
    parser.add_argument('-n', '--network', type=str, default='jessi_finetuned_rl_out_turtlebot.pkl', help='Network weights pickle file name')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parser.add_argument('-d', '--diagnostics', action='store_true', help='Activate diagnostic during control to debug')
    parser.add_argument('-a', '--align', action='store_true', help='Activate alignement of waypoints with Hough Transform of LiDAR scan')
    parser.add_argument('-f', '--frequency', type=float, default=4.0, help='Control frequency in Hz')
    parser.add_argument('-l', '--lidar_rays', type=int, default=300, help='Number of rays used to infer the policy action')
    parser.add_argument('-sn', '--san_niccolo', action='store_true', help='Activare mode San Niccolò experiment with hardcoded waypoints')

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
        save_lists=parsed_args.collect,
        align=parsed_args.align,
        diagnostics=parsed_args.diagnostics,
        san_niccolo=parsed_args.san_niccolo,
    )
    mode_text = "🔄 PATROL MODE" if parsed_args.patrol else "🛑 SINGLE TARGET MODE"
    node.get_logger().info(f"Goals set to: {rc_goals_list} in the robot frame | {mode_text}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.pub_cmd.publish(stop)
        node.get_logger().info("🛑 Stopping control...")
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