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

from socialjym.policies.jessi import JESSI

class JessiController(Node):
    def __init__(self, rc_goal, patrol_mode, interp_mode, network_name, save_file_name, save_lists):
        super().__init__('jessi_controller')
        self.init_time = time.time()
        self.previous_control_time = time.time()

        self.n_stack = 5 
        self.dt = 0.25 # 4 Hz
        self.radius = 0.3
        self.patrol = patrol_mode # Back and forth from initial position to goal
        self.save_file_name = save_file_name

        self.obs_stack = deque(maxlen=self.n_stack)
        self.recorded_data = []

        self.latest_scan = None
        self.latest_odom = None
        self.odom_buffer = deque(maxlen=200)
        self.odom_time_offset = None
        self.robot_goal = rc_goal
        self.initial_position = jnp.array([0.,0.]) # Odometry is reset at the beginning
        self.goal_reached = False
        self.interp_mode = interp_mode

        #self.original_lidar_num_rays = 1081
        self.lidar_num_rays = 100
        self.lidar_min_angle = -jnp.pi
        self.lidar_max_angle = jnp.pi
        self.lidar_max_dist = 10
        self.angular_res = (float(self.lidar_max_angle) - float(self.lidar_min_angle)) / self.lidar_num_rays
        self.previous_scan_time = 0.

        self.jessi = JESSI(
            v_max=0.42,
            wheels_distance=2*0.42/1.9,
            robot_radius=self.radius,
            lidar_num_rays=self.lidar_num_rays,
            lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
            lidar_max_dist=self.lidar_max_dist,
            n_stack_for_action_space_bounding=1
        )
        self.rng_key = random.PRNGKey(0)
        with open(os.path.join(os.path.dirname(__file__), network_name), 'rb') as f:
            self.network_params, _, _ = pickle.load(f)
        
        # Reset turtlebot odometry
        self.odom_reset_confirmed = False
        self.reset_odom_client = self.create_client(ResetPose, '/reset_pose')
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
            self.get_logger().warn("WARNING: Odometry reset service /reset_pose not found.\nControl loop will not start...")

        # ROS 2 Subscribers
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_scan)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.sub_cmd = self.create_subscription(TwistStamped, '/cmd_vel_stamped', self.cmd_callback, qos_profile_sensor_data)
        
        # ROS 2 Publisher
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', qos_cmd)
        self.pub_cmd_stamped = self.create_publisher(TwistStamped, '/cmd_vel_stamped', qos_cmd)
        
        # ROS 2 Timer
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.previous_action = jnp.array([0.,0.])

        # Saving Lists
        self.save_lists = save_lists
        self.scan_list = []
        self.odom_list = []
        self.cmd_list = []

        self.get_logger().info("JESSI Controller initialized at 4Hz!")

    def odom_reset_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info("OK: Odometry reset on turtlebot4")
            self.odom_reset_confirmed = True
        except Exception as e:
            self.get_logger().error(f"Error during odometry reset: {e}\nControl loop will not start...")

    def scan_callback(self, msg):
        self.latest_scan = msg
        # Since odomoetry runs at higher freq. we save the latest odometry at the moment of receiving the scan, to have them synchronized for the control loop
        self.latest_scan_odom = self.latest_odom
        ### DEBUG
        # t_scan = self.latest_scan.header.stamp
        # scan_time_sec = t_scan.sec + t_scan.nanosec * 1e-9
        # t_odom = self.latest_scan_odom.header.stamp
        # odom_time_sec = t_odom.sec + t_odom.nanosec * 1e-9
        # print(f"Scan received - Scan time delta. {abs(scan_time_sec - self.previous_scan_time):.2f} s | Sync delta: {abs(scan_time_sec - odom_time_sec):.2f} s")
        # self.previous_scan_time = scan_time_sec
        if self.save_lists:
            self.scan_list.append(msg)

    def odom_callback(self, msg):
        raw_odom_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.odom_time_offset is None:
            local_t = self.get_clock().now().nanoseconds * 1e-9
            self.odom_time_offset = local_t - raw_odom_t
            self.get_logger().info(f"🕒 Software Time-Sync: Applicato offset di {self.odom_time_offset:+.3f} secondi all'Odometria!")
        corrected_t = raw_odom_t + self.odom_time_offset
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
        # Ranges cleaning
        ranges = np.array(self.latest_scan.ranges)
        cleaned = np.nan_to_num(ranges, nan=30., posinf=30., neginf=30.)
        cleaned[cleaned < 0.15] = self.lidar_max_dist
        cleaned = np.clip(cleaned, 0.0, self.lidar_max_dist)
        # Ranges shifting (first ray of TB4 is at -90, in JESSI first ray is at -self.lidar_angular_range/2)
        tb4_angle_min = self.latest_scan.angle_min
        tb4_angle_max = self.latest_scan.angle_max
        tb4_num_rays = len(cleaned)
        angular_res_tb4 = (tb4_angle_max - tb4_angle_min) / (tb4_num_rays - 1)
        jessi_angle_min = float(self.lidar_min_angle)
        shift_rad = (tb4_angle_min - jessi_angle_min) + jnp.deg2rad(jnp.array([90]))[0]
        shift_bins = int(round(shift_rad / angular_res_tb4))
        shifted_cleaned = np.roll(cleaned, shift_bins)
        # Ranges resampling (from self.original_num_rays to self.lidar_num_rays)
        x_old_indices = np.linspace(0, tb4_num_rays - 1, self.lidar_num_rays).round().astype(int)
        lidar_scan = shifted_cleaned[x_old_indices]
        lidar_scan = np.clip(lidar_scan, 0, self.lidar_max_dist)

        # Observation
        current_step_obs = np.concatenate(([rx, ry, r_theta, self.radius, self.previous_action[0], self.previous_action[1]], lidar_scan))
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

        if dist < self.radius:
            if self.patrol:
                self.get_logger().info(f"🏆🔄 Goal reached, back to the previous goal...")
                temp_goal = self.robot_goal
                self.robot_goal = self.initial_position
                self.initial_position = temp_goal
                info_dict["robot_goal"] = jnp.array(self.robot_goal)
            else:
                self.get_logger().info("🏆 Goal reached. Stopping the robot...")
                stop = Twist()
                self.pub_cmd.publish(stop)
                self.goal_reached = True
        else:
            # JESSI INFERENCE
            try:
                action, self.rng_key, _, _, _, _, perception_output, actor_distr, _, _ = self.jessi.act(
                    key=self.rng_key,
                    obs=obs_matrix,
                    info=info_dict,
                    e2e_network_params=self.network_params,
                    sample=False # Use mean action
                )
                v_cmd, w_cmd = float(action[0]), float(action[1])
                # DEBUG action override
                # v_cmd = 0.
                # if (time.time() - self.init_time) % 5.0 > 2.5:
                #     w_cmd = 2.
                # else:
                #     w_cmd = -2.
                # #
                cmd_msg = Twist()
                cmd_msg.linear.x = v_cmd
                cmd_msg.angular.z = w_cmd
                self.pub_cmd.publish(cmd_msg)
                cmd_stamped = TwistStamped()
                cmd_stamped.header.stamp = self.get_clock().now().to_msg()
                cmd_stamped.twist = cmd_msg
                self.pub_cmd_stamped.publish(cmd_stamped)
                self.previous_action = jnp.array([v_cmd, w_cmd])
                self.recorded_data.append({
                    'observation': np.array(obs_matrix),
                    'robot_goal': np.array(self.robot_goal),
                    'action': np.array([v_cmd, w_cmd]),
                    'perception_distr': perception_output,
                    'actor_distr': actor_distr,
                    'scan_timestamp': scan_time_sec,
                    'odom_timestamp': odom_time_sec if not self.interp_mode else scan_time_sec,
                })
            except Exception as e:
                self.get_logger().error(f"Error during JESSI inference: {e}")

    def save_data(self):
        if len(self.recorded_data) > 0:
            save_path = os.path.join(os.path.dirname(__file__), self.save_file_name)
            try:
                out = {
                    'trajectory': self.recorded_data,
                    'jessi_params': {
                        'v_max':self.jessi.v_max,
                        'wheels_distance':self.jessi.wheels_distance,
                        'robot_radius':self.radius,
                        'n_stack':self.jessi.n_stack,
                        'lidar_num_rays':self.lidar_num_rays,
                        'lidar_angular_range':self.lidar_max_angle-self.lidar_min_angle,
                        'lidar_max_dist':self.lidar_max_dist,
                        'n_stack_for_action_space_bounding':self.jessi.n_stack_for_action_space_bounding
                    },
                }
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
    parser = argparse.ArgumentParser(description='JESSI Robot Controller')
    parser.add_argument('-x', '--goal_x', type=float, default=2.0, help='Goal X (in meters)')
    parser.add_argument('-y', '--goal_y', type=float, default=0.0, help='Goal Y (in meters)')
    parser.add_argument('--patrol', action='store_true', help='Activate Patrol Mode (back and forth continuously)')
    parser.add_argument('--interp', action='store_true', help='Activate Interpolation Mode for pose with respect to LiDAR timestamp (instead of using the latest odometry)')
    parser.add_argument('--collect', action='store_true', help='Activate Full Data Collection Mode')
    parser.add_argument('-n', '--network', type=str, default='jessi_finetuned_rl_out_turtlebot.pkl', help='Network weights pickle file name')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)
    rc_goal = np.array([parsed_args.goal_x, parsed_args.goal_y])

    rclpy.init(args=ros_args)
    
    node = JessiController(
        rc_goal=rc_goal, 
        patrol_mode=parsed_args.patrol,
        interp_mode=parsed_args.interp,
        network_name=parsed_args.network,
        save_file_name=parsed_args.save_file,
        save_lists=parsed_args.collect
    )
    mode_text = "🔄 PATROL MODE" if parsed_args.patrol else "🛑 SINGLE TARGET MODE"
    node.get_logger().info(f"Goal set to: X={rc_goal[0]}m, Y={rc_goal[1]}m in the robot frame | {mode_text}")
    
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