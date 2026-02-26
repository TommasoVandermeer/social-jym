#!/usr/bin/env python3
import sys
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
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

from socialjym.policies.jessi import JESSI

class JessiController(Node):
    def __init__(self, rc_goal, patrol_mode, network_name, save_file_name):
        super().__init__('jessi_controller')
        
        self.n_stack = 5 
        self.dt = 0.25 # 4 Hz
        self.radius = 0.3
        self.patrol = patrol_mode # Back and forth from initial position to goal
        self.save_file_name = save_file_name

        self.obs_stack = deque(maxlen=self.n_stack)
        self.recorded_data = []

        self.latest_scan = None
        self.latest_odom = None
        self.robot_goal = rc_goal
        self.initial_position = jnp.array([0.,0.]) # Odometry is reset at the beginning
        self.goal_reached = False
        
        #self.original_lidar_num_rays = 1081
        self.lidar_num_rays = 540
        self.lidar_min_angle = -jnp.pi
        self.lidar_max_angle = jnp.pi
        self.lidar_max_dist = 10
        self.angular_res = (float(self.lidar_max_angle) - float(self.lidar_min_angle)) / self.lidar_num_rays

        self.jessi = JESSI(
            v_max=0.3,
            wheels_distance=0.235,
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
        if self.reset_odom_client.wait_for_service(timeout_sec=3.0):
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
        self.sub_scan = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # ROS 2 Publisher
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', qos_cmd)
        
        # ROS 2 Timer
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.previous_action = jnp.array([0.,0.])

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

    def odom_callback(self, msg):
        self.latest_odom = msg

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if self.latest_scan is None or self.latest_odom is None or not self.odom_reset_confirmed:
            self.get_logger().warn("Waiting data from sensors...")
            return

        # Ranges cleaning
        ranges = np.array(self.latest_scan.ranges)
        cleaned = np.nan_to_num(ranges, nan=self.lidar_max_dist, posinf=self.lidar_max_dist, neginf=self.lidar_max_dist)
        cleaned[cleaned < 0.15] = self.lidar_max_dist
        cleaned = np.clip(cleaned, 0.0, self.lidar_max_dist)
        # Ranges shifting (first ray of TB4 is at -90, in JESSI first ray is at -self.lidar_angular_range/2)
        tb4_angle_min = self.latest_scan.angle_min
        tb4_angle_max = self.latest_scan.angle_max
        tb4_num_rays = len(cleaned)
        angular_res_tb4 = (tb4_angle_max - tb4_angle_min) / (tb4_num_rays - 1)
        jessi_angle_min = float(self.lidar_min_angle)
        shift_rad = tb4_angle_min - jessi_angle_min
        shift_bins = int(round(shift_rad / angular_res_tb4))
        shifted_cleaned = np.roll(cleaned, shift_bins)
        # Ranges resampling (from self.original_num_rays to self.lidar_num_rays)
        x_old = np.linspace(0, 1, tb4_num_rays)
        x_new = np.linspace(0, 1, self.lidar_num_rays)
        lidar_scan = np.interp(x_new, x_old, shifted_cleaned)

        # Odometry
        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        r_theta = self.get_yaw_from_quaternion(self.latest_odom.pose.pose.orientation)
        
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
                cmd_msg = Twist()
                cmd_msg.linear.x = v_cmd
                cmd_msg.angular.z = w_cmd
                self.pub_cmd.publish(cmd_msg)
                self.previous_action = jnp.array([v_cmd, w_cmd])
                self.recorded_data.append({
                    'observation': np.array(obs_matrix),
                    'robot_goal': np.array(self.robot_goal),
                    'action': np.array([v_cmd, w_cmd]),
                    'perception_distr': perception_output,
                    'actor_distr': actor_distr,
                })
            except Exception as e:
                self.get_logger().error(f"Error during JESSI inference: {e}")

    def save_data(self):
        if len(self.recorded_data) > 0:
            save_path = os.path.join(os.path.dirname(__file__), self.save_file_name)
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
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
    parser.add_argument('-n', '--network', type=str, default='jessi_finetuned_rl_out_turtlebot.pkl', help='Network weights pickle file name')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)
    rc_goal = np.array([parsed_args.goal_x, parsed_args.goal_y])

    rclpy.init(args=ros_args)
    
    node = JessiController(
        rc_goal=rc_goal, 
        patrol_mode=parsed_args.patrol,
        network_name=parsed_args.network,
        save_file_name=parsed_args.save_file
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()