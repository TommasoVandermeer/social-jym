#!/usr/bin/env python3
import sys
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import os
import pickle
from jax import random, lax
from collections import deque
import math
import jax.numpy as jnp
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from socialjym.policies.jessi import JESSI

class JessiController(Node):
    def __init__(self, rc_goal, patrol_mode, network_name, save_file_name, use_virtual):
        super().__init__('jessi_controller')
        
        self.n_stack = 5
        self.dt = 0.25 # 4 Hz
        self.radius = 0.3
        self.v_max = .7
        self.wheels_distance = 0.7
        self.bounding_radius = 0.4
        self.patrol = patrol_mode # Back and forth from initial position to goal
        self.save_file_name = save_file_name
        
        self.obs_stack = deque(maxlen=self.n_stack)
        self.recorded_data = []

        self.latest_scan = None
        self.latest_odom = None
        self.robot_goal = rc_goal
        self.initial_position = jnp.array([0.,0.]) # Odometry is reset at the beginning
        self.goal_reached = False
        
        self.lidar_num_rays = 100
        self.lidar_min_angle = -0.46981275  
        self.lidar_max_angle = 0.46981275   
        self.lidar_max_dist = 4

        self.use_virtual = use_virtual
        self.num_virtual_points = 100
        self.jessi_virtual = JESSI(
            n_stack=self.n_stack,
            robot_radius = self.bounding_radius,
            v_max = self.v_max,
            wheels_distance=self.wheels_distance,
            lidar_num_rays=self.lidar_num_rays + self.num_virtual_points,
            lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
            lidar_max_dist=self.lidar_max_dist,
            n_stack_for_action_space_bounding=2
        )
        ## SIDE WALLS
        # num_points_per_wall = self.num_virtual_points // 2
        # x_coords = np.linspace(0.0, 10.0, num_points_per_wall)
        # y_left = np.full(num_points_per_wall, 0.5)
        # y_right = np.full(num_points_per_wall, -0.5)
        # left_wall = np.column_stack((x_coords, y_left))
        # right_wall = np.column_stack((x_coords, y_right))
        # self.virtual_points = jnp.array(np.vstack((left_wall, right_wall)))
        ## CIRCULAR WALL
        center_displacement = jnp.array([1.5,0.])
        radius = 2
        angles = jnp.linspace(-jnp.pi, jnp.pi, self.num_virtual_points)
        self.virtual_points = radius * jnp.column_stack((jnp.cos(angles), jnp.sin(angles))) + center_displacement

        self.jessi = JESSI(
            n_stack=self.n_stack,
            robot_radius = self.bounding_radius,
            v_max = self.v_max,
            wheels_distance=self.wheels_distance,
            lidar_num_rays=self.lidar_num_rays,
            lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
            lidar_max_dist=self.lidar_max_dist,
            n_stack_for_action_space_bounding=2
        )
        self.rng_key = random.PRNGKey(0)
        with open(os.path.join(os.path.dirname(__file__), network_name), 'rb') as f:
            self.network_params, _, _ = pickle.load(f)
        
        # ROS 2 Subscribers
        qos_realtime = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_scan = self.create_subscription(
            LaserScan, 
            '/loomo/scan', 
            self.scan_callback, 
            qos_realtime
        )
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/loomo/odom', 
            self.odom_callback, 
            qos_realtime
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        self.last_control_time = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info("JESSI Controller initialized at 4Hz!")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if self.latest_scan is None or self.latest_odom is None:
            self.get_logger().warn("Waiting data from sensors...")
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        print(f"Delta t: {current_time-self.last_control_time}") 
        self.last_control_time = current_time
        t_scan = self.latest_scan.header.stamp
        scan_time_sec = t_scan.sec + t_scan.nanosec * 1e-9
        t_odom = self.latest_odom.header.stamp
        odom_time_sec = t_odom.sec + t_odom.nanosec * 1e-9

        try:
            trans = self.tf_buffer.lookup_transform(
                'odom', 
                'base_link', 
                rclpy.time.Time(), 
                rclpy.duration.Duration(seconds=0.1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Impossible to sincronize odom and scan (desync): {e}", throttle_duration_sec=2.0)
            return
        rx = trans.transform.translation.x
        ry = trans.transform.translation.y
        r_theta = self.get_yaw_from_quaternion(trans.transform.rotation)

        print(f"Interp. Robot pose: {rx:.3f}, {ry:.3f}, {r_theta:.3f}")

        ranges = np.array(self.latest_scan.ranges)
        safe_ranges = np.where(np.isnan(ranges) | np.isinf(ranges), self.jessi.lidar_max_dist, ranges)
        indices = np.round(np.linspace(0, len(safe_ranges) - 1, self.lidar_num_rays)).astype(int)
        safe_ranges = safe_ranges[indices]

        curr_rx = self.latest_odom.pose.pose.position.x
        curr_ry = self.latest_odom.pose.pose.position.y
        curr_r_theta = self.get_yaw_from_quaternion(self.latest_odom.pose.pose.orientation)
        print(f"Robot pose:         {curr_rx:.3f}, {curr_ry:.3f}, {curr_r_theta:.3f}")

        current_step_obs = np.concatenate(([rx, ry, r_theta, self.radius, self.previous_action[0], self.previous_action[1]], [scan_time_sec], [odom_time_sec], [current_time], safe_ranges))
        self.obs_stack.appendleft(current_step_obs)
        while len(self.obs_stack) < self.n_stack:
            self.obs_stack.appendleft(current_step_obs) 
        obs_matrix = jnp.array(self.obs_stack) # Shape: (n_stack, 326)
        info_dict = {
            "robot_goal": jnp.array(self.robot_goal)
        }

        # Virtual points
        if self.use_virtual:
            v_points = jnp.array(self.virtual_points)
            dx = v_points[:, 0] - rx
            dy = v_points[:, 1] - ry
            cos_theta = math.cos(-r_theta)
            sin_theta = math.sin(-r_theta)
            raw_local_x = dx * cos_theta - dy * sin_theta
            raw_local_y = dx * sin_theta + dy * cos_theta
            virtual_dists = jnp.sqrt(raw_local_x**2 + raw_local_y**2)
            virtual_angles = jnp.arctan2(raw_local_y, raw_local_x)
            scale_factor = jnp.where(
                virtual_dists > self.lidar_max_dist, 
                self.lidar_max_dist / (virtual_dists + 1e-6), 
                1.0
            )
            virtual_dists = jnp.minimum(virtual_dists, self.lidar_max_dist)
            local_x = raw_local_x * scale_factor
            local_y = raw_local_y * scale_factor
            rc_virtual_points = jnp.column_stack((local_x, local_y))
            base_features = jnp.column_stack((
                virtual_dists / self.jessi.max_beam_range,            # norm_dist
                jnp.where(virtual_dists < self.jessi.lidar_max_dist, 1.0, 0.0), # hit
                local_x,                                              # x
                local_y,                                              # y
                jnp.sin(virtual_angles),                              # sin_theta
                jnp.cos(virtual_angles)                               # cos_theta
            ))
            delta_ts = jnp.arange(self.n_stack) * self.dt
            delta_t_matrix = jnp.broadcast_to(
                delta_ts[:, None, None], 
                (self.n_stack, len(local_x), 1)
            )
            base_features_broadcasted = jnp.broadcast_to(
                base_features[None, :, :], 
                (self.n_stack, len(local_x), 6)
            )
            virtual_tokens = jnp.concatenate([base_features_broadcasted, delta_t_matrix], axis=2)

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
                action, self.rng_key, _, _, _, _, perception_output, actor_distr, _, _, _, _ = self.jessi.act(
                    key=self.rng_key,
                    obs=obs_matrix,
                    info=info_dict,
                    e2e_network_params=self.network_params,
                    sample=False # Use mean action
                )
                # Compute encoder input and last lidar point cloud (for action bounding)
                # perception_input: lidar_tokens (n_stack, lidar_num_rays, 7): aligned LiDAR tokens for transformer encoder.
                # 7 features per token: [norm_dist, hit, x, y, sin_theta (theta of beam in the robot frame), cos_theta (theta of beam in the robot frame), delta_t (time difference from the most recent scan)].
                # Compute encoder input and last lidar point cloud (for action bounding)
                # perception_input, point_cloud_for_bounding = self.jessi.compute_perception_input(obs_matrix)
                # if self.use_virtual:
                #     perception_input = jnp.concatenate(
                #         [perception_input, virtual_tokens], 
                #         axis=1
                #     )
                #     repeated_virtual_points = jnp.tile(
                #         rc_virtual_points, 
                #         (self.jessi_virtual.n_stack_for_action_space_bounding, 1)
                #     )
                #     point_cloud_for_bounding = jnp.concatenate(
                #         [point_cloud_for_bounding, repeated_virtual_points], 
                #         axis=0
                #     )
                #     # Compute bounded action space parameters and add it to the input
                #     bounding_parameters = self.jessi_virtual.bound_action_space(
                #         point_cloud_for_bounding,  
                #     )
                # else:
                #     bounding_parameters = self.jessi.bound_action_space(
                #         point_cloud_for_bounding,  
                #     )
                # # Prepare input for network
                # robot_position = obs_matrix[0,:2]
                # robot_orientation = obs_matrix[0,2]
                # c, s = jnp.cos(-robot_orientation), jnp.sin(-robot_orientation)
                # R = jnp.array([[c, -s],
                #             [s,  c]])
                # translated_position = info_dict["robot_goal"] - robot_position
                # rc_robot_goal = R @ translated_position
                # robot_state_input = self.jessi.compute_robot_state_input(
                #     bounding_parameters,
                #     rc_robot_goal,
                # )
                # # Compute action
                # perception_output, _, _, actor_distr, _, _, _, _, _, _ = self.jessi.e2e.apply(
                #     self.network_params, 
                #     None, 
                #     perception_input,
                #     robot_state_input,
                #     random_key=self.rng_key
                # )
                # action = self.jessi.dirichlet.mean(actor_distr)

                v_cmd, w_cmd = float(action[0]), float(action[1])
                
                cmd_msg = Twist()
                cmd_msg.linear.x = v_cmd
                cmd_msg.angular.z = w_cmd
                self.pub_cmd.publish(cmd_msg)

                v_real = self.latest_odom.twist.twist.linear.x
                w_real = self.latest_odom.twist.twist.angular.z

                self.recorded_data.append({
                    'observation': np.array(obs_matrix),
                    'virtual_points': rc_virtual_points if self.use_virtual else 0.,
                    'robot_goal': np.array(self.robot_goal),
                    'action': np.array([v_cmd, w_cmd]),
                    'action_registered': np.array([v_real, w_real]),
                    'perception_distr': perception_output,
                    'actor_distr': actor_distr,
                })

                self.previous_action = jnp.array([v_cmd, w_cmd])
                
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
    parser.add_argument('--virtual', action='store_true', help='Use virtual points as well')
    parser.add_argument('-n', '--network', type=str, default='jessi_finetuned_rl_out.pkl', help='Network weights pickle file name')
    parser.add_argument('-s', '--save_file', type=str, default='jessi_recorded_obs.pkl', help='Output pickle file name for recorded data')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)
    rc_goal = np.array([parsed_args.goal_x, parsed_args.goal_y])

    rclpy.init(args=ros_args)
    
    node = JessiController(
        rc_goal=rc_goal,
        patrol_mode=parsed_args.patrol,
        network_name=parsed_args.network,
        save_file_name=parsed_args.save_file,
        use_virtual=parsed_args.virtual,
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