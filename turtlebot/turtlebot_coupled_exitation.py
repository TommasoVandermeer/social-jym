#!/usr/bin/env python3
import sys
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import os
import pickle
import matplotlib.pyplot as plt

class CoupledExcitationTracker(Node):
    def __init__(self, step_duration, save_file_name):
        super().__init__('coupled_excitation_tracker')
        self.start_time = time.time()
        self.dt = 0.05
        self.step_duration = step_duration
        self.save_file_name = save_file_name

        self.recorded_data = {'cmd': [], 'odom': []}
        self.odom_time_offset = None

        self.sequence = [
            (0.4, 0.0),    # 1. Pure Forward
            (-0.4, 0.0),   # 2. Pure Backward
            (0.0, 2.0),    # 3. Pure Rotation Left
            (0.0, -2.0),   # 4. Pure Rotation Right
            (0.3, 1.5),    # 5. Coupled Forward-Left (Right Wheel Max)
            (-0.3, -1.5),  # 6. Coupled Backward-Right
            (0.3, -1.5),   # 7. Coupled Forward-Right (Left Wheel Max)
            (-0.3, 1.5)    # 8. Coupled Backward-Left
        ]
        self.total_phases = len(self.sequence)

        self.sub_odom = self.create_subscription(
            Odometry, 
            '/turtlebot1/odom', 
            self.odom_callback, 
            qos_profile_sensor_data
        )
        
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_cmd = self.create_publisher(Twist, '/turtlebot1/cmd_vel', qos_cmd)
        self.pub_cmd_stamped = self.create_publisher(TwistStamped, '/turtlebot1/cmd_vel_stamped', qos_cmd)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"Starting Coupled Excitation... {self.total_phases} phases of {self.step_duration}s each.")

    def odom_callback(self, msg):
        raw_odom_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.odom_time_offset is None:
            local_t = self.get_clock().now().nanoseconds * 1e-9
            self.odom_time_offset = local_t - raw_odom_t
            
        corrected_odom_t = raw_odom_t + self.odom_time_offset
        
        self.recorded_data['odom'].append({
            'timestamp': corrected_odom_t,
            'v': msg.twist.twist.linear.x,
            'w': msg.twist.twist.angular.z
        })

    def control_loop(self):
        elapsed_time = time.time() - self.start_time
        
        # Calcola quale fase della sequenza stiamo eseguendo
        current_phase_idx = int((elapsed_time / self.step_duration) % self.total_phases)
        target_v, target_w = self.sequence[current_phase_idx]
        
        cmd_msg = Twist()
        cmd_msg.linear.x = target_v
        cmd_msg.angular.z = target_w
            
        self.pub_cmd.publish(cmd_msg)
        
        now = self.get_clock().now()
        cmd_stamped = TwistStamped()
        cmd_stamped.header.stamp = now.to_msg()
        cmd_stamped.twist = cmd_msg
        self.pub_cmd_stamped.publish(cmd_stamped)
        
        self.recorded_data['cmd'].append({
            'timestamp': now.nanoseconds * 1e-9,
            'v': cmd_msg.linear.x,
            'w': cmd_msg.angular.z
        })

    def save_data(self):
        if len(self.recorded_data['cmd']) > 0 or len(self.recorded_data['odom']) > 0:
            save_path = os.path.join(os.path.dirname(__file__), self.save_file_name)
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
                self.get_logger().info(f"Data saved to {save_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save data: {e}")

    def plot_data(self):
        if not self.recorded_data['cmd'] or not self.recorded_data['odom']:
            return

        cmd_t = [d['timestamp'] for d in self.recorded_data['cmd']]
        odom_t = [d['timestamp'] for d in self.recorded_data['odom']]
        
        t0 = min(cmd_t[0], odom_t[0])
        cmd_t = [t - t0 for t in cmd_t]
        odom_t = [t - t0 for t in odom_t]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Linear Subplot
        ax1.step(cmd_t, [d['v'] for d in self.recorded_data['cmd']], label='Cmd V', where='post', color='black', linestyle='--')
        ax1.plot(odom_t, [d['v'] for d in self.recorded_data['odom']], label='Odom V', color='blue', linewidth=2)
        ax1.set_ylabel('Linear Vel [m/s]')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Coupled Excitation Trajectory')

        # Angular Subplot
        ax2.step(cmd_t, [d['w'] for d in self.recorded_data['cmd']], label='Cmd W', where='post', color='black', linestyle='--')
        ax2.plot(odom_t, [d['w'] for d in self.recorded_data['odom']], label='Odom W', color='red', linewidth=2)
        ax2.set_ylabel('Angular Vel [rad/s]')
        ax2.set_xlabel('Time [s]')
        ax2.legend()
        ax2.grid(True)

        eps_filename = os.path.splitext(self.save_file_name)[0] + '.eps'
        eps_path = os.path.join(os.path.dirname(__file__), eps_filename)
        try:
            plt.savefig(eps_path, format='eps', bbox_inches='tight')
        except Exception:
            pass
        plt.close()

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate a coupled excitation trajectory for system identification")
    parser.add_argument('--step_duration', type=float, default=5.0, help="Duration of each phase in seconds")
    parser.add_argument('--save_file', type=str, default='coupled_excitation_data.pkl')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)

    rclpy.init(args=ros_args)
    
    node = CoupledExcitationTracker(
        step_duration=parsed_args.step_duration,
        save_file_name=parsed_args.save_file
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.pub_cmd.publish(stop)
    finally:
        node.save_data()
        node.plot_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()