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

class SquareWaveTracker(Node):
    def __init__(self, axis, amplitude, period, save_file_name):
        super().__init__('square_wave_tracker')
        self.start_time = time.time()
        self.dt = 0.02
        self.axis = axis
        self.amplitude = amplitude
        self.period = period
        self.save_file_name = save_file_name

        self.recorded_data = {'cmd': [], 'odom': []}
        self.odom_time_offset = None

        self.sub_odom = self.create_subscription(
            Odometry, 
            '/turtlebot1/odom', 
            self.odom_callback, 
            qos_profile_sensor_data
        )
        
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_cmd = self.create_publisher(Twist, '/turtlebot1/cmd_vel', qos_cmd)
        self.pub_cmd_stamped = self.create_publisher(TwistStamped, '/turtlebot1/cmd_vel_stamped', qos_cmd)
        
        self.timer = self.create_timer(self.dt, self.control_loop)

    def odom_callback(self, msg):
        raw_odom_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.odom_time_offset is None:
            local_t = self.get_clock().now().nanoseconds * 1e-9
            self.odom_time_offset = local_t - raw_odom_t
            print(f"Recorded odometry time offset: {self.odom_time_offset}")
            
        corrected_odom_t = raw_odom_t + self.odom_time_offset
        
        measured_v = msg.twist.twist.linear.x
        measured_w = msg.twist.twist.angular.z
        
        self.recorded_data['odom'].append({
            'timestamp': corrected_odom_t,
            'v': measured_v,
            'w': measured_w
        })

    def control_loop(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        phase = (elapsed_time % self.period) / self.period
        current_value = self.amplitude if phase < 0.5 else -self.amplitude
        
        cmd_msg = Twist()
        if self.axis == 'linear':
            cmd_msg.linear.x = current_value
        elif self.axis == 'angular':
            cmd_msg.angular.z = current_value
            
        self.pub_cmd.publish(cmd_msg)
        
        now = self.get_clock().now()
        cmd_stamped = TwistStamped()
        cmd_stamped.header.stamp = now.to_msg()
        cmd_stamped.twist = cmd_msg
        self.pub_cmd_stamped.publish(cmd_stamped)
        
        cmd_t = now.nanoseconds * 1e-9
        
        self.recorded_data['cmd'].append({
            'timestamp': cmd_t,
            'v': cmd_msg.linear.x,
            'w': cmd_msg.angular.z
        })

    def save_data(self):
        if len(self.recorded_data['cmd']) > 0 or len(self.recorded_data['odom']) > 0:
            save_file_name = self.save_file_name + f"_{self.axis}" + ".pkl"
            save_path = os.path.join(os.path.dirname(__file__), save_file_name)
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
            except Exception:
                pass

    def plot_data(self):
        if not self.recorded_data['cmd'] or not self.recorded_data['odom']:
            return

        cmd_t = [d['timestamp'] for d in self.recorded_data['cmd']]
        odom_t = [d['timestamp'] for d in self.recorded_data['odom']]
        
        t0 = min(cmd_t[0], odom_t[0])
        cmd_t = [t - t0 for t in cmd_t]
        odom_t = [t - t0 for t in odom_t]

        plt.figure(figsize=(10, 5))
        
        if self.axis == 'linear':
            cmd_val = [d['v'] for d in self.recorded_data['cmd']]
            odom_val = [d['v'] for d in self.recorded_data['odom']]
            ylabel = 'Linear Velocity [m/s]'
        else:
            cmd_val = [d['w'] for d in self.recorded_data['cmd']]
            odom_val = [d['w'] for d in self.recorded_data['odom']]
            ylabel = 'Angular Velocity [rad/s]'

        plt.step(cmd_t, cmd_val, label='Command', where='post', linewidth=2)
        plt.plot(odom_t, odom_val, label='Odometry', linewidth=2)
        
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)
        plt.title(f'Tracking Performance ({self.axis})')
        plt.grid(True)
        plt.legend()
        
        eps_filename = self.save_file_name + f"_{self.axis}" + '.eps'
        eps_path = os.path.join(os.path.dirname(__file__), eps_filename)
        
        try:
            plt.savefig(eps_path, format='eps', bbox_inches='tight')
        except Exception:
            pass
            
        plt.close()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', type=str, choices=['linear', 'angular'], default='linear')
    parser.add_argument('--amplitude', type=float, default=0.4)
    parser.add_argument('--period', type=float, default=1.0)
    parser.add_argument('--save_file', type=str, default='square_wave_data')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)

    rclpy.init(args=ros_args)
    
    node = SquareWaveTracker(
        axis=parsed_args.axis,
        amplitude=parsed_args.amplitude,
        period=parsed_args.period,
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