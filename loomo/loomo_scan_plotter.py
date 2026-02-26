#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class LoomoScanPlotter(Node):
    def __init__(self):
        super().__init__('loomo_scan_plotter')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/loomo/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Loomo 2D LiDAR (Depth to Scan)")
        self.ax.set_xlabel("Y (meters - Left/Right)")
        self.ax.set_ylabel("X (meters - Front)")
        
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(0.0, 4.0)
        self.ax.grid(True)
        
        self.ax.plot(0, 0, 'ro', markersize=10, label='Loomo')
        self.ax.legend()
        
        self.scatter = self.ax.scatter([], [], s=10, c='blue')
        
        self.get_logger().info("📊 Plotter Matplotlib started. Waiting data on /loomo/scan...")

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        
        valid_indices = np.isfinite(ranges)
        valid_ranges = ranges[valid_indices]
        
        if len(valid_ranges) == 0:
            return 
            
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid_indices]
        
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        y = -y 
        
        self.scatter.set_offsets(np.c_[y, x])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    plotter = LoomoScanPlotter()
    
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plotter.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()