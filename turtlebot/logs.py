import pickle
import numpy as np
import math
import os

def get_yaw_from_q(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

filename = "lists_jessi_recorded_obs.pkl"
with open(os.path.join(os.path.dirname(__file__), filename), 'rb') as f:
    data = pickle.load(f)

scans = data['scan']
odoms = data['odom']

print("--- TIMESTAMPS OVERVIEW ---")
print(f"First Scan T: {scans[0].header.stamp.sec + scans[0].header.stamp.nanosec*1e-9:.3f}")
print(f"First Odom T: {odoms[0].header.stamp.sec + odoms[0].header.stamp.nanosec*1e-9:.3f}")

odom_t = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in odoms])
odom_x = np.array([m.pose.pose.position.x for m in odoms])
odom_y = np.array([m.pose.pose.position.y for m in odoms])
odom_th = np.unwrap([get_yaw_from_q(m.pose.pose.orientation) for m in odoms])

print("\n--- ODOMETRY INTERPOLATION CHECK (First 5 Scans) ---")
previous_t = None
for i in range(5):
    t_scan = scans[i].header.stamp.sec + scans[i].header.stamp.nanosec * 1e-9
    rx = np.interp(t_scan, odom_t, odom_x)
    ry = np.interp(t_scan, odom_t, odom_y)
    r_th = np.interp(t_scan, odom_t, odom_th)
    dt = t_scan - previous_t if previous_t is not None else 0.0
    previous_t = t_scan
    
    print(f"Scan {i}: dt={dt:.3f}s | rx={rx:.3f}, ry={ry:.3f}, th={r_th:.3f} rad")