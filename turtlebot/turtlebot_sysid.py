#!/usr/bin/env python3
"""
TurtleBot4 System Identification Script
========================================
Three independent modes covering:
  lidar    - Section 1: Static LiDAR noise + dropout acquisition
  velocity - Section 3: Velocity step-response (low-level controller dynamics)
  latency  - Section 4: Actuator dead-time + differential drive drift

Usage examples:
  python3 turtlebot_sysid.py lidar    -n 1000 -s sysid_lidar.pkl
  python3 turtlebot_sysid.py velocity -v 0.5  -s sysid_velocity.pkl
  python3 turtlebot_sysid.py latency  -v 0.5 -d 5.0 -s sysid_latency.pkl
"""
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
import time
import math
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_cmd_pub(node):
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    return node.create_publisher(Twist, '/cmd_vel', qos)


def publish_vel(pub, v=0.0, w=0.0):
    msg = Twist()
    msg.linear.x = float(v)
    msg.angular.z = float(w)
    pub.publish(msg)


def reset_odom(node, client):
    """Send async odometry reset. Returns the future."""
    req = ResetPose.Request()
    req.pose.orientation.w = 1.0
    return client.call_async(req)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — LiDAR Static Noise Acquisition  (Section 1)
# ─────────────────────────────────────────────────────────────────────────────

class LidarSysIdNode(Node):
    """
    Acquires `n_scans` full-resolution LiDAR scans while the robot is completely
    static (motors off / wheels on blocks).  Stores:
      - Full ranges array (all rays at native resolution)
      - Scan metadata (angle_min/max, angle_increment, range_min/max)
      - ROS and wall timestamps for temporal analysis
      - Intensity channel if available

    Offline analysis (not done here):
      1. Per-ray mean µ_i and std σ_i  →  linear noise model σ(d) ≈ a·d + b
      2. NaN/inf dropout sequences  →  Markov chain p_start, p_continue
    """

    def __init__(self, n_scans: int, save_file: str):
        super().__init__('lidar_sysid')
        self.n_scans = n_scans
        self.save_file = save_file
        self._scans: list[dict] = []

        self._sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f"[LiDAR SysId] Waiting for {n_scans} scans on /scan …\n"
            f"  → Make sure the robot is STATIC and motors are OFF.")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        if len(self._scans) >= self.n_scans:
            return

        self._scans.append({
            # timing
            'ros_time_ns':   self.get_clock().now().nanoseconds,
            'wall_time':     time.monotonic(),
            'header_stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            # scan geometry
            'angle_min':       msg.angle_min,
            'angle_max':       msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment':  msg.time_increment,
            'scan_time':       msg.scan_time,
            'range_min':       msg.range_min,
            'range_max':       msg.range_max,
            # data
            'ranges':      np.array(msg.ranges,      dtype=np.float32),
            'intensities': np.array(msg.intensities, dtype=np.float32)
                           if len(msg.intensities) > 0 else np.array([], dtype=np.float32),
        })

        n = len(self._scans)
        if n % 100 == 0 or n == self.n_scans:
            self.get_logger().info(f"  Collected {n}/{self.n_scans} scans")

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return len(self._scans) >= self.n_scans

    def save(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_file)
        payload = {
            'mode':    'lidar_sysid',
            'n_scans': self.n_scans,
            'scans':   self._scans,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        self.get_logger().info(
            f"[LiDAR SysId] Saved {len(self._scans)} scans → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Velocity Step-Response  (Section 3)
# ─────────────────────────────────────────────────────────────────────────────

class VelocitySysIdNode(Node):
    """
    Sends a step command (0 → v_target), records high-rate odometry for `hold_time`
    seconds, then returns to 0 for `rest_time` seconds.  Repeats `n_steps` times.

    Saved data per trial:
      - Command issue wall/ROS timestamp
      - All (t, vx, wz, px, py, yaw) odom samples during rest + step phases

    Offline analysis:
      - Fit first-order system to vx(t) after step onset → time constant τ
    """

    # State machine phases
    _PHASE_WAIT  = 'wait_reset'
    _PHASE_REST  = 'rest'
    _PHASE_STEP  = 'step'
    _PHASE_DONE  = 'done'

    def __init__(self, v_target: float, hold_time: float,
                 rest_time: float, n_steps: int, save_file: str):
        super().__init__('velocity_sysid')
        self.v_target  = v_target
        self.hold_time = hold_time
        self.rest_time = rest_time
        self.n_steps   = n_steps
        self.save_file = save_file

        self._odom_buffer: list[dict] = []
        self._trials:      list[dict] = []
        self._step_idx   = 0
        self._phase      = self._PHASE_WAIT
        self._phase_t0   = 0.0

        self.pub_cmd = make_cmd_pub(self)
        self._sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self._reset_client = self.create_client(ResetPose, '/reset_pose')
        if self._reset_client.wait_for_service(timeout_sec=3.0):
            future = reset_odom(self, self._reset_client)
            future.add_done_callback(self._reset_cb)
        else:
            self.get_logger().warn("Reset service not found – skipping odom reset.")
            self._transition_rest()

        # 100 Hz control loop for precise timing
        self._timer = self.create_timer(0.01, self._loop)

        self.get_logger().info(
            f"[Velocity SysId] {n_steps} steps | v={v_target} m/s | "
            f"hold={hold_time}s | rest={rest_time}s")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _reset_cb(self, future):
        try:
            future.result()
            self.get_logger().info("Odometry reset OK.")
        except Exception as e:
            self.get_logger().warn(f"Odom reset failed: {e}")
        self._transition_rest()

    def _odom_cb(self, msg: Odometry):
        self._odom_buffer.append({
            'ros_time_ns':     self.get_clock().now().nanoseconds,
            'wall_time':       time.monotonic(),
            'header_stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            'vx':  msg.twist.twist.linear.x,
            'vy':  msg.twist.twist.linear.y,
            'wz':  msg.twist.twist.angular.z,
            'px':  msg.pose.pose.position.x,
            'py':  msg.pose.pose.position.y,
            'yaw': quaternion_to_yaw(msg.pose.pose.orientation),
            'phase':     self._phase,
            'step_idx':  self._step_idx,
        })

    def _loop(self):
        if self._phase in (self._PHASE_WAIT, self._PHASE_DONE):
            return
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == self._PHASE_REST:
            publish_vel(self.pub_cmd, v=0.0)   # keep sending 0 to satisfy watchdog
            if elapsed >= self.rest_time:
                self._transition_step()

        elif self._phase == self._PHASE_STEP:
            if elapsed >= self.hold_time:
                publish_vel(self.pub_cmd, v=0.0)
                self._step_idx += 1
                if self._step_idx >= self.n_steps:
                    self._phase = self._PHASE_DONE
                    self.get_logger().info("[Velocity SysId] All steps complete.")
                else:
                    self._transition_rest()
            else:
                publish_vel(self.pub_cmd, v=self.v_target)  # keep sending v_target to satisfy watchdog

    # ── transitions ────────────────────────────────────────────────────────

    def _transition_rest(self):
        publish_vel(self.pub_cmd, v=0.0)
        self._phase    = self._PHASE_REST
        self._phase_t0 = time.monotonic()
        self.get_logger().info(
            f"  Step {self._step_idx + 1}/{self.n_steps}: resting {self.rest_time}s …")

    def _transition_step(self):
        t_wall = time.monotonic()
        t_ros  = self.get_clock().now().nanoseconds
        publish_vel(self.pub_cmd, v=self.v_target)
        self._phase    = self._PHASE_STEP
        self._phase_t0 = t_wall
        self._trials.append({
            'step_idx':      self._step_idx,
            'cmd_wall_time': t_wall,
            'cmd_ros_ns':    t_ros,
            'v_cmd':         self.v_target,
        })
        self.get_logger().info(
            f"  Step {self._step_idx + 1}/{self.n_steps}: v={self.v_target} m/s for {self.hold_time}s")

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._phase == self._PHASE_DONE

    def save(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_file)
        payload = {
            'mode':        'velocity_sysid',
            'v_target':    self.v_target,
            'hold_time':   self.hold_time,
            'rest_time':   self.rest_time,
            'n_steps':     self.n_steps,
            'trials':      self._trials,
            'odom_buffer': self._odom_buffer,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        self.get_logger().info(
            f"[Velocity SysId] Saved {len(self._odom_buffer)} odom samples, "
            f"{len(self._trials)} trials → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Actuator Latency + Differential Drive Drift  (Section 4)
# ─────────────────────────────────────────────────────────────────────────────

class LatencyDriftSysIdNode(Node):
    """
    Two sequential sub-tests:

    A) Dead-time (n_lat_trials repetitions):
       - Robot at rest → publish step command → record odom at ≥200 Hz
       - Detect first odom velocity crossing VX_THRESHOLD after command
       - Tdelay = t_first_response − t_cmd

    B) Differential drift (straight-line run):
       - Command v=v_cruise, ω=0 for dist_target metres
       - Record full odometry trajectory
       - Lateral deviation = final |y|  (straight-line = x-axis after odom reset)

    Saved data:
      - All odom readings (high-rate) with phase/trial annotation
      - Per-trial latency estimates
    """

    _PHASE_WAIT       = 'wait_reset'
    _PHASE_LAT_REST   = 'lat_rest'
    _PHASE_LAT_STEP   = 'lat_step'
    _PHASE_PRE_DRIFT  = 'pre_drift'    # short pause + odom reset before drift
    _PHASE_DRIFT      = 'drift'
    _PHASE_DONE       = 'done'

    # Velocity threshold to detect first encoder response [m/s]
    VX_THRESHOLD = 0.02

    # Pause between latency trials [s]
    LAT_REST_DUR = 1.5
    # Duration of each step command in latency trials [s]
    LAT_STEP_DUR = 1.0
    # Pre-drift pause after last latency trial [s]
    PRE_DRIFT_DUR = 2.0

    def __init__(self, v_cruise: float, dist_target: float,
                 n_lat_trials: int, save_file: str):
        super().__init__('latency_drift_sysid')
        self.v_cruise      = v_cruise
        self.dist_target   = dist_target
        self.n_lat_trials  = n_lat_trials
        self.save_file     = save_file

        self._odom_buffer:    list[dict] = []
        self._lat_trials:     list[dict] = []
        self._lat_trial_idx  = 0
        self._phase          = self._PHASE_WAIT
        self._phase_t0       = 0.0
        self._cmd_wall_time  = None   # set when step command is issued
        self._response_found = False  # prevent multi-trigger per trial
        self._drift_start_px = None
        self._drift_start_py = None

        self.pub_cmd = make_cmd_pub(self)
        self._sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self._reset_client = self.create_client(ResetPose, '/reset_pose')
        if self._reset_client.wait_for_service(timeout_sec=3.0):
            future = reset_odom(self, self._reset_client)
            future.add_done_callback(self._initial_reset_cb)
        else:
            self.get_logger().warn("Reset service not found – skipping odom reset.")
            self._start_lat_rest()

        # 200 Hz timer for precise latency measurement
        self._timer = self.create_timer(0.005, self._loop)

        self.get_logger().info(
            f"[Latency+Drift SysId] {n_lat_trials} latency trials | "
            f"drift: v={v_cruise} m/s for {dist_target} m")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _initial_reset_cb(self, future):
        try:
            future.result()
            self.get_logger().info("Odometry reset OK.")
        except Exception as e:
            self.get_logger().warn(f"Odom reset failed: {e}")
        self._start_lat_rest()

    def _pre_drift_reset_cb(self, future):
        try:
            future.result()
            self.get_logger().info("Pre-drift odom reset OK.")
        except Exception as e:
            self.get_logger().warn(f"Pre-drift odom reset failed: {e}")

    def _odom_cb(self, msg: Odometry):
        now_wall = time.monotonic()
        now_ros  = self.get_clock().now().nanoseconds

        entry = {
            'ros_time_ns':     now_ros,
            'wall_time':       now_wall,
            'header_stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            'vx':  msg.twist.twist.linear.x,
            'vy':  msg.twist.twist.linear.y,
            'wz':  msg.twist.twist.angular.z,
            'px':  msg.pose.pose.position.x,
            'py':  msg.pose.pose.position.y,
            'yaw': quaternion_to_yaw(msg.pose.pose.orientation),
            'phase':     self._phase,
            'lat_trial': self._lat_trial_idx,
        }
        self._odom_buffer.append(entry)

        # ── Latency detection ─────────────────────────────────────────────
        # First odom sample after step command where |vx| crosses the threshold
        if (self._phase == self._PHASE_LAT_STEP
                and not self._response_found
                and self._cmd_wall_time is not None
                and abs(entry['vx']) >= self.VX_THRESHOLD):
            delay_s = now_wall - self._cmd_wall_time
            self._lat_trials[-1].update({
                'detected_delay_s':       delay_s,
                'first_response_ros_ns':  now_ros,
                'first_response_wall':    now_wall,
                'first_vx':               entry['vx'],
            })
            self._response_found = True
            self.get_logger().info(
                f"  Trial {self._lat_trial_idx + 1}: Tdelay ≈ {delay_s * 1000:.1f} ms "
                f"(first |vx| = {entry['vx']:.3f} m/s)")

    def _loop(self):
        if self._phase in (self._PHASE_WAIT, self._PHASE_DONE):
            return

        elapsed = time.monotonic() - self._phase_t0

        # ── Latency rest ─────────────────────────────────────────────────
        if self._phase == self._PHASE_LAT_REST:
            publish_vel(self.pub_cmd, v=0.0)   # keep sending 0 to satisfy watchdog
            if elapsed >= self.LAT_REST_DUR:
                self._start_lat_step()

        # ── Latency step ─────────────────────────────────────────────────
        elif self._phase == self._PHASE_LAT_STEP:
            if elapsed >= self.LAT_STEP_DUR:
                publish_vel(self.pub_cmd, v=0.0)
                self._lat_trial_idx += 1
                if self._lat_trial_idx >= self.n_lat_trials:
                    self.get_logger().info(
                        "[Latency SysId] All latency trials done. "
                        f"Pausing {self.PRE_DRIFT_DUR}s before drift test …")
                    self._phase    = self._PHASE_PRE_DRIFT
                    self._phase_t0 = time.monotonic()
                    # Issue async odom reset for drift baseline
                    if self._reset_client.service_is_ready():
                        future = reset_odom(self, self._reset_client)
                        future.add_done_callback(self._pre_drift_reset_cb)
                else:
                    self._start_lat_rest()
            else:
                publish_vel(self.pub_cmd, v=self.v_cruise)  # keep sending v_cruise to satisfy watchdog

        # ── Pre-drift pause ───────────────────────────────────────────────
        elif self._phase == self._PHASE_PRE_DRIFT:
            publish_vel(self.pub_cmd, v=0.0)   # keep sending 0 to satisfy watchdog
            if elapsed >= self.PRE_DRIFT_DUR:
                self._start_drift()

        # ── Drift run ─────────────────────────────────────────────────────
        elif self._phase == self._PHASE_DRIFT:
            if not self._odom_buffer:
                return
            last = self._odom_buffer[-1]
            if self._drift_start_px is None:
                # Latch starting position on first sample after drift begins
                self._drift_start_px = last['px']
                self._drift_start_py = last['py']
                publish_vel(self.pub_cmd, v=self.v_cruise)
                return
            dist = math.hypot(
                last['px'] - self._drift_start_px,
                last['py'] - self._drift_start_py,
            )
            if dist >= self.dist_target:
                publish_vel(self.pub_cmd, v=0.0)
                self._phase = self._PHASE_DONE
                # Express deviation relative to straight-line direction (initial yaw)
                # After odom reset the robot starts at (0,0) heading along x-axis,
                # so lateral deviation is simply the final y component.
                lateral_m = last['py'] - self._drift_start_py
                self.get_logger().info(
                    f"[Drift SysId] Done. Distance: {dist:.3f} m | "
                    f"Lateral deviation: {lateral_m * 100:.1f} cm")
            else:
                publish_vel(self.pub_cmd, v=self.v_cruise)  # keep sending v_cruise to satisfy watchdog

    # ── transitions ────────────────────────────────────────────────────────

    def _start_lat_rest(self):
        publish_vel(self.pub_cmd, v=0.0)
        self._phase    = self._PHASE_LAT_REST
        self._phase_t0 = time.monotonic()

    def _start_lat_step(self):
        t_wall = time.monotonic()
        t_ros  = self.get_clock().now().nanoseconds
        publish_vel(self.pub_cmd, v=self.v_cruise)
        self._phase          = self._PHASE_LAT_STEP
        self._phase_t0       = t_wall
        self._cmd_wall_time  = t_wall
        self._response_found = False
        self._lat_trials.append({
            'trial_idx':       self._lat_trial_idx,
            'cmd_wall_time':   t_wall,
            'cmd_ros_ns':      t_ros,
            'v_cmd':           self.v_cruise,
            'detected_delay_s':       None,
            'first_response_ros_ns':  None,
            'first_response_wall':    None,
            'first_vx':               None,
        })
        self.get_logger().info(
            f"  Latency trial {self._lat_trial_idx + 1}/{self.n_lat_trials}: "
            f"step v={self.v_cruise} m/s")

    def _start_drift(self):
        publish_vel(self.pub_cmd, v=self.v_cruise)
        self._phase          = self._PHASE_DRIFT
        self._phase_t0       = time.monotonic()
        self._drift_start_px = None
        self._drift_start_py = None
        self.get_logger().info(
            f"[Drift SysId] Starting: v={self.v_cruise} m/s for {self.dist_target} m …")

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._phase == self._PHASE_DONE

    def save(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_file)
        payload = {
            'mode':          'latency_drift_sysid',
            'v_cruise':      self.v_cruise,
            'dist_target':   self.dist_target,
            'n_lat_trials':  self.n_lat_trials,
            'vx_threshold':  self.VX_THRESHOLD,
            'lat_trials':    self._lat_trials,
            'odom_buffer':   self._odom_buffer,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        self.get_logger().info(
            f"[Latency+Drift SysId] Saved {len(self._odom_buffer)} odom samples, "
            f"{len(self._lat_trials)} latency trials → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 — Velocity Staircase Tracking  (Section 5)
# ─────────────────────────────────────────────────────────────────────────────

def _build_staircase_sequence(seg_strings: list[str]) -> list[tuple[float, float, float]]:
    """
    Parse a list of segment strings into a flat sequence of
    (v_cmd, w_cmd, hold_s) triples.

    Segment formats (order on command line is preserved):
      lin:v_start:v_end:v_step:hold_s   — linear ramp,  ω = 0
      ang:w_start:w_end:w_step:hold_s   — angular ramp, v = 0
      v_start:v_end:v_step:hold_s       — legacy, treated as lin:...

    Consecutive duplicate (v, w) pairs at segment boundaries are merged.

    Examples:
        ["lin:0:0.3:0.05:0.5", "lin:0.3:0:0.1:0.5"]
        ["ang:0:0.5:0.1:0.5", "ang:0.5:0:0.1:0.5"]
        ["lin:0:0.3:0.05:0.5", "lin:0.3:0:0.1:0.5",
         "ang:0:0.5:0.1:0.5", "ang:0.5:0:0.1:0.5"]
    """
    def _ramp(start: float, end: float, step: float) -> list[float]:
        direction = 1.0 if end >= start else -1.0
        n = round(abs(end - start) / step) + 1
        return [round(start + direction * step * i, 6) for i in range(n)]

    sequence: list[tuple[float, float, float]] = []

    for seg_str in seg_strings:
        parts = seg_str.split(':')

        # Detect prefix
        if parts[0] in ('lin', 'ang'):
            kind   = parts[0]
            values = parts[1:]
        else:
            kind   = 'lin'
            values = parts

        if len(values) != 4:
            raise ValueError(
                f"Segment must be '[lin|ang:]start:end:step:hold_s', got: {seg_str!r}")
        start, end, step, hold = (float(p) for p in values)
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")

        for level in _ramp(start, end, step):
            v = level if kind == 'lin' else 0.0
            w = level if kind == 'ang' else 0.0
            # Collapse exact duplicate at boundary with previous step
            if sequence and abs(sequence[-1][0] - v) < 1e-9 and abs(sequence[-1][1] - w) < 1e-9:
                continue
            sequence.append((v, w, hold))

    return sequence


class StaircaseSysIdNode(Node):
    """
    Commands a pre-computed staircase profile and records odometry.

    The profile is a flat sequence of (v_cmd, w_cmd, hold_s) triples built
    from one or more segments (lin: or ang: prefixes).  Each level is held
    for hold_s seconds with the command republished at 100 Hz (Create3 watchdog).

    Saved data:
      - sequence:    list of (v_cmd, w_cmd, hold_s) triples actually commanded
      - odom_buffer: all odom samples with 'v_cmd', 'w_cmd', 'step_idx' annotations
    """

    _PHASE_WAIT = 'wait_reset'
    _PHASE_RUN  = 'run'
    _PHASE_DONE = 'done'

    def __init__(self, sequence: list[tuple[float, float, float]], save_file: str):
        super().__init__('staircase_sysid')
        self._sequence  = sequence
        self.save_file  = save_file
        self._step_idx  = 0
        self._phase     = self._PHASE_WAIT
        self._phase_t0  = 0.0
        self._odom_buffer: list[dict] = []

        self.pub_cmd = make_cmd_pub(self)
        self._sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self._reset_client = self.create_client(ResetPose, '/reset_pose')
        if self._reset_client.wait_for_service(timeout_sec=3.0):
            future = reset_odom(self, self._reset_client)
            future.add_done_callback(self._reset_cb)
        else:
            self.get_logger().warn("Reset service not found – skipping odom reset.")
            self._start()

        self._timer = self.create_timer(0.01, self._loop)   # 100 Hz

        total_t = sum(h for _, _, h in sequence)
        v_vals  = [v for v, _, _ in sequence]
        w_vals  = [w for _, w, _ in sequence]
        self.get_logger().info(
            f"[Staircase SysId] {len(sequence)} levels | "
            f"v ∈ [{min(v_vals):.2f}, {max(v_vals):.2f}] m/s | "
            f"ω ∈ [{min(w_vals):.2f}, {max(w_vals):.2f}] rad/s | "
            f"total ≈ {total_t:.1f} s")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _reset_cb(self, future):
        try:
            future.result()
            self.get_logger().info("Odometry reset OK.")
        except Exception as e:
            self.get_logger().warn(f"Odom reset failed: {e}")
        self._start()

    def _odom_cb(self, msg: Odometry):
        if self._step_idx < len(self._sequence):
            v_cmd, w_cmd, _ = self._sequence[self._step_idx]
        else:
            v_cmd, w_cmd = 0.0, 0.0
        self._odom_buffer.append({
            'ros_time_ns':     self.get_clock().now().nanoseconds,
            'wall_time':       time.monotonic(),
            'header_stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            'vx':  msg.twist.twist.linear.x,
            'vy':  msg.twist.twist.linear.y,
            'wz':  msg.twist.twist.angular.z,
            'px':  msg.pose.pose.position.x,
            'py':  msg.pose.pose.position.y,
            'yaw': quaternion_to_yaw(msg.pose.pose.orientation),
            'v_cmd':    v_cmd,
            'w_cmd':    w_cmd,
            'step_idx': self._step_idx,
            'phase':    self._phase,
        })

    def _loop(self):
        if self._phase in (self._PHASE_WAIT, self._PHASE_DONE):
            return
        elapsed = time.monotonic() - self._phase_t0
        v_cmd, w_cmd, hold = self._sequence[self._step_idx]

        if elapsed >= hold:
            self._step_idx += 1
            if self._step_idx >= len(self._sequence):
                publish_vel(self.pub_cmd, v=0.0, w=0.0)
                self._phase = self._PHASE_DONE
                self.get_logger().info("[Staircase SysId] All levels complete.")
            else:
                self._phase_t0 = time.monotonic()
                v_next, w_next, _ = self._sequence[self._step_idx]
                publish_vel(self.pub_cmd, v=v_next, w=w_next)
                self.get_logger().info(
                    f"  Level {self._step_idx + 1}/{len(self._sequence)}: "
                    f"v={v_next:.3f} m/s  ω={w_next:.3f} rad/s")
        else:
            publish_vel(self.pub_cmd, v=v_cmd, w=w_cmd)  # keep sending to satisfy watchdog

    # ── helpers ────────────────────────────────────────────────────────────

    def _start(self):
        self._phase    = self._PHASE_RUN
        self._phase_t0 = time.monotonic()
        v0, w0, _ = self._sequence[0]
        publish_vel(self.pub_cmd, v=v0, w=w0)
        self.get_logger().info(
            f"  Level 1/{len(self._sequence)}: v={v0:.3f} m/s  ω={w0:.3f} rad/s")

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._phase == self._PHASE_DONE

    def save(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_file)
        payload = {
            'mode':        'staircase_sysid',
            'sequence':    self._sequence,
            'odom_buffer': self._odom_buffer,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        self.get_logger().info(
            f"[Staircase SysId] Saved {len(self._odom_buffer)} odom samples → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TurtleBot4 System Identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  lidar      Section 1 — Static LiDAR noise + dropout (robot MUST be stationary)
  velocity   Section 3 — Velocity step-response (identifies low-level τ)
  latency    Section 4 — Actuator dead-time + differential drive drift
  staircase  Section 5 — Multi-level velocity tracking (staircase profile)
        """,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── lidar ─────────────────────────────────────────────────────────────
    p_lidar = sub.add_parser('lidar', help='LiDAR static noise acquisition (Section 1)')
    p_lidar.add_argument('-n', '--n_scans', type=int, default=1000,
                         help='Number of scans to collect (default: 1000)')
    p_lidar.add_argument('-s', '--save_file', type=str, default='sysid_lidar.pkl',
                         help='Output pickle filename')

    # ── velocity ──────────────────────────────────────────────────────────
    p_vel = sub.add_parser('velocity', help='Velocity step-response (Section 3)')
    p_vel.add_argument('-v', '--v_target', type=float, default=0.3,
                       help='Step target velocity in m/s (default: 0.3)')
    p_vel.add_argument('--hold', type=float, default=4.0,
                       help='Seconds at v_target per step (default: 4.0)')
    p_vel.add_argument('--rest', type=float, default=2.0,
                       help='Seconds at rest between steps (default: 2.0)')
    p_vel.add_argument('-n', '--n_steps', type=int, default=5,
                       help='Number of step repetitions (default: 5)')
    p_vel.add_argument('-s', '--save_file', type=str, default='sysid_velocity.pkl',
                       help='Output pickle filename')

    # ── latency ───────────────────────────────────────────────────────────
    p_lat = sub.add_parser('latency', help='Actuator latency + differential drift (Section 4)')
    p_lat.add_argument('-v', '--v_cruise', type=float, default=0.3,
                       help='Cruise velocity for both sub-tests in m/s (default: 0.3)')
    p_lat.add_argument('-d', '--dist', type=float, default=5.0,
                       help='Straight-line distance for drift test in m (default: 5.0)')
    p_lat.add_argument('-n', '--n_trials', type=int, default=10,
                       help='Number of latency trials (default: 10)')
    p_lat.add_argument('-s', '--save_file', type=str, default='sysid_latency.pkl',
                       help='Output pickle filename')

    # ── staircase ─────────────────────────────────────────────────────────
    p_stair = sub.add_parser('staircase',
                             help='Multi-level velocity/angular tracking (Section 5)',
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog="""
Each --seg defines one ramp segment, executed in order.

Formats:
  lin:v_start:v_end:v_step:hold_s   linear velocity ramp  (ω = 0)
  ang:w_start:w_end:w_step:hold_s   angular velocity ramp (v = 0)
  v_start:v_end:v_step:hold_s       legacy shorthand for lin:...

Duplicate (v,ω) pairs at segment boundaries are automatically merged.

Examples:
  # Linear up then down:
  staircase --seg lin:0:0.3:0.05:0.5 --seg lin:0.3:0:0.1:0.5

  # Angular up then down:
  staircase --seg ang:0:0.5:0.1:0.5 --seg ang:0.5:0:0.1:0.5

  # Linear staircase followed by angular staircase:
  staircase --seg lin:0:0.3:0.05:0.5 --seg lin:0.3:0:0.1:0.5 \\
            --seg ang:0:0.5:0.1:0.5  --seg ang:0.5:0:0.1:0.5
""")
    p_stair.add_argument('--seg', dest='segs',
                         metavar='[lin|ang:]start:end:step:hold_s',
                         action='append', required=True,
                         help='Staircase segment (repeatable, executed in order)')
    p_stair.add_argument('-s', '--save_file', type=str, default='sysid_staircase.pkl',
                         help='Output pickle filename (default: sysid_staircase.pkl)')

    parsed, ros_args = parser.parse_known_args(sys.argv[1:])
    rclpy.init(args=ros_args)

    # ── run selected mode ─────────────────────────────────────────────────
    if parsed.mode == 'lidar':
        node = LidarSysIdNode(parsed.n_scans, parsed.save_file)
        try:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            node.get_logger().info("Interrupted – saving partial data …")
        finally:
            node.save()
            node.destroy_node()

    elif parsed.mode == 'velocity':
        node = VelocitySysIdNode(
            parsed.v_target, parsed.hold, parsed.rest, parsed.n_steps, parsed.save_file)
        try:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.01)
        except KeyboardInterrupt:
            publish_vel(node.pub_cmd)
            node.get_logger().info("Interrupted – saving partial data …")
        finally:
            node.save()
            node.destroy_node()

    elif parsed.mode == 'latency':
        node = LatencyDriftSysIdNode(
            parsed.v_cruise, parsed.dist, parsed.n_trials, parsed.save_file)
        try:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.005)
        except KeyboardInterrupt:
            publish_vel(node.pub_cmd)
            node.get_logger().info("Interrupted – saving partial data …")
        finally:
            node.save()
            node.destroy_node()

    elif parsed.mode == 'staircase':
        try:
            sequence = _build_staircase_sequence(parsed.segs)
        except ValueError as e:
            print(f"[staircase] Bad segment spec: {e}", file=sys.stderr)
            sys.exit(1)
        node = StaircaseSysIdNode(sequence, parsed.save_file)
        try:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.01)
        except KeyboardInterrupt:
            publish_vel(node.pub_cmd)
            node.get_logger().info("Interrupted – saving partial data …")
        finally:
            node.save()
            node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
