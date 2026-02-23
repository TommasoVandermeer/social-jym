#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cv_bridge import CvBridge
import socket
import struct
import time
import threading
import numpy as np
import math
import warnings

class LoomoJessiBridge(Node):
    def __init__(self):
        super().__init__('loomo_jessi_bridge')
        
        self.declare_parameter('loomo_ip', '10.186.105.67')
        self.loomo_ip = self.get_parameter('loomo_ip').value
        
        self.port_cmd = 8000
        self.port_odom = 8001
        self.port_depth = 8002

        self.img_width = 320
        self.img_height = 240
        self.bridge = CvBridge()
        
        # Publishers & Subscribers
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.pub_odom = self.create_publisher(Odometry, '/loomo/odom', 10)
        
        # Publisher per i Sensori (Profilo Best Effort)
        self.pub_depth = self.create_publisher(Image, '/loomo/depth', qos_profile_sensor_data)
        self.pub_camera_info = self.create_publisher(CameraInfo, '/loomo/camera_info', qos_profile_sensor_data)        
        self.pub_scan = self.create_publisher(LaserScan, '/loomo/scan', qos_profile_sensor_data)
        
        # TF Broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        self.running = True
        self.sock_cmd = None
        
        self.publish_static_transforms()
        self.connect_cmd_socket()
        
        self.thread_depth = threading.Thread(target=self.receive_depth_loop, daemon=True)
        self.thread_odom = threading.Thread(target=self.receive_odom_loop, daemon=True)
        self.thread_depth.start()
        self.thread_odom.start()

        self.get_logger().info(f"🚀 Bridge JESSI All-In-One avviato su {self.loomo_ip}!")

    def publish_static_transforms(self):
        # 1. TF per la telecamera ottica (Z in avanti) per le immagini
        t_opt = TransformStamped()
        t_opt.header.stamp = self.get_clock().now().to_msg()
        t_opt.header.frame_id = "base_link"
        t_opt.child_frame_id = "loomo_depth_optical_frame"
        t_opt.transform.translation.x = 0.1
        t_opt.transform.translation.y = 0.0
        t_opt.transform.translation.z = 0.5
        t_opt.transform.rotation.x = -0.5
        t_opt.transform.rotation.y = 0.5
        t_opt.transform.rotation.z = -0.5
        t_opt.transform.rotation.w = 0.5

        # 2. TF per il LaserScan (X in avanti, Y a sinistra, come la base_link)
        t_laser = TransformStamped()
        t_laser.header.stamp = t_opt.header.stamp
        t_laser.header.frame_id = "base_link"
        t_laser.child_frame_id = "loomo_laser_frame"
        t_laser.transform.translation.x = 0.1
        t_laser.transform.translation.y = 0.0
        t_laser.transform.translation.z = 0.5
        t_laser.transform.rotation.w = 1.0 # Nessuna rotazione

        self.static_broadcaster.sendTransform([t_opt, t_laser])

    def connect_cmd_socket(self):
        while rclpy.ok() and self.running:
            try:
                self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock_cmd.connect((self.loomo_ip, self.port_cmd))
                self.get_logger().info("✅ Connesso al Controllo Motori (8000)")
                break
            except Exception:
                self.get_logger().warn("Attesa server comandi Loomo...")
                time.sleep(2)

    def cmd_vel_callback(self, msg):
        if not self.sock_cmd: return
        try:
            command_bytes = struct.pack('<ff', float(msg.linear.x), float(msg.angular.z))
            self.sock_cmd.sendall(command_bytes)
        except Exception as e:
            self.get_logger().error(f"Errore invio comandi: {e}")
            self.connect_cmd_socket()

    def recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return data

    def receive_depth_loop(self):
        # Parametri Intrinseci della Telecamera Loomo
        fx = 314.14313
        cx = 159.5
        u = np.arange(self.img_width)
        # Calcolo angoli pre-computato per efficienza (da sinistra a destra)
        theta = np.arctan2(cx - u, fx)

        while rclpy.ok() and self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.loomo_ip, self.port_depth))
                self.get_logger().info("📸 Connesso al Server Depth e Laser (8002)")
                
                while self.running:
                    header = self.recvall(s, 4)
                    if not header: break
                    msg_len = struct.unpack('>i', header)[0]
                    
                    if msg_len != (self.img_width * self.img_height * 2):
                        s.recv(1024)
                        continue

                    img_data = self.recvall(s, msg_len)
                    if not img_data: break
                    
                    # 1. Creazione Immagine Numpy
                    image_np = np.frombuffer(img_data, dtype=np.uint16).reshape((self.img_height, self.img_width))
                    sync_stamp = self.get_clock().now().to_msg()
                    
                    # 2. Pubblicazione Depth Grezza (Opzionale, puoi rimuoverlo in futuro se non serve)
                    ros_img = self.bridge.cv2_to_imgmsg(image_np, encoding="16UC1")
                    ros_img.header.stamp = sync_stamp
                    ros_img.header.frame_id = "loomo_depth_optical_frame"
                    self.pub_depth.publish(ros_img)

                    # --- 3. CONVERSIONE NATIVA IN LASERSCAN ---
                    # Prendiamo 10 righe centrali e ne calcoliamo la media
                    center_rows = image_np[115:125, :].astype(np.float32)
                    center_rows[center_rows == 0] = np.nan # Ignora gli zeri
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        z_mm = np.nanmean(center_rows, axis=0) # array di 320 elementi
                    
                    z_m = z_mm / 1000.0 # Converti in metri
                    
                    # Calcola l'ipotenusa (distanza radiale reale)
                    r = z_m * np.sqrt(1 + ((cx - u) / fx)**2)
                    
                    # Creiamo il messaggio LaserScan
                    scan_msg = LaserScan()
                    scan_msg.header.stamp = sync_stamp
                    scan_msg.header.frame_id = "loomo_laser_frame"
                    scan_msg.angle_min = float(theta[-1]) # Destra (negativo)
                    scan_msg.angle_max = float(theta[0])  # Sinistra (positivo)
                    scan_msg.angle_increment = float((scan_msg.angle_max - scan_msg.angle_min) / (self.img_width - 1))
                    scan_msg.time_increment = 0.0
                    scan_msg.scan_time = 0.1 # circa 10Hz
                    scan_msg.range_min = 0.3
                    scan_msg.range_max = 4.0
                    
                    # Binning dei raggi (allineamento perfetto agli incrementi angolari)
                    ranges = np.full(self.img_width, np.inf)
                    valid = (r >= scan_msg.range_min) & (r <= scan_msg.range_max) & ~np.isnan(r)
                    
                    bins = ((theta - scan_msg.angle_min) / scan_msg.angle_increment).astype(int)
                    valid_bins = valid & (bins >= 0) & (bins < self.img_width)
                    
                    for i in range(self.img_width):
                        if valid_bins[i]:
                            b = bins[i]
                            if r[i] < ranges[b]:
                                ranges[b] = float(r[i])
                                
                    scan_msg.ranges = ranges.tolist()
                    self.pub_scan.publish(scan_msg)

            except Exception as e:
                self.get_logger().error(f"Errore Depth/Scan: {e}")
                time.sleep(2)

    def receive_odom_loop(self):
        self.port_odom = 8001
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_time = time.time()

        while rclpy.ok() and self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.loomo_ip, self.port_odom))
                self.get_logger().info("📍 Connesso al Server Odometria (8001)")
                self.last_time = time.time()
                
                while self.running:
                    raw_data = self.recvall(s, 12)
                    if not raw_data: break
                    v, w, reset_flag = struct.unpack('<fff', raw_data)
                    
                    if reset_flag > 0.5:
                        self.odom_x = 0.0
                        self.odom_y = 0.0
                        self.odom_theta = 0.0
                        self.get_logger().info("🔄 Odometria AZZERATA dal display del robot!")
                    
                    current_time = time.time()
                    dt = current_time - self.last_time
                    self.last_time = current_time
                    
                    if abs(w) < 1e-6:
                        self.odom_x += v * math.cos(self.odom_theta) * dt
                        self.odom_y += v * math.sin(self.odom_theta) * dt
                    else:
                        self.odom_x += (v / w) * (math.sin(self.odom_theta + w * dt) - math.sin(self.odom_theta))
                        self.odom_y -= (v / w) * (math.cos(self.odom_theta + w * dt) - math.cos(self.odom_theta))
                    
                    self.odom_theta += w * dt
                    self.odom_theta = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))
                    
                    odom_msg = Odometry()
                    odom_msg.header.stamp = self.get_clock().now().to_msg()
                    odom_msg.header.frame_id = "odom"
                    odom_msg.child_frame_id = "base_link"
                    
                    odom_msg.pose.pose.position.x = float(self.odom_x)
                    odom_msg.pose.pose.position.y = float(self.odom_y)
                    odom_msg.pose.pose.orientation.z = math.sin(self.odom_theta / 2.0)
                    odom_msg.pose.pose.orientation.w = math.cos(self.odom_theta / 2.0)
                    odom_msg.twist.twist.linear.x = float(v)
                    odom_msg.twist.twist.angular.z = float(w)

                    t = TransformStamped()
                    t.header.stamp = odom_msg.header.stamp
                    t.header.frame_id = "odom"
                    t.child_frame_id = "base_link"
                    t.transform.translation.x = float(self.odom_x)
                    t.transform.translation.y = float(self.odom_y)
                    t.transform.translation.z = 0.0
                    t.transform.rotation.z = math.sin(self.odom_theta / 2.0)
                    t.transform.rotation.w = math.cos(self.odom_theta / 2.0)
                    
                    self.tf_broadcaster.sendTransform(t)
                    self.pub_odom.publish(odom_msg)
            except Exception as e:
                self.get_logger().warn(f"Riconnessione Odometria... ({e})")
                time.sleep(2)

def main(args=None):
    rclpy.init(args=args)
    bridge = LoomoJessiBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.running = False
        rclpy.shutdown()

if __name__ == '__main__':
    main()