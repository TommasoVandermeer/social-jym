#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import socket
import struct
import time
import threading
import numpy as np
import math

class LoomoROS2Bridge(Node):
    def __init__(self):
        super().__init__('loomo_socket_bridge')
        
        self.declare_parameter('loomo_ip', '10.186.105.67')
        self.loomo_ip = self.get_parameter('loomo_ip').value
        
        self.port_cmd = 8080
        self.port_img = 8081
        self.port_odom = 8082

        self.num_odom_floats = 5 
        
        # --- RISOLUZIONE MAGICA TROVATA (230400 bytes) ---
        self.img_width = 320
        self.img_height = 240
        self.img_channels = 3  # Ora usiamo 3 canali! RGB
        self.img_bytes = self.img_width * self.img_height * self.img_channels
        
        # Se 640x360 dovesse dare un'immagine tutta corrotta, commenta sopra e scommenta sotto (Ipotesi RGB):
        # self.img_width = 320
        # self.img_height = 240
        # self.img_channels = 3
        # self.img_bytes = self.img_width * self.img_height * self.img_channels

        self.bridge = CvBridge()
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.pub_odom = self.create_publisher(Odometry, '/loomo/odom', 10)
        self.pub_depth = self.create_publisher(Image, '/loomo/depth', 10)
        
        self.sock_cmd = None
        self.running = True
        
        self.sock_cmd = None
        self.running = True
        
        self.last_cmd_time = time.time()
        
        self.connect_cmd_socket()

        # Timer che gira a 10 Hz per l'Heartbeat
        self.heartbeat_timer = self.create_timer(0.1, self.cmd_heartbeat)

        self.thread_img = threading.Thread(target=self.receive_image_loop, daemon=True)
        self.thread_odom = threading.Thread(target=self.receive_odom_loop, daemon=True)
        self.thread_img.start()
        self.thread_odom.start()

        self.get_logger().info("Bridge con Heartbeat attivato! In attesa di scoprire i byte dell'Odometria...")

    # ==========================================
    # HELPER PER RICEZIONE ROBUSTA TCP
    # ==========================================
    def recvall(self, sock, n, name="Dati"):
        """Versione aggiornata con il nome del thread per i log"""
        data = bytearray()
        while len(data) < n and self.running:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                self.get_logger().warn(f"[{name}] Attesa... Ricevuti finora: {len(data)}/{n} byte.")
                continue 
            except Exception:
                return None
        return data

    # ==========================================
    # GESTIONE COMANDI (PORTA 8080)
    # ==========================================
    def connect_cmd_socket(self):
        while rclpy.ok() and self.running:
            try:
                self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock_cmd.connect((self.loomo_ip, self.port_cmd))
                self.get_logger().info("Connesso al Controllo Motori (8080)")
                break
            except socket.error:
                time.sleep(2)

    def cmd_vel_callback(self, msg):
        if not self.sock_cmd: return
        try:
            self.last_cmd_time = time.time() # Aggiorniamo il timer quando ricevi un comando vero
            command_bytes = struct.pack('<ff', msg.linear.x, msg.angular.z)
            self.sock_cmd.sendall(command_bytes)
        except socket.error:
            self.sock_cmd.close()
            self.connect_cmd_socket()

    def cmd_heartbeat(self):
        """Se per 0.2 secondi non riceve comandi da ROS, manda zeri per tenere sveglia l'app"""
        if self.sock_cmd and (time.time() - self.last_cmd_time > 0.2):
            try:
                self.sock_cmd.sendall(struct.pack('<ff', 0.0, 0.0))
            except: pass
    # ==========================================
    # GESTIONE IMMAGINI DEPTH (PORTA 8081)
    # ==========================================
    def receive_image_loop(self):
        while rclpy.ok() and self.running:
            try:
                sock_img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_img.settimeout(2.0)
                sock_img.connect((self.loomo_ip, self.port_img))
                self.get_logger().info("Connesso alla Camera (8081)")
            except Exception:
                time.sleep(2)
                continue

            while rclpy.ok() and self.running:
                # Usiamo il recvall corretto
                raw_data = self.recvall(sock_img, self.img_bytes, name="Camera")
                if not raw_data:
                    break
                
                try:
                    # Convertiamo in numpy array (3D)
                    image_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.img_height, self.img_width, self.img_channels))
                    ros_image = self.bridge.cv2_to_imgmsg(image_np, encoding="bgr8")
                    
                    # Se usiamo 1 canale usiamo mono8, se 3 canali bgr8
                    enc = "mono8" if self.img_channels == 1 else "bgr8"
                    ros_image = self.bridge.cv2_to_imgmsg(image_np, encoding=enc)
                    
                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = "loomo_camera"
                    self.pub_depth.publish(ros_image)
                    self.get_logger().info("📸 Frame pubblicato con successo!") # Per festeggiare!
                except Exception as e:
                    self.get_logger().error(f"Errore reshape immagine: {e}")
            
            sock_img.close()

    # ==========================================
    # GESTIONE ODOMETRIA/STATO (PORTA 8082)
    # ==========================================
    def receive_odom_loop(self):
        self.port_odom = 8082
        self.num_odom_floats = 5
        byte_count = 20  # 5 floats * 4 bytes
        unpack_format = '<fffff'

        while rclpy.ok() and self.running:
            try:
                sock_odom = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_odom.settimeout(2.0)
                sock_odom.connect((self.loomo_ip, self.port_odom))
                self.get_logger().info("Connesso all'Odometria (8082) - Flusso ROS Attivo")
            except Exception:
                time.sleep(2)
                continue

            while rclpy.ok() and self.running:
                # Usiamo il recvall corretto per aspettare esattamente 20 byte senza sfasamenti
                raw_data = self.recvall(sock_odom, byte_count, name="Odometria")
                if not raw_data:
                    break
                
                try:
                    vals = struct.unpack(unpack_format, raw_data)
                    x, y, theta, v, w = vals
                    
                    # Creazione del messaggio ROS 2 Odometry
                    odom_msg = Odometry()
                    odom_msg.header.stamp = self.get_clock().now().to_msg()
                    odom_msg.header.frame_id = "odom"
                    odom_msg.child_frame_id = "base_link"
                    
                    odom_msg.pose.pose.position.x = float(x)
                    odom_msg.pose.pose.position.y = float(y)
                    
                    # Convertiamo l'angolo theta in un Quaternione per ROS
                    odom_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
                    odom_msg.pose.pose.orientation.w = math.cos(theta / 2.0)
                    
                    odom_msg.twist.twist.linear.x = float(v)
                    odom_msg.twist.twist.angular.z = float(w)
                    
                    self.pub_odom.publish(odom_msg)

                except Exception as e:
                    self.get_logger().error(f"Errore pubblicazione Odometria: {e}")
            
            sock_odom.close()

    def stop_loomo(self):
        self.running = False
        if self.sock_cmd:
            try:
                self.get_logger().info("Fermo i motori del Loomo...")
                self.sock_cmd.sendall(struct.pack('<ff', 0.0, 0.0))
                self.sock_cmd.close()
            except: pass

def main(args=None):
    rclpy.init(args=args)
    bridge = LoomoROS2Bridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop_loomo()
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()