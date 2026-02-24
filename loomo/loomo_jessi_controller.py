#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import os
import pickle
from jax import random
from jax.tree_util import tree_map
from collections import deque
import math
import jax.numpy as jnp
from rclpy.qos import qos_profile_sensor_data

from socialjym.policies.jessi import JESSI

NETWORK_NAME = 'jessi_finetuned_rl_out.pkl'
SAVE_FILE_NAME = 'jessi_recorded_obs.pkl'

class JessiController(Node):
    def __init__(self):
        super().__init__('jessi_controller')
        
        # --- Parametri JESSI ---
        self.n_stack = 5 # Cambia in base al tuo modello
        self.lidar_num_rays = 320 # Il nostro bridge manda 320 raggi
        self.dt = 0.25 # 4 Hz
        
        # Inizializza il buffer per lo stacking. 
        # Usiamo appendleft così l'indice 0 è sempre l'osservazione più recente.
        self.obs_stack = deque(maxlen=self.n_stack)
        
        # Buffer per salvare i dati per il pickle
        self.recorded_data = []

        # Variabili di stato correnti
        self.latest_scan = None
        self.latest_odom = None
        self.robot_goal = np.array([0., 2.]) # Esempio: goal a 5 metri dritti in frame globale (da aggiornare in runtime)
        
        self.lidar_num_rays = 320
        self.lidar_min_angle = -0.46981275  # Radianti
        self.lidar_max_angle = 0.46981275   # Radianti
        self.lidar_angle_increment = 0.00294553 # Radianti
        self.lidar_max_dist = 4

        # Inizializza il modello JESSI
        self.jessi = JESSI(
            lidar_num_rays=320,
            lidar_angular_range=self.lidar_max_angle-self.lidar_min_angle,
            lidar_max_dist=self.lidar_max_dist,
            n_stack_for_action_space_bounding=1
        )
        self.rng_key = random.PRNGKey(0)
        with open(os.path.join(os.path.dirname(__file__), NETWORK_NAME), 'rb') as f:
            self.network_params, _, _ = pickle.load(f)
        
        # ROS 2 Subscribers
        self.sub_scan = self.create_subscription(
            LaserScan, 
            '/loomo/scan', 
            self.scan_callback, 
            qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(Odometry, '/loomo/odom', self.odom_callback, 10)
        
        # ROS 2 Publisher
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # ROS 2 Timer (Il cuore di JESSI)
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.previous_action = jnp.array([0.,0.])

        self.get_logger().info("🧠 JESSI Controller inizializzato a 4Hz!")

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
            self.get_logger().warn("In attesa dei dati dai sensori...")
            return

        # 1. ESTARZIONE E PULIZIA LIDAR
        ranges = np.array(self.latest_scan.ranges)
        
        # TRUCCO PER JAX: Sostituisci i NaN e Inf con un valore fuori range (es. 10.0)
        # In questo modo len(ranges) è sempre ESATTAMENTE 320 e il flag 'hit' di JESSI sarà 0.
        safe_ranges = np.where(np.isnan(ranges) | np.isinf(ranges), self.jessi.lidar_max_dist, ranges)
        # safe_ranges = np.where(safe_ranges < 0.3, 0.0, safe_ranges)

        # 2. ESTRAZIONE ODOMETRIA
        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        r_theta = self.get_yaw_from_quaternion(self.latest_odom.pose.pose.orientation)
        print(rx,ry, r_theta)

        # Parametri cinematici del robot (da adattare al tuo setup)
        r_radius = 0.3 # Raggio ingombro Loomo
        
        # 3. CREAZIONE DELLO STEP CORRENTE
        # Forma richiesta da JESSI: [rx, ry, r_theta, r_radius, r_a1, r_a2, lidar_measurements]
        current_step_obs = np.concatenate(([rx, ry, r_theta, r_radius, self.previous_action[0], self.previous_action[1]], safe_ranges))
        
        # Aggiungiamo alla pila (in cima, index 0)
        self.obs_stack.appendleft(current_step_obs)
        
        # Se non abbiamo ancora abbastanza cronologia, duplichiamo l'osservazione attuale
        while len(self.obs_stack) < self.n_stack:
            self.obs_stack.appendleft(current_step_obs)

        # 4. PREPARAZIONE INPUT PER JAX
        obs_matrix = jnp.array(self.obs_stack) # Shape: (n_stack, 326)
        
        info_dict = {
            "robot_goal": jnp.array(self.robot_goal)
        }

        # 5. INFERENZA JESSI
        try:
            action, self.rng_key, _, _, _, _, perception_output, actor_distr, _, _ = self.jessi.act(
                key=self.rng_key,
                obs=obs_matrix,
                info=info_dict,
                e2e_network_params=self.network_params,
                sample=False # Azione deterministica
            )
            v_cmd, w_cmd = float(action[0]), float(action[1])
            
            # # Dummy action per testare il loop
            # v_cmd, w_cmd = 0.0, 0.0 
            
            # 6. PUBBLICAZIONE AZIONE
            cmd_msg = Twist()
            cmd_msg.linear.x = v_cmd
            cmd_msg.angular.z = w_cmd
            self.pub_cmd.publish(cmd_msg)

            self.previous_action = jnp.array([v_cmd, w_cmd])

            # 7. SALVATAGGIO DATI PER IL PICKLE
            # Convertiamo i tensori JAX in array Numpy standard per evitare problemi di unpickling offline
            self.recorded_data.append({
                'observation': np.array(obs_matrix),
                'robot_goal': np.array(self.robot_goal),
                'action': np.array([v_cmd, w_cmd]),
                'perception_distr': perception_output,
                'actor_distr': actor_distr,
            })
            
        except Exception as e:
            self.get_logger().error(f"Errore durante inferenza JESSI: {e}")

    def save_data(self):
        """Metodo chiamato allo spegnimento per salvare le osservazioni su disco"""
        if len(self.recorded_data) > 0:
            save_path = os.path.join(os.path.dirname(__file__), SAVE_FILE_NAME)
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
                self.get_logger().info(f"💾 Salvataggio completato! {len(self.recorded_data)} frame salvati in: {save_path}")
            except Exception as e:
                self.get_logger().error(f"Errore durante il salvataggio dei dati: {e}")
        else:
            self.get_logger().info("Nessun dato da salvare.")

def main(args=None):
    rclpy.init(args=args)
    node = JessiController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()