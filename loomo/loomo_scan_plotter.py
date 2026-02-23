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
        
        # Ci iscriviamo al topic del LaserScan
        self.subscription = self.create_subscription(
            LaserScan,
            '/loomo/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # Setup di Matplotlib in modalità interattiva
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Loomo 2D LiDAR (Depth to Scan)")
        self.ax.set_xlabel("Y (Metri - Sinistra/Destra)")
        self.ax.set_ylabel("X (Metri - Avanti)")
        
        # Fissiamo i limiti del grafico: 4 metri in avanti, +/- 2 metri di lato
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(0.0, 4.0)
        self.ax.grid(True)
        
        # Disegniamo il robot nell'origine (0,0)
        self.ax.plot(0, 0, 'ro', markersize=10, label='Loomo')
        self.ax.legend()
        
        # Inizializziamo lo scatter plot per i punti del laser (vuoto all'inizio)
        self.scatter = self.ax.scatter([], [], s=10, c='blue')
        
        self.get_logger().info("📊 Plotter Matplotlib Avviato. In attesa di dati su /loomo/scan...")

    def scan_callback(self, msg):
        # Convertiamo la lista di range in un array numpy
        ranges = np.array(msg.ranges)
        
        # Troviamo gli indici dei punti validi (non nan e non inf)
        valid_indices = np.isfinite(ranges)
        valid_ranges = ranges[valid_indices]
        
        if len(valid_ranges) == 0:
            return # Nessun ostacolo rilevato
            
        # Calcoliamo gli angoli corrispondenti per i punti validi
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid_indices]
        
        # Trasformazione Polare -> Cartesiana
        # Nota: per la convenzione standard dei robot (X avanti, Y sinistra)
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        # Dato che OpenCV/Depth usa una convenzione degli assi diversa, 
        # invertiamo l'asse orizzontale per far combaciare lo schermo con la realtà
        y = -y 
        
        # Aggiorniamo i dati del grafico
        self.scatter.set_offsets(np.c_[y, x]) # Invertiamo x/y solo per il disegno in modo da avere il robot in basso
        
        # Disegniamo il frame aggiornato
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