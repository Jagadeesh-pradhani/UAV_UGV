import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import csv
from builtin_interfaces.msg import Time
import time

class PowerCalculator(Node):
    def __init__(self):
        super().__init__('energy_model_node')
        
        # Parameters from the table
        self.rho = 1.225       # Air density (kg/m³)
        self.A = 0.503         # Rotor disc area (m²)
        self.U_tip = 120.0     # Tip speed (m/s)
        self.s = 0.05          # Rotor solidity
        self.d0 = 0.6          # Fuselage drag ratio
        self.v0 = 4.03         # Mean rotor induced velocity (m/s)

        # Assume
        self.Cd0 = 0.011        # Blade drag coefficient 

        # Derived constants
        # self.p_bl = (self.rho * self.s * self.A * self.U_tip**3) / 8    # Blade profile power
        # self.p_ind = (1 / (2 * self.rho * self.A)) * (1 / self.v0)      # Induced power

        self.p_bl = (self.Cd0 * self.rho * self.s * self.A * self.U_tip**3) / 8    # Blade profile power
        self.p_ind = 2 * self.rho * self.A * (self.v0**3)     # Induced power

        # State variables
        self.current_velocity = 0.0
        self.current_position = None
        self.active_mission = False
        self.mission_start_time = None
        self.last_update_time = None

        # CSV setup
        self.csv_file = open('power_timing_log.csv', 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 
            'power(W)', 
            'velocity(m/s)',
            'position_x',
            'position_y',
            'position_z',
            'mission_duration(s)'
        ])

        # Subscribers
        self.create_subscription(Path, '/drone_path', self.path_callback, 10)
        self.create_subscription(Odometry, '/simple_drone/odom', self.odom_callback, 10)
        
        # Publisher
        self.power_pub = self.create_publisher(Float32, '/drone_power', 10)

    def path_callback(self, msg):
        if len(msg.poses) > 0:
            self.active_mission = True
            self.mission_start_time = time.time()
            self.get_logger().info("New path received. Starting logging.")

    def odom_callback(self, msg):
        if not self.active_mission:
            return

        # Calculate velocity magnitude
        current_time = time.time()
        self.current_velocity = np.linalg.norm([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        # Calculate power using equation 10
        term1 = self.p_bl * (1 + (3 * self.current_velocity**2) / self.U_tip**2)
        sqrt_term = np.sqrt(1 + (self.current_velocity**4) / (4 * self.v0**4))
        term2 = self.p_ind * (sqrt_term - (self.current_velocity**2) / (2 * self.v0**2))**0.5
        term3 = 0.5 * self.d0 * self.rho * self.s * self.A * self.current_velocity**3
        total_power = term1 + term2 + term3

        # Store position data
        position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

        # Calculate mission duration
        mission_duration = current_time - self.mission_start_time

        # Write to CSV
        self.csv_writer.writerow([
            current_time,
            total_power,
            self.current_velocity,
            position[0],
            position[1],
            position[2],
            mission_duration
        ])

        # Publish power value
        self.power_pub.publish(Float32(data=total_power))

        # Update last position
        self.current_position = position

        # Update timing
        self.last_update_time = current_time

    def __del__(self):
        self.csv_file.close()

def main():
    rclpy.init()
    node = PowerCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()