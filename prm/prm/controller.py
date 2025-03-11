#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

# ROS2 message imports
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        self.get_logger().info("Path Follower Node Initialized")
        
        # Subscribers: planned path and odometry
        self.path_sub = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odom_callback,
            10)
        
        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        
        # Internal state variables
        self.current_path = []  # List of PoseStamped
        self.current_waypoint_idx = 0
        self.current_pose = None  # Current drone position (x, y, z)
        
        # Controller parameters
        self.waypoint_tolerance = 0.5  # meters
        self.Kp = 0.5  # proportional gain for linear velocity
        self.max_speed = 1.0  # maximum speed (m/s)
        
        # Timer for control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty path.")
            return
        self.current_path = msg.poses
        self.current_waypoint_idx = 0
        self.get_logger().info(f"Received new path with {len(self.current_path)} waypoints.")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def control_loop(self):
        if self.current_pose is None or not self.current_path:
            return  # Nothing to do until we have a pose and a path
        
        # Check if we have reached the final waypoint
        if self.current_waypoint_idx >= len(self.current_path):
            self.get_logger().info("Path complete. Stopping drone.")
            self.publish_velocity(0.0, 0.0, 0.0)
            return

        # Get current target waypoint
        target_wp = self.current_path[self.current_waypoint_idx].pose.position
        
        # Compute error vector
        error_x = target_wp.x - self.current_pose.x
        error_y = target_wp.y - self.current_pose.y
        error_z = target_wp.z - self.current_pose.z
        distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)

        # Check if the waypoint has been reached
        if distance < self.waypoint_tolerance:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx+1}/{len(self.current_path)}")
            self.current_waypoint_idx += 1
            return

        # Simple proportional controller for linear velocity
        vx = self.Kp * error_x
        vy = self.Kp * error_y
        vz = self.Kp * error_z

        # Saturate velocities if needed
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
            vz *= scale

        self.publish_velocity(vx, vy, vz)

    def publish_velocity(self, vx, vy, vz):
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        # For this simple example, angular velocities are set to zero.
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Path Follower Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
