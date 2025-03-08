import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Bool, Int8, Float64
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math
from collections import deque
import time

# Function to convert quaternion to Euler angles
def euler_from_quaternion(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class FellowPathController(Node):
    def __init__(self):
        super().__init__("drone_fellow_path_control")

        # Initialize publishers and subscribers
        self.takeoff_publisher = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.mode_switch_publisher = self.create_publisher(Bool, '/simple_drone/cmd_mode', 10)
        self.velocity_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.goal_reached_publisher = self.create_publisher(Empty, '/goal_reached', 10)
        
        self.state_subscriber = self.create_subscription(Int8, '/simple_drone/state', self.state_callback, 10)
        self.odometry = self.create_subscription(Odometry, "simple_drone/odom", self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, "drone_path", self.set_path, 10)
        self.error_distance_sub = self.create_subscription(Float64, '/error_distance', self.error_distance_callback, 10)
        self.drone_state = 0  # Default state is "Landed"
        self.takeoff_initiated = False
        self.vel_mode_set = False
        self.odom = None  # Set to None initially
        self.path_queue = deque()  # Initialize an empty deque for storing interpolated path poses
        self.initial_vel = 1.3
        self.desired_linear_vel = self.initial_vel  # Linear velocity
        self.lookahead_distance = 0.3
        self.min_angle_tolerance=0.3 
        self.max_linear_vel_z = 0.43  # m/s
        self.approach_velocity_scaling_dist_ = 0.1 # meters
        self.initial_alignment_done = False  # Flag for initial alignment
        self.error = None  # Error distance to the drone

        # Set up a timer to call compute_velocity_commands at a frequency of 10 Hz
        self.timer = self.create_timer(0.1, self.compute_velocity_commands)  # 10 Hz frequency

    def compute_velocity_commands(self):
        # Check if error distance is greater than 5.0
        if self.error is not None and self.error > 3.0:  # Start slowing down early
            self.desired_linear_vel -= 0.1  # Slow down linear velocity
            self.get_logger().info(f"Slowing down {self.desired_linear_vel}.")
            if self.desired_linear_vel <= 0.0:
                self.desired_linear_vel = 0.0
        else:
            self.desired_linear_vel += 0.1
            self.get_logger().info(f"Reset {self.desired_linear_vel}.")
            if self.desired_linear_vel >= self.initial_vel:
                self.desired_linear_vel = self.initial_vel

        if not self.path_queue or self.odom is None:
            return  # Wait for a path and odometry data to be available

        current_position = np.array([self.odom.pose.pose.position.x,
                                    self.odom.pose.pose.position.y,
                                    self.odom.pose.pose.position.z]) 
          
        # Remove points from the path that are behind the current position
        # Look ahead for the carrot point, skipping points that are too close
        carrot_point = None
        while self.path_queue:
            next_point = self.path_queue[0]
            next_position = np.array([next_point.pose.position.x,
                                    next_point.pose.position.y,
                                    next_point.pose.position.z])
            distance_to_next_point = np.linalg.norm(current_position[:2] - next_position[:2])

            # if distance to next point > lookahead_distance this mean this is the carrot point we need to go for it 
            if distance_to_next_point > self.lookahead_distance:
                carrot_point=np.array([next_point.pose.position.x,   # set carrot point
                                       next_point.pose.position.y,
                                       next_point.pose.position.z])
                break
            else:  # If we're beyond this point, remove it
                self.path_queue.popleft()
                

        # If no points remain in the path queue, we are done
        if not self.path_queue:
            self.get_logger().info("Path complete.")
            self.set_velocity(0.0, 0.0, 0.0)
            self.goal_reached_publisher.publish(Empty())
            return

        # If no carrot point is found, path is complete
        if carrot_point is None:
            self.get_logger().info("Path complete.")
            self.goal_reached_publisher.publish(Empty())
            return

        # Calculate angle to the carrot point
        angle_to_carrot = np.arctan2(carrot_point[1] - current_position[1],
                                    carrot_point[0] - current_position[0])

        # Use euler_from_quaternion to get the current yaw
        _, _, current_yaw = euler_from_quaternion(self.odom.pose.pose.orientation)
        
        # Calculate the yaw error
        yaw_error = angle_to_carrot - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi  # ensure Normalization  to [-pi, pi]

        # Ensure alignment before moving forward
        if abs(yaw_error) > self.min_angle_tolerance:
            # Rotate to align with carrot point
            angular_vel = min(0.6, abs(yaw_error)) * np.sign(yaw_error)
            self.set_velocity(0.0, angular_vel, 0.0)  # No forward movement until aligned
            self.get_logger().info(f"Rotating to align: angular z: {angular_vel}")
        else:
            # Proceed with linear velocity if aligned
            linear_vel = self.desired_linear_vel
            dist_square = np.square(carrot_point[0] - current_position[0]) + np.square(carrot_point[1] - current_position[1])
            curvature = 2.0 * carrot_point[1] / dist_square if dist_square > 0.001 else 0.0

            angular_vel = curvature * self.desired_linear_vel
            angular_vel = min(0.6, abs(angular_vel)) * np.sign(yaw_error)  # Ensure bounded angular velocity
            #calculate vertical velocity
            # calculate altitude error 
            altitude_error=carrot_point[2]-current_position[2]
            if np.abs(altitude_error)< self.approach_velocity_scaling_dist_:
                scaling_factor=altitude_error/self.approach_velocity_scaling_dist_
            else:
                scaling_factor=altitude_error/np.abs(altitude_error) # this will be -1 or 1
            linear_vel_z=scaling_factor*self.max_linear_vel_z
            self.set_velocity(linear_vel, angular_vel, linear_vel_z)
            self.get_logger().info(f"eror: z {altitude_error}")

            self.get_logger().info(f"Moving to carrot point: x: {linear_vel}, angular z: {angular_vel},linear_vel_z: {linear_vel_z}")
    
    def error_distance_callback(self, msg):
        # Log the distance to the drone for debugging
        self.error = msg.data

    def set_path(self, msg):
        # Clear current path queue and interpolate the entire path
        self.path_queue.clear()
        self.initial_alignment_done = False  # Reset alignment flag for a new path
        
        self.path_queue = deque(msg.poses)
        self.get_logger().info(f"Received path with {len(self.path_queue)} points.")

    def odom_callback(self, msg):
        # Update odometry
        self.odom = msg

    def state_callback(self, msg):
        # Update drone state based on topic subscription
        self.drone_state = msg.data

        # Step 1: Initiate takeoff when in "Landed" state
        if self.drone_state == 0 and not self.takeoff_initiated:
            self.takeoff()
            self.takeoff_initiated = True
        # Step 2: Once in "Flying" mode, switch to velocity control mode
        elif self.drone_state == 1 and not self.vel_mode_set:
            self.switch_to_velocity_mode()
            self.vel_mode_set = True
            time.sleep(0.5)  # Wait briefly to ensure mode switch

    def takeoff(self):
        # Publish an empty message to takeoff topic to initiate takeoff
        takeoff_msg = Empty()
        self.takeoff_publisher.publish(takeoff_msg)
        self.get_logger().info("Takeoff command executed.")

    def switch_to_velocity_mode(self):
        # Publish a boolean message to switch control mode to "velocity"
        mode_msg = Bool()
        mode_msg.data = True  # True for velocity mode
        self.mode_switch_publisher.publish(mode_msg)
        self.get_logger().info("Switched to velocity mode.")

    def set_velocity(self, x, angular_z, z):
        # Publish the desired velocity in x, y, z directions
        vel_msg = Twist()
        vel_msg.linear.x = x
        vel_msg.angular.z = angular_z
        vel_msg.linear.z = z
        self.velocity_publisher.publish(vel_msg)
        self.get_logger().info(f"Setting velocity to x: {x}, z: {z}, angular z: {angular_z}")

def main(args=None):
    rclpy.init(args=args)
    fellow_path_controller_node = FellowPathController()
    rclpy.spin(fellow_path_controller_node)
    fellow_path_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()