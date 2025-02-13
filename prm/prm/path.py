import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
import numpy as np

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        # Create a publisher for the path
        self.publisher_ = self.create_publisher(Path, 'drone_path', QoSProfile(depth=10))

        # Define the path message
        self.path = Path()
        self.path.header.frame_id = 'world'  # Set to the coordinate frame you are using, e.g., 'map' or 'world'

        # List of 3D points defining the path (x, y, z)
        points = [
            (0.4, -0.550124, 0.41),
            (-2.8,-0.950124, 1.41),
            (-5.8,-0.550124, 1.41),
            (-8.4, 0.049875, 1.81),
            (-9.4, 0.249875, 3.41),
            (-11.8, 0.049875, 3.81),
            (-13.8, -0.450124, 4.21),
            (-14.4, -1.150124, 4.21),
            (-15.4, -2.750124, 4.21)
        ]



        # # Interpolate points
        interpolated_points = self.interpolate_path(points, step_size=0.1)
        # Convert interpolated points to PoseStamped and add to the path
        for point in interpolated_points:
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0  # Set orientation to neutral (no rotation)
            self.path.poses.append(pose)

        # Publish the path only once
        self.publish_path()

    def interpolate_path(self, points, step_size=0.1):
        """
        Interpolates between given 3D points with a specified step size.
        :param points: List of tuples [(x, y, z), ...] representing waypoints
        :param step_size: Distance between consecutive interpolated points
        :return: List of interpolated points
        """
        interpolated_points = []
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            distance = np.linalg.norm(p2 - p1)
            num_steps = max(1, int(distance / step_size))
            for t in np.linspace(0, 1, num_steps):
                interpolated_points.append((p1 + t * (p2 - p1)).tolist())
        interpolated_points.append(points[-1])  # Add the last point to the path
        return interpolated_points

    def publish_path(self):
        self.path.header.stamp = self.get_clock().now().to_msg()  # Update timestamp
        self.publisher_.publish(self.path)
        self.get_logger().info('Publishing interpolated path once')

def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    rclpy.spin_once(path_publisher)  # Publish once and process callbacks once
    path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
