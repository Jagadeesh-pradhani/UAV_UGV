import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import math

class TurtleBotFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_follower')
        
        # Initialize the BasicNavigator
        self.navigator = BasicNavigator()

        # Subscriber to the drone's odometry
        self.drone_odom_sub = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.drone_odom_callback,
            10
        )

        self.error_distance_pub = self.create_publisher(Float64, '/error_distance', 10)

        # Subscriber to the TurtleBot's odometry
        self.turtlebot_odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.turtlebot_odom_callback,
            10
        )

        self.drone_position = None
        self.turtlebot_position = None

        self.timer = self.create_timer(0.1, self.update_goal)


    def drone_odom_callback(self, msg):
        self.drone_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # self.update_goal()

    def turtlebot_odom_callback(self, msg):
        self.turtlebot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def update_goal(self):
        if self.drone_position and self.turtlebot_position:
            drone_x, drone_y = self.drone_position
            turtlebot_x, turtlebot_y = self.turtlebot_position

            # Calculate the distance to the drone
            distance = math.sqrt((drone_x - turtlebot_x)**2 + (drone_y - turtlebot_y)**2)
            self.error_distance_pub.publish(Float64(data=float(distance)))
            # Set a goal if the TurtleBot is not close enough
            if distance > 2.0:  # Adjust the threshold as needed
                if distance > 5.0:
                    self.get_logger().warn('TurtleBot is too far from the drone!')
                    
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'world'
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = drone_x
                goal_pose.pose.position.y = drone_y
                goal_pose.pose.position.z = 0.0
                goal_pose.pose.orientation.w = 1.0

                self.navigator.goToPose(goal_pose)

                # Log the goal for debugging
            # self.get_logger().info(f'Distance : {distance} Bot Pose: x={turtlebot_x}, y={turtlebot_y}, Drone pose: x={drone_x}, y={drone_y}')

    # def spin(self):
    #     while rclpy.ok():
    #         rclpy.spin_once(self)
    #         # Check if the robot has reached its goal
    #         if self.navigator.isTaskComplete():
    #             self.get_logger().info('Goal reached!')


def main(args=None):
    rclpy.init(args=args)
    
    follower = TurtleBotFollower()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
