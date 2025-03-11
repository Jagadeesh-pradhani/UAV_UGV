#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import time

# ROS2 message imports
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from octomap_msgs.msg import Octomap

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* Planner Node Initialized")
        
        # Subscribers for goal (from RViz), odometry, and Octomap
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odom_callback,
            10)
        self.octomap_sub = self.create_subscription(
            Octomap,
            '/octomap_full',
            self.octomap_callback,
            10)
        
        # Publisher for the planned path
        self.path_pub = self.create_publisher(Path, '/drone_path', 10)
        
        # Internal state variables
        self.current_pose = None      # geometry_msgs/Point
        self.goal_pose = None         # geometry_msgs/Point (will override z)
        self.occupancy_grid = None    # 3D numpy array representing the occupancy grid
        
        # Grid parameters (adjust these to suit your simulation)
        self.grid_resolution = 1.0    # meters per grid cell
        self.grid_size = (100, 100, 20)  # grid dimensions (x, y, z)
        
        # Altitude parameters:
        self.fixed_goal_z = 2.0   # Fixed altitude for a 2D goal from RViz
        self.over_wall_z = 5.0    # Altitude to try if no path found at fixed altitude

    def goal_callback(self, msg: PoseStamped):
        # Override the goal's z value with our fixed altitude
        self.goal_pose = msg.pose.position
        self.goal_pose.z = self.fixed_goal_z
        self.get_logger().info(
            f"Received new goal (fixed z): x={self.goal_pose.x}, y={self.goal_pose.y}, z={self.goal_pose.z}"
        )
        self.attempt_plan()

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def octomap_callback(self, msg: Octomap):
        # For this example we create an empty occupancy grid.
        # In a real application, you would convert msg.data into a grid marking obstacles.
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        self.get_logger().info("Octomap received and occupancy grid updated.")
        # self.attempt_plan()

    def attempt_plan(self):
        # Only attempt planning if we have all necessary data.
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            return
        if self.goal_pose is None:
            self.get_logger().warn("Waiting for goal pose...")
            return
        if self.occupancy_grid is None:
            self.get_logger().warn("Waiting for octomap data...")
            return

        # Convert the current pose and goal pose (with fixed z) to grid indices.
        start = (int(self.current_pose.x / self.grid_resolution),
                 int(self.current_pose.y / self.grid_resolution),
                 int(self.current_pose.z / self.grid_resolution))
        goal_fixed = (int(self.goal_pose.x / self.grid_resolution),
                      int(self.goal_pose.y / self.grid_resolution),
                      int(self.goal_pose.z / self.grid_resolution))

        self.get_logger().info(f"Attempting path planning at fixed altitude from {start} to {goal_fixed}...")
        path = self.astar(start, goal_fixed, self.occupancy_grid)

        # If no path is found at the fixed altitude, try the "over wall" option.
        if path is None:
            self.get_logger().warn("No path found at fixed altitude, attempting over wall option.")
            goal_over = (int(self.goal_pose.x / self.grid_resolution),
                         int(self.goal_pose.y / self.grid_resolution),
                         int(self.over_wall_z / self.grid_resolution))
            path = self.astar(start, goal_over, self.occupancy_grid)
            if path is None:
                self.get_logger().error("No path found even with over wall option!")
                return
            else:
                self.get_logger().info("Path found using over wall option.")
        else:
            self.get_logger().info("Path found at fixed altitude.")
        print(f'current_pose: {self.current_pose} :: path: {path}')
        self.publish_path(path)

    def astar(self, start, goal, grid):
        """A simple 3D A* search on a grid."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)
            closed_set.add(current)
            for neighbor in self.get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + 1  # uniform cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def get_neighbors(self, node, grid):
        neighbors = []
        x, y, z = node
        # Check 26-connected neighbors in 3D.
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                        if grid[nx, ny, nz] == 0:
                            neighbors.append((nx, ny, nz))
        return neighbors

    def heuristic(self, node, goal):
        # Euclidean distance heuristic.
        return ((node[0] - goal[0]) ** 2 +
                (node[1] - goal[1]) ** 2 +
                (node[2] - goal[2]) ** 2) ** 0.5

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def publish_path(self, grid_path):
        """Convert grid indices back to world coordinates and publish as a nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for grid_point in grid_path:
            real_x = grid_point[0] * self.grid_resolution
            real_y = grid_point[1] * self.grid_resolution
            real_z = grid_point[2] * self.grid_resolution
            waypoint = PoseStamped()
            waypoint.header.stamp = self.get_clock().now().to_msg()
            waypoint.header.frame_id = "map"
            waypoint.pose.position.x = real_x
            waypoint.pose.position.y = real_y
            waypoint.pose.position.z = real_z
            path_msg.poses.append(waypoint)
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published planned path.")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("A* Planner Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
