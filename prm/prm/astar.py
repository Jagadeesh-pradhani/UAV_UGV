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
        
        # Subscribers for goal, odometry, and octomap
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
        
        # Publisher for the full path
        self.path_pub = self.create_publisher(Path, '/drone_path', 10)
        
        # Internal state
        self.current_pose = None   # geometry_msgs/Point
        self.goal_pose = None      # geometry_msgs/Point
        self.occupancy_grid = None # 3D numpy array (occupancy grid)
        
        # Grid parameters (adjust these as needed)
        self.grid_resolution = 1.0  # meters per grid cell
        self.grid_size = (100, 100, 20)  # grid dimensions (x, y, z)

    def goal_callback(self, msg: PoseStamped):
        # Save the target goal
        self.goal_pose = msg.pose.position
        self.get_logger().info(f"Received new goal: x={self.goal_pose.x}, y={self.goal_pose.y}, z={self.goal_pose.z}")
        self.attempt_plan()

    def odom_callback(self, msg: Odometry):
        # Save current position of the drone
        self.current_pose = msg.pose.pose.position

    def octomap_callback(self, msg: Octomap):
        # Here, convert the incoming Octomap message to a 3D occupancy grid.
        # This placeholder simply creates an empty grid.
        # In your real application, parse msg.data to mark obstacles.
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        self.get_logger().info("Octomap received and occupancy grid updated.")
        # self.attempt_plan()

    def attempt_plan(self):
        if self.goal_pose:
            self.goal_pose.z = 1.0  # Set a fixed altitude for now
        # Ensure that current pose, goal, and occupancy grid are available
        if self.current_pose is None:
            self.get_logger().warn("Current pose not received yet; waiting for odometry...")
            return
        if self.goal_pose is None:
            self.get_logger().warn("No goal received yet; waiting for goal pose...")
            return
        if self.occupancy_grid is None:
            self.get_logger().warn("Occupancy grid not available yet; waiting for octomap data...")
            return

        # Convert real-world positions to grid indices
        start = (int(self.current_pose.x / self.grid_resolution),
                 int(self.current_pose.y / self.grid_resolution),
                 int(self.current_pose.z / self.grid_resolution))
        goal = (int(self.goal_pose.x / self.grid_resolution),
                int(self.goal_pose.y / self.grid_resolution),
                int(self.goal_pose.z / self.grid_resolution))
                
        self.get_logger().info(f"Planning from grid {start} to {goal}...")
        path = self.astar(start, goal, self.occupancy_grid)
        print(f'current_pose: {self.current_pose} :: path: {path}')
        if path is None:
            self.get_logger().error("No path found!")
        else:
            self.get_logger().info(f"Path found with {len(path)} waypoints.")
            self.publish_path(path)

    def astar(self, start, goal, grid):
        """A simple 3D A* implementation on a grid."""
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
        # 26-connected neighbors in 3D
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    # Check grid boundaries
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                        # Only add free cells (cell value 0)
                        if grid[nx, ny, nz] == 0:
                            neighbors.append((nx, ny, nz))
        return neighbors

    def heuristic(self, node, goal):
        # Euclidean distance heuristic
        return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2 + (node[2] - goal[2]) ** 2) ** 0.5

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def publish_path(self, grid_path):
        """Convert grid indices back to world coordinates and publish as a nav_msgs/Path message."""
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
            # Orientation can be set if needed
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
