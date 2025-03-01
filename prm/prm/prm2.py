import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import open3d as o3d
from ament_index_python.packages import get_package_share_directory

class PRMNode(Node):
    def __init__(self):
        super().__init__('prm_node')

        prm_path = get_package_share_directory('sjtu_drone_bringup')
        pcd_path = os.path.join(prm_path, 'map', 'map.pcd')

        # Parameters
        self.declare_parameter('pcd_path', pcd_path)
        self.declare_parameter('voxel_size', 0.2)
        self.declare_parameter('num_samples', 500)
        self.declare_parameter('k_neighbors', 10)
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('num_attempts', 5)  # Number of PRM attempts

        # Initialize parameters
        self.pcd_path = self.get_parameter('pcd_path').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        self.k_neighbors = self.get_parameter('k_neighbors').get_parameter_value().integer_value
        self.max_cost = self.get_parameter('max_cost').get_parameter_value().integer_value
        self.num_attempts = self.get_parameter('num_attempts').get_parameter_value().integer_value

        self.get_logger().info(f'Sampling {self.num_samples} points with {self.k_neighbors} neighbors...')

        # Subscribers and Publishers
        self.goal_subscriber = self.create_subscription(
            Point,
            '/drone_goal',
            self.goal_callback,
            10
        )
        self.odometry = self.create_subscription(
            Odometry, 
            "simple_drone/odom", 
            self.odom_callback, 
            10
        )
        self.path_publisher = self.create_publisher(Path, '/drone_path', 10)

        # Load Point Cloud and Initialize PRM-related maps
        self.get_logger().info('Loading point cloud...')
        self.pcd = self.generate_sample_pointcloud(self.pcd_path)
        self.occupancy_grid, self.min_bound, self.voxel_size = self.create_occupancy_map(self.pcd, self.voxel_size)
        self.costmap = self.generate_costmap(self.occupancy_grid, self.max_cost)
        self.get_logger().info('PRM initialized.')

    def odom_callback(self, msg):
        self.odom = msg

    def generate_sample_pointcloud(self, path):
        return o3d.io.read_point_cloud(path)

    def create_occupancy_map(self, pcd, voxel_size):
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        extent = bbox.get_extent()
        grid_shape = np.ceil(extent / voxel_size).astype(int)
        occupancy_grid = np.zeros(grid_shape, dtype=np.int8)
        points = np.asarray(pcd.points)
        for point in points:
            voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
            occupancy_grid[tuple(voxel_index)] = 1
        return occupancy_grid, min_bound, voxel_size

    def generate_costmap(self, occupancy_grid, max_cost):
        free_space = (occupancy_grid == 0).astype(np.float32)
        distance_to_obstacles = distance_transform_edt(free_space)
        normalized_distance = distance_to_obstacles / np.max(distance_to_obstacles)
        costmap = max_cost * (1 - normalized_distance)
        return costmap

    def sample_free_points(self, occupancy_grid, min_bound, voxel_size, num_samples):
        free_voxels = np.argwhere(occupancy_grid == 0)
        num_voxels = len(free_voxels)
        if num_voxels > 0:
            sampled_indices = np.random.choice(num_voxels, size=min(num_samples, num_voxels), replace=False)
            sampled_voxels = free_voxels[sampled_indices]
        else:
            sampled_voxels = np.array([])
        points_sample = sampled_voxels * voxel_size + np.array(min_bound)
        return points_sample

    def build_prm_roadmap_with_costmap(self, sampled_points, occupancy_grid, costmap, min_bound, voxel_size, k):
        kdtree = cKDTree(sampled_points)
        roadmap = nx.Graph()
        for i, point in enumerate(sampled_points):
            roadmap.add_node(i, pos=point)
            distances, indices = kdtree.query(point, k=k + 1)
            for j in indices[1:]:
                neighbor_point = sampled_points[j]
                if self.is_collision_free(point, neighbor_point, occupancy_grid, min_bound, voxel_size):
                    distance = np.linalg.norm(point - neighbor_point)
                    cost = self.calculate_path_cost(point, neighbor_point, costmap, min_bound, voxel_size)
                    roadmap.add_edge(i, j, weight=distance + cost)
        return roadmap

    def is_collision_free(self, point1, point2, occupancy_grid, min_bound, voxel_size):
        line_points = np.linspace(point1, point2, num=100)
        for point in line_points:
            voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
            if np.any(voxel_index < 0) or np.any(voxel_index >= occupancy_grid.shape):
                continue
            if occupancy_grid[tuple(voxel_index)] == 1:
                return False
        return True

    def calculate_path_cost(self, point1, point2, costmap, min_bound, voxel_size):
        line_points = np.linspace(point1, point2, num=100)
        cost = 0
        for point in line_points:
            voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
            if np.any(voxel_index < 0) or np.any(voxel_index >= costmap.shape):
                continue
            cost += costmap[tuple(voxel_index)]
        return cost / len(line_points)

    def find_shortest_path(self, roadmap, start_point, goal_point, sampled_points):
        # Find the nearest node indices for start and goal
        start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - goal_point, axis=1))
        try:
            path_indices = nx.astar_path(roadmap, start_idx, goal_idx, weight='weight')
            return [sampled_points[i] for i in path_indices]
        except nx.NetworkXNoPath:
            self.get_logger().error("No path found in current attempt.")
            return []

    def simplify_path(self, points, occupancy_grid, min_bound, voxel_size):
        if len(points) == 0:
            return []
        simplified_points = [points[0]]  # Always keep the start point
        end_point = points[-1]
        i = 0  # Start from the first point

        while i < len(points) - 1:
            low = i + 1
            high = len(points) - 1
            candidate = i  # Farthest collision-free index from points[i]

            # Binary search for the farthest collision-free point from points[i]
            while low <= high:
                mid = (low + high) // 2
                if self.is_collision_free(points[i], points[mid], occupancy_grid, min_bound, voxel_size):
                    candidate = mid  # Segment is collision-free; try to go further
                    low = mid + 1
                else:
                    high = mid - 1

            if candidate > i:
                simplified_points.append(points[candidate])
                i = candidate
            else:
                # Advance one step if no candidate found to avoid infinite loop
                i += 1
                simplified_points.append(points[i])
        # if simplified_points[-1] != end_point:
        simplified_points.append(end_point)  # Always keep the end point

        return simplified_points

    def goal_callback(self, msg):
        goal_point = np.array([msg.x, msg.y, msg.z])
        start_point = np.array([
            self.odom.pose.pose.position.x, 
            self.odom.pose.pose.position.y, 
            self.odom.pose.pose.position.z
        ])

        best_path = []
        best_length = float('inf')

        # Run PRM for a number of attempts to find multiple candidate paths
        for attempt in range(self.num_attempts):
            self.get_logger().info(f'Attempt {attempt + 1} of {self.num_attempts}')
            sampled_points = self.sample_free_points(self.occupancy_grid, self.min_bound, self.voxel_size, self.num_samples)
            roadmap = self.build_prm_roadmap_with_costmap(
                sampled_points, self.occupancy_grid, self.costmap, self.min_bound, self.voxel_size, self.k_neighbors
            )
            current_path = self.find_shortest_path(roadmap, start_point, goal_point, sampled_points)
            if not current_path:
                continue

            # Compute the total Euclidean length of the path
            current_length = sum(np.linalg.norm(current_path[i] - current_path[i-1]) for i in range(1, len(current_path)))
            self.get_logger().info(f'Attempt {attempt + 1}: Path length = {current_length}')

            if current_length < best_length:
                best_length = current_length
                best_path = current_path

        if best_path:
            simplified_path_points = self.simplify_path(best_path, self.occupancy_grid, self.min_bound, self.voxel_size)
            self.publish_path(best_path)
        else:
            self.get_logger().error('No valid path found after all attempts.')

    def publish_path(self, path_points):
        path_msg = Path()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for point in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PRMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
