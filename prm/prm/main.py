import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx

# Generate a point cloud from a PCD file
def generate_sample_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd

# Function to calculate the bounding box dimensions
def get_bounding_box(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    return bbox

# Create an occupancy map based on the point cloud and bounding box
def create_occupancy_map(pcd, voxel_size=0.5):
    bbox = get_bounding_box(pcd)
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = bbox.get_extent()
    grid_shape = np.ceil(extent / voxel_size).astype(int)
    occupancy_grid = np.zeros(grid_shape, dtype=np.int8)
    
    points = np.asarray(pcd.points)
    for point in points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        occupancy_grid[tuple(voxel_index)] = 1
    
    return occupancy_grid, min_bound, voxel_size

def sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=1000):
    # Convert the occupancy grid to a boolean mask where True means free
    free_voxels = np.argwhere(occupancy_grid == 0)
  
    # Randomly sample from the free voxels
    num_voxels = len(free_voxels)
    if num_voxels > 0:
        sampled_indices = np.random.choice(num_voxels, size=min(num_samples, num_voxels), replace=False)
        sampled_voxels = free_voxels[sampled_indices]
    else:
        sampled_voxels = np.array([])

    # Convert voxel indices to world coordinates (multiply by voxel_size and add min_bound)
    points_sample = sampled_voxels * voxel_size + np.array(min_bound)

    return points_sample

# Check if a straight-line path between two points is collision-free
def is_collision_free(point1, point2, occupancy_grid, min_bound, voxel_size):
    line_points = np.linspace(point1, point2, num=100)
    for point in line_points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        if occupancy_grid[tuple(voxel_index)] == 1:  # Collision with occupied voxel
            return False
    return True

# Build the PRM roadmap
def build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10):
    kdtree = cKDTree(sampled_points)
    roadmap = nx.Graph()

    for i, point in enumerate(sampled_points):
        roadmap.add_node(i, pos=point)
        distances, indices = kdtree.query(point, k=k + 1)  # k + 1 because query returns the point itself
        for j in indices[1:]:  # Skip the first neighbor (itself)
            neighbor_point = sampled_points[j]
            if is_collision_free(point, neighbor_point, occupancy_grid, min_bound, voxel_size):
                distance = np.linalg.norm(point - neighbor_point)
                roadmap.add_edge(i, j, weight=distance)
    
    return roadmap

# Find the shortest path using A* search
def find_shortest_path(roadmap, start_point, goal_point, sampled_points):
    start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - start_point, axis=1))
    goal_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - goal_point, axis=1))
    
    try:
        path = nx.astar_path(roadmap, start_idx, goal_idx, weight='weight')
        return [sampled_points[i] for i in path]
    except nx.NetworkXNoPath:
        print("No path found")
        return []

def simplify_path(points, occupancy_grid, min_bound, voxel_size):
    """
    Simplify a list of points by removing intermediate points that lie on collision-free paths.

    :param points: List of points (numpy array or list of lists) representing the path.
    :param occupancy_grid: 3D occupancy grid (numpy array) indicating obstacles (1 = occupied, 0 = free).
    :param min_bound: Minimum bound of the occupancy grid (numpy array of 3 values).
    :param voxel_size: Size of each voxel in the occupancy grid (single float or numpy array of 3 values).
    :return: Simplified list of points.
    """
    simplified_points = [points[0]]  # Always keep the start point
    i = 0  # Start from the first point

    while i < len(points) - 1:
        # Find the farthest point that is collision-free from the current point
        for j in range(len(points) - 1, i, -1):
            if is_collision_free(points[i], points[j], occupancy_grid, min_bound, voxel_size):
                # If a collision-free path exists, add this point and skip intermediate points
                simplified_points.append(points[j])
                i = j  # Move the starting point to the current endpoint
                break
        else:
            # If no further point is collision-free, just move to the next point
            i += 1

    return simplified_points

def visualize_prm_over_pointcloud(pcd, sampled_points, roadmap, shortest_path_points=None, simple_path=None):
    """
    Visualize the PRM roadmap over the point cloud, including the shortest path and simple path.

    :param pcd: Open3D point cloud object for the environment.
    :param sampled_points: List of sampled points in the roadmap.
    :param roadmap: Graph representing the PRM roadmap (edges and nodes).
    :param shortest_path_points: List of points representing the shortest path.
    :param simple_path: List of points representing the simplified path.
    """
    # Create an Open3D point cloud for the sampled points
    prm_pcd = o3d.geometry.PointCloud()
    prm_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # Create line segments for the roadmap connections (edges)
    lines = []
    for (i, j) in roadmap.edges():
        lines.append([i, j])
    
    # Create line set for the roadmap
    line_set = o3d.geometry.LineSet()
    line_set.points = prm_pcd.points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Color the roadmap connections (edges) - default color is white
    line_colors = [[0, 0, 0] for _ in range(len(lines))]  # White color for the roadmap lines
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    # Add the shortest path (in red) if provided
    visualization_objects = [pcd, prm_pcd, line_set]  # Start with basic objects
    if shortest_path_points is not None and len(shortest_path_points) > 1:
        path_lines = []
        for i in range(len(shortest_path_points) - 1):
            # Find the indices of the points in the sampled_points array
            start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - shortest_path_points[i], axis=1))
            end_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - shortest_path_points[i + 1], axis=1))
            path_lines.append([start_idx, end_idx])
        
        # Create a line set for the shortest path
        path_line_set = o3d.geometry.LineSet()
        path_line_set.points = prm_pcd.points  # Use the same points as the roadmap
        path_line_set.lines = o3d.utility.Vector2iVector(path_lines)
        
        # Color the shortest path in red
        path_colors = [[1, 0, 0] for _ in range(len(path_lines))]  # Red color for the shortest path
        path_line_set.colors = o3d.utility.Vector3dVector(path_colors)
        visualization_objects.append(path_line_set)
    
    # Add the simplified path (in blue) if provided
    if simple_path is not None and len(simple_path) > 1:
        simple_path_lines = []
        for i in range(len(simple_path) - 1):
            # Find the indices of the points in the sampled_points array
            start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - simple_path[i], axis=1))
            end_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - simple_path[i + 1], axis=1))
            simple_path_lines.append([start_idx, end_idx])
        
        # Create a line set for the simplified path
        simple_path_line_set = o3d.geometry.LineSet()
        simple_path_line_set.points = prm_pcd.points  # Use the same points as the roadmap
        simple_path_line_set.lines = o3d.utility.Vector2iVector(simple_path_lines)
        
        # Color the simplified path in blue
        simple_path_colors = [[1, 1, 0] for _ in range(len(simple_path_lines))]  # Blue color for the simplified path
        simple_path_line_set.colors = o3d.utility.Vector3dVector(simple_path_colors)
        visualization_objects.append(simple_path_line_set)
    
    # Visualize all objects
    o3d.visualization.draw_geometries(visualization_objects)

# Main function
def main():
    # Load the point cloud
    pcd = generate_sample_pointcloud("/home/intel/fiverr/md/drone_ws/src/sjtu_drone_bringup/map/map/map.pcd")

    # Create an occupancy map
    voxel_size = 0.2
    occupancy_grid, min_bound, voxel_size = create_occupancy_map(pcd, voxel_size=voxel_size)

    # Sample 1000 free voxels
    sampled_points = sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=1000)

    # Build the PRM roadmap
    roadmap = build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10)

    # Define the start and goal points
    start_point = np.array([0.0,0.0,1.0])  # Replace with actual start point
    goal_point = np.array([14.0,0.0, 5.0])   # Replace with actual goal point

    # Find the shortest path using A* search
    shortest_path_points = find_shortest_path(roadmap, start_point, goal_point, sampled_points)
    print(f"Path : {shortest_path_points}")
    #semplify path to get the shortest path 
    simplified_path=simplify_path(shortest_path_points, occupancy_grid, min_bound, voxel_size)
    print(f"Simplified Path : {simplified_path}")
    # Visualize the PRM roadmap and the shortest path (if found)
    visualize_prm_over_pointcloud(pcd, sampled_points, roadmap, shortest_path_points,simple_path=simplified_path)
    # visualized simplified path 
    
# Run the main function
main()