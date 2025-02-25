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
    # Find indices of voxels that are free (i.e. not occupied)
    free_voxels = np.argwhere(occupancy_grid == 0)
  
    num_voxels = len(free_voxels)
    if num_voxels > 0:
        sampled_indices = np.random.choice(num_voxels, size=min(num_samples, num_voxels), replace=False)
        sampled_voxels = free_voxels[sampled_indices]
    else:
        sampled_voxels = np.array([])

    # Convert voxel indices to world coordinates
    points_sample = sampled_voxels * voxel_size + np.array(min_bound)

    return points_sample

def sample_free_points_above(occupancy_grid, min_bound, voxel_size, num_samples=500, z_altitude=0):
    """
    Sample points over the same horizontal area as the occupancy grid but
    with the z-coordinate fixed to z_altitude.
    """
    grid_shape = occupancy_grid.shape
    # Compute x and y limits based on occupancy grid dimensions.
    x_max = min_bound[0] + grid_shape[0] * voxel_size
    y_max = min_bound[1] + grid_shape[1] * voxel_size
    xs = np.random.uniform(min_bound[0], x_max, num_samples)
    ys = np.random.uniform(min_bound[1], y_max, num_samples)
    zs = np.full(num_samples, z_altitude)
    points = np.vstack((xs, ys, zs)).T
    return points

def is_collision_free(point1, point2, occupancy_grid, min_bound, voxel_size):
    """
    Check if a straight-line path between two points is collision-free.
    If a sample point falls outside the occupancy grid bounds, assume it is free.
    """
    line_points = np.linspace(point1, point2, num=100)
    grid_shape = occupancy_grid.shape
    for point in line_points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        # If the index is outside the grid, assume free (i.e. no obstacle information)
        if np.any(voxel_index < 0) or np.any(voxel_index >= grid_shape):
            continue
        if occupancy_grid[tuple(voxel_index)] == 1:  # Collision with an obstacle voxel
            return False
    return True

def build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10):
    """
    Build a probabilistic roadmap (PRM) graph using k-nearest neighbors.
    """
    kdtree = cKDTree(sampled_points)
    roadmap = nx.Graph()

    for i, point in enumerate(sampled_points):
        roadmap.add_node(i, pos=point)
        distances, indices = kdtree.query(point, k=k + 1)  # k+1 because the query returns the point itself first
        for j in indices[1:]:
            neighbor_point = sampled_points[j]
            if is_collision_free(point, neighbor_point, occupancy_grid, min_bound, voxel_size):
                distance = np.linalg.norm(point - neighbor_point)
                roadmap.add_edge(i, j, weight=distance)
    
    return roadmap

def find_shortest_path(roadmap, start_point, goal_point, sampled_points):
    """
    Use A* search on the PRM graph to find a path between the nodes
    nearest to start_point and goal_point.
    """
    start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - start_point, axis=1))
    goal_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - goal_point, axis=1))
    
    try:
        path = nx.astar_path(roadmap, start_idx, goal_idx, weight='weight')
        return [sampled_points[i] for i in path]
    except nx.NetworkXNoPath:
        print("No path found in current roadmap")
        return []

def simplify_path(points, occupancy_grid, min_bound, voxel_size):
    """
    Simplify a list of points by removing unnecessary intermediate points.
    """
    if len(points) < 2:
        return points
    
    simplified_points = [points[0]]
    i = 0

    while i < len(points) - 1:
        # Try to skip as many intermediate points as possible.
        for j in range(len(points) - 1, i, -1):
            if is_collision_free(points[i], points[j], occupancy_grid, min_bound, voxel_size):
                simplified_points.append(points[j])
                i = j
                break
        else:
            i += 1

    return simplified_points

def visualize_prm_over_pointcloud(pcd, sampled_points, roadmap, shortest_path_points=None, simple_path=None):
    """
    Visualize the point cloud, the PRM nodes and edges, and (if available)
    the planned shortest and simplified paths.
    """
    # Create a point cloud for the sampled PRM points
    prm_pcd = o3d.geometry.PointCloud()
    prm_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # Create line segments for all roadmap edges
    lines = []
    for (i, j) in roadmap.edges():
        lines.append([i, j])
    
    # Roadmap as a line set (default color black)
    line_set = o3d.geometry.LineSet()
    line_set.points = prm_pcd.points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_colors = [[0, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    visualization_objects = [pcd, prm_pcd, line_set]
    
    # Visualize the shortest path (in red)
    if shortest_path_points is not None and len(shortest_path_points) > 1:
        path_lines = []
        for i in range(len(shortest_path_points) - 1):
            start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - shortest_path_points[i], axis=1))
            end_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - shortest_path_points[i + 1], axis=1))
            path_lines.append([start_idx, end_idx])
        
        path_line_set = o3d.geometry.LineSet()
        path_line_set.points = prm_pcd.points
        path_line_set.lines = o3d.utility.Vector2iVector(path_lines)
        path_colors = [[1, 0, 0] for _ in range(len(path_lines))]  # Red color
        path_line_set.colors = o3d.utility.Vector3dVector(path_colors)
        visualization_objects.append(path_line_set)
    
    # Visualize the simplified path (in yellow)
    if simple_path is not None and len(simple_path) > 1:
        simple_path_lines = []
        for i in range(len(simple_path) - 1):
            start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - simple_path[i], axis=1))
            end_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - simple_path[i + 1], axis=1))
            simple_path_lines.append([start_idx, end_idx])
        
        simple_path_line_set = o3d.geometry.LineSet()
        simple_path_line_set.points = prm_pcd.points
        simple_path_line_set.lines = o3d.utility.Vector2iVector(simple_path_lines)
        simple_path_colors = [[1, 1, 0] for _ in range(len(simple_path_lines))]  # Yellow color
        simple_path_line_set.colors = o3d.utility.Vector3dVector(simple_path_colors)
        visualization_objects.append(simple_path_line_set)
    
    o3d.visualization.draw_geometries(visualization_objects)

# Main function
def main():
    # Load the point cloud (adjust the path as needed)
    pcd = generate_sample_pointcloud("/home/jagadeesh/yasyn/src/UAV_UGV/sjtu_drone_bringup/map/map.pcd")

    # Create an occupancy grid from the point cloud
    voxel_size = 0.2
    occupancy_grid, min_bound, voxel_size = create_occupancy_map(pcd, voxel_size=voxel_size)

    # Sample free points from the occupancy grid (ground-level samples)
    sampled_points = sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=1000)

    # Build the initial PRM roadmap from ground-level free points
    roadmap = build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10)

    # Define the start and goal points (adjust as needed)
    start_point = np.array([1.0, 0.0, 1.0])
    goal_point  = np.array([-2.0, 3.0, 1.0])

    # Try to find a path using the ground-level roadmap
    shortest_path_points = find_shortest_path(roadmap, start_point, goal_point, sampled_points)
    
    # If a path is found, use it; otherwise, try to plan a path by "lifting" above obstacles.
    if shortest_path_points:
        print("Path found with initial ground-level sampling.")
    else:
        print("No path found at ground level. Attempting to plan above obstacles...")
        # Get the highest z-value in the point cloud (assumed obstacle top)
        obstacle_max_z = np.max(np.asarray(pcd.points)[:,2])
        found_path = False
        max_margin = 10.0  # maximum vertical offset to try (in meters)
        margin = 1.0
        
        while margin <= max_margin and not found_path:
            z_altitude = obstacle_max_z + margin
            # Create lifted versions of start and goal (vertical climb points)
            lifted_start = np.array([start_point[0], start_point[1], z_altitude])
            lifted_goal  = np.array([goal_point[0],  goal_point[1],  z_altitude])
            
            # Sample extra free points at this higher altitude
            extra_points = sample_free_points_above(occupancy_grid, min_bound, voxel_size, num_samples=500, z_altitude=z_altitude)
            
            # Combine the original ground samples, the extra high-altitude samples,
            # and the lifted start and goal points.
            combined_points = np.vstack((sampled_points, extra_points, 
                                         lifted_start.reshape(1, 3), lifted_goal.reshape(1, 3)))
            
            # Rebuild the roadmap with the combined set of points
            roadmap = build_prm_roadmap(combined_points, occupancy_grid, min_bound, voxel_size, k=10)
            
            # Plan between the lifted start and goal points
            lifted_path = find_shortest_path(roadmap, lifted_start, lifted_goal, combined_points)
            
            if lifted_path:
                # Connect vertically from the original start to the lifted start,
                # then follow the lifted path, then drop vertically from the lifted goal to the original goal.
                shortest_path_points = [start_point, lifted_start] + lifted_path[1:-1] + [lifted_goal, goal_point]
                sampled_points = combined_points  # update samples for visualization
                print(f"Path found by lifting to z = {z_altitude:.2f} (vertical offset = {margin} m)")
                found_path = True
            else:
                margin += 1.0  # Increase the vertical offset and try again
        
        if not found_path:
            print("No path found even after attempting to plan above obstacles.")

    print(f"Path: {shortest_path_points}")
    # Simplify the path
    simplified_path = simplify_path(shortest_path_points, occupancy_grid, min_bound, voxel_size)
    print(f"Simplified Path: {simplified_path}")

    # Visualize the point cloud, roadmap, and paths
    visualize_prm_over_pointcloud(pcd, sampled_points, roadmap, shortest_path_points, simple_path=shortest_path_points)

# Run the main function
if __name__ == "__main__":
    main()
