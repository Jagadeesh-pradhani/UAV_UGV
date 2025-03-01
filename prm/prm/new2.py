import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import json

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
    extent = bbox.get_extent()
    grid_shape = np.ceil(extent / voxel_size).astype(int)
    occupancy_grid = np.zeros(grid_shape, dtype=np.int8)
    
    points = np.asarray(pcd.points)
    for point in points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        occupancy_grid[tuple(voxel_index)] = 1
    
    return occupancy_grid, min_bound, voxel_size

def sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=1000):
    free_voxels = np.argwhere(occupancy_grid == 0)
    num_voxels = len(free_voxels)
    if num_voxels > 0:
        sampled_indices = np.random.choice(num_voxels, size=min(num_samples, num_voxels), replace=False)
        sampled_voxels = free_voxels[sampled_indices]
    else:
        sampled_voxels = np.array([])
    points_sample = sampled_voxels * voxel_size + np.array(min_bound)
    return points_sample

# Check if a straight-line path between two points is collision-free
def is_collision_free(point1, point2, occupancy_grid, min_bound, voxel_size):
    line_points = np.linspace(point1, point2, num=100)
    for point in line_points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        # Ensure the index is within the grid bounds
        if np.any(voxel_index < 0) or np.any(voxel_index >= occupancy_grid.shape):
            continue
        if occupancy_grid[tuple(voxel_index)] == 1:  # Collision with occupied voxel
            return False
    return True

# Build the PRM roadmap
def build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10):
    kdtree = cKDTree(sampled_points)
    roadmap = nx.Graph()

    for i, point in enumerate(sampled_points):
        roadmap.add_node(i, pos=point)
        distances, indices = kdtree.query(point, k=k + 1)  # k+1 because the point itself is included
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
    simplified_points = [points[0]]  # Always keep the start point
    i = 0  # Start from the first point

    while i < len(points) - 1:
        low = i + 1
        high = len(points) - 1
        candidate = i  # Will store the farthest collision-free index found

        # Binary search to find the farthest collision-free point from points[i]
        while low <= high:
            mid = (low + high) // 2
            if is_collision_free(points[i], points[mid], occupancy_grid, min_bound, voxel_size):
                candidate = mid  # Update candidate because the segment is collision-free
                low = mid + 1  # Try to see if we can go further
            else:
                high = mid - 1  # Look in the lower half

        if candidate > i:
            simplified_points.append(points[candidate])
            i = candidate  # Jump to the farthest collision-free point
        else:
            i += 1
            simplified_points.append(points[i])
    return simplified_points

# Main function (no visualization)
def main():
    # Load the point cloud
    pcd_path = "/home/intel/fiverr/md/drone_ws/src/UAV_UGV/sjtu_drone_bringup/map/map.pcd"
    pcd = generate_sample_pointcloud(pcd_path)

    # Create an occupancy map
    voxel_size = 0.2
    occupancy_grid, min_bound, voxel_size = create_occupancy_map(pcd, voxel_size=voxel_size)

    # Parameters for PRM and attempts
    k_neighbors = 10
    num_attempts = 5

    # Define start and goal points
    start_point = np.array([0.0, 0.0, 1.0])  # Replace with actual start point if needed
    goal_point = np.array([-12.0, 0.0, 2.0])   # Replace with actual goal point if needed

    results = []  # List to hold results for each sample size

    # Run simulation for sample sizes from 500 to 3000 in steps of 500
    for current_samples in range(500, 3001, 500):
        print(f"Running simulation for num_samples = {current_samples}")
        best_path = None
        best_length = float('inf')
        best_sampled_points = None
        best_roadmap = None

        # Run multiple attempts for the current sample size
        for attempt in range(num_attempts):
            print(f"  Attempt {attempt+1} of {num_attempts}")
            sampled_points = sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=current_samples)
            roadmap = build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=k_neighbors)
            candidate_path = find_shortest_path(roadmap, start_point, goal_point, sampled_points)
            
            if candidate_path:
                path_length = 0.0
                for i in range(1, len(candidate_path)):
                    path_length += np.linalg.norm(candidate_path[i] - candidate_path[i-1])
                print(f"    Attempt {attempt+1}: Path length = {path_length}")
                
                if path_length < best_length:
                    best_length = path_length
                    best_path = candidate_path
                    best_sampled_points = sampled_points
                    best_roadmap = roadmap
            else:
                print(f"    Attempt {attempt+1}: No path found.")
        
        if best_path:
            simplified_path = simplify_path(best_path, occupancy_grid, min_bound, voxel_size)
            print(f"Best path length for {current_samples} samples: {best_length}")
            # Convert numpy arrays to lists for JSON serialization
            best_path_list = [point.tolist() for point in best_path]
            simplified_path_list = [point.tolist() for point in simplified_path]
        else:
            best_path_list = []
            simplified_path_list = []
            print(f"No valid path found for {current_samples} samples.")
        
        results.append({
            "num_samples": current_samples,
            "best_path_length": best_length if best_path else None,
            "best_path": best_path_list,
            "simplified_path": simplified_path_list
        })
    
    # Save results to a file (JSON format)
    output_filename = "prm_paths_results.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_filename}")

# Run the main function
if __name__ == "__main__":
    main()
