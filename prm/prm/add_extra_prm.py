import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx

# ----------------------- Utility Functions -----------------------

def generate_sample_pointcloud(path):
    """Load a point cloud from a PCD file."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def get_bounding_box(pcd):
    """Return the axis aligned bounding box of the point cloud."""
    bbox = pcd.get_axis_aligned_bounding_box()
    return bbox

def create_occupancy_map(pcd, voxel_size=0.5):
    """Creates an occupancy grid from a point cloud using a voxel grid."""
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
    """Randomly sample free voxels and return their world coordinates."""
    free_voxels = np.argwhere(occupancy_grid == 0)
    num_voxels = len(free_voxels)
    if num_voxels > 0:
        sampled_indices = np.random.choice(num_voxels, size=min(num_samples, num_voxels), replace=False)
        sampled_voxels = free_voxels[sampled_indices]
    else:
        sampled_voxels = np.array([])
    points_sample = sampled_voxels * voxel_size + np.array(min_bound)
    return points_sample

def is_collision_free(point1, point2, occupancy_grid, min_bound, voxel_size):
    """Check if the straight-line between two points is collision free."""
    line_points = np.linspace(point1, point2, num=100)
    for point in line_points:
        voxel_index = np.floor((point - min_bound) / voxel_size).astype(int)
        if occupancy_grid[tuple(voxel_index)] == 1:
            return False
    return True

def build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10):
    """Build a PRM roadmap connecting sampled points."""
    kdtree = cKDTree(sampled_points)
    roadmap = nx.Graph()
    for i, point in enumerate(sampled_points):
        roadmap.add_node(i, pos=point)
        distances, indices = kdtree.query(point, k=k + 1)
        for j in indices[1:]:
            neighbor_point = sampled_points[j]
            if is_collision_free(point, neighbor_point, occupancy_grid, min_bound, voxel_size):
                distance = np.linalg.norm(point - neighbor_point)
                roadmap.add_edge(i, j, weight=distance)
    return roadmap

def find_shortest_path(roadmap, start_point, goal_point, sampled_points):
    """Use A* search to find a path in the roadmap."""
    start_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - start_point, axis=1))
    goal_idx = np.argmin(np.linalg.norm(np.array(sampled_points) - goal_point, axis=1))
    try:
        path = nx.astar_path(roadmap, start_idx, goal_idx, weight='weight')
        return [sampled_points[i] for i in path]
    except nx.NetworkXNoPath:
        print("No path found")
        return []

def simplify_path(points, occupancy_grid, min_bound, voxel_size):
    """Simplify the computed path using a binary search approach."""
    simplified_points = [points[0]]
    i = 0
    while i < len(points) - 1:
        low = i + 1
        high = len(points) - 1
        candidate = i
        while low <= high:
            mid = (low + high) // 2
            if is_collision_free(points[i], points[mid], occupancy_grid, min_bound, voxel_size):
                candidate = mid
                low = mid + 1
            else:
                high = mid - 1
        if candidate > i:
            simplified_points.append(points[candidate])
            i = candidate
        else:
            i += 1
            simplified_points.append(points[i])
    return simplified_points

def compute_path_cost(path_points):
    """Compute the total Euclidean length of a path."""
    cost = 0.0
    for i in range(len(path_points) - 1):
        cost += np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i]))
    return cost

def get_color(idx, total):
    """Generate a distinct color for the idx-th path out of total paths (using a simple gradient)."""
    r = idx / total
    g = 1 - idx / total
    b = 0.5
    return [r, g, b]

def create_path_line_set(path_points, color):
    """Create an Open3D LineSet from a list of 3D points, using the given color."""
    vertices = o3d.utility.Vector3dVector(path_points)
    lines = [[i, i+1] for i in range(len(path_points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = vertices
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

def visualize_final_map(pcd, sampled_points, roadmap, path_line_sets, optimum_line_set):
    """Visualize the final map with the point cloud, roadmap, all computed paths, and the optimum path."""
    prm_pcd = o3d.geometry.PointCloud()
    prm_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    roadmap_lines = []
    for (i, j) in roadmap.edges():
        roadmap_lines.append([i, j])
    roadmap_line_set = o3d.geometry.LineSet()
    roadmap_line_set.points = prm_pcd.points
    roadmap_line_set.lines = o3d.utility.Vector2iVector(roadmap_lines)
    roadmap_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(roadmap_lines))])
    
    vis_objects = [pcd, prm_pcd, roadmap_line_set]
    for ls in path_line_sets:
        vis_objects.append(ls)
    
    # Add optimum path with a distinct color (magenta)
    if optimum_line_set is not None:
        vis_objects.append(optimum_line_set)
    
    o3d.visualization.draw_geometries(vis_objects)

# ----------------------- Main Function -----------------------

def main():
    # ------------------- User Parameters -------------------
    pointcloud_path = "/home/intel/fiverr/md/drone_ws/src/UAV_UGV/sjtu_drone_bringup/map/map.pcd"
    voxel_size = 0.2
    start_point = np.array([0.0, 0.0, 1.0])
    goal_point = np.array([-12.0, 0.0, 5.0])
    # Generalized sample range parameters
    sample_min = 500
    sample_max = 3000
    sample_gap = 500

    # ------------------- Initialization -------------------
    pcd = generate_sample_pointcloud(pointcloud_path)
    occupancy_grid, min_bound, voxel_size = create_occupancy_map(pcd, voxel_size=voxel_size)
    
    # List to hold computed path data (each dict stores sample count, simplified path, cost, and color)
    path_data = []
    
    total_samples = sample_min
    sampled_points = sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=total_samples)
    iteration = 0
    num_iterations = ((sample_max - sample_min) // sample_gap) + 1

    # ------------------- Compute Paths for Each Sample Count -------------------
    while total_samples <= sample_max:
        print(f"\nComputing roadmap and path with {total_samples} samples...")
        roadmap = build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10)
        shortest_path_points = find_shortest_path(roadmap, start_point, goal_point, sampled_points)
        if shortest_path_points:
            simplified = simplify_path(shortest_path_points, occupancy_grid, min_bound, voxel_size)
            cost = compute_path_cost(simplified)
            color = get_color(iteration, num_iterations)
            path_data.append({
                "samples": total_samples,
                "path": simplified,
                "cost": cost,
                "color": color
            })
            print(f"Path found with {total_samples} samples, distance: {cost:.2f}")
        else:
            print(f"No path found with {total_samples} samples.")
        
        if total_samples == sample_max:
            break
        
        extra = sample_free_points(occupancy_grid, min_bound, voxel_size, num_samples=sample_gap)
        sampled_points = np.vstack((sampled_points, extra))
        total_samples += sample_gap
        iteration += 1

    # ------------------- Determine the Optimum Path -------------------
    # Here the optimum path is defined as the one with the minimum total distance (cost) from start to goal.
    optimum = None
    if path_data:
        optimum = min(path_data, key=lambda x: x["cost"])
        print(f"\nOptimum path found with {optimum['samples']} samples, distance: {optimum['cost']:.2f}")
        optimum_line_set = create_path_line_set(optimum["path"], [1, 0, 1])  # Magenta for optimum
    else:
        optimum_line_set = None
        print("No valid path was found in any iteration.")
    
    # ------------------- Final Visualization -------------------
    final_roadmap = build_prm_roadmap(sampled_points, occupancy_grid, min_bound, voxel_size, k=10)
    path_line_sets = [create_path_line_set(data["path"], data["color"]) for data in path_data]
    
    visualize_final_map(pcd, sampled_points, final_roadmap, path_line_sets, optimum_line_set)

if __name__ == '__main__':
    main()
