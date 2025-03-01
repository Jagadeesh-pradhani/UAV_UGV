import open3d as o3d
import numpy as np
import json

def create_line_set_from_path(path, color):
    """
    Create an Open3D LineSet object from a list of points.
    
    :param path: List of points (each a list of 3 floats) representing the path.
    :param color: List of 3 floats representing the RGB color for the lines.
    :return: Open3D LineSet object.
    """
    if len(path) < 2:
        return None
    points = np.array(path)
    # Create line segments connecting consecutive points
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Set each line with the provided color
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

def main():
    # File paths (adjust these paths as needed)
    json_filename = "prm_paths_results.json"
    pcd_path = "/home/intel/fiverr/md/drone_ws/src/UAV_UGV/sjtu_drone_bringup/map/map.pcd"
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Load the results JSON file
    with open(json_filename, "r") as f:
        results = json.load(f)
    
    # Define colors for best paths and simplified paths for each sample size.
    # (If there are more than 6 sample sizes, colors will repeat.)
    best_colors = [
        [1, 0, 0],   # red
        [0, 1, 0],   # green
        [0, 0, 1],   # blue
        [1, 1, 0],   # yellow
        [1, 0, 1],   # magenta
        [0, 1, 1]    # cyan
    ]
    simplified_colors = [
        [0.5, 0, 0],   # dark red
        [0, 0.5, 0],   # dark green
        [0, 0, 0.5],   # dark blue
        [0.5, 0.5, 0], # olive
        [0.5, 0, 0.5], # purple
        [0, 0.5, 0.5]  # teal
    ]
    
    # List to hold all geometries to visualize
    geometries = [pcd]
    
    # Iterate through each result (each corresponding to a different sample size)
    for i, result in enumerate(results):
        num_samples = result["num_samples"]
        best_path = result["best_path"]
        simplified_path = result["simplified_path"]
        
        print(f"Visualizing sample size {num_samples}")
        if best_path and len(best_path) >= 2:
            ls_best = create_line_set_from_path(best_path, best_colors[i % len(best_colors)])
            if ls_best is not None:
                geometries.append(ls_best)
        if simplified_path and len(simplified_path) >= 2:
            ls_simplified = create_line_set_from_path(simplified_path, simplified_colors[i % len(simplified_colors)])
            if ls_simplified is not None:
                geometries.append(ls_simplified)
    
    # Visualize all objects together
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()
