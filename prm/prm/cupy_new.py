import cupy as cp
import faiss
import open3d as o3d
import numpy as np
import networkx as nx
import json
import time
import os
from joblib import Parallel, delayed

# --------------------------
# Configuration
# --------------------------
CUDA_VISIBLE_DEVICES = "0"
VOXEL_SIZE = 0.3            # Optimal voxel size for performance/accuracy
COLLISION_STEPS = 50        # Collision checking resolution
K_NEIGHBORS = 15            # Neighbor connections per node
BATCH_SIZE = 1024           # Memory-optimized batch processing
NUM_JOBS = -1               # Use all available CPU cores

# Initialize CUDA context
cp.cuda.Device().use()

# --------------------------
# FAISS GPU Configuration
# --------------------------
class FAISSGPUNeighbors:
    """GPU-accelerated nearest neighbor search using FAISS"""
    def __init__(self, dim=3):
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory
        self.index = faiss.GpuIndexFlatL2(self.res, dim)

    def build(self, points):
        self.index.reset()
        self.index.add(points.astype(np.float32))

    def search(self, queries, k):
        return self.index.search(queries.astype(np.float32), k)

# --------------------------
# Core PRM Components
# --------------------------
def create_occupancy_map_gpu(pcd):
    """Create GPU-accelerated occupancy grid"""
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = cp.asarray(bbox.get_min_bound())
    extent = cp.asarray(bbox.get_extent())
    
    grid_shape = cp.ceil(extent / VOXEL_SIZE).astype(int)
    points_gpu = cp.asarray(pcd.points)
    
    # Calculate voxel indices
    voxel_indices = cp.floor((points_gpu - min_bound) / VOXEL_SIZE).astype(int)
    valid_mask = cp.all((voxel_indices >= 0) & (voxel_indices < grid_shape), axis=1)
    
    # Build occupancy grid
    occupancy_gpu = cp.zeros(grid_shape.get(), dtype=cp.int8)
    valid_indices = voxel_indices[valid_mask]
    
    if valid_indices.size > 0:
        occupancy_gpu[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
    
    return occupancy_gpu, cp.asnumpy(min_bound)

def sample_free_points_gpu(occupancy_gpu, min_bound, num_samples):
    """GPU-accelerated free space sampling"""
    min_bound_gpu = cp.asarray(min_bound)
    free_voxels = cp.argwhere(occupancy_gpu == 0)
    
    if free_voxels.size == 0:
        return np.empty((0, 3))
    
    sample_indices = cp.random.choice(
        free_voxels.shape[0], 
        size=min(num_samples, free_voxels.shape[0]), 
        replace=False
    )
    
    sampled_points = (free_voxels[sample_indices] * VOXEL_SIZE) + min_bound_gpu
    return cp.asnumpy(sampled_points)

def is_collision_free_gpu(point1, point2, occupancy_gpu, min_bound):
    """Vectorized GPU collision checking"""
    # Convert inputs to CuPy arrays
    point1_gpu = cp.asarray(point1)
    point2_gpu = cp.asarray(point2)
    min_bound_gpu = cp.asarray(min_bound)
    
    line_points = cp.linspace(point1_gpu, point2_gpu, num=COLLISION_STEPS)
    voxel_indices = cp.floor((line_points - min_bound_gpu) / VOXEL_SIZE).astype(int)
    
    # Batch bounds checking
    valid_mask = cp.all((voxel_indices >= 0) & (voxel_indices < occupancy_gpu.shape), axis=1)
    if not valid_mask.any():
        return True
    
    # Batch collision check
    collisions = occupancy_gpu[
        voxel_indices[valid_mask, 0],
        voxel_indices[valid_mask, 1], 
        voxel_indices[valid_mask, 2]
    ]
    return not cp.any(collisions)

def build_prm_roadmap(points, occupancy_gpu, min_bound):
    """Memory-safe roadmap construction"""
    # Convert points to float32 numpy array for FAISS
    points_np = np.asarray(points, dtype=np.float32)
    
    # Create main index
    main_finder = FAISSGPUNeighbors()
    main_finder.build(points_np)
    
    roadmap = nx.Graph()
    roadmap.add_nodes_from(range(len(points)))

    def process_batch(batch_indices):
        # Thread-local FAISS resources
        with faiss.StandardGpuResources() as thread_res:
            thread_res.setTempMemory(512 * 1024 * 1024)
            
            # Clone index for thread safety
            index = faiss.GpuIndexFlatL2(thread_res, 3)
            index.copyFrom(main_finder.index)
            
            batch_edges = []
            for i in batch_indices:
                # Search using thread-local index
                distances, indices = index.search(points_np[i:i+1], K_NEIGHBORS+1)
                
                for j in indices[0][1:]:
                    if j < len(points) and is_collision_free_gpu(points[i], points[j], occupancy_gpu, min_bound):
                        distance = np.linalg.norm(points[i] - points[j])
                        batch_edges.append((i, j, distance))
            return batch_edges

    # Process with reduced parallelism
    batches = [range(i, min(i+BATCH_SIZE, len(points))) 
              for i in range(0, len(points), BATCH_SIZE)]
    
    results = Parallel(n_jobs=4, prefer="threads")(
        delayed(process_batch)(batch) for batch in batches
    )
    
    # Build roadmap
    for batch in results:
        for i, j, d in batch:
            roadmap.add_edge(i, j, weight=d)
    
    return roadmap


def simplify_path(points, occupancy_gpu, min_bound):
    """GPU-accelerated path simplification"""
    simplified = [points[0]]
    current_idx = 0
    
    while current_idx < len(points) - 1:
        next_idx = len(points) - 1
        while next_idx > current_idx + 1:
            mid = (current_idx + next_idx) // 2
            if is_collision_free_gpu(points[current_idx], points[mid], occupancy_gpu, min_bound):
                next_idx = mid
            else:
                next_idx -= 1
        simplified.append(points[next_idx])
        current_idx = next_idx
    
    return simplified

# --------------------------
# Main Execution
# --------------------------
def main():
    # Environment setup
    pcd_path = "/home/intel/fiverr/md/drone_ws/src/UAV_UGV/sjtu_drone_bringup/map/map.pcd"
    start_point = np.array([0.0, 0.0, 1.0])
    goal_point = np.array([-12.0, 0.0, 2.0])
    
    # Load and process point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    occupancy_gpu, min_bound_np = create_occupancy_map_gpu(pcd)
    min_bound = cp.asarray(min_bound_np) 
    
    # Main processing loop
    results = []
    for num_samples in range(3000, 3001, 1000):
        print(f"Processing {num_samples} samples...")
        t_start = time.time()
        
        # GPU-accelerated sampling
        sampled_points = sample_free_points_gpu(occupancy_gpu, min_bound, num_samples)
        
        # Build roadmap
        roadmap = build_prm_roadmap(sampled_points, occupancy_gpu, min_bound)
        
        # Path finding
        try:
            start_idx = np.argmin(np.linalg.norm(sampled_points - start_point, axis=1))
            goal_idx = np.argmin(np.linalg.norm(sampled_points - goal_point, axis=1))
            path = nx.astar_path(roadmap, start_idx, goal_idx, weight='weight')
            simplified = simplify_path([sampled_points[i] for i in path], occupancy_gpu, min_bound)
            path_length = sum(np.linalg.norm(simplified[i]-simplified[i-1]) for i in range(1,len(simplified)))
        except Exception as e:
            path_length = float('inf')
            simplified = []
        
        results.append({
            "samples": num_samples,
            "time": time.time() - t_start,
            "length": path_length,
            "path": [p.tolist() for p in simplified]
        })
    
    # Save results
    with open("prm_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    main()