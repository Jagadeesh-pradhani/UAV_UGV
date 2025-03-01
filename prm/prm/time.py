import json
import numpy as np

def calculate_travel_time(path,
                          desired_linear_vel=1.3,
                          lookahead_distance=0.3,
                          min_angle_tolerance=0.3,
                          max_linear_vel_z=0.43,
                          approach_velocity_scaling_dist=0.1):
    """
    Calculate the travel time along a path based on a simple velocity model.
    
    :param path: List of 3D points (each a list of three floats).
    :param desired_linear_vel: Nominal linear velocity (m/s).
    :param lookahead_distance: Distance threshold (m) for scaling velocity on short segments.
    :param min_angle_tolerance: Minimum turning angle (radians) to trigger deceleration.
    :param max_linear_vel_z: Maximum allowed velocity (m/s) when turning.
    :param approach_velocity_scaling_dist: Distance (m) to the goal over which speed is scaled down.
    :return: Estimated travel time (seconds).
    """
    travel_time = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        segment_length = np.linalg.norm(p2 - p1)
        
        # Start with the nominal desired velocity.
        effective_vel = desired_linear_vel
        
        # Scale velocity for short segments.
        if segment_length < lookahead_distance:
            effective_vel = desired_linear_vel * (segment_length / lookahead_distance)
        
        # Check turning angle if there's a previous segment.
        if i > 0:
            prev_segment = np.array(path[i]) - np.array(path[i-1])
            curr_segment = p2 - p1
            norm_prev = np.linalg.norm(prev_segment)
            norm_curr = np.linalg.norm(curr_segment)
            if norm_prev > 0 and norm_curr > 0:
                cos_angle = np.dot(prev_segment, curr_segment) / (norm_prev * norm_curr)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > min_angle_tolerance:
                    effective_vel = min(effective_vel, max_linear_vel_z)
        
        # Scale down velocity when near the goal.
        distance_to_goal = np.linalg.norm(np.array(path[-1]) - p1)
        if distance_to_goal < approach_velocity_scaling_dist:
            effective_vel = min(effective_vel, desired_linear_vel * (distance_to_goal / approach_velocity_scaling_dist))
        
        # Avoid division by zero.
        if effective_vel < 1e-3:
            effective_vel = 1e-3
        
        travel_time += segment_length / effective_vel
    return travel_time

def main():
    input_filename = "prm_paths_results.json"   # Input JSON file with stored paths.
    output_filename = "prm_paths_with_travel_time.json"  # Output JSON file with travel time added.
    
    # Load the existing results JSON.
    with open(input_filename, "r") as f:
        results = json.load(f)
    
    # For each entry (each sample size) in the JSON, calculate travel time.
    for entry in results:
        # Compute travel time for the best path if available.
        if "best_path" in entry and entry["best_path"]:
            best_path = entry["best_path"]
            best_travel_time = calculate_travel_time(
                best_path,
                desired_linear_vel=1.3,
                lookahead_distance=0.3,
                min_angle_tolerance=0.3,
                max_linear_vel_z=0.43,
                approach_velocity_scaling_dist=0.1
            )
            entry["best_path_travel_time"] = best_travel_time
        else:
            entry["best_path_travel_time"] = None

        # Compute travel time for the simplified path if available.
        if "simplified_path" in entry and entry["simplified_path"]:
            simplified_path = entry["simplified_path"]
            simplified_travel_time = calculate_travel_time(
                simplified_path,
                desired_linear_vel=1.3,
                lookahead_distance=0.3,
                min_angle_tolerance=0.3,
                max_linear_vel_z=0.43,
                approach_velocity_scaling_dist=0.1
            )
            entry["simplified_path_travel_time"] = simplified_travel_time
        else:
            entry["simplified_path_travel_time"] = None

    # Save the updated results to a new JSON file.
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Travel time results saved to {output_filename}")

if __name__ == "__main__":
    main()
