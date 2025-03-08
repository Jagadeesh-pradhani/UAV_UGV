import json
import pandas as pd
import matplotlib.pyplot as plt

# Load travel time data from JSON file
with open('prm_paths_with_travel_time.json', 'r') as f:
    travel_json = json.load(f)

# Extract travel time data from JSON
num_samples = []
astar_travel_time = []
saastar_travel_time = []

for entry in travel_json:
    num_samples.append(entry['num_samples'])
    astar_travel_time.append(entry['best_path_travel_time'])
    saastar_travel_time.append(entry['simplified_path_travel_time'])

# Create travel_data dictionary and DataFrame
travel_data = {
    'num_samples': num_samples,
    'A*': astar_travel_time,
    'SA*': saastar_travel_time
}
travel_df = pd.DataFrame(travel_data)

# Load timing data from CSV file
df = pd.read_csv('prm_timing_results_60k.csv')

# Create figure and axis for plotting
fig, ax_left = plt.subplots()

# Left Y-axis (Travel Time) - Normal Scale (removed semilog scale)
ax_left.set_ylabel('Travel Time (seconds)', color='black')
ax_left.plot(travel_df['num_samples'].to_numpy(), travel_df['A*'].to_numpy(), 
             color='purple', linestyle='--', marker='o', label='A* Travel Time')
ax_left.plot(travel_df['num_samples'].to_numpy(), travel_df['SA*'].to_numpy(), 
             color='orange', linestyle='-.', marker='s', label='A* + Pruning path Travel Time')
ax_left.tick_params(axis='y', labelcolor='black')

# Right Y-axis (Timing Metrics) - Semilog Scale remains
ax_right = ax_left.twinx()
ax_right.set_yscale('log')
ax_right.set_ylabel('Time (seconds)', color='black')

num_samples_csv = df['num_samples'].to_numpy()
sampling_roadmap = (df['sampling_time'] + df['roadmap_time']).to_numpy()
pathfinding = df['pathfinding_time'].to_numpy()
simplification_pathfinding = (df['simplification_time'] + df['pathfinding_time']).to_numpy()

ax_right.plot(num_samples_csv, sampling_roadmap, color='blue', linestyle='-', marker='x', label='PRM')
ax_right.plot(num_samples_csv, pathfinding, color='green', linestyle='-', marker='d', label='A*')
ax_right.plot(num_samples_csv, simplification_pathfinding, color='red', linestyle='-', marker='^', label='A* + Pruning Path')

ax_left.set_xlabel('Number of Nodes')
plt.title('Time vs. Number of Samples')

# Legends and Grid
ax_left.legend(loc='upper left', fontsize=10)
ax_right.legend(loc='upper right', fontsize=10)
ax_left.grid(True, which='both', linestyle='--', alpha=0.5)

# Show plot
plt.show()
