import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Added import for linear regression

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

# Extract data for regression
x = travel_df['num_samples'].to_numpy()
y_astar = travel_df['A*'].to_numpy()
y_sastar = travel_df['SA*'].to_numpy()

# Perform linear regression for A* and SA*
coefficients_astar = np.polyfit(x, y_astar, 1)
poly_astar = np.poly1d(coefficients_astar)
regression_astar = poly_astar(x)

coefficients_sastar = np.polyfit(x, y_sastar, 1)
poly_sastar = np.poly1d(coefficients_sastar)
regression_sastar = poly_sastar(x)

# Load timing data from CSV file
df = pd.read_csv('prm_timing_results_60k.csv')

# Create figure and axis for plotting
fig, ax_left = plt.subplots(figsize=(10, 6))  # Adjust figure size

# Left Y-axis (Travel Time) - Plot original data and regression lines
ax_left.set_ylabel('Travel Time (seconds)', color='black', fontsize=30)
# Original data points
ax_left.plot(x, y_astar, color='purple', linestyle='--', marker='o')
ax_left.plot(x, y_sastar, color='orange', linestyle='-.', marker='s')
# Regression lines
ax_left.plot(x, regression_astar, color='purple', linestyle='-', label='A* Travel Time')
ax_left.plot(x, regression_sastar, color='orange', linestyle='-', label='A* + Pruning path')
ax_left.tick_params(axis='y', labelcolor='black', labelsize=12)

# Right Y-axis (Timing Metrics) - Semilog Scale remains
ax_right = ax_left.twinx()
ax_right.set_yscale('log')
ax_right.set_ylabel('Execution Time (seconds)', color='black', fontsize=30)

num_samples_csv = df['num_samples'].to_numpy()
sampling_roadmap = (df['sampling_time'] + df['roadmap_time']).to_numpy()
pathfinding = df['pathfinding_time'].to_numpy()
simplification_pathfinding = (df['simplification_time'] + df['pathfinding_time']).to_numpy()

ax_right.plot(num_samples_csv, sampling_roadmap, color='blue', linestyle='-', marker='x', label='PRM')
ax_right.plot(num_samples_csv, pathfinding, color='green', linestyle='-', marker='d', label='A*')
ax_right.plot(num_samples_csv, simplification_pathfinding, color='red', linestyle='-', marker='^', label='A* + Pruning Path')

ax_left.set_xlabel('Number of Nodes', fontsize=30)
plt.title('Time vs. Number of Nodes', fontsize=30)

# Legends and Grid
ax_left.legend(loc='upper left', fontsize=14)
ax_right.legend(loc='upper right', fontsize=14)
ax_left.grid(True, which='both', linestyle='--', alpha=0.5)

# Show plot
plt.show()
