import pandas as pd
import matplotlib.pyplot as plt

# Load timing data
df = pd.read_csv('prm_timing_results.csv')

# Travel time data
travel_data = {
    'num_samples': [500,                1000,               1500,               2000,               2500,               3000],
    'A*':          [20.603855180994145, 26.06183188662354,  27.890861786537727, 29.597998654173413,  33.15293318637222,  27.84188083302814],
    'SA*':         [20.603855180994145, 16.386078334951208, 16.874425613529127, 20.004724378222978, 27.383456421478193, 19.139848147704363]
}

# Convert to DataFrame
travel_df = pd.DataFrame(travel_data)

# Create figure and axis
fig, ax_left = plt.subplots()

# Left Y-axis (Travel Time) - Semilog Scale
ax_left.set_yscale('log')
ax_left.set_ylabel('Travel Time (seconds)', color='black')
ax_left.plot(travel_df['num_samples'].to_numpy(), travel_df['A*'].to_numpy(), 
             color='purple', linestyle='--', marker='o', label='A* Travel Time')
ax_left.plot(travel_df['num_samples'].to_numpy(), travel_df['SA*'].to_numpy(), 
             color='orange', linestyle='-.', marker='s', label='SA* Travel Time')
ax_left.tick_params(axis='y', labelcolor='black')

# Right Y-axis (Timing Metrics) - Semilog Scale
ax_right = ax_left.twinx()
ax_right.set_yscale('log')
ax_right.set_ylabel('Time (seconds)', color='black')

num_samples = df['num_samples'].to_numpy()
sampling_roadmap = (df['sampling_time'] + df['roadmap_time']).to_numpy()
pathfinding = df['pathfinding_time'].to_numpy()
simplification_pathfinding = (df['simplification_time'] + df['pathfinding_time']).to_numpy()

ax_right.plot(num_samples, sampling_roadmap, color='blue', linestyle='-', marker='x', label='Sampling + Roadmap')
ax_right.plot(num_samples, pathfinding, color='green', linestyle='-', marker='d', label='Pathfinding')
ax_right.plot(num_samples, simplification_pathfinding, color='red', linestyle='-', marker='^', label='Simplification + Pathfinding')

ax_left.set_xlabel('Number of Samples')
plt.title('Semilog Plot: Time vs. Number of Samples')

# Legends
ax_left.legend(loc='upper left', fontsize=10)
ax_right.legend(loc='upper right', fontsize=10)

# Grid
ax_left.grid(True, which='both', linestyle='--', alpha=0.5)

# Show plot
plt.show()
