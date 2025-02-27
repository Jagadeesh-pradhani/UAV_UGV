import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('prm_timing_results.csv')

# Create figure and axes
fig, ax = plt.subplots()
ax.yaxis.set_visible(False)  # Hide the left y-axis

# Create a twin axis for the right side with a logarithmic scale
ax_right = ax.twinx()
ax_right.set_yscale('log')
ax_right.set_ylabel('Time (seconds)')

# Convert DataFrame columns to numpy arrays for plotting
num_samples = df['num_samples'].to_numpy()
sampling_roadmap = (df['sampling_time'] + df['roadmap_time']).to_numpy()
pathfinding = df['pathfinding_time'].to_numpy()
simplification_pathfinding = (df['simplification_time'] + df['pathfinding_time']).to_numpy()

# Plot each series with different colors
ax_right.plot(num_samples, sampling_roadmap, color='blue', label='Sampling + Roadmap')
ax_right.plot(num_samples, pathfinding, color='green', label='Pathfinding')
ax_right.plot(num_samples, simplification_pathfinding, color='red', label='Simplification + Pathfinding')

# Set labels and title
ax.set_xlabel('Number of Samples')
plt.title('Semilog Plot of Time vs. Number of Samples')

# Add legend and grid
ax_right.legend(loc='upper left')
ax_right.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()