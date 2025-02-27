import pandas as pd
import numpy as np

def compute_energy_and_duration(file_path):
    """
    Reads a CSV file and computes:
      - Energy consumption by numerically integrating the power over mission_duration(s).
      - Mission duration as the final value in the 'mission_duration(s)' column.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure the data is sorted by mission duration
    df = df.sort_values(by='mission_duration(s)')
    
    # Compute energy using the trapezoidal rule:
    # Energy (J) = integral of power (W) over mission duration (s)
    energy = np.trapz(df['power(W)'], df['mission_duration(s)'])
    
    # The mission duration is taken as the last logged value in the mission_duration(s) column
    mission_duration = df['mission_duration(s)'].iloc[-1]
    
    return energy, mission_duration

# List the CSV file names (update these if needed)
csv_files = ['/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log1_-6.6_-4.9_5.0.csv', 
             '/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log2_9.72_-3.75_3.0.csv', 
             '/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log3_-2.94_5.41_3.0.csv']

energies = []
durations = []

for file in csv_files:
    energy, duration = compute_energy_and_duration(file)
    energies.append(energy)
    durations.append(duration)

# Calculate the average energy consumption and average mission duration across the three files
avg_energy = np.mean(energies)
avg_duration = np.mean(durations)

print("Average Energy Consumed (J):", avg_energy)
print("Average Mission Duration (s):", avg_duration)
