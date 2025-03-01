import pandas as pd
import numpy as np

def compute_energy_and_duration(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='mission_duration(s)')
    energy = np.trapz(df['power(W)'], df['mission_duration(s)'])
    mission_duration = df['mission_duration(s)'].iloc[-1]
    return energy, mission_duration

csv_files = ['/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log1_-6.6_-4.9_5.0.csv', 
             '/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log2_9.72_-3.75_3.0.csv', 
             '/home/intel//fiverr/md/drone_ws/src/UAV_UGV/prm/results/UAV_UGV/Log3_-2.94_5.41_3.0.csv']

energies = []
durations = []

for file in csv_files:
    energy, duration = compute_energy_and_duration(file)
    energies.append(energy)
    durations.append(duration)

avg_energy = np.mean(energies)
avg_duration = np.mean(durations)

print("Average Energy Consumed (J):", avg_energy)
print("Average Mission Duration (s):", avg_duration)
