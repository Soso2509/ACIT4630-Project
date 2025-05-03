import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

# Reload the files after kernel reset
rl_path = "goal_timestamps_RL.json"
hybrid_path = "goal_timestamps_Hybrid.json"

# Load the JSON files
with open(rl_path, "r") as f:
    data_rl = json.load(f)

with open(hybrid_path, "r") as f:
    data_hybrid = json.load(f)

# Define the start times for each dataset
start_time_rl = datetime.strptime("2025-05-01 18:07", "%Y-%m-%d %H:%M")
start_time_hybrid = datetime.strptime("2025-05-02 21:35", "%Y-%m-%d %H:%M") 

# Process RL data
times_rl = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_rl]
hours_since_start_rl = [(t - start_time_rl).total_seconds() / 3600 for t in times_rl]
counts_rl = list(range(1, len(hours_since_start_rl) + 1))

# Process Hybrid data
times_hybrid = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_hybrid]
hours_since_start_hybrid = [(t - start_time_hybrid).total_seconds() / 3600 for t in times_hybrid]
counts_hybrid = list(range(1, len(hours_since_start_hybrid) + 1))

# Determine the x-axis range
max_hour = int(max(max(hours_since_start_rl, default=0), max(hours_since_start_hybrid, default=0))) + 1
x_ticks = np.arange(0, max_hour + 1, 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(hours_since_start_rl, counts_rl, label='RL Agent', color='blue')
plt.plot(hours_since_start_hybrid, counts_hybrid, label='Hybrid Agent', color='green')
plt.xlabel("Hours Since Start")
plt.ylabel("Total Timestamps")
plt.title("Goal Reaches Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
