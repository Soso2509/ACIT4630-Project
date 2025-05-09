import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

# Reload the files after kernel reset
rl_path = "goal_timestamps_RL.json"
hybrid_path = "goal_timestamps_Hybrid.json"
hybrid_j_path = "goal_timestamps_Hybrid_J.json"
hybrid_slow_path = "goal_timestamps_Hybrid_slower_failed.json"
hybrid_slow2_path = "goal_timestamps_slow2.json"

# Load the JSON files
with open(rl_path, "r") as f:
    data_rl = json.load(f)

with open(hybrid_path, "r") as f:
    data_hybrid_p = json.load(f)

with open(hybrid_j_path, "r") as f:
    data_hybrid_j = json.load(f)

with open(hybrid_slow_path, "r") as f:
    data_hybrid_slow = json.load(f)

with open(hybrid_slow2_path, "r") as f:
    data_hybrid_slow2 = json.load(f)

# Define the start times for each dataset
start_time_rl = datetime.strptime("2025-05-01 18:07", "%Y-%m-%d %H:%M")
start_time_hybrid = datetime.strptime("2025-05-02 21:35", "%Y-%m-%d %H:%M") 
start_time_hybrid_j = datetime.strptime("2025-05-04 18:40", "%Y-%m-%d %H:%M") 
start_time_hybrid_slow = datetime.strptime("2025-05-06 15:23", "%Y-%m-%d %H:%M") 
start_time_hybrid_slow2 = datetime.strptime("2025-05-07 12:23", "%Y-%m-%d %H:%M") 

# Process RL data
times_rl = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_rl]
hours_since_start_rl = [(t - start_time_rl).total_seconds() / 3600 for t in times_rl]
counts_rl = list(range(1, len(hours_since_start_rl) + 1))

# Process Hybrid P data
times_hybrid_p = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_hybrid_p]
hours_since_start_hybrid_p = [(t - start_time_hybrid).total_seconds() / 3600 for t in times_hybrid_p]
counts_hybrid_p = list(range(1, len(hours_since_start_hybrid_p) + 1))

# Process Hybrid J data
times_hybrid_j = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_hybrid_j]
hours_since_start_hybrid_j = [(t - start_time_hybrid_j).total_seconds() / 3600 for t in times_hybrid_j]
counts_hybrid_j = list(range(1, len(hours_since_start_hybrid_j) + 1))

# Process Hybrid Slow (failed) data
times_hybrid_slow = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_hybrid_slow]
hours_since_start_hybrid_slow = [(t - start_time_hybrid_slow).total_seconds() / 3600 for t in times_hybrid_slow]
counts_hybrid_slow = list(range(1, len(hours_since_start_hybrid_slow) + 1))

# Process Hybrid Slow2 data
times_hybrid_slow2 = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data_hybrid_slow2]
hours_since_start_hybrid_slow2 = [(t - start_time_hybrid_slow2).total_seconds() / 3600 for t in times_hybrid_slow2]
counts_hybrid_slow2 = list(range(1, len(hours_since_start_hybrid_slow2) + 1))


# Determine the x-axis range
max_hour = int(max(max(hours_since_start_rl, default=0), max(hours_since_start_hybrid_p, default=0), max(hours_since_start_hybrid_j, default=0), max(hours_since_start_hybrid_slow, default=0), max(hours_since_start_hybrid_slow2, default=0))) + 1
x_ticks = np.arange(0, max_hour + 1, 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(hours_since_start_rl, counts_rl, label='RL Agent', color='blue')
plt.plot(hours_since_start_hybrid_p, counts_hybrid_p, label='Hybrid Agent P', color='green')
plt.plot(hours_since_start_hybrid_j, counts_hybrid_j, label='Hybrid Agent J', color='orange')
plt.plot(hours_since_start_hybrid_slow, counts_hybrid_slow, label='Hybrid Agent Slow', color='yellow')
plt.plot(hours_since_start_hybrid_slow2, counts_hybrid_slow2, label='Hybrid Agent Slow2', color='red')
plt.xlabel("Hours Since Start")
plt.ylabel("Total Timestamps")
plt.title("Goal Reaches Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
