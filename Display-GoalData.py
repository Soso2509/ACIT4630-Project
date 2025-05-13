import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Load the JSON data from the uploaded file
file_path = "goal_timestamps_slow2.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Define the start time 
'''
Test RL1: 2025-05-01 18:07
Test_RL2: 2025-05-08 12:48 

Test P1: 2025-05-02 21:35 **(3-layer)**
Test P2: 2025-05-10 12:56 **(4-layer)**

Test J1: 2025-05-12 15:38 **(3-layer)**
Test J2: 2025-05-04 18:40 **(4-layer)**
Test J3 (J-good): 2025-05-09 12:48 **(6-layer)**

Test slow2: 2025-05-11 15:26 **(4-layer)**
Test slow3: 2025-05-07 12:23 **( 6-layer)**
'''
start_time = datetime.strptime("2025-05-11 15:26", "%Y-%m-%d %H:%M")

# Parse timestamps and count occurrences
timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
df = pd.DataFrame(timestamps, columns=["timestamp"])
df["count"] = 1

# Group by each minute (or second if finer resolution needed)
df_grouped = df.set_index("timestamp").resample("1min").sum().fillna(0).cumsum()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df_grouped.index, df_grouped["count"], marker='o')
plt.axvline(start_time, color='red', linestyle='--', label='Start Time')
plt.xlabel("Time")
plt.ylabel("Cumulative Goal Completions")
plt.title("Goal Completions Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
