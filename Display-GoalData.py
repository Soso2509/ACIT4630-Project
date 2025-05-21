import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Load the JSON data from the uploaded file
file_path = "goal_timestamps_j1.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Define the start time 
'''
Test RL1: 2025-05-01 18:07
Test_RL2: 2025-05-08 12:48 
Test RL3: 2025-05-13 15:50

Test P1: 2025-05-02 21:35 **(3-layer)**
Test P2: 2025-05-10 12:56 **(4-layer)**
Test P3: 2025-05-14 16:02 **(6-layer)**

Test J1: 2025-05-12 15:38 **(3-layer)**
Test J2: 2025-05-04 18:40 **(4-layer)**
Test J3 (J-good): 2025-05-09 12:48 **(6-layer)**

Test slow1: 2025-05-15 16:40 **(3-layer)**
Test slow2: 2025-05-11 15:26 **(4-layer)**
Test slow3: 2025-05-07 12:23 **( 6-layer)**
'''
start_time = datetime.strptime("2025-05-12 15:38", "%Y-%m-%d %H:%M")

# Parse timestamps and create a DataFrame
timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
il_chances = [entry.get("IL_chance", None) for entry in data]

df = pd.DataFrame({"timestamp": timestamps, "IL_chance": il_chances})
df["count"] = 1

# Group for first plot (goal completions over time)
df_grouped = df.set_index("timestamp").resample("1min").sum().fillna(0).cumsum()

# First plot: Cumulative goal completions
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

# Second plot: Lap occurrence (binary) and IL_chance over time
df_binary = df.copy()
df_binary["binary"] = 1  # always 1 at the timestamp of a goal
df_binary.set_index("timestamp", inplace=True)

# Reindex to same time range as original for consistency
resample_index = pd.date_range(start=min(df_binary.index), end=max(df_binary.index), freq="1min")
df_binary = df_binary.reindex(resample_index, method='pad').fillna(0)

plt.figure(figsize=(12, 6))
plt.plot(df_binary.index, df_binary["binary"], label="Lap Registered", linestyle='-', drawstyle='steps-post')
plt.plot(df_binary.index, df_binary["IL_chance"], label="IL_chance", linestyle='--')
plt.axvline(start_time, color='red', linestyle='--', label='Start Time')
plt.xlabel("Time")
plt.ylabel("Value (0 to 1)")
plt.title("Lap Detection and IL_chance Over Time")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
