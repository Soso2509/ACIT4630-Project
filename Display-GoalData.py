import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Load the JSON data from the uploaded file
file_path = "goal_timestamps_Hybrid.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Define the start time 
start_time = datetime.strptime("2025-05-02 21:35", "%Y-%m-%d %H:%M") #2025-05-01 18:07

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
