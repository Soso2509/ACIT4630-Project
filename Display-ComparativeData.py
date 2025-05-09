import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define agents and the number of max runs per agent type
agent_types = ['rl', 'j', 'p', 'slow']
max_runs_per_agent = 5

# Define the configurations for all agents and runs
configs = {
    'rl': ['goal_timestamps_RL1.json', 'goal_timestamps_RL2.json'],         # , 'goal_timestamps_RL3.json'
    'j': ['goal_timestamps_j2.json'],             # 'goal_timestamps_j1.json', , 'goal_timestamps_j3.json'
    'p': ['goal_timestamps_p1.json'],             # , 'goal_timestamps_p2.json', 'goal_timestamps_p3.json'
    'slow': ['goal_timestamps_slow3.json']  # 'goal_timestamps_slow1.json', 'goal_timestamps_slow2.json', 
}

start_times = {
    'rl1': "2025-05-01 18:07", 'rl2': "2025-05-08 12:37",                 # 'rl3': "2025-05-10 09:00",
    'j2': "2025-05-04 18:40",                   # 'j1': "2025-05-04 18:40",  , 'j3': "2025-05-12 10:00" 
    'p1': "2025-05-02 21:35",                   # , 'p2': "2025-05-13 12:00", 'p3': "2025-05-14 12:00"
    'slow3': "2025-05-07 12:23"           # 'slow1': "2025-05-06 15:23", 'slow2': "2025-05-07 12:23", 
}

colors = {
    'rl': 'blue',
    'j': 'orange',
    'p': 'green',
    'slow': 'red'
}

# Storage for plotting
all_hours = []
all_counts = []
labels = []
plot_colors = []

# Loop through each agent type and attempt up to 5 runs
for agent in agent_types:
    for i in range(1, max_runs_per_agent + 1):
        name = f"{agent}{i}"
        filename = f"goal_timestamps_{name.upper()}.json"
        if not os.path.exists(filename):
            continue
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            start_str = start_times[name]
            start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
            times = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
            hours_since_start = [(t - start_time).total_seconds() / 3600 for t in times]
            counts = list(range(1, len(hours_since_start) + 1))
            all_hours.append(hours_since_start)
            all_counts.append(counts)
            labels.append(name.upper())
            plot_colors.append(colors[agent])
        except KeyError:
            print(f"Missing start time for {name}, skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Determine the x-axis range for all data
max_hour = int(max([max(h, default=0) for h in all_hours], default=0)) + 1
x_ticks = np.arange(0, max_hour + 1, 1)

# Plotting
plt.figure(figsize=(14, 7))
for hours, counts, label, color in zip(all_hours, all_counts, labels, plot_colors):
    plt.plot(hours, counts, label=label, color=color)

plt.xticks(ticks=x_ticks)
plt.xlabel("Hours Since Start")
plt.ylabel("Total Timestamps")
plt.title("Goal Reaches Over Time for All Agents")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
