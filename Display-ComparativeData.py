import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define agents and the number of max runs per agent type
agent_types = ['rl', 'j', 'p', 'slow']
max_runs_per_agent = 5

# Slow4 = Rerun of slow1
# J4 = Rerun of J1
# P4 = Rerun of P3

# Define the configurations for all agents and runs
configs = {
    'rl': ['goal_timestamps_RL1.json', 'goal_timestamps_RL2.json', 'goal_timestamps_RL3.json'],         # 
    'j': ['goal_timestamps_j1.json','goal_timestamps_j2.json', 'goal_timestamps_j3.json', 'goal_timestamps_j4.json'],             #  
    'p': ['goal_timestamps_p1.json', 'goal_timestamps_p2.json', 'goal_timestamps_p3.json', 'goal_timestamps_p4.json'],             # 
    'slow': ['goal_timestamps_slow1.json', 'goal_timestamps_slow2.json', 'goal_timestamps_slow3.json', 'goal_timestamps_slow4.json']  # 
}

start_times = {
    'rl1': "2025-05-01 18:07", 'rl2': "2025-05-08 12:37", 'rl3': "2025-05-13 15:50",                # 
    'j1': "2025-05-12 15:38", 'j2': "2025-05-04 18:40", 'j3': "2025-05-09 12:48", 'j4': "2025-05-17 23:06",                  #   
    'p1': "2025-05-02 21:35", 'p2': "2025-05-10 12:56", 'p3': "2025-05-14 16:02", 'p4': "2025-05-18 23:23",                  # 
    'slow1': "2025-05-15 16:40", 'slow2': "2025-05-11 15:26", 'slow3': "2025-05-07 12:23", 'slow4': "2025-05-16 16:42"           #  
}

colors = {
    'rl1': '#1f77b4', 'rl2': '#2a5d9f', 'rl3': '#3c83c0',   #14507a #4179a5
    'j1': '#ff7f0e', 'j2': '#e66c00', 'j3': '#ffa133', 'j4': '#cc5f00',      #cc5f00 #ff9933
    'p1': '#2ca02c', 'p2': '#247b24', 'p3': '#3ea33e', 'p4': '#1f661f',  #    #1f661f #4ab04a
    'slow1': '#e04344', 'slow2': '#d62728', 'slow3': '#b61e1f', 'slow4': '#8c1b1c'  #     #c93030
}

# Data containers
all_hours = []
all_counts = []
labels = []
plot_colors = []

# Main loop
for agent in agent_types:
    for i in range(1, max_runs_per_agent + 1):
        name = f"{agent}{i}"
        filename = f"goal_timestamps_{name.upper()}.json"
        if not os.path.exists(filename):
            continue
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            start_time = datetime.strptime(start_times[name], "%Y-%m-%d %H:%M")
            timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
            hours_since_start = [(t - start_time).total_seconds() / 3600 for t in timestamps]
            counts = list(range(1, len(hours_since_start) + 1))

            # Metric: hours until first goal
            hours_to_first = round(hours_since_start[0], 2) if hours_since_start else None

            # Metric: finishes/hour after 5th
            if len(hours_since_start) > 5:
                hours_range = hours_since_start[5:]  # After 5th
                time_span = hours_range[-1] - hours_range[0] if len(hours_range) > 1 else 0.0001
                finishes_per_hour = round((len(hours_range)) / time_span, 2)
            else:
                finishes_per_hour = 0.0

            print(f"{name.upper()} | First goal after: {hours_to_first} hrs | Finishes/hr (after 5th): {finishes_per_hour}")

            # Append plotting data
            all_hours.append(hours_since_start)
            all_counts.append(counts)
            labels.append(f"{name.upper()}: {len(data)} laps")
            plot_colors.append(colors.get(name, "#000000"))

        except Exception as e:
            print(f"Error with {filename}: {e}")

# Plot
max_hour = int(max([max(h, default=0) for h in all_hours], default=0)) + 1
x_ticks = np.arange(0, max_hour + 1, 1)

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