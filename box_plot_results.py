import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Configurations
configs = {
    'rl': ['./goalTimestamps/goal_timestamps_RL1.json', './goalTimestamps/goal_timestamps_RL2.json', './goalTimestamps/goal_timestamps_RL3.json'],
    'j': ['./goalTimestamps/goal_timestamps_j1.json', './goalTimestamps/goal_timestamps_j2.json', './goalTimestamps/goal_timestamps_j3.json', './goalTimestamps/goal_timestamps_j4.json'],
    'p': ['./goalTimestamps/goal_timestamps_p1.json', './goalTimestamps/goal_timestamps_p2.json', './goalTimestamps/goal_timestamps_p3.json', './goalTimestamps/goal_timestamps_p4.json'],
    's': ['./goalTimestamps/goal_timestamps_s1.json', './goalTimestamps/goal_timestamps_s2.json', './goalTimestamps/goal_timestamps_s3.json', './goalTimestamps/goal_timestamps_s4.json']
}

start_times = {
    'rl1': "2025-05-01 18:07", 'rl2': "2025-05-08 12:37", 'rl3': "2025-05-13 15:50",
    'j1': "2025-05-12 15:38", 'j2': "2025-05-04 18:40", 'j3': "2025-05-09 12:48", 'j4': "2025-05-17 23:06",
    'p1': "2025-05-02 21:35", 'p2': "2025-05-10 12:56", 'p3': "2025-05-14 16:02", 'p4': "2025-05-18 23:23",
    's1': "2025-05-15 16:40", 's2': "2025-05-11 15:26", 's3': "2025-05-07 12:23", 's4': "2025-05-16 16:42"
}

colors = {
    'rl': '#1f77b4',
    'j': '#ff7f0e',
    'p': '#2ca02c',
    's': '#d62728',
    'All IL models': '#e377c2'
}

# Containers
box_data = {}
positions = {}
average_first_goals = {}
combined_data = []
combined_positions = []

# Data processing
for model_type, files in configs.items():
    values = []
    first_goal_times = []

    for i, file in enumerate(files, start=1):
        name = f"{model_type}{i}"
        if not os.path.exists(file) or name not in start_times:
            continue

        with open(file, "r") as f:
            data = json.load(f)

        start_time = datetime.strptime(start_times[name], "%Y-%m-%d %H:%M")
        timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
        hours_since_start = [(t - start_time).total_seconds() / 3600 for t in timestamps]
        hours_since_start = [h for h in hours_since_start if h <= 24]

        # Default value for first goal time
        if hours_since_start:
            first_goal_time = hours_since_start[0]
        else:
            first_goal_time = 25.0  # no goals

        first_goal_times.append(first_goal_time)

        if len(hours_since_start) > 5:
            post_five = hours_since_start[5:]
            duration = 24 - hours_since_start[5]
            rate = len(post_five) / duration if duration > 0 else 0
            values.append(rate)

            if model_type in ['j', 'p', 'slow']:
                combined_data.append(rate)

        # Always add to combined_positions if j/p/slow (even if no goals)
        if model_type in ['j', 'p', 'slow']:
            combined_positions.append(first_goal_time)

    if values:
        box_data[model_type] = values
    avg_pos = sum(first_goal_times) / len(first_goal_times)
    positions[model_type] = avg_pos
    average_first_goals[model_type] = avg_pos

# Add combined data
if combined_data:
    box_data['All IL models'] = combined_data
    avg_comb_pos = sum(combined_positions) / len(combined_positions)
    positions['All IL models'] = avg_comb_pos
    average_first_goals['All IL models'] = avg_comb_pos

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
labels, data, pos, face_colors = [], [], [], []

for k in sorted(box_data.keys(), key=lambda x: positions[x]):
    labels.append(k.upper())
    data.append(box_data[k])
    pos.append(positions[k])
    face_colors.append(colors.get(k, "#000000"))

bp = ax.boxplot(data, positions=pos, widths=0.5, patch_artist=True)

# Color boxes
for patch, color in zip(bp['boxes'], face_colors):
    patch.set_facecolor(color)

# Axis settings
ax.set_xlim(0, 24)
ax.set_ylim(0, 60)
ax.set_xticks(np.arange(1, 25, 1))
ax.set_xticklabels([str(i) for i in range(1, 25)])
ax.set_xlabel("Average First Completed Lap (Hour)")
ax.set_ylabel("Avg Completed Laps per Hour (after 5th Completed Lap)")
ax.set_title("Box Plot of Completed Lap Rates by Model Type")

# Legend with first goal time
legend_patches = []
for k in sorted(average_first_goals.keys(), key=lambda x: positions[x]):
    color = colors.get(k, "#000000")
    label = f"{k.upper()} (first: {average_first_goals[k]:.2f}h)"
    legend_patches.append(Patch(color=color, label=label))

ax.legend(handles=legend_patches, loc='upper left', title="Avg First Completed Lap")

plt.tight_layout()
plt.show()
