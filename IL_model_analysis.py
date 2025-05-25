import json
import matplotlib.pyplot as plt

# Load the IL_rew.json file
with open('IL_rew.json', 'r') as f:
    data = json.load(f)

# Limit to the first X iterations
X = 30
data = data[:X]

# Extract and compute values for plotting
iterations = list(range(1, len(data) + 1))
progressions = []
colors = []

for entry in data:
    rew = entry['total_rew']
    run_time = entry['run_time']
    if rew >= 270:
        progress = 100
        colors.append('green')
    else:
        progress = min(100, (rew / 220) * 100)
        if run_time < 50:
            colors.append('red')
        else:
            colors.append('blue')
    progressions.append(progress)

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(iterations, progressions, color=colors)
plt.xlabel('Lap nr.')
plt.ylabel('Lap Progression (%)')
plt.title('IL Analysis of Model Runs')
plt.ylim(0, 110)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
