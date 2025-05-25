import pandas as pd
import ast
import csv

# === Load original dataset ===
df = pd.read_csv('demonstration_data.csv')

# === Clean and filter ===
filtered_rows = []

for _, row in df.iterrows():
    fwd = row['Forward Throttle']
    back = row['Backward throttle']
    steer = row['Steering']
    velocity = ast.literal_eval(row['Velocity'])[0] if isinstance(row['Velocity'], str) else row['Velocity']
    
    # Skip if velocity < 5 and both throttle values are 0 or less
    if velocity <= 5 and fwd <= 0 and back <= 0 and steer == 0:
        continue

    # Build action
    action = [fwd, back, steer]
    
    # Format LiDAR (ensure it stays a string, as expected)
    lidar_str = row['Lidar']
    
    # Create mock LiDAR object like original dataset: [lidar_array, [0.0, 0.0, 0.0], None]
    formatted_lidar = f"[{lidar_str}, [0.0, 0.0, 0.0], None]"
    
    # Append formatted row
    filtered_rows.append([str(action), velocity, formatted_lidar])

# === Write to new CSV ===
with open('demonstration_filtered.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Action', 'Velocity', 'LiDAR'])
    writer.writerows(filtered_rows)

print(f"Filtered data saved to demonstration_filtered.csv")
