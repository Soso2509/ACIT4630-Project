import pandas as pd
import ast
import csv

INPUT_CSV = 'expert.csv'
OUTPUT_CSV = 'expert_filtered.csv'

def should_keep(action_str, velocity):
    try:
        action = ast.literal_eval(action_str)
        if velocity < 5 and not (action[0] > 0 or action[1] > 0):
            return False
    except:
        return False 
    return True

# Load and filter
with open(INPUT_CSV, 'r') as infile, open(OUTPUT_CSV, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        velocity = float(row['Velocity'])
        if should_keep(row['Action'], velocity):
            writer.writerow(row)

print(f"Filtered rows saved to {OUTPUT_CSV}")
