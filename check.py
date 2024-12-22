import json

# Load JSON file
with open('eval_logs.json', 'r') as f:
    data = json.load(f)

# Ensure it's a list and print the length
if isinstance(data, list):
    print(f"Length of the list: {len(data)}")
else:
    print("The JSON data is not a list.")
