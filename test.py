import json

# Open the input files and load the data
with open('eval_log/ablation/90%/eval_flickr30k.json', 'r') as f:
    data = json.load(f)

with open('eval_log/flickr30k/eval_flickr30k.json', 'r') as file:
    ref_data = json.load(file)

# Ensure data and ref_data are lists for iteration
if isinstance(data, list) and isinstance(ref_data, list):
    for i, (d, r) in enumerate(zip(data, ref_data)):
        d["image_name"] = r["image_name"]

    # Write the modified data back to the file
    with open('eval_log/ablation/90%/eval_flickr30k.json', 'w') as f:
        json.dump(data, f, indent=4)
else:
    print("Error: Both 'data' and 'ref_data' must be lists.")
