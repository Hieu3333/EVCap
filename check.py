import json

# Load the Flickr30k test split file
with open('flickr30k_test_split.json', 'r') as f:
    flickr30k_test_split = json.load(f)

# Load the evaluation results file
with open('eval_flickr30k.json', 'r') as f:
    eval_flickr30k = json.load(f)

# Extract the image IDs from flickr30k_test_split.json
image_ids = [image["imgid"] for image in flickr30k_test_split["images"]]

# Update eval_flickr30k.json with the corresponding image IDs
for i, result in enumerate(eval_flickr30k):
    if i < len(image_ids):
        result["image_name"] = image_ids[i]
    else:
        print(f"Warning: More predictions in eval_flickr30k.json than images in flickr30k_test_split.json. Skipping extra predictions.")

# Save the updated eval_flickr30k.json
with open('eval_flickr30k_updated.json', 'w') as f:
    json.dump(eval_flickr30k, f, indent=4)

print("Successfully added image IDs to eval_flickr30k.json!")
