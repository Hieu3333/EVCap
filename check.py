import json

# Load the Karpathy split JSON file
with open("dataset_flickr32k.json", "r") as split_file:  # Replace with your split file
    split_data = json.load(split_file)

# Load the Flickr30k dataset JSON file
with open("dataset_flickr32k.json", "r") as dataset_file:  # Replace with your dataset file
    dataset_data = json.load(dataset_file)

# Extract test split image filenames
test_filenames = {item["filename"] for item in split_data["images"] if item["split"] == "test"}

# Filter images in the test split
test_images = [image for image in dataset_data["images"] if image["filename"] in test_filenames]

# Save filtered test images to a new file (optional)
with open("flickr30k_test_split.json", "w") as output_file:
    json.dump({"dataset": "flickr30k", "images": test_images}, output_file, indent=4)

print(f"Filtered {len(test_images)} test images.")
