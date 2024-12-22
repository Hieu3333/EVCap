import json

def filter_and_count_test_split(inpath, outpath):
    # Load the dataset_coco.json file
    with open(inpath, 'r') as infile:
        data = json.load(infile)
    
    # Filter images based on the "split" field being "test"
    test_images = [image for image in data['images'] if image.get('split') == 'test']
    
    # Count the number of images in the "test" split
    test_image_count = len(test_images)
    
    # Create a new dictionary with only the "test" images and structure
    filtered_data = {
        "dataset": "coco",
        "images": test_images
    }
    
    # Write the filtered data to a new JSON file
    with open(outpath, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)
    
    # Print the number of images in the "test" split
    print(f"Number of images in the 'test' split: {test_image_count}")
    
    return test_image_count, filtered_data

# Path to the original dataset_coco.json file
inpath = 'dataset_coco.json'
# Path where the filtered JSON will be saved
outpath = 'dataset_coco_test_split.json'

# Call the function
filter_and_count_test_split(inpath, outpath)
