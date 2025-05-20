import json
from collections import defaultdict
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Separate COCO format JSON file by video ID.")
parser.add_argument('input_file', type=str, help='Path to the input JSON file')
parser.add_argument('output_folder', type=str, help='Path to the output folder')

# Parse the arguments
args = parser.parse_args()
input_file = args.input_file
output_folder = args.output_folder

# Load the JSON file
with open(input_file, 'r') as infile:
    coco_data = json.load(infile)

# Extract categories from the original JSON file
categories = coco_data.get("categories", [])

# Group images and annotations by vid_id
images_by_vid = defaultdict(list)
annotations_by_vid = defaultdict(list)

# Group images based on their 'vid_id'
for image in coco_data.get("images", []):
    vid_id = image.get("vid_id")
    if vid_id:
        images_by_vid[vid_id].append(image)

# Group annotations based on the 'vid_id' of the corresponding image
for annotation in coco_data.get("annotations", []):
    image_id = annotation.get("image_id")
    # Find the 'vid_id' associated with this image
    vid_id = None
    for vid, images in images_by_vid.items():
        if any(img['id'] == image_id for img in images):
            vid_id = vid
            break
    if vid_id:
        annotations_by_vid[vid_id].append(annotation)

# Create separate JSON files for each video, including images, annotations, and categories
for vid_id, images in images_by_vid.items():
    output_data = {
        "categories": categories,
        "images": images,
        "annotations": annotations_by_vid[vid_id]
    }
    output_file = f'{vid_id}.json'  # Create a filename based on the vid_id

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, output_file)

    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

print("COCO format JSON files have been separated based on the video.")

#python separate_json.py /home/pace/Poseidon/dataJHMDB/jsons/split3/posetrack_val.json /home/pace/Poseidon/dataJHMDB/annotations/split3/val