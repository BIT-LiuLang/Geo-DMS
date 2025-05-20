import json
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Update JSON file with additional keypoints.")
parser.add_argument('file_path', type=str, help='Path to the JSON file')

# Parse the arguments
args = parser.parse_args()
file_path = args.file_path

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Add "left_ear" and "right_ear" keypoints to the 'categories'
for category in data['categories']:
    if 'keypoints' in category:
        if len(category['keypoints']) == 15:  # Only update if there are 15 keypoints
            category['keypoints'].insert(3, "left_ear")  # Insert as 4th
            category['keypoints'].insert(4, "right_ear")  # Insert as 5th

        # Add 0 to 'skeleton' to maintain alignment if needed
        if 'skeleton' in category:
            for i in range(2):
                category['skeleton'].insert(3, [0, 0])

# Update each annotation to add empty keypoints for "left_ear" and "right_ear"
for instance in data['annotations']:
    instance['iscrowd'] = 0
    # Calculate area based on bbox
    bbox = instance['bbox']
    instance['area'] = bbox[2] * bbox[3]

    if 'keypoints' in instance:
        if len(instance['keypoints']) == 15 * 3:  # Each keypoint has (x, y, visibility)
            # Insert empty keypoints with format [0, 0, 0]
            instance['keypoints'][3 * 3:3 * 3] = [0, 0, 0]  # for "left_ear"
            instance['keypoints'][4 * 3:4 * 3] = [0, 0, 0]  # for "right_ear"

# Write the updated JSON file
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("JSON file updated successfully.")