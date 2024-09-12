import json
import os
import argparse
from shapely.geometry import shape, mapping
from tqdm import tqdm

# Function to calculate IoU (Intersection over Union)
def calculate_iou(polygon1, polygon2):
    poly1 = shape(polygon1)
    poly2 = shape(polygon2)
    
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return intersection_area / union_area

# Function to filter trees by height and overlap with the passed max_iou threshold
def filter_trees_by_height(trees_data, max_iou, min_height):
    # Sort trees by height in descending order
    trees_data.sort(key=lambda tree: tree['properties']['height'], reverse=True)
    
    # List to hold the filtered trees
    filtered_trees = []
    
    # Loop through each tree and add it to the filtered list if no taller tree overlaps too much with it
    for tree in tqdm(trees_data):
        tree_polygon = tree['geometry']

        # Exclude tree if it is too short
        if tree['properties']['height'] < min_height:
            continue
        
        # Loop through the filtered trees to check for overlap
        include_tree = True
        for other_tree in filtered_trees:
            other_polygon = other_tree['geometry']
            
            # Calculate the IoU and exclude if it exceeds the threshold
            iou = calculate_iou(tree_polygon, other_polygon)
            if iou > max_iou:
                include_tree = False
                break
        
        if include_tree:
            filtered_trees.append(tree)
    
    return filtered_trees

def main(file_path, max_iou, min_height):
    # Load GeoJSON data
    with open(file_path, 'r') as file:
        geojson_data = json.load(file)

    # Filter trees based on the overlap and height criteria with IoU threshold
    filtered_trees = filter_trees_by_height(geojson_data['features'], max_iou=max_iou, min_height=min_height)

    # Create a new GeoJSON structure with the filtered trees
    filtered_data = {
        "type": "FeatureCollection",
        "features": filtered_trees
    }

    # Construct the output file path
    directory = os.path.dirname(file_path)
    iou_str = str(max_iou).replace('.', '_')
    output_file = os.path.join(directory, f'filtered_trees-{iou_str}iou-{min_height}m.geojson')

    # Save the filtered data to a new GeoJSON file
    with open(output_file, 'w') as file:
        json.dump(filtered_data, file, indent=4)

    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter tree data based on height and bounding box overlap.")
    parser.add_argument("-f", "--file", required=True, help="Relative path to the GeoJSON file containing tree data.")
    parser.add_argument("-t", "--max_iou", type=float, default=0.25, help="Maximum IoU threshold for bounding box overlap. Default is 0.25.")
    parser.add_argument("-s", "--min_height", type=float, default=3, help="Minimum height for a tree to be considered. Default is 3 meters.")
    
    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the provided file path
    main(args.file, args.max_iou, args.min_height)
