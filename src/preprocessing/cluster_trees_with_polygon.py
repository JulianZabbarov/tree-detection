import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pyproj import Transformer
import argparse

def load_trees_from_json(file_path):
    """Loads tree data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['trees']

def get_tree_coordinates(trees):
    """Extracts ground height coordinates and attributes from tree data, including polygons."""
    transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
    coords = []
    attributes = []
    polygons = []
    for tree in trees:
        x, y = tree['groundHeightCircle']['x'], tree['groundHeightCircle']['y']
        lat, lon = transformer.transform(x, y)
        coords.append([lon, lat])  # GeoJSON uses (lon, lat)
        attributes.append([
            tree['groundHeight'],
            tree['height'],
            tree['crownRadius'],
            tree['crownArea'],
        ])
        polygons.append(tree.get('crownPolygon', []))  # Get crownPolygon if it exists
    return np.array(coords), np.array(attributes), polygons

def merge_tree_attributes(tree_coords, tree_attributes, polygons, radius=0.00001):
    """Clusters trees using DBSCAN and merges tree attributes within clusters."""
    dbscan = DBSCAN(eps=radius, min_samples=1)
    cluster_labels = dbscan.fit(tree_coords).labels_
    cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    unique_cluster_sizes = np.unique(cluster_sizes)
    print("INFO: Unique cluster sizes", unique_cluster_sizes)

    num_unmerged_trees = len(tree_attributes)
    num_merged_trees = len(cluster_ids)
    num_tree_attributes = tree_attributes.shape[1]

    merged_tree_attributes = np.zeros((num_merged_trees, num_tree_attributes), dtype=np.float64)
    merged_tree_counts = np.zeros((num_merged_trees, num_tree_attributes), dtype=np.int64)
    merged_polygons = [None] * num_merged_trees  # Initialize list for merged polygons
    ones = np.ones((num_unmerged_trees, num_tree_attributes), dtype=np.int64)

    # Handle missing values
    ones[np.logical_not(np.isfinite(tree_attributes))] = 0
    tree_attributes[np.logical_not(np.isfinite(tree_attributes))] = 0

    # Aggregate attributes over clusters
    for idx in range(num_tree_attributes):
        np.add.at(merged_tree_attributes[:, idx], cluster_labels, tree_attributes[:, idx])
        np.add.at(merged_tree_counts[:, idx], cluster_labels, ones[:, idx])

    # Store the polygon of the first tree in each cluster
    for i, cluster_id in enumerate(cluster_ids):
        if merged_polygons[cluster_id] is None:  # Only assign the first encountered polygon
            merged_polygons[cluster_id] = polygons[i]

    # Average the tree attributes over merged trees
    tree_attributes_merged = merged_tree_attributes / merged_tree_counts
    return tree_attributes_merged, cluster_labels, merged_polygons

def transform_coordinates_back_to_epsg25833(clustered_coords):
    """Transforms coordinates from EPSG:4326 to EPSG:25833."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    transformed_coords = []
    for coord in clustered_coords:
        x, y = transformer.transform(coord[1], coord[0])  # (lat, lon) to (x, y)
        transformed_coords.append([x, y])
    return np.array(transformed_coords)

def export_to_json(clustered_coords, cluster_attributes, polygons, output_file="merged_trees.json"):
    """Exports clustered tree locations and attributes to a JSON file."""
    features = []
    for i, (coord, attributes, polygon) in enumerate(zip(clustered_coords, cluster_attributes, polygons)):
        feature = {
            "treeId": i,  # Assign a unique ID for each cluster
            "groundHeight": attributes[0],
            "height": attributes[1],
            "crownRadius": attributes[2],
            "crownArea": attributes[3],
            "crownPolygon": polygon,  # Include the crownPolygon
            "groundHeightCircle": {
                "radius": None,  # Placeholder, add actual value if needed
                "x": coord[0],
                "y": coord[1],
            },
            "breastHeightCircle": {
                "radius": None,  # Placeholder, add actual value if needed
                "x": coord[0],
                "y": coord[1],
            },
        }
        features.append(feature)
    
    with open(output_file, "w") as f:
        json.dump({"trees": features}, f, indent=4)

    print(f"JSON file '{output_file}' created successfully.")

def main():
    # Parse the folder name with tree locations
    parser = argparse.ArgumentParser(description="Cluster and merge tree locations")
    parser.add_argument("-f", "--folder", help="Relative path to folder with tree locations")
    args = parser.parse_args()

    # Get path names of files that end with .json
    folder = os.path.join(os.getcwd(), args.folder)
    json_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".json") and "clustered" not in file]

    all_tree_coords = []
    all_tree_attributes = []
    all_polygons = []

    # Load and aggregate tree data from all JSON files
    for json_file in json_files:
        trees = load_trees_from_json(json_file)
        tree_coords, tree_attributes, polygons = get_tree_coordinates(trees)
        all_tree_coords.append(tree_coords)
        all_tree_attributes.append(tree_attributes)
        all_polygons.extend(polygons)

    # Convert lists to numpy arrays
    all_tree_coords = np.vstack(all_tree_coords)
    all_tree_attributes = np.vstack(all_tree_attributes)

    # Merge tree attributes based on clustering
    tree_attributes_merged, cluster_ids, merged_polygons = merge_tree_attributes(all_tree_coords, all_tree_attributes, all_polygons)

    # Get clustered coordinates (averaged)
    clustered_coords = []
    for cluster_id in np.unique(cluster_ids):
        cluster_coords = all_tree_coords[cluster_ids == cluster_id]
        mean_coords = cluster_coords.mean(axis=0)
        clustered_coords.append(mean_coords)
    
    clustered_coords = np.array(clustered_coords)

    # Transform coordinates back to EPSG:25833
    clustered_coords_epsg25833 = transform_coordinates_back_to_epsg25833(clustered_coords)

    # Export merged tree locations and attributes to JSON
    export_path = os.path.join(folder, "treeDetails-20230720_Sauen_PLS_clustered_with_polygon.json")
    export_to_json(clustered_coords_epsg25833, tree_attributes_merged, merged_polygons, output_file=export_path)

if __name__ == "__main__":
    main()