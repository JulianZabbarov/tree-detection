import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pyproj import Transformer
import geojson
import argparse

def load_trees_from_json(file_path):
    """Loads tree data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['trees']

def get_tree_coordinates(trees):
    """Extracts ground height coordinates from tree data."""
    transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
    coords = []
    attributes = []
    for tree in trees:
        x, y = tree['groundHeightCircle']['x'], tree['groundHeightCircle']['y']
        lat, lon = transformer.transform(x, y)
        coords.append([lon, lat])  # GeoJSON uses (lon, lat)
        attributes.append([
            tree['groundHeight'],
            tree['height'],
            tree['crownRadius'],
            tree['crownArea']
        ])
    return np.array(coords), np.array(attributes)

def merge_tree_attributes(tree_coords, tree_attributes, radius=0.00001):
    """Clusters trees using DBSCAN and merges tree attributes within clusters."""
    dbscan = DBSCAN(eps=radius, min_samples=1)
    cluster_labels = dbscan.fit(tree_coords).labels_
    cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    unique_cluster_sizes = np.unique(cluster_sizes)
    print("INFO: Unique luster sizes", unique_cluster_sizes)

    num_unmerged_trees = len(tree_attributes)
    num_merged_trees = len(cluster_ids)
    num_tree_attributes = tree_attributes.shape[1]

    merged_tree_attributes = np.zeros((num_merged_trees, num_tree_attributes), dtype=np.float64)
    merged_tree_counts = np.zeros((num_merged_trees, num_tree_attributes), dtype=np.int64)
    ones = np.ones((num_unmerged_trees, num_tree_attributes), dtype=np.int64)

    # Handle missing values
    ones[np.logical_not(np.isfinite(tree_attributes))] = 0
    tree_attributes[np.logical_not(np.isfinite(tree_attributes))] = 0

    # Aggregate attributes over clusters
    for idx in range(num_tree_attributes):
        np.add.at(merged_tree_attributes[:, idx], cluster_labels, tree_attributes[:, idx])
        np.add.at(merged_tree_counts[:, idx], cluster_labels, ones[:, idx])

    # Average the tree attributes over merged trees
    tree_attributes_merged = merged_tree_attributes / merged_tree_counts
    return tree_attributes_merged, cluster_labels

def export_to_geojson(clustered_coords, cluster_attributes, output_file="merged_trees.geojson"):
    """Exports clustered tree locations and attributes to a GeoJSON file."""
    features = []
    for i, (coord, attributes) in enumerate(zip(clustered_coords, cluster_attributes)):
        point = geojson.Point((coord[1], coord[0]))  # (lon, lat)
        properties = {
            "segmentid": i,
            "groundHeight": attributes[0],
            "height": attributes[1],
            "crownRadius": attributes[2],
            "crownArea": attributes[3],
        }
        features.append(geojson.Feature(geometry=point, properties=properties))
    
    feature_collection = geojson.FeatureCollection(features)
    
    with open(output_file, "w") as f:
        geojson.dump(feature_collection, f)

    print(f"GeoJSON file '{output_file}' created successfully.")

def main():
    # Parse the folder name with tree locations
    parser = argparse.ArgumentParser(description="Cluster and merge tree locations")
    parser.add_argument("-f", "--folder", help="Relative path to folder with tree locations")
    args = parser.parse_args()

    # Get path names of files that end with .json
    folder = os.path.join(os.getcwd(), args.folder)
    json_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".json")]

    all_tree_coords = []
    all_tree_attributes = []

    # Load and aggregate tree data from all JSON files
    for json_file in json_files:
        trees = load_trees_from_json(json_file)
        tree_coords, tree_attributes = get_tree_coordinates(trees)
        all_tree_coords.append(tree_coords)
        all_tree_attributes.append(tree_attributes)

    # Convert lists to numpy arrays
    all_tree_coords = np.vstack(all_tree_coords)
    all_tree_attributes = np.vstack(all_tree_attributes)

    # Merge tree attributes based on clustering
    tree_attributes_merged, cluster_ids = merge_tree_attributes(all_tree_coords, all_tree_attributes)

    # Get clustered coordinates (averaged)
    clustered_coords = []
    print("INFO: Number of clusters", len(np.unique(cluster_ids)))
    print("INFO: Number of cluster_ids", len(cluster_ids))
    print("INFO: Number of trees", len(all_tree_coords))
    for cluster_id in np.unique(cluster_ids):
        cluster_coords = all_tree_coords[cluster_ids == cluster_id]
        mean_coords = cluster_coords.mean(axis=0)
        clustered_coords.append(mean_coords)
    
    clustered_coords = np.array(clustered_coords)

    # Export merged tree locations and attributes to GeoJSON
    export_path = os.path.join(folder, "treeDetails-20230720_Sauen_PLS_clustered.geojson")
    export_to_geojson(clustered_coords, tree_attributes_merged, output_file=export_path)

if __name__ == "__main__":
    main()
