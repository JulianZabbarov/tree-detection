import os
import json
import geojson
from pyproj import Transformer
import argparse
from pathlib import Path


# Function to create a bounding box
def create_bounding_box(center_x, center_y, radius):
    min_x = center_x - radius
    max_x = center_x + radius
    min_y = center_y - radius
    max_y = center_y + radius
    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)]


# Function to create a GeoJSON Feature for each tree
def create_feature(tree):
    # Initialize the transformer to convert coordinates to EPSG:4326
    transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

    # Transform ground and breast height circle centers
    ground_x, ground_y = transformer.transform(
        tree["groundHeightCircle"]["x"], tree["groundHeightCircle"]["y"]
    )
    breast_x, breast_y = transformer.transform(
        tree["breastHeightCircle"]["x"], tree["breastHeightCircle"]["y"]
    )

    # Create bounding box based on groundHeightCircle
    bbox = create_bounding_box(
        tree["groundHeightCircle"]["x"], tree["groundHeightCircle"]["y"], tree["crownRadius"]
    )

    # Transform the bounding box coordinates to EPSG:4326
    bbox_transformed = [transformer.transform(x, y) for x, y in bbox]

    return geojson.Feature(
        geometry=geojson.Polygon([bbox_transformed]),
        properties={
            "treeId": tree["treeId"],
            "groundHeight": tree["groundHeight"],
            "height": tree["height"],
            "crownRadius": tree["crownRadius"],
            "crownArea": tree["crownArea"],
            # "deletionReason": tree["deletionReason"],
            "breastHeightCircle": {
                "radius": tree["breastHeightCircle"]["radius"],
                "x": breast_x,
                "y": breast_y,
            },
            "groundHeightCircle": {
                "radius": tree["groundHeightCircle"]["radius"],
                "x": ground_x,
                "y": ground_y,
            },
        },
    )


def main():
    # Parse the folder name with tree locations
    parser = argparse.ArgumentParser(description="Convert JSON tree data to GeoJSON")
    parser.add_argument("-f", "--folder", help="Folder name with tree locations")
    args = parser.parse_args()

    folder = os.path.join(Path(os.getcwd()), Path(args.folder))
    # Load all JSON files in the folder
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)

                # Create GeoJSON Features for all trees
                features = [create_feature(tree) for tree in data["trees"]]

                # Create a GeoJSON FeatureCollection
                feature_collection = geojson.FeatureCollection(features)

                # Output the GeoJSON to a file
                filename = file.split(".")[0]
                with open(os.path.join(folder, f"{filename}-bboxs.geojson"), "w") as f:
                    geojson.dump(feature_collection, f)

                print(f"GeoJSON file created successfully for {file}.")


if __name__ == "__main__":
    main()
