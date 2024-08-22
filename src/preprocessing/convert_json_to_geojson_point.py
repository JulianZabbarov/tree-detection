import os
import json
import geojson
from pyproj import Transformer
import argparse
from pathlib import Path


# Function to create a GeoJSON Feature for each tree
def create_feature(tree):

    # Initialize the transformer to convert coordinates to EPSG:25833
    transformer = Transformer.from_crs(
        "EPSG:25833", "EPSG:4326", always_xy=True
    )

    ground_x, ground_y = transformer.transform(
        tree["groundHeightCircle"]["x"], tree["groundHeightCircle"]["y"]
    )
    breast_x, breast_y = transformer.transform(
        tree["breastHeightCircle"]["x"], tree["breastHeightCircle"]["y"]
    )

    return geojson.Feature(
        geometry=geojson.Point((ground_x, ground_y)),
        properties={
            "treeId": tree["treeId"],
            "groundHeight": tree["groundHeight"],
            "height": tree["height"],
            "crownRadius": tree["crownRadius"],
            "crownArea": tree["crownArea"],
            "deletionReason": tree["deletionReason"],
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
    parser = argparse.ArgumentParser(
        description="Convert JSON tree data to GeoJSON"
    )
    parser.add_argument("-f", "--folder", help="Folder name with tree locations")
    args = parser.parse_args()

    folder = os.path.join(Path(os.getcwd()), Path(args.folder))
    # load all json files in the folder
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
                with open(os.path.join(folder, f"{filename}.geojson"), "w") as f:
                    geojson.dump(feature_collection, f)

                print(f"GeoJSON file created successfully for {file}.")


if __name__ == "__main__":
    main()
