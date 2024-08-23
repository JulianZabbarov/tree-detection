import os
import json
import geojson
from pyproj import Transformer
import argparse
from pathlib import Path

# Function to calculate the bounding box that surrounds the polygon
def create_bounding_box_from_polygon(polygon):
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)]

# Function to create a GeoJSON Feature for each tree
def create_feature(tree):
    # Initialize the transformer to convert coordinates from EPSG:25833 to EPSG:4326
    transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

    # Transform ground and breast height circle centers
    ground_x, ground_y = transformer.transform(
        tree["groundHeightCircle"]["x"], tree["groundHeightCircle"]["y"]
    )
    breast_x, breast_y = transformer.transform(
        tree["breastHeightCircle"]["x"], tree["breastHeightCircle"]["y"]
    )

    # Transform the crown polygon coordinates
    crown_polygon = tree.get("crownPolygon", [])
    crown_polygon_transformed = []
    if crown_polygon:
        for i in range(0, len(crown_polygon), 2):
            x = crown_polygon[i] + tree["groundHeightCircle"]["x"]
            y = crown_polygon[i + 1] + tree["groundHeightCircle"]["y"]
            lon, lat = transformer.transform(x, y)
            crown_polygon_transformed.append((lon, lat))

    # Ensure the polygon is closed by repeating the first point at the end
    if crown_polygon_transformed:
        crown_polygon_transformed.append(crown_polygon_transformed[0])

    # Create a bounding box that surrounds the polygon
    bbox_transformed = create_bounding_box_from_polygon(crown_polygon_transformed)

    return geojson.Feature(
        geometry=geojson.Polygon([bbox_transformed]),
        properties={
            "treeId": tree["treeId"],
            "groundHeight": tree["groundHeight"],
            "height": tree["height"],
            "crownRadius": tree["crownRadius"],
            "crownArea": tree["crownArea"],
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
                with open(os.path.join(folder, f"{filename}-bboxs_using_polygons.geojson"), "w") as f:
                    geojson.dump(feature_collection, f)

                print(f"GeoJSON file created successfully for {file}.")

if __name__ == "__main__":
    main()
