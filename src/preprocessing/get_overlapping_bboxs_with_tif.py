import os
import argparse
import geopandas as gpd
from shapely.geometry import box
import rasterio
import xml.etree.ElementTree as ET
from pathlib import Path

def world_to_pixel(geo_transform, x, y):
    """Convert geospatial coordinates (x, y) to pixel coordinates (col, row)."""
    a = geo_transform.a
    b = geo_transform.b
    c = geo_transform.c
    d = geo_transform.d
    e = geo_transform.e
    f = geo_transform.f

    print(a, b, c, d, e, f)
    if a == 0.0 or e == 0.0:
        raise ValueError("Affine transform scaling factors cannot be zero.")
    
    col = (x - c) / a
    row = (y - f) / e
    return col, row

def create_xml_bbox(xmin, ymin, xmax, ymax):
    """Create XML structure for a bounding box."""
    obj = ET.Element('object')
    name = ET.SubElement(obj, 'name')
    name.text = 'AddedTree'

    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Unspecified'

    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = '0'

    occluded = ET.SubElement(obj, 'occluded')
    occluded.text = '0'

    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')

    xmin_elem = ET.SubElement(bndbox, 'xmin')
    xmin_elem.text = f"{xmin:.6f}"

    ymin_elem = ET.SubElement(bndbox, 'ymin')
    ymin_elem.text = f"{ymin:.6f}"

    xmax_elem = ET.SubElement(bndbox, 'xmax')
    xmax_elem.text = f"{xmax:.6f}"

    ymax_elem = ET.SubElement(bndbox, 'ymax')
    ymax_elem.text = f"{ymax:.6f}"

    return obj

def filter_bounding_boxes_to_xml(tif_path, geojson_path, output_xml, folder_name, filename):
    # Load the TIF file
    with rasterio.open(tif_path) as src:

        # Get the bounding box of the forest area (entire TIF file)
        forest_bounds = src.bounds
        geo_transform = src.transform  # Geospatial transformation to convert coordinates to pixels
        width = src.width
        height = src.height
        depth = src.count

        # Convert the forest bounds to a Shapely Polygon (still in the TIF CRS)
        forest_polygon = box(*forest_bounds)
        print(f"Forest bounds: {forest_polygon}")

        # Load the GeoJSON file containing the bounding boxes (in EPSG:4326)
        gdf = gpd.read_file(geojson_path)

        # List to hold the XML objects
        xml_objects = []

        # Iterate through each bounding box (polygon geometry)
        for index, row in gdf.iterrows():
            bbox = row['geometry']
            
            print(bbox)
            print(forest_polygon)
            # Clip the bounding box to the forest polygon
            clipped_bbox = bbox.intersection(forest_polygon)
            
            # If the clipped bounding box has an area greater than 0, keep it
            if not clipped_bbox.is_empty:
                # Convert the bounding box to pixel coordinates
                minx, miny, maxx, maxy = clipped_bbox.bounds
                try:
                    xmin_pixel, ymin_pixel = world_to_pixel(geo_transform, minx, miny)
                    xmax_pixel, ymax_pixel = world_to_pixel(geo_transform, maxx, maxy)
                except ValueError as e:
                    print("Error in conversion:", e)
                    continue  # Skip this bounding box if conversion fails
                
                # Create the XML object for the bounding box
                xml_obj = create_xml_bbox(xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel)
                xml_objects.append(xml_obj)

        # Create XML root
        root = ET.Element("annotation")
        
        # Add folder and filename
        folder_elem = ET.SubElement(root, 'folder')
        folder_elem.text = folder_name

        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = filename

        # Add image size
        size_elem = ET.SubElement(root, 'size')
        width_elem = ET.SubElement(size_elem, 'width')
        width_elem.text = str(width)

        height_elem = ET.SubElement(size_elem, 'height')
        height_elem.text = str(height)

        depth_elem = ET.SubElement(size_elem, 'depth')
        depth_elem.text = str(depth)

        # Append all the XML objects (bounding boxes)
        for xml_obj in xml_objects:
            root.append(xml_obj)

        # Write the XML tree to the output XML file
        tree = ET.ElementTree(root)
        tree.write(output_xml, encoding='utf-8', xml_declaration=True)
        print(f"Bounding boxes saved as XML to: {output_xml}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filter and adjust bounding boxes that overlap with the forest area in a TIF file.")
    parser.add_argument("-t", "--tif_file", type=str, help="Path to the TIF file representing the forest area.")
    parser.add_argument("-g", "--geojson_file", type=str, help="Path to the GeoJSON file containing the bounding boxes.")
    parser.add_argument("-o", "--output_xml", type=str, help="Path to save the bounding boxes as pixel coordinates in XML format.")
    
    args = parser.parse_args()
    
    # Run the filtering function
    folder_name = os.path.dirname(Path(args.tif_file)).split('/')[-1]
    file_name = os.path.basename(Path(args.tif_file))
    print(folder_name)
    filter_bounding_boxes_to_xml(args.tif_file, args.geojson_file, args.output_xml, folder_name, file_name)