import os
import argparse
import rasterio
from PIL import Image
import numpy as np


def convert_tif_to_png(input_folder, output_folder=None):
    # Use the input folder for output if no output folder is provided
    if output_folder is None:
        output_folder = input_folder

    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    # Create the output folder if it does not exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a .tif or .tiff image
        if file_name.lower().endswith((".tif", ".tiff")):
            input_file_path = os.path.join(input_folder, file_name)

            try:
                # Open the .tif image using rasterio
                with rasterio.open(input_file_path) as src:
                    # Read the image data into a numpy array
                    img_array = src.read()

                    # Transpose the image array to be in the correct shape for Pillow
                    # This is necessary because rasterio returns the array as (band, height, width),
                    # while Pillow expects (height, width, band).
                    img_array = np.transpose(img_array, (1, 2, 0))

                    # Convert the numpy array to a PIL image
                    img = Image.fromarray(img_array)

                    # Generate the output file path with a .png extension
                    new_file_name = os.path.splitext(file_name)[0] + ".png"
                    output_file_path = os.path.join(
                        output_folder, new_file_name
                    )

                    # Save the image in .png format
                    img.save(output_file_path, "PNG")

                    print(f"Converted: {file_name} -> {new_file_name}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Convert TIF images to PNG format."
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Relative path to the folder containing TIF images.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Relative path to the folder where PNG images will be saved (optional). If not provided, PNG images are saved in the input folder.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Convert .tif images to .png
    convert_tif_to_png(args.input_folder, args.output_folder)
