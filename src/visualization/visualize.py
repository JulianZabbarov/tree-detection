import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from pathlib import Path

import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from deepforest.visualize import plot_prediction_and_targets

from src.utils.imports import load_config

def get_filenames_in_folder(path: str, type: str) -> set:
    files = os.listdir(Path(path))
    files = [file for file in files if file.endswith(type)]
    files = set([file.split(".")[0] for file in files])
    return files

if __name__ == "__main__":
    config = load_config().visualization

    image_names = get_filenames_in_folder(
        path=config.image_folder,
        type=config.image_format,
    )
    prediction_names = get_filenames_in_folder(
        path=config.predictions_folder,
        type=".csv"
    )
    files = image_names.intersection(prediction_names)
    if files == set():
        raise ValueError("No image-predictions pair found in specified folder.")
    
    for file in files:
        image_path = Path(config.image_folder) / f"{file}.{config.image_format}"
        # read in image
        if config.image_format == "png":
            image = plt.imread(image_path)

        if config.image_format == "tif":
            image = tiff.imread(image_path)

        label_path = os.path.join(Path(config.label_folder), f"{file}.csv")
        prediction_path = os.path.join(Path(config.predictions_folder), f"{file}.csv")
        export_path = os.path.join(Path(config.export_folder), f"{file}.png")
        path_of_exported_image = plot_prediction_and_targets(
            image=image,
            targets=pd.read_csv(label_path),
            predictions= pd.read_csv(prediction_path),
            export_path=export_path,
        )
        print("Image exported to: ", path_of_exported_image)
