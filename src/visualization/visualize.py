import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from pathlib import Path

import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import cv2

from deepforest.visualize import plot_predictions

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
        # read in image
        image_path = Path(config.image_folder) / f"{file}.{config.image_format}"
        if config.image_format == "png":
            image = cv2.imread(image_path)

        if config.image_format == "tif":
            image = tiff.imread(image_path)
        image = np.array(image).astype(np.uint8).copy()


        # plot labels on image
        labels = pd.read_csv(os.path.join(Path(config.label_folder), f"{file}.csv"))
        labels = labels[["xmin", "ymin", "xmax", "ymax", "label"]]
        image = plot_predictions(
            image=image,
            df=labels,
            color=(0, 165, 255), # orange
            thickness=4
        )

        # plot predictions
        predictions = pd.read_csv(os.path.join(Path(config.predictions_folder), f"{file}.csv"))
        predictions = predictions[["xmin", "ymin", "xmax", "ymax", "label"]]
        image = plot_predictions(
            image=image,
            df=predictions,
            color=(255, 89, 0), # blue
            thickness=2
        )

        # export image to config.export_folder
        export_path = os.path.join(Path(config.export_folder), f"{file}.png")
        cv2.imwrite(export_path, image)
        print(f"Exported image to {export_path}")