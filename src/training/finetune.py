import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

from deepforest import utilities, main, preprocess

from src.utils.imports import load_config


def get_annotations(path: str):
    return utilities.xml_to_annotations(path)


if __name__ == "__main__":
    # load config file
    config = load_config()

    # get annotations from xml file
    folder = config.training.annotations_folder
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            annotations = get_annotations(os.path.join(folder, file))
            annotations["label"] = "Tree"
            print(annotations.head())
            annotations.to_csv(
                os.path.join(folder, str(file).replace(".xml", ".csv")),
                index=False,
            )

    # load path for image and annotations used for training
    raster = os.listdir(config.training.images_folder)[1]
    print("raster", raster)
    annotation = os.listdir(config.training.annotations_folder)[0]
    print("annotation", annotation)
    # create crops for the raster
    crop_dir = os.path.join(os.getcwd(), config.training.images_folder, "tiles")
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    train_annotations = preprocess.split_raster(
        path_to_raster=os.path.join(config.training.images_folder, raster),
        annotations_file=os.path.join(
            config.training.annotations_folder, annotation
        ),
        save_dir=crop_dir,
        patch_size=config.training.patch_size,
        patch_overlap=0.05,
    )

    # configure model
    print("train.csv_file", os.path.join(crop_dir, annotation))
    print("train.root_dir", os.path.dirname(os.path.join(crop_dir, annotation)))
    model = main.deepforest()
    model.use_release()
    # model.config["gpus"] = "-1"
    model.config["epochs"] = 1
    model.config["save-snapshot"] = False
    model.config["train"]["csv_file"] = os.path.join(crop_dir, annotation)
    model.config["train"]["root_dir"] = os.path.dirname(
        os.path.join(crop_dir, annotation)
    )
    model.create_trainer(precision=16)
    model.trainer.fit(model)
