import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

from deepforest import utilities, main, preprocess

from src.utils.imports import load_config

def get_annotations(path: str):
    return utilities.xml_to_annotations(path)

if __name__=="__main__":
    # load config file
    config = load_config()

    # get annotations from xml file
    folder = config.training.annotations_folder
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            annotations = get_annotations(os.path.joinfile)
            annotations.to_csv(os.path.join(folder, file), index=False)


    # split large image into raster
    raster = os.path.join(config.training.images_folder, os.listdir(config.training.images_folder)[0])
    annotation = os.path.join(config.training.annotations_folder, os.listdir(config.training.annotations_folder)[0])
    crop_dir = os.path.join(os.getcwd(), config.training.images_folder, "tiles")
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    train_annotations= preprocess.split_raster(path_to_raster=raster,
                                 annotations_file=annotation,
                                 save_dir=crop_dir,
                                 patch_size=400,
                                 patch_overlap=0.05)

    # configure model
    model = main.deepforest()
    model.use_release()
    model.config['gpus'] = '-1'
    model.config["epochs"] = 1
    model.config["save-snapshot"] = False
    model.config["train"]["csv_file"] = annotations
    model.config["train"]["root_dir"] = config.training.images_path
    model.create_trainer()

    model.trainer.fit(model)


