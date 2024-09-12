import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import wandb
from deepforest import utilities, main, preprocess
from pytorch_lightning.loggers import WandbLogger

from src.utils.imports import load_config
from src.prediction.run_tree_detection import start_prediction

def load_annotations(path: str):
    return utilities.xml_to_annotations(path)


def transform_annotations(folder: str):
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            annotations = load_annotations(os.path.join(folder, file))
            annotations["label"] = "Tree"
            annotations["image_path"] = annotations["image_path"].str.replace("tif", "png")
            annotations.to_csv(
                os.path.join(folder, str(file).replace(".xml", ".csv")),
                index=False,
            )
            print(f"Exported {file} to {str(file).replace('.xml', '.csv')}")


def get_rastered_annotations(path_to_images: str, path_to_annotations: str):
    # load path for image and annotations used for training
    print(os.listdir(path_to_images))
    for file in os.listdir(path_to_images):
        if file.endswith("png"):
            raster = file
            break

    annotation_filename = None
    for file in os.listdir(path_to_annotations):
        print(file)
        if file.endswith(".csv") and file.startswith(raster.strip(".png")):
            annotation_filename = file
            break

    # create crops for the raster
    crop_dir = os.path.join(os.getcwd(), path_to_images, "tiles")
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    print(os.path.join(path_to_images, raster))
    print(os.path.join(path_to_annotations, annotation_filename))
    _ = preprocess.split_raster(
        path_to_raster=os.path.join(path_to_images, raster),
        annotations_file=os.path.join(path_to_annotations, annotation_filename),
        save_dir=crop_dir,
        patch_size=config.training.patch_size,
        patch_overlap=0.05,
    )
    return crop_dir, annotation_filename


if __name__ == "__main__":
    # load config file
    config = load_config()

    print("\nLoading data ...")

    # transform annotations from xml to csv
    transform_annotations(folder=config.training.annotations_folder)
    transform_annotations(folder=config.training.unsupervised_annotations_folder)

    # get rastered annotations for unsupervised annotations
    if (
        config.training.unsupervised_annotations_folder
        and config.training.unsupervised_images_folder
    ):
        unsupervised_crop_dir, unsupervised_annotation_filename = (
            get_rastered_annotations(
                path_to_images=config.training.unsupervised_images_folder,
                path_to_annotations=config.training.unsupervised_annotations_folder,
            )
        )
    else:
        raise ValueError(
            "Unsupervised annotations or images folder is not provided but required to run this script."
        )

    # get rastered annotations for hand-labelled annotations
    crop_dir, annotation_filename = get_rastered_annotations(
        path_to_images=config.training.images_folder,
        path_to_annotations=config.training.annotations_folder,
    )

    # logger = WandbLogger(project="tree-detection_sauen")
    # wandb.init(project="tree_detection-sauen", entity="julianzabbarov")

    print("\nStarting training ...")

    # configure model
    model = main.deepforest()
    model.use_release()
    model.config["gpus"] = "-1"
    model.config["train"]["epochs"] = config.training.num_epochs
    model.config["save-snapshot"] = False

    # set training data for unsupervised annotations
    model.config["train"]["csv_file"] = os.path.join(
        unsupervised_crop_dir, unsupervised_annotation_filename
    )
    model.config["train"]["root_dir"] = os.path.dirname(
        os.path.join(unsupervised_crop_dir, unsupervised_annotation_filename)
    )
    model.create_trainer(precision=16, log_every_n_steps=1)# , logger=logger)
    model.trainer.fit(model)

    # set training data for hand-labelled annotations
    model.config["train"]["csv_file"] = os.path.join(
        crop_dir, annotation_filename
    )
    model.config["train"]["root_dir"] = os.path.dirname(
        os.path.join(crop_dir, annotation_filename)
    )
    model.create_trainer(precision=16, log_every_n_steps=1)# , logger=logger)
    model.trainer.fit(model)

    print("\nPredicting ...")

    start_prediction(model, config=config)

    # save model
    # model.save(os.path.join(os.getcwd(), "experiments/sauen/saved_models/finetuned_model.pth"))
