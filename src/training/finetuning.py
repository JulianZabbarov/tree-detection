import os
import sys
import time
import copy

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import wandb
from deepforest import utilities, main, preprocess
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np

from src.utils.imports import load_pipeline_config
from src.prediction.run_tree_detection import start_prediction
from src.configs.config_definition import PipelineConfig


def load_annotations(path: str):
    """
    Load annotations from xml file
    """
    return utilities.xml_to_annotations(path)


def adjust_export_paths(config: PipelineConfig, added_subfolder_name: str):
    config.export.annotations_path = (
        config.export.annotations_path + "/" + added_subfolder_name
    )
    if not os.path.exists(config.export.annotations_path):
        os.makedirs(config.export.annotations_path)
    config.export.image_path = (
        config.export.image_path + "/" + added_subfolder_name
    )
    if not os.path.exists(config.export.image_path):
        os.makedirs(config.export.image_path)
    return config


def evaluate_model_on_train_set(model, config: PipelineConfig, seed: int):
    """
    Evaluate model on training data. Adjusts config to evaluate model on training data.

    Args:
        model: deepforest model
        config: pipeline config
    """
    config.data.path_to_images = config.training.images_folder.strip("/png")
    print(
        "INFO: Training evaluation: path_to_images: ",
        config.data.path_to_images,
    )
    config.data.tile_size = config.training.patch_size
    print("INFO: Training evaluation: tile_size: ", config.data.tile_size)

    adjust_export_paths(config, added_subfolder_name="train-seed-" + str(seed))
    print(
        "INFO: Training evaluation: export path: ",
        config.export.annotations_path,
    )
    print(
        "INFO: Training evaluation: export image path: ",
        config.export.image_path,
    )

    start_prediction(model, config=config)


def transform_annotations(folder: str):
    """
    Transform annotations from xml to csv. Required for deepforest training.

    Args:
        folder: path to folder containing xml files
    """
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            annotations = load_annotations(os.path.join(folder, file))
            annotations["label"] = "Tree"
            annotations["image_path"] = annotations["image_path"].str.replace(
                "tif", "png"
            )
            print(annotations.head())
            # drop all rows where xmin == xmax or ymin == ymax
            # these are invalid annotations
            annotations = annotations[
                (annotations["xmin"] != annotations["xmax"])
                & (annotations["ymin"] != annotations["ymax"])
            ]
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
            image_name = file
            break

    annotation_filename = None
    for file in os.listdir(path_to_annotations):
        print(file)
        print(image_name)
        if file.endswith(".csv") and file.startswith(image_name.strip(".png")):
            annotation_filename = file
            break

    # create crops for the raster
    crop_dir = os.path.join(
        os.getcwd(),
        path_to_annotations,
        f"annotations_for_subtiles-run_at_{str(round(time.time()*1000))}",
    )
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    print(os.path.join(path_to_images, image_name))
    print(os.path.join(path_to_annotations, annotation_filename))
    _ = preprocess.split_raster(
        path_to_raster=os.path.join(path_to_images, image_name),
        annotations_file=os.path.join(path_to_annotations, annotation_filename),
        save_dir=crop_dir,
        patch_size=config.training.patch_size,
        patch_overlap=0.05,
    )
    return crop_dir, annotation_filename


def clean_up():
    """
    Clean up crop directories
    """
    if (
        config.training.unsupervised_annotations_folder
        and config.training.unsupervised_images_folder
    ):
        os.system(f"rm -r {unsupervised_crop_dir}")
    os.system(f"rm -r {crop_dir}")


if __name__ == "__main__":
    # load config file
    config = load_pipeline_config()

    print("\nLoading data ...")

    # transform annotations from xml to csv
    transform_annotations(folder=config.training.annotations_folder)
    transform_annotations(
        folder=config.training.unsupervised_annotations_folder
    )

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
        print(
            "INFO: Unsupervised annotations or images folder is not provided. Single-stage fine-tuning will be performed."
        )

    # get rastered annotations for hand-labelled annotations
    crop_dir, annotation_filename = get_rastered_annotations(
        path_to_images=config.training.images_folder,
        path_to_annotations=config.training.annotations_folder,
    )

    # logger = WandbLogger(project="tree-detection_sauen")
    # wandb.init(project="tree_detection-sauen", entity="julianzabbarov")

    print("\nStarting training ...")

    for seed in config.training.seeds:
        print("INFO: Training with seed: ", seed)
        
        # copy config to avoid overwriting
        current_config = copy.deepcopy(config)

        # set seeds for reproducibility
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        # configure model
        model = main.deepforest()
        model.use_release()
        model.config["gpus"] = "-1"
        model.config["train"]["epochs"] = current_config.training.num_epochs
        model.config["save-snapshot"] = False

        if (
            current_config.training.unsupervised_annotations_folder
            and current_config.training.unsupervised_images_folder
        ):
            # set training data for unsupervised annotations
            model.config["train"]["csv_file"] = os.path.join(
                unsupervised_crop_dir, unsupervised_annotation_filename
            )
            model.config["train"]["root_dir"] = os.path.dirname(
                os.path.join(
                    unsupervised_crop_dir, unsupervised_annotation_filename
                )
            )
            model.create_trainer(
                precision=16, log_every_n_steps=1
            )  # , logger=logger)
            model.trainer.fit(model)

        # set training data for hand-labelled annotations
        model.config["train"]["csv_file"] = os.path.join(
            crop_dir, annotation_filename
        )
        model.config["train"]["root_dir"] = os.path.dirname(
            os.path.join(crop_dir, annotation_filename)
        )
        model.create_trainer(precision=16, log_every_n_steps=1)  # , logger=logger)
        model.trainer.fit(model)

        # start prediction on training data
        print("\nPredictions on last train set ...")
        config_copy = copy.deepcopy(current_config)
        evaluate_model_on_train_set(model, config=config_copy, seed=seed)

        # evaluate model on target data
        print("\nPredictions on target dataset ...")
        current_config = adjust_export_paths(current_config, added_subfolder_name="test-seed-" + str(seed))
        start_prediction(model, config=current_config)

    # clean up crop directories
    clean_up()

    # save model
    # model.save(os.path.join(os.getcwd(), "experiments/sauen/saved_models/finetuned_model.pth"))
