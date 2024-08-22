import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import wandb
from deepforest import utilities, main, preprocess
from pytorch_lightning.loggers import WandbLogger

from src.utils.imports import load_config
from src.prediction.run_tree_detection import start_prediction


def get_annotations(path: str):
    return utilities.xml_to_annotations(path)


if __name__ == "__main__":
    # load config file
    config = load_config()

    print("\nLoading data ...")

    # get annotations from xml file
    folder = config.training.annotations_folder
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            annotations = get_annotations(os.path.join(folder, file))
            annotations["label"] = "Tree"
            annotations.to_csv(
                os.path.join(folder, str(file).replace(".xml", ".csv")),
                index=False,
            )

    # load path for image and annotations used for training
    raster = os.listdir(config.training.images_folder)[1]
    annotation = None
    for file in os.listdir(config.training.annotations_folder):
        if file.endswith(".csv") and file.startswith(raster.split(".")[0]):
            annotation = file
            break

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

    logger = WandbLogger(project="tree-detection_sauen")
    wandb.init(project='tree_detection-sauen', entity='julianzabbarov')

    print("\nStarting training ...")

    # configure model
    model = main.deepforest()
    model.use_release()
    model.config["gpus"] = "-1"
    model.config["train"]["epochs"] = config.training.num_epochs
    model.config["save-snapshot"] = False
    model.config["train"]["csv_file"] = os.path.join(crop_dir, annotation)
    model.config["train"]["root_dir"] = os.path.dirname(
        os.path.join(crop_dir, annotation)
    )
    
    model.create_trainer(precision=16, log_every_n_steps=1, logger=logger)
    model.trainer.fit(model)

    print("\nPredicting ...")

    start_prediction(model, config=config)

    # save model
    # model.save(os.path.join(os.getcwd(), "experiments/sauen/saved_models/finetuned_model.pth"))

    
