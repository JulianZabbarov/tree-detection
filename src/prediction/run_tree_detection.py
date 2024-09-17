import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tifffile import imread
import numpy as np
from tqdm import tqdm
from deepforest import main
import tomllib
from PIL import Image
from dacite import from_dict
from src.configs.config_definition import PipelineConfig
import torch

from argparse import ArgumentParser

from src.utils import export, imports

class TreeDataset(Dataset):
    def __init__(
        self, config, transform: bool = True, target_transform: bool = False
    ):
        self.config = config
        self.img_names = os.listdir(
            os.getcwd() + "/" + self.config.data.path_to_images
        )
        # filter out non-image files
        self.img_names = [img for img in self.img_names if img.endswith(".tif")]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        print(f"Image names: {self.img_names}")
        img_path = os.path.join(
            os.getcwd(), self.config.data.path_to_images, self.img_names[idx]
        )
        print(f"Loading image: {img_path}")
        image_array = np.array(imread(img_path))[:, :, :3].astype(np.uint8)
        image = Image.fromarray(image_array)

        if not os.path.exists(
            os.path.join(os.getcwd(), self.config.export.image_path)
        ):
            os.mkdir(os.path.join(os.getcwd(), self.config.export.image_path))
        image.save(
            f"{os.getcwd()}/{self.config.export.image_path}/{self.img_names[idx].replace('.tif', '.png')}",
            type="PNG",
        )

        if self.transform:
            # resize image to 400x400, default for deepforest
            image = image.resize((400, 400))
        if self.target_transform:
            # don't resize for tile prediction
            label = self.target_transform(label)

        image = np.array(image)
        return image

    def __getname__(self, idx):
        return self.img_names[idx]


def start_prediction(model, config, seed):
    print("\nLoading dataset and model ...")

    if not config.data.predict_tile:
        tree_dataset = TreeDataset(
            config, transform=True, target_transform=False
        )
    else:
        tree_dataset = TreeDataset(
            config, transform=False, target_transform=False
        )

    all_predictions = (
        pd.DataFrame() if config.export.type == "combined" else None
    )

    # predict images
    print(f"\nRunning predictions for {len(tree_dataset)} image(s) ...")
    for img_idx in tqdm(range(len(tree_dataset))):
        if config.data.predict_tile:
            pred = model.predict_tile(
                image=tree_dataset.__getitem__(img_idx).astype(np.float32),
                return_plot=False,
                patch_size=config.data.tile_size,
                patch_overlap=0.1,
            )
        else:
            pred = model.predict_image(
                image=tree_dataset.__getitem__(img_idx).astype(np.float32),
                return_plot=False,
            )
        pred["image_path"] = tree_dataset.__getname__(img_idx)
        if all_predictions is not None:
            all_predictions = pd.concat([all_predictions, pred], axis=0)
        else:
            img_name = tree_dataset.__getname__(img_idx)
            img_name_with_seed = img_name.split(".")[0] + "-seed-" + str(seed) + "." + img_name.split(".")[1]
            if config.export.annotations_format == "XML":
                export.export_predictions_as_xml(
                    pred=pred,
                    image_name=img_name_with_seed,
                    image_folder=config.data.path_to_images,
                    export_config=config.export,
                    image_size=config.export.image_size,
                    scale_annotations=not (config.data.predict_tile),
                )
            if config.export.annotations_format == "CSV":
                export.export_predictions_as_csv(
                    pred_df=pred,
                    export_config=config.export,
                    image_name=img_name_with_seed,
                )

    if all_predictions is not None:
        if config.export.annotations_format == "CSV":
            export.export_predictions_as_csv(
                pred_df=all_predictions,
                export_config=config.export,
                image_name=config.data.path_to_images.split("/")[1],
            )

    print(
        "\nPredictions exported to: {export}.".format(
            export=os.path.join(os.getcwd(), config.export.annotations_path)
        )
    )


if __name__ == "__main__":
    config = imports.load_pipeline_config()

    for seed in [0, 1, 2, 3, 4]:

        # set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # loading model
        model = main.deepforest()
        model.use_release()

        start_prediction(model=model, config=config, seed=seed)
