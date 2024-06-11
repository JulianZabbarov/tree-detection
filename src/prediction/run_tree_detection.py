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
from experiments.config_definition import PipelineConfig


from argparse import ArgumentParser

from src.utils import export

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--config-path",
    dest="config",
    action="store",
    help="relative path to config file for experiment configurations",
)
args = parser.parse_args()

with open(args.config, "rb") as f:
    config = from_dict(data_class=PipelineConfig, data=tomllib.load(f))


class TreeDataset(Dataset):
    def __init__(
        self, img_dir, transform: bool = True, target_transform: bool = False
    ):
        self.img_dir = img_dir
        self.img_names = os.listdir(os.getcwd() + "/" + self.img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(os.getcwd() + "/" + self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(os.getcwd(), self.img_dir, self.img_names[idx])
        image_array = np.array(imread(img_path))[:, :, :3].astype(np.uint8)
        image = Image.fromarray(image_array)

        if self.transform:
            image.save(
                f"{os.getcwd()}/{config.export.image_path}/{self.img_names[idx].replace('.tif', '.png')}",
                type="PNG",
            )
            image = image.resize((400, 400))
        if self.target_transform:
            label = self.target_transform(label)

        image = np.array(image)
        return image

    def __getname__(self, idx):
        return self.img_names[idx]


print("\nLoading dataset and model ...")
tree_dataset = TreeDataset(
    config.data.path_to_images, transform=True, target_transform=False
)

model = main.deepforest()
model.use_release()

all_predictions = pd.DataFrame() if config.export.type == "combined" else None

# predict images
print("\nRunning predictions ...")
for img_idx in tqdm(range(len(tree_dataset))):
    pred = model.predict_image(
        image=tree_dataset.__getitem__(img_idx).astype(np.float32),
        return_plot=False,
    )
    pred["image_path"] = tree_dataset.__getname__(img_idx)
    if all_predictions is not None:
        all_predictions = pd.concat([all_predictions, pred], axis=0)
    else:
        if config.export.annotations_format == "XML":
            export.export_predictions_as_xml(
                pred=pred,
                image_name=tree_dataset.__getname__(img_idx),
                image_folder=config.data.path_to_images,
                export_config=config.export,
                width=config.data.image_size,
                height=config.data.image_size,
            )
        if config.export.annotations_format == "CSV":
            export.export_predictions_as_csv(
                pred_df=pred,
                export_config=config.export,
                image_name=tree_dataset.__getname__(img_idx),
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
