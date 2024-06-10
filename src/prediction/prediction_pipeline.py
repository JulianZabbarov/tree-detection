import os

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tifffile import imread
import numpy as np
from tqdm import tqdm
from deepforest import main

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-p",
    "--path-to-images",
    dest="images_path",
    action="store",
    help="relative path to folder with images for which annotations are available",
)
parser.add_argument(
    "-l",
    "--path-to-labels",
    dest="annotations_path",
    action="store",
    help="relative path to csv with labels for images in defined image path",
)
parser.add_argument(
    "-e",
    "--export-path",
    dest="export_path",
    action="store",
    help="relative path to folder where predictions should be exported to",
)
args = parser.parse_args()


class NeonTreeEvaluation(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(os.getcwd() + annotations_file)
        self.img_dir = img_dir
        self.img_names = os.listdir(os.getcwd() + self.img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(os.getcwd() + self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(os.getcwd() + self.img_dir, self.img_names[idx])
        image = imread(img_path)
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image

    def __getname__(self, idx):
        return self.img_names[idx]


print("\nLoading dataset and model ...")
neontree = NeonTreeEvaluation(
    args.annotations_path, args.images_path, None, None
)

model = main.deepforest()
model.use_release()

predictions = pd.DataFrame()

# predict images
print("\nRunning predictions ...")
for img_idx in tqdm(range(len(neontree))):
    pred = model.predict_image(
        image=neontree.__getitem__(img_idx).astype(np.float32),
        return_plot=False,
    )
    pred["image_path"] = neontree.__getname__(img_idx)
    predictions = pd.concat([predictions, pred])

# format predictions to for readible inspection
predictions = predictions[
    ["image_path", "xmin", "ymin", "xmax", "ymax", "label", "score"]
]
predictions.sort_values(by="image_path", inplace=True).reset_index(drop=True)

# export predictions
predictions.to_csv(
    os.path.join(os.getcwd(), args.export_path, "benchmark_predictions.csv"),
    index=False,
)

print(
    "\nPredictions exported to: {export}.".format(
        export=os.path.join(os.getcwd(), args.export_path)
    )
)
