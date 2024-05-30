import os

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tifffile import imread
import numpy as np
from tqdm import tqdm
from deepforest import main

ANNOTATIONS_PATH = "/data/neontree/evaluation/benchmark_annotations.csv"
IMAGES_PATH = "/data/neontree/evaluation/RGB_with_annotations"
EXPORT_PRED_PATH = "src/evaluation"

class NeonTreeEvaluation(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
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
        image = np.array(image)#.astype(np.float32)
        
        xmin = self.img_labels.iloc[idx, 1]
        ymin = self.img_labels.iloc[idx, 2]
        xmax = self.img_labels.iloc[idx, 3]
        ymax = self.img_labels.iloc[idx, 4]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image
    
    def __getname__(self, idx):
        return self.img_names[idx]

neontree = NeonTreeEvaluation(ANNOTATIONS_PATH, IMAGES_PATH, None, None)

model = main.deepforest()
model.use_release()

predictions = pd.DataFrame()

print("Predictions are running ...")
for img_idx in tqdm(range(len(neontree))):
    pred = model.predict_image(image=neontree.__getitem__(img_idx).astype(np.float32), return_plot=False)
    # append pred to predictions with value for image_path set to neontree.get_name(img_idx)
    pred["image_path"] = neontree.__getname__(img_idx)
    predictions = pd.concat([predictions, pred])

# format predictions
predictions = predictions[["image_path", "xmin", "ymin", "xmax", "ymax", "label", "score"]]
predictions.sort_values(by="image_path", inplace=True)
predictions.reset_index(drop=True, inplace=True)

# export predictions
predictions.to_csv(os.path.join(os.getcwd(), EXPORT_PRED_PATH, "benchmark_predictions.csv"), index=False)

