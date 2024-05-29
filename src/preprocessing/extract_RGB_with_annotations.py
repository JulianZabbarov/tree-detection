import os
import shutil

import pandas as pd

IMG_PATH = os.path.join(os.getcwd(), "data/evaluation/RGB")
ANNOTATIONS_PATH = os.path.join(os.getcwd(), "data/evaluation/benchmark_annotations.csv")
EXPORT_PATH = os.path.join(os.getcwd(), "data/evaluation/RGB_with_annotations")

# create folder at export path
if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# load annotations
annotations = pd.read_csv(ANNOTATIONS_PATH)['image_path'].to_list()
annotations = list(set(annotations))
print(f"Loaded annotations for {len(annotations)} images.")

# copy images with annotations to export path
print(f"{len(os.listdir(IMG_PATH))} images available.")
copied = 0
imgs = 0
for img in os.listdir(IMG_PATH):
    imgs += 1
    if img in annotations:
        # save image as png in export path
        shutil.copy(os.path.join(IMG_PATH, img), EXPORT_PATH)
        copied += 1

print(f"Copied {copied} images with annotations to defined export path.")