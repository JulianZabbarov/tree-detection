
import pandas as pd

ANNOTATIONS_PATH = "/Users/julianzabbarov/Documents/HPI/Analysis_and_Visualization_of_Spatial_Data/tree-detection/data/evaluation"

annotations = pd.read_csv(ANNOTATIONS_PATH + "/benchmark_annotations.csv", header=None, names=["image_path", "xmin", "ymin", "xmax", "ymax", "label"], skiprows=1)
annotations = annotations.sort_values(by="image_path")
annotations.to_csv(ANNOTATIONS_PATH + "/benchmark_annotations.csv", index=False)

