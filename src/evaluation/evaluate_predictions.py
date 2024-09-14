import warnings
import os

import pandas as pd
from deepforest import main
from deepforest.evaluate import evaluate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-p",
    "--path-to-predictions",
    dest="pred",
    action="store",
    help="relative path to csv with predictions",
)
parser.add_argument(
    "-l",
    "--path-to-labels",
    dest="ann",
    action="store",
    help="relative path to csv with annotations",
)
parser.add_argument(
    "-t",
    "--iou-threshold",
    dest="iou_threshold",
    action="store",
    help="threshold for intersection over union",
)
# add path to folder for export
parser.add_argument(
    "-e",
    "--export-folder-path",
    dest="export_folder_path",
    action="store",
    help="relative path to folder for export",
)

args = parser.parse_args()

pred = pd.read_csv(os.path.join(os.getcwd(), args.pred))
ann = pd.read_csv(os.path.join(os.getcwd(), args.ann))
absolute_export_folder_path = os.path.join(os.getcwd(), args.export_folder_path)

# ignore deprecated warnings from pandas raised by deepforest.IoU (line 113: iou_df = pd.concat(iou_df))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = evaluate(
        predictions=pred,
        ground_df=ann,
        root_dir="experiments/sauen/predictions_120m_1140px_3510b2",
        iou_threshold=float(args.iou_threshold),
    )

# rename box precision and box recall to precision and recall in dict
results["precision"] = results.pop("box_precision")
results["recall"] = results.pop("box_recall")
results["f1"] = (
    2
    * (results["precision"] * results["recall"])
    / (results["precision"] + results["recall"])
)

# print results to console
print(
    "Precision:\t{precision}\nRecall:\t\t{recall}\nF1:\t\t{f1}".format(
        precision=results["precision"],
        recall=results["recall"],
        f1=results["f1"],
    )
)

# export results
metrics = []
metrics.append({"metric": "precision", "score": results["precision"]})
metrics.append({"metric": "recall", "score": results["recall"]})
metrics.append({"metric": "f1", "score": results["f1"]})
df = pd.DataFrame.from_dict(metrics)

# save dict as csv
file_name = args.pred.split("/")[-2]
df.to_csv(
    os.path.join(absolute_export_folder_path, file_name + ".csv"), index=False
)
