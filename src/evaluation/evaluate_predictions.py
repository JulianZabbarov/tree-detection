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

args = parser.parse_args()

pred = pd.read_csv(os.path.join(os.getcwd(), args.pred))
ann = pd.read_csv(os.path.join(os.getcwd(), args.ann))

print("Evaluating predictions ...")
# ignore deprecated warnings from pandas raised by deepforest.IoU (line 113: iou_df = pd.concat(iou_df))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = evaluate(
        predictions=pred,
        ground_df=ann,
        root_dir="experiments/sauen/predictions_120m_1140px_3510b2",
        iou_threshold=float(args.iou_threshold),
    )

print(
    "Precision:\t{precision}\nRecall:\t\t{recall}".format(
        precision=results["box_precision"], recall=results["box_recall"]
    )
)
