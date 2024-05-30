import warnings
import os

import pandas as pd
from deepforest import main
from deepforest.evaluate import evaluate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-predpath", "--pred", action="store", help="relative ")
parser.add_argument(
    "-annpath", "--ann", action="store", help="don't print status messages to stdout"
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
        root_dir="data/neontree/evaluation/RGB_with_annotations",
        iou_threshold=0.4,
    )

print(
    "Precision:\t{precision}\nRecall:\t\t{recall}".format(
        precision=results["box_precision"], recall=results["box_recall"]
    )
)
