# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import numpy as np
from tifffile import imread
from matplotlib import pyplot as plt
from PIL import Image
from deepforest import main
import pandas as pd
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-i", "--image_folder", dest="image_folder", type=str, required=True)
parser.add_argument("-o", "--output_folder", dest="output_folder", type=str, required=True)
parser.add_argument("-p", "--png_folder", dest="png_folder", type=str, required=True)
parser.add_argument("-s", "--size", dest="image_size", type=int, default=4000)
args = parser.parse_args()

# img = np.array(
#     imread(
#         "/Users/julianzabbarov/Documents/HPI/Analysis_and_Visualization_of_Spatial_Data/tree-detection/data/sauen/tiles/unfiltered/1/20230809_Sauen_3515b1_4_4.tif"
#     )
# )[:, :, :3].astype(np.uint8)

# img = Image.fromarray(img)
# img = img.resize((400, 400))
# img = np.array(img)

# model = main.deepforest()
# model.use_release()

# pred = model.predict_image(image=img.astype(np.float32), return_plot=False)
# print(pred)


def export_prediction_as_xml(
    pred: pd.DataFrame,
    image_folder: str,
    export_folder: str,
    filename: str,
    path="/my/path/GeneratedData_Train/",
    width=224,
    height=224,
    depth=3,
) -> None:
    annotation = ET.Element("annotation")

    folder_element = ET.SubElement(annotation, "folder")
    folder_element.text = image_folder

    filename_element = ET.SubElement(annotation, "filename")
    filename_element.text = filename

    path_element = ET.SubElement(annotation, "path")
    path_element.text = f"{path}{filename}"

    # source = ET.SubElement(annotation, "source")

    size = ET.SubElement(annotation, "size")
    width_element = ET.SubElement(size, "width")
    width_element.text = str(width)

    height_element = ET.SubElement(size, "height")
    height_element.text = str(height)

    depth_element = ET.SubElement(size, "depth")
    depth_element.text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for _, row in pred.iterrows():
        object_element = ET.SubElement(annotation, "object")

        name = ET.SubElement(object_element, "name")
        name.text = str(row["label"])

        bndbox = ET.SubElement(object_element, "bndbox")

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(row["xmin"] * args.image_size / 400)

        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(row["xmax"] * args.image_size / 400)

        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(row["ymin"] * args.image_size / 400)

        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(row["ymax"] * args.image_size / 400)

    tree = ET.ElementTree(annotation)
    xml_file = filename.replace(".tif", ".xml")
    tree.write(os.path.join(os.getcwd(), export_folder, xml_file), encoding="unicode", xml_declaration=True)


for tile in os.listdir(os.path.join(os.getcwd(), args.image_folder)):
    img = np.array(imread(f"{args.image_folder}/{tile}"))[:, :, :3].astype(np.uint8)

    img = Image.fromarray(img)
    img = img.resize((args.image_size, args.image_size))
    img.save(f"{os.getcwd()}/{args.png_folder}/{tile.replace('.tif', '.png')}", type="PNG")
    img = img.resize((400, 400))
    img = np.array(img)

    model = main.deepforest()
    model.use_release()

    pred = model.predict_image(image=img.astype(np.float32), return_plot=False)
    pred = pred.sort_values(by=["xmin", "ymin", "xmax", "ymax"]).reset_index(drop=True)
    pred["label"] = pred["label"] + pred.index.astype(str)
    print(pred)

    export_prediction_as_xml(
        pred=pred,
        image_folder=args.image_folder,
        export_folder=args.output_folder,
        filename=tile,
        path=args.image_folder,
        width=args.image_size,
        height=args.image_size,
        depth=3,
    )
