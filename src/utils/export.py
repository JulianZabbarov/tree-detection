import os
import xml.etree.ElementTree as ET
from src.configs.config_definition import ExportConfig

import pandas as pd


def create_folder_if_not_exists(folder: str):
    if not os.path.exists(folder):
        print("Creating folder:", folder, "...")
        os.makedirs(folder)


def export_predictions_as_csv(
    pred_df: pd.DataFrame, export_config: ExportConfig, image_name: str
) -> None:
    # replace tif with png in image path to match annotation format
    if export_config.image_format == "PNG":
        pred_df["image_path"] = pred_df["image_path"].str.replace(".tif", ".png")

    # sort by label
    if export_config.sort_values:
        pred_df.sort_values(
            by=export_config.sort_values,
            inplace=True,
        )  # .reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)

    # add index as label suffix
    if export_config.index_as_label_suffix:
        pred_df["label"] = pred_df["label"] + pred_df.index.astype(str)

    # reorder columns
    if export_config.column_order:
        pred_df = pred_df[export_config.column_order]

    # create export folder if not exists
    export_folder = os.path.join(os.getcwd(), export_config.annotations_path)
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
        
    # export to csv
    export_path = os.path.join(os.getcwd(), export_config.annotations_path, image_name.split(".")[0] + ".csv")
    pred_df.to_csv(
        export_path,
        index=False,
    )
    print(f"Exported {image_name} to {export_path}.")


def export_predictions_as_xml(
    pred: pd.DataFrame,
    image_name: str,
    image_folder: str,
    export_config: ExportConfig,
    image_size: int,
    depth: int = 3,
    scale_annotations: bool = False,
) -> None:
    annotation = ET.Element("annotation")

    folder_element = ET.SubElement(annotation, "folder")
    folder_element.text = image_folder

    filename_element = ET.SubElement(annotation, "filename")
    filename_element.text = image_name

    path_element = ET.SubElement(annotation, "path")
    path_element.text = f"{image_folder}{image_name}"

    size = ET.SubElement(annotation, "size")
    width_element = ET.SubElement(size, "width")
    width_element.text = str(image_size)

    height_element = ET.SubElement(size, "height")
    height_element.text = str(image_size)

    depth_element = ET.SubElement(size, "depth")
    depth_element.text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    pred = pred.sort_values(by=export_config.sort_values).reset_index()

    if scale_annotations:
        pred["xmin"] = pred["xmin"] * image_size / 400
        pred["xmax"] = pred["xmax"] * image_size / 400
        pred["ymin"] = pred["ymin"] * image_size / 400
        pred["ymax"] = pred["ymax"] * image_size / 400

    for idx, row in pred.iterrows():
        object_element = ET.SubElement(annotation, "object")

        name = ET.SubElement(object_element, "name")
        name.text = str(row["label"] + str(idx))

        bndbox = ET.SubElement(object_element, "bndbox")

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(row["xmin"])

        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(row["xmax"])

        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(row["ymin"])

        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(row["ymax"])

    tree = ET.ElementTree(annotation)
    xml_file = image_name.replace(".tif", ".xml")
    if not os.path.exists(os.path.join(os.getcwd(), export_config.annotations_path)):
        os.mkdir(os.path.join(os.getcwd(), export_config.annotations_path))
    tree.write(
        os.path.join(os.getcwd(), export_config.annotations_path, xml_file),
        encoding="unicode",
        xml_declaration=True,
    )
