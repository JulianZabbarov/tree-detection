import os
import xml.etree.ElementTree as ET
from experiments.config_definition import ExportConfig

import pandas as pd


def export_predictions_as_csv(
    pred_df: pd.DataFrame, export_config: ExportConfig, image_name: str
) -> None:
    # export predictions
    if export_config.sort_values:
        pred_df.sort_values(
            by=export_config.sort_values,
            inplace=True,
        )  # .reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)
    if export_config.index_as_label_suffix:
        pred_df["label"] = pred_df["label"] + pred_df.index.astype(str)
    if export_config.column_order:
        pred_df = pred_df[export_config.column_order]
    pred_df.to_csv(
        os.path.join(os.getcwd(), export_config.annotations_path, image_name + ".csv"),
        index=False,
    )


def export_predictions_as_xml(
    pred: pd.DataFrame,
    image_name: str,
    image_folder: str,
    export_config: ExportConfig,
    width: int = 4000,
    height: int = 4000,
    depth: int = 3,
) -> None:
    annotation = ET.Element("annotation")

    folder_element = ET.SubElement(annotation, "folder")
    folder_element.text = image_folder

    filename_element = ET.SubElement(annotation, "filename")
    filename_element.text = image_name

    path_element = ET.SubElement(annotation, "path")
    path_element.text = f"{image_folder}{image_name}"

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

    pred = pred.sort_values(by=export_config.sort_values).reset_index()

    for idx, row in pred.iterrows():
        object_element = ET.SubElement(annotation, "object")

        name = ET.SubElement(object_element, "name")
        name.text = str(row["label"] + str(idx))

        bndbox = ET.SubElement(object_element, "bndbox")

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(row["xmin"] * width / 400)

        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(row["xmax"] * width / 400)

        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(row["ymin"] * height / 400)

        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(row["ymax"] * height / 400)

    tree = ET.ElementTree(annotation)
    xml_file = image_name.replace(".tif", ".xml")
    tree.write(
        os.path.join(os.getcwd(), export_config.annotations_path, xml_file),
        encoding="unicode",
        xml_declaration=True,
    )
