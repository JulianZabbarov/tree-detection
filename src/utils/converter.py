import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from src.utils.imports import load_config
from deepforest import utilities

def get_annotations(path: str):
    return utilities.xml_to_annotations(path)

def convert_xml_to_csv(folder_path: str):
    for file in os.listdir(folder_path):
        if file.endswith(".xml"):
            annotations = get_annotations(os.path.join(folder_path, file))
            annotations["label"] = "Tree"
            annotations.to_csv(
                os.path.join(folder_path, str(file).replace(".xml", ".csv")),
                index=False,
            )


if __name__ == "__main__":
    folder = "experiments/sauen/labels/edited_annotations_120m_1240px_3510a1"
    convert_xml_to_csv(folder)