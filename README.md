# Tree Crown Detection with DeepForest

This is the repository for the course "Algorithms for Analysis and Visualization of Spatial Data" (SS 2024) at Hasso Plattner Institute. It provides the required functionalities to fine-tune and evaluate the available RetinaNet in the DeepForest Python package on hand-labelled data from a custom site.

## Getting Started

### Installation

Install the required packages in an Python 3.11 environment:

```
pip install -r requirements.txt
```

### Run DeepForest on NeonTree benchmark

To assess the capabilities of the shipped RetinaNet in DeepForest, you can run predictions on NeonTreeDataset, a popular benchmark for tree crown detection. To do so, import the RGB images into [this](data/neontree/evaluation/RGB_with_annotations) path. Please contact josafat.burmeister@hpi.de for data access. After that you can call:
```
python src/prediction/run_tree_detection.py -c experiments/neontree/config.toml
```

Evaluate predictions:
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/neontree/annotations/neontree.csv" --path-to-labels "data/neontree/evaluation/benchmark_annotations.csv" --iou-threshold 0.4
```

## How to Run DeepForest on Sauen data

In the following, I provide an example on how to the code of this repository for tree-crown detections using hand-labelled data from the Sauen forest. You find the hand-annotated labels [here](experiments/sauen/labels).

### Predictions without fine-tuning

Run predictions on dataset 3510b3 ([link](experiments/sauen/edited_annotations_120m_1140px_3510b2)) from Sauen. Note that I made a naming consistent namign error, each reference to 3510b2 in this repository should be 3510b3.
```
python src/prediction/run_tree_detection.py -c experiments/sauen/configs/prediction_on_3510b2.toml
```

Evaluate predictions
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/sauen/predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --path-to-labels "experiments/sauen/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --iou-threshold 0.4
```

### Prediction using single-stage fine-tuning

Finetune and predict with RetinaNet from DeepForest on defined images in finetuning_config.toml:
```
python src/training/finetune.py -c experiments/sauen/finetuning_config.toml
```

Evaluate predictions after finetuning:
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/sauen/finetuning_predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --path-to-labels "experiments/sauen/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --iou-threshold 0.4
```

### Prediction with two-stage semi-supervised fine-tuning



##### Pre-processing tree locations extracted from point clouds

This repository contains a preprocessing pipeline that builds upon algorithmically derived tree locations and crown polygon information to derive bounding boxes. Here is a pipeline overview: tree locations -> cluster tree locations (file as json) -> extract bboxs (file as geojson) -> filter bboxs (file as geojson) -> filtered bounding boxes.

To run the preprocessing pipeline you have to add tif files that include geolocation information to the repository. Tif files are required to filter out the bounding boxes for the areas of interest. To reproduce the findings from the Sauen experiments add the three tif files for the three areas to the three folders in [this](data/sauen/tiles) directory. The name of the tif file should match the png in the respective subdirectory called "png".

If done so, you can derive bounding boxes from the tree details stored [here](data/sauen/treelocations). To do so you can call one of the two following scripts:

```
sh src/preprocessing/run_tree_processing_with_polygons.sh
sh src/preprocessing/run_tree_processing_without_polygons.sh
```
The first script creates bounding boxes using the provided polyon information in the tree details. The second script uses bounding boxes by using the radius of the tree crown.

The filtered bounding boxes are exported to the following folder: "data/sauen/treelocations/for_annotation". The folder is created dynamically when executing the scripts above. To use the derived bounding boxes as labels, copy them to a labels directory for your experiments and specify the path in the experiments config.toml like [here](experiments/sauen/configs/semisupervised_finetuning_with_bboxs_from_polygons.toml)