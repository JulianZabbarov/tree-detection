# Tree Crown Detection with DeepForest

This is the repository for the course "Algorithms for Analysis and Visualization of Spatial Data" (SS 2024) at Hasso Plattner Institute. It provides the required functionalities to fine-tune and evaluate the available RetinaNet in the DeepForest Python package on hand-labelled data from a custom site.

## Getting Started

### Installation

Install the required packages in an Python 3.1 environment:

```
pip install -r requirements.txt
```

### Run DeepForest on NeonTree benchmark

To assess the capabilities of the shipped RetinaNet in DeepForest, you can run predictions on NeonTreeDataset, a popular benchmark for tree crown detection:
```
python src/prediction/run_tree_detection.py -c experiments/neontree/config.toml
```

Evaluate predictions:
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/neontree/annotations/neontree.csv" --path-to-labels "data/neontree/evaluation/benchmark_annotations.csv" --iou-threshold 0.4
```

## How to Run DeepForest on Sauen data

In the following, I provide an example on how to the code of this repository for tree-crown detections using hand-labelled data from the Sauen forest. You find the hand-annotated labels [here](experiments/sauen).

### Predictions without fine-tuning

Run predictions on data ([link](experiments/sauen/edited_annotations_120m_1140px_3510b2)) from Sauen:
```
python src/prediction/run_tree_detection.py -c experiments/sauen/config.toml
```

Evaluate predictions
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/sauen/predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --path-to-labels "experiments/sauen/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --iou-threshold 0.4
```

### Prediction with fine-tuning

Finetune and predict with RetinaNet from DeepForest on defined images in finetuning_config.toml:
```
python src/training/finetune.py -c experiments/sauen/finetuning_config.toml
```

Evaluate predictions after finetuning:
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/sauen/finetuning_predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --path-to-labels "experiments/sauen/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --iou-threshold 0.4
```

### Prediction with two-step fine-tuning

##### Pre-processing tree locations extracted from point clouds

Pipeline overview: tree locations -> cluster tree locations (files in json) -> extract bboxs (files in geojson) -> filter bboxs (files in geojson).

This repository provides the functionality to create bounding boxes from json data that contains tree locations extracted from point clouds. As the tree locations of individual trees can appear multiple times in the data, the following script allows you to cluster tree locations.
```
python src/preprocessing/cluster_trees.py
```
To retain polygon information, use "python src/preprocessing/cluster_trees_with_polygons.py".


You can then use the following script to create bounding boxes for the clustered trees.
```
src/preprocessing/convert_json_to_geojson_bbox.py
```

To improve the bounding boxes for fine-tuning tree-detection models that use UAV data, the following script can be run to extract bounding boxes of the highests trees and a certain IoU threshold.
```
python src/preprocessing/filter_bboxs.py -f data/sauen/treelocations/treeDetails-20230720_Sauen_PLS_clustered-bboxs.geojson
```