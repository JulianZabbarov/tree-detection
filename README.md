# tree-detection
Repository for the course "Algorithms for Analysis and Visualization of Spatial Data" (SS 2024) at Hasso Plattner Institute

## Getting Started

### Installation

Install the required packages in an Python 3.1 environment:

```
pip install -r requirements.txt
```

### Run DeepForest on NeonTree benchmark

Run predictions on NeonTreeDataset:
```
python src/prediction/run_tree_detection.py -c experiments/neontree/config.toml
```

Evaluate predictions:
```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/neontree/annotations/neontree.csv" --path-to-labels "data/neontree/evaluation/benchmark_annotations.csv" --iou-threshold 0.4
```

## Run DeepForest on Sauen data

### Predictions without fine-tuning

Run predictions on data (tile with indicator 3510b2) from Sauen
```
python src/prediction/run_tree_detection.py -c experiments/sauen/config.toml
```

```
python src/evaluation/evaluate_predictions.py --path-to-predictions "experiments/sauen/predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --path-to-labels "experiments/sauen/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv" --iou-threshold 0.4
```

### Prediction with fine-tuning

```
python src/training/finetune.py -c experiments/sauen/config.toml
```