# tree-detection
Repository for the course "Algorithms for Analysis and Visualization of Spatial Data" (SS 2024) at Hasso Plattner Institute

## Getting Started

### Predict with DeepForest model

```
python src/evaluation/prediction_pipeline.py --path-to-labels "/data/neontree/evaluation/benchmark_annotations.csv" --path-to-images "/data/neontree/evaluation/RGB_with_annotations" --export-path "src/evaluation"
```


### Evaluation predictions

```
python src/evaluation/evaluate_predictions.py --path-to-predictions "src/evaluation/benchmark_predictions.csv" --path-to-labels "data/neontree/evaluation/benchmark_annotations.csv" --iou-threshold 0.4
```

### Split tiff into tiles

Alternative:
```
vips dzsave data/sauen/raw/20230720_Sauen_3512a1.tif data/sauen/tiles/unfiltered --depth one --tile-size 4000 --overlap 0 --suffix .tif
```
