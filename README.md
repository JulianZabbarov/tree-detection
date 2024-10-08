# Tree Crown Detection with DeepForest

This is the repository for the course "Algorithms for Analysis and Visualization of Spatial Data" (SS 2024) at Hasso Plattner Institute. It provides the required functionalities to fine-tune and evaluate the available RetinaNet in the DeepForest Python package on hand-labelled data from a custom site.

## Getting started

Create a new Python 3.11 environment like so ...
```
conda create -n aavsd python=3.11
```
... and install the required dependencies like so:
```
pip install -r requirements.txt
```

If you name the environment "aavsd" you have to make less changes to the sbatch files for scheduling the experiments. Otherwise, make sure to make the required adjustments there. This also includes the root directory and the export folder of logging files.

> [!NOTE]  
> If you have troubles getting the code to run and get errors with numpy, double-check that numpy version 1.26.4 is installed. Numpy version of 2.x and above can cause errors with the deep-forest package.

## Run DeepForest on NeonTree benchmark

> [!IMPORTANT]  
> To run the benchmark, you first have to import the RGB images of the benchmark to [this](data/neontree/evaluation/RGB_with_annotations) folder. The folder should contain all the RGB images from the benchmark for which ground truth labels are available. Please contact josafat-mattias.burm@hpi.de for to access this data. For testing purposes, one benchmarking image is provided under the path above.

You can run predictions on NeonTreeDataset, a popular benchmark for tree crown detection. To start the benchmarking process call:
```
python src/prediction/run_tree_detection.py -c experiments/neontree/config.toml
```
The predictions are exported to the path defined in export.annotations_path in the config.toml.

You can evaluate the predictions for the benchmark like so:
```
python src/evaluation/evaluate_predictions.py -p "experiments/neontree/annotations/neontree.csv" -l "data/neontree/evaluation/benchmark_annotations.csv" -t 0.4 -e experiments/neontree/results
```

If you get "No predictions are made" in the command line and some metrics show NaN values, manually check the csv files with the predictions and labels. A typical error is that the image's data types under "image_path" to not match, e.g. are like "image1.png" and "image.tif". You can change the image type for the predictions using the "image_format" under export in the config. For an example, see [here](experiments/neontree/config.toml).

## Evaluate results from Sauen experiments

[Here](experiments/sauen/results) you find the results from different fine-tuning approaches defined in [this](experiments/sauen/configs) configurations folder. To evaluate and visualize the results, you can use [this](experiments/sauen/evaluation/evaluation.ipynb) Python notebook.

## How to run DeepForest on Sauen data

> [!IMPORTANT]  
>  To reproduce the findings from the Sauen experiments, add the tif files for the three areas to the three folders in [this](data/sauen/tiles) directory. The tif files contain geolocation information on the relevent forest areas and are required to filter out the bounding boxes for the areas of interest in step 4 of the pipeline. The name of the tif file should match the png in the respective subdirectory called "png" like [here](data/sauen/tiles/120m_1140px_3510b2/png). Please contact josafat-mattias.burm@hpi.de for to access this data.

> [!NOTE]  
> I made a consistent naming mistake: Each reference to dataset 3510b2 in this repository should be 3510b3.

In the following, I provide an example on how to the code of this repository for tree-crown detection using data from the Sauen forest. All experiments are configured [here](experiments/sauen/configs). You find the labels for training (bounding boxes) [here](experiments/sauen/labels). 

### Predictions without fine-tuning

First, you can run predictions on dataset 3510b3 (see [here](experiments/sauen/labels/edited_annotations_120m_1140px_3510b2)) from Sauen without any fine-tuning.
```
srun python src/prediction/run_tree_detection.py -c experiments/sauen/configs/prediction_on_3510b2.toml
```

You can also use [this](experiments/sauen/schedule/schedule_prediction_on_3510b2.sh) sbatch script to schedule the call on the DELab cluster. In the following, I link the sbatch scripts. They contain the simple Python calls.

The predictions are exported to experiments/sauen/predictions. The folder should be created when running the predictions.

### Prediction using single-stage fine-tuning

> [!NOTE]  
> Fine-tuning the DeepForest model is currently not available on Macs with M chips due to some missing implementations in torchvision. Please run the model fine-tuning on Linux machines.

Second, you can fine-tune the DeepForest model using hand-annotated labels:
```
sbatch experiments/sauen/schedule/schedule_prediction_on_3512a1.sh
```

Other fine-tuning experiments can be started using the sbatch files stored [here](experiments/sauen/schedule). The relevant files for single-stage fine-tuning contain "finetuning" in their names.

### Prediction with two-stage semi-supervised fine-tuning

Third, you can fine-tune the DeepForest model in a two-stage, semi-supervised fashion. For example, you can call:

```
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1_2x3_with_bboxs_from_polygons.sh
```

In this case, the DeepForest model is first fine-tuned on algorithmically derived bounding boxes using tree details from point clouds stored [here](experiments/sauen/labels/computed_annotations_from_polygons_120m_1240px_3512a1_2x3) and then fitted to manually annotated bounding boxes for 3512a1 stored [here](experiments/sauen/labels/edited_annotations_120m_1240px_3512a1)

##### Pre-processing tree locations extracted from point clouds

This repository contains a preprocessing pipeline that builds upon algorithmically derived tree locations and crown polygon information to derive bounding boxes. Here is a pipeline overview: tree locations -> cluster tree locations (file as json) -> extract bboxs (file as geojson) -> filter bboxs (file as geojson) -> filtered bounding boxes.

After importing the required tif files (see message above), you can generate bounding boxes from the tree details stored [here](data/sauen/treelocations). To start the pipeline, call one of the two following scripts:

```
sh src/preprocessing/run_tree_processing_with_polygons.sh
```
```
sh src/preprocessing/run_tree_processing_without_polygons.sh
```
The first script creates bounding boxes using the provided polyon information in the tree details. The second script uses bounding boxes by using the radius of the tree crown.

The filtered bounding boxes are exported to the following folder: "data/sauen/treelocations/for_annotation". The folder is created dynamically when executing the scripts above. To use the derived bounding boxes as labels, copy them to a labels directory for your experiments and specify the path in the experiments config.toml like [here](experiments/sauen/configs/semisupervised_finetuning_with_bboxs_from_polygons.toml)

You can visualize an exemplary bounding box that was derived from the provided polygon information for the first tree from [this](data/sauen/treelocations/treeDetails-20230720_Sauen_PLS_part1.json) file using [this](experiments/sauen/evaluation/plot_bounding_box_from_polygon.ipynb) Python notebook.

### Evaluation of predictions

To evaluate the predictions from all experiments that were performed as part of the AAVSD seminar, call the following files for train and test scores, respectively:
```
sbatch experiments/sauen/schedule/run_evaluation_on_test.sh
```
or
```
sbatch experiments/sauen/schedule/run_evaluation_on_test.sh
```
The results are exported to experiments/sauen/new_results.