#!/bin/sh
EXPORT_FOLDER="experiments/sauen/results/epochs-13/train"

# evaluate single finetuning predictions
echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1/train/20230720_Sauen_3512a1_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1/20230720_Sauen_3512a1_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3/train/20230720_Sauen_3512a1_2x3-tile-rendered.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1_2x3/20230720_Sauen_3512a1_2x3-tile-rendered.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3 with bboxs from polygons) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3_with_bboxs_from_polygons/train/20230720_Sauen_3512a1_2x3-tile-rendered.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1_2x3/20230720_Sauen_3512a1_2x3-tile-rendered.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate double-finetuning predictions
echo "Evaluating performance of RetinaNet with double-stage fine-tuning (3512a1_2x3 & 3512a1) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/double_finetuning/train/20230720_Sauen_3512a1_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1/20230720_Sauen_3512a1_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate semi-supervised finetuning predictions
echo "Evaluating performance of RetinaNet with semi-supervised fine-tuning (3512a1_2x3 & 3512a1) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning/train/20230720_Sauen_3512a1_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1/20230720_Sauen_3512a1_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate semi-supervised finetuning predictions
echo "Evaluating performance of RetinaNet with semi-supervised fine-tuning (3512a1_2x3 with bboxs from polygons & 3512a1) on train set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning_with_bboxs_from_polygons/train/20230720_Sauen_3512a1_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1240px_3512a1/20230720_Sauen_3512a1_tile.csv -t 0.4 -e $EXPORT_FOLDER