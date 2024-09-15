#!/bin/sh
EXPORT_FOLDER="experiments/sauen/results/epochs-13/test"

# evaluate predictions without fine-tuning
echo "Evaluating RetinaNet without fine-tuning on test set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/prediction_on_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1) on test set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1/test/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3) on test set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3/test/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate double-finetuning predictions
echo "Evaluating performance of RetinaNet with double-stage fine-tuning (3512a1_2x3 & 3512a1) on test set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/double_finetuning/test/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate semi-supervised fine-tuning predictions
echo "Evaluating performance of RetinaNet with semi-supervised fine-tuning (3512a1_2x3 & 3512a1) on test set ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning/test/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER