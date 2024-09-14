#!/bin/sh
EXPORT_FOLDER="experiments/sauen/results"

# evaluate predictions without fine-tuning
echo "Evaluating RetinaNet without fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/prediction_on_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating RetinaNet with single-stage fine-tuning on 3512a1 ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
echo "Evaluating RetinaNet with single-stage fine-tuning on 3512a1_2x3 ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate double-finetuning predictions
echo "Evaluating RetinaNet with double-stage fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/double_finetuning/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate double-finetuning predictions
echo "Evaluating RetinaNet with semi-supervised fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER