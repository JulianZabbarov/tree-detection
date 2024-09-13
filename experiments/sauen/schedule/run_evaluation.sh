# evaluate predictions without fine-tuning
echo "Evaluating without fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4

# evaluate single finetuning predictions
echo "Evaluating single fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4

# evaluate double-finetuning predictions
echo "Evaluating double fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/double_finetuning_predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4

# evaluate double-finetuning predictions
echo "Evaluating semi-supervised fine-tuning ..."
srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning_predictions_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4