#!/bin/bash
#SBATCH --job-name="aavsd"
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-gpu=8
#SBATCH --account=renard
#SBATCH --partition=sorcery # alternative: cauldron # sorcery when training
#SBATCH --time=0-00:15:00
#SBATCH --chdir=/hpi/fs00/home/julian.zabbarov/documents/tree-detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julian.zabbarov@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/evaluation_on_test.txt
#SBATCH --error=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/evaluation_on_test.txt

echo "START"
source /hpi/fs00/home/julian.zabbarov/software/miniconda3/etc/profile.d/conda.sh
conda activate aavsd

#!/bin/sh
EXPORT_FOLDER="experiments/sauen/results/epochs-9/test"

# # evaluate predictions without fine-tuning
# echo "Evaluating RetinaNet without fine-tuning on test set ..."
# srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/prediction_on_3510b2/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER

# evaluate single finetuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate single finetuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3 with handlabeled bboxs) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3_with_handlabeled_bboxs/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate single finetuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3 with computed bboxs) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3_with_computed_bboxs/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate single finetuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with single-stage fine-tuning (3512a1_2x3 with bboxs from polygons) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/finetuning_on_3512a1_2x3_with_bboxs_from_polygons/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate double-finetuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with double-stage fine-tuning (3512a1_2x3 & 3512a1) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/double_finetuning/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate semi-supervised fine-tuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with semi-supervised fine-tuning (3512a1_2x3 & 3512a1) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done

# evaluate semi-supervised fine-tuning predictions
for SEED in 0 1 2 3 4
do
    echo "Evaluating performance of RetinaNet with semi-supervised fine-tuning (3512a1_2x3 with bboxs from polygons & 3512a1) on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/semisupervised_finetuning_with_bboxs_from_polygons/test-seed-$SEED/20230809_Sauen_3510b2_tile.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER-seed-$SEED
done