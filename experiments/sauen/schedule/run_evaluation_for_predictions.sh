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
#SBATCH --output=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/evaluation_of_predictions.txt
#SBATCH --error=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/evaluation_of_predictions.txt

echo "START"
source /hpi/fs00/home/julian.zabbarov/software/miniconda3/etc/profile.d/conda.sh
conda activate aavsd

#!/bin/sh
EXPORT_FOLDER="experiments/sauen/results/baseline"

# evaluate predictions without fine-tuning
for SEED in 0 1 2 3 4
do
    echo "Evaluating RetinaNet without fine-tuning on test set ..."
    srun python src/evaluation/evaluate_predictions.py -p experiments/sauen/predictions/prediction_on_3510b2/20230809_Sauen_3510b2_tile-seed-$SEED.csv -l experiments/sauen/labels/edited_annotations_120m_1140px_3510b2/20230809_Sauen_3510b2_tile.csv -t 0.4 -e $EXPORT_FOLDER/test-seed-$SEED
done