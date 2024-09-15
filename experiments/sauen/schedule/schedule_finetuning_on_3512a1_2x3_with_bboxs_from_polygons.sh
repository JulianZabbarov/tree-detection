#!/bin/bash
#SBATCH --job-name="aavsd"
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-gpu=8
#SBATCH --account=renard
#SBATCH --partition=sorcery # alternative: cauldron # sorcery when training
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/hpi/fs00/home/julian.zabbarov/documents/tree-detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julian.zabbarov@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/finetuning_on_3512a1_2x3_with_bboxs_from_polygons.txt
#SBATCH --error=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/slurm/finetuning_on_3512a1_2x3_with_bboxs_from_polygons_error.txt

echo "START"
source /hpi/fs00/home/julian.zabbarov/software/miniconda3/etc/profile.d/conda.sh
conda activate aavsd
srun python src/training/finetuning.py -c experiments/sauen/configs/finetuning_on_3512a1_2x3_with_bboxs_from_polygons.toml -s 42
echo "FINISHED"
exit 0