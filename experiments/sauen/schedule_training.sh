#!/bin/bash
#SBATCH --job-name="aavsd"
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-gpu=8
#SBATCH --account=renard
#SBATCH --partition=sorcery # cauldron # sorcery when training
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/hpi/fs00/home/julian.zabbarov/documents/tree-detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julian.zabbarov@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/training.txt
#SBATCH --error=/hpi/fs00/home/julian.zabbarov/documents/tree-detection/experiments/sauen/error.txt

echo "START"
source /hpi/fs00/home/julian.zabbarov/software/miniconda3/etc/profile.d/conda.sh
conda activate aavsd
# export WANDB_API_KEY="565b0acd482690fc7eecd3eaec6f503f55bfa969"
python --version
srun python src/prediction/run_tree_detection.py -c experiments/sauen/config.toml
# srun python src/training/finetune.py -c experiments/sauen/config.toml
echo "FINISHED"
exit 0