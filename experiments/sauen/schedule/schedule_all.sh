sbatch experiments/sauen/schedule/schedule_prediction_on_3510b2.sh
sleep 1
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1.sh
sleep 1
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1_2x3.sh
sleep 1
sbatch experiments/sauen/schedule/schedule_double_finetuning.sh
sleep 1
sbatch experiments/sauen/schedule/schedule_semisupervised_finetuning.sh
