sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1_2x3_with_handlabeled_bboxs.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1_2x3_with_computed_bboxs.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_finetuning_on_3512a1_2x3_with_bboxs_from_polygons.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_double_finetuning.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_semisupervised_finetuning.sh
sleep 3
sbatch experiments/sauen/schedule/schedule_semisupervised_finetuning_with_bboxs_from_polygons.sh