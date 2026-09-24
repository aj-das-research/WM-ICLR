# Language-Table lane A: features (if needed) + ShiftWM and AR, seed 0 (droid_v1s recipe, 8k steps; test eval in train.py).
source slurm_jebel/tasks/lt_common.sh
bash scripts/v2/launch_pack.sh configs/v2/language_table.json 0 'shiftwm ar'
