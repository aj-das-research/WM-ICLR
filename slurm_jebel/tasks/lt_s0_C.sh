# Language-Table lane C: AR-TF (DINO-WM-style) seed 0, same recipe as lanes A/B (fills the '--' cell of Table 1).
source slurm_jebel/tasks/lt_common.sh
bash scripts/v2/launch_pack.sh configs/v2/language_table.json 0 'ar_tf'
