#!/bin/bash
# sbatch -J dwm-pusht-base slurm_jebel/job.sbatch slurm_jebel/tasks/dinowm_pusht_base.sh   (resumable; re-submit if it times out)
source /home/Test/abhijit.das/projects/WM-ICLR/slurm_jebel/tasks/dinowm_common.sh
train_arm pusht dinowm
