#!/bin/bash
# sbatch -J dwm-wall-plugin slurm_jebel/job.sbatch slurm_jebel/tasks/dinowm_wall_plugin.sh   (resumable; re-submit if it times out)
source /home/Test/abhijit.das/projects/WM-ICLR/slurm_jebel/tasks/dinowm_common.sh
train_arm wall dinowm_shiftwm
