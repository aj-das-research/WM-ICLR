#!/bin/bash
# sbatch -J dwm-pusht-plan slurm_jebel/job.sbatch slurm_jebel/tasks/dinowm_pusht_plan.sh   (needs both arms trained; skips finished cells)
source /home/Test/abhijit.das/projects/WM-ICLR/slurm_jebel/tasks/dinowm_common.sh
for seed in ${PLAN_SEEDS:-99 7 13}; do
  for planner in ${PLANNERS:-mpc_cem cem}; do
    for arm in dinowm dinowm_shiftwm; do plan_arm pusht $arm $planner $seed; done
  done
done
python scripts/external_dinowm_plugin/summarize.py pusht
