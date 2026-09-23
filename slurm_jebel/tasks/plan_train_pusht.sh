#!/bin/bash
# Needs plan_prep_pusht.sh done. Two packs per env (<=2 learned arms per job):
#   sbatch -J v2-plan-tr-pusht-A slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_pusht.sh              # shiftwm + ar
#   sbatch -J v2-plan-tr-pusht-B --export=ALL,ARMS="direct ar_tf" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_pusht.sh
#   sbatch -J v2-plan-tr-pusht-C --export=ALL,ARMS="shiftwm_ctr direct_ctr" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_pusht.sh
PENV=pusht SEED=${SEED:-0} ARMS="${ARMS:-shiftwm ar}" bash slurm_jebel/tasks/plan_train.sh
