#!/bin/bash
# Needs plan_prep_tworoom.sh done. Two packs per env (<=2 learned arms per job):
#   sbatch -J v2-plan-tr-tworoom-A slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_tworoom.sh              # shiftwm + ar
#   sbatch -J v2-plan-tr-tworoom-B --export=ALL,ARMS="direct ar_tf" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_tworoom.sh
#   sbatch -J v2-plan-tr-tworoom-C --export=ALL,ARMS="shiftwm_ctr direct_ctr" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_tworoom.sh
PENV=tworoom SEED=${SEED:-0} ARMS="${ARMS:-shiftwm ar}" bash slurm_jebel/tasks/plan_train.sh
