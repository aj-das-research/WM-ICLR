#!/bin/bash
# Needs plan_prep_reacher.sh done. Two packs per env (<=2 learned arms per job):
#   sbatch -J v2-plan-tr-reacher-A slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_reacher.sh              # shiftwm + ar
#   sbatch -J v2-plan-tr-reacher-B --export=ALL,ARMS="direct ar_tf" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_reacher.sh
#   sbatch -J v2-plan-tr-reacher-C --export=ALL,ARMS="shiftwm_ctr direct_ctr" slurm_jebel/job.sbatch slurm_jebel/tasks/plan_train_reacher.sh
PENV=reacher SEED=${SEED:-0} ARMS="${ARMS:-shiftwm ar}" bash slurm_jebel/tasks/plan_train.sh
