#!/bin/bash
# sbatch -J v2-plan-prep-reacher slurm_jebel/job.sbatch slurm_jebel/tasks/plan_prep_reacher.sh
PENV=reacher bash slurm_jebel/tasks/plan_prep.sh
