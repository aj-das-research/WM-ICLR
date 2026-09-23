#!/bin/bash
# sbatch -J v2-plan-prep-tworoom slurm_jebel/job.sbatch slurm_jebel/tasks/plan_prep_tworoom.sh
PENV=tworoom bash slurm_jebel/tasks/plan_prep.sh
