#!/bin/bash
# Move every PENDING job of this user from h200 to the A100 partitions (keeps job IDs and dependencies).
# Slurm starts each job on whichever listed partition frees a GPU first.
C=external-5709d29b6e754513af0d4104274c53b5
for j in $(squeue -u "$USER" -h -t PD -p h200 -o %i); do
  scontrol update job "$j" Partition=a100_80,a100 Comment="$C" && echo "moved $j"
done
sleep 10
squeue -u "$USER" -o "%.8i %.10P %.12j %.2t %.10M %R"
