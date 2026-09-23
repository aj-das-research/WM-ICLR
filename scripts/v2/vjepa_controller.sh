#!/usr/bin/env bash
# Prioritise the V-JEPA 2-AC plug-in pipeline without idling GPUs: hold others until our job starts, then release.
cd "$(dirname "$0")/../.."
release_all() { for j in $(squeue -u $USER -t PD -h -o "%i %r" | awk '$2=="JobHeldUser"{print $1}'); do [ "$j" != "6758" ] && scontrol release $j; done; }
hold_others() { for j in $(squeue -u $USER -t PD -h -o %i); do case " $* " in *" $j "*) ;; *) scontrol hold $j 2>/dev/null;; esac; done; }
wait_running() { until squeue -h -j "$1" -o %T | grep -q RUNNING || ! squeue -h -j "$1" | grep -q .; do sleep 60; done; }
echo "$(date -Is) waiting for cache job 6895 to start"; wait_running 6895; release_all
echo "$(date -Is) cache running; others released"
until ! squeue -h -j 6895 | grep -q .; do sleep 120; done
if python3 -c "import json,sys; sys.exit(0 if json.load(open('data/v2/features/droid/vjepa2g/index.json')).get('complete') else 1)"; then
  B=$(sbatch -J vjepa_ft_B slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_ft_B.sh | awk '{print $4}')
  C=$(sbatch -J vjepa_ft_C slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_ft_C.sh | awk '{print $4}')
  hold_others $B $C; echo "$(date -Is) submitted B=$B C=$C; others held"
  wait_running $B; wait_running $C; release_all; echo "$(date -Is) B and C running; others released"
else
  echo "$(date -Is) cache incomplete; resubmitting"; sbatch -J vjepa_cache slurm_jebel/job.sbatch slurm_jebel/tasks/vjepa_cache.sh
fi
