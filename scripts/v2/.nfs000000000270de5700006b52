#!/usr/bin/env bash
# Poll every 10 min; refresh the paper when the set of finished runs changes.
cd "$(dirname "$0")/../.."
last=""
while true; do
  cur=$( (find results/v2 results/v2s -name 'summary.json' -o -name 'test_summary.json' -o -name 'openloop.json' -o -name 'final.json' -o -name 'eval_test.npz' 2>/dev/null; ls results/v2/analysis/*/*.json 2>/dev/null) | xargs -r stat -c '%n %Y' 2>/dev/null | sort | md5sum)
  if [ "$cur" != "$last" ]; then
    echo "== $(date -Is) change detected"; J=$(squeue -u $USER -h -t R -o %i | head -1)
    if [ -n "$J" ]; then srun --jobid=$J --overlap bash scripts/v2/refresh_paper.sh; else scripts/v2/refresh_paper.sh; fi
    [ -f paper/submission_main.pdf ] && last=$cur
  fi
  sleep 600
done
