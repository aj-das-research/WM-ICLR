#!/usr/bin/env bash
# Poll every 10 min; refresh the paper when the set of finished runs changes.
cd "$(dirname "$0")/../.."
last=""
while true; do
  cur=$(ls results/v2/*/*/*/s*/summary.json results/v2/*/*/ablations/*/s*/summary.json results/v2/planning/*/*/*.json results/v2/external/*/*.json 2>/dev/null | md5sum)
  if [ "$cur" != "$last" ]; then
    echo "== $(date -Is) change detected"; scripts/v2/refresh_paper.sh && last=$cur
  fi
  sleep 600
done
