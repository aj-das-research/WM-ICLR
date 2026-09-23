set -o pipefail
python scripts/v2/region_eval.py; python scripts/v2/sharpness.py
for t in analysis_transfer analysis_probes analysis_flow analysis_decode_eval analysis_figures; do
  echo "== $(date -Is) $t"; bash slurm_jebel/tasks/$t.sh || echo "FAILED $t"
done
