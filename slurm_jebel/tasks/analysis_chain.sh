set -o pipefail
for t in analysis_transfer analysis_probes analysis_flow analysis_decode_eval analysis_figures; do
  echo "== $(date -Is) $t"; bash slurm_jebel/tasks/$t.sh || echo "FAILED $t"
done
