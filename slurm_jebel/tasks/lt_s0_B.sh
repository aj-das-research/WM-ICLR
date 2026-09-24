# Language-Table lane B: features (if needed) + Direct (learned) with the param-free Persistence/Linear evals, seed 0,
# and the feature->RGB decoder for the qualitative figures (results/v2/analysis/decoder/language_table).
source slurm_jebel/tasks/lt_common.sh
mkdir -p results/v2/analysis/logs
python scripts/v2/train_decoder.py --dataset language_table > results/v2/analysis/logs/decoder_language_table.log 2>&1 &
pd=$!
rc=0
bash scripts/v2/launch_pack.sh configs/v2/language_table.json 0 'direct persistence linear' || rc=1
wait $pd || rc=1
tail -n 2 results/v2/analysis/logs/decoder_language_table.log
exit $rc
