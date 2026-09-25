# tanh-bounded correction ablation (rerun alone; its packed run OOMed at eval on a shared 40 GB card).
scripts/v2/run_variant.sh configs/v2/droid_v1s.json 0 tanh shiftwm 'model.correction="tanh"'
