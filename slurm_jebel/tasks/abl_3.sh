scripts/v2/run_variant.sh configs/v2/droid_v1s.json 0 tanh shiftwm 'model.correction="tanh"' & scripts/v2/run_variant.sh configs/v2/droid_v1s.json 0 nocorr shiftwm 'model.correction="none"' & wait
