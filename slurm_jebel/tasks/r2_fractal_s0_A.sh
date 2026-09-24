# Fractal train features (~70 GB fp16) are held on the GPU per process: run arms sequentially, one per GPU at a time.
bash scripts/v2/launch_pack.sh configs/v2/fractal_r2.json 0 'persistence'
bash scripts/v2/launch_pack.sh configs/v2/fractal_r2.json 0 'shiftwm'
bash scripts/v2/launch_pack.sh configs/v2/fractal_r2.json 0 'ar'
