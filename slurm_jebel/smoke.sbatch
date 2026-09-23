#!/bin/bash
#SBATCH --job-name=shiftwm-smoke
#SBATCH --partition=h200
#SBATCH --comment="h200test2609"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/Test/abhijit.das/projects/WM-ICLR
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source .venv/bin/activate; export PYTHONPATH=src HF_HUB_OFFLINE=1
python - <<'PY'
import torch, time
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); torch.cuda.synchronize(); t=time.time()
for _ in range(50): y=x@x
torch.cuda.synchronize(); print('TFLOPS', 50*2*8192**3/(time.time()-t)/1e12)
PY
curl -sI --max-time 8 https://huggingface.co | head -1 || echo "NO_EGRESS"
df -h /tmp | tail -1
