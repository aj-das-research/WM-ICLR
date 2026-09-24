# Table 17 (regions) on the final results/v2s DROID checkpoints (all seeds) + transport identity mass (blind review 2).
set -o pipefail
python scripts/v2/region_eval.py --dataset droid
python scripts/v2/identity_mass.py
