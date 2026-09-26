# Rerun the DROID geometry (oracle move / shares) and anatomy analyses with the final v2s AR checkpoint (Table 1 recipe).
set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR
python scripts/v2/geometry.py
python scripts/v2/anatomy.py
