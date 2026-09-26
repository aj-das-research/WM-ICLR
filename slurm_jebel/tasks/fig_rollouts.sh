# Re-render the appendix rollout figure (largest-margin windows only).
set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR/paper/submission_folder/figures/src
export PYTHONPATH=/home/Test/abhijit.das/projects/WM-ICLR/src:/home/Test/abhijit.das/projects/WM-ICLR/scripts/v2
python make_analysis_figures.py --only rollouts --device cuda
