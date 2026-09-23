set -eo pipefail
cd /home/Test/abhijit.das/projects/WM-ICLR; source .venv/bin/activate; export PYTHONPATH=src HF_HUB_OFFLINE=1 MUJOCO_GL=egl
bash slurm_jebel/tasks/planning_lewm_repro.sh
for e in tworoom pusht reacher; do bash slurm_jebel/tasks/plan_prep_$e.sh; done
