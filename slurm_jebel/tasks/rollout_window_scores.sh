# Score every stride-2 test window (k=10 MSE of ShiftWM/Direct/AR + true change) for the rollout-figure selection
# rule (make_analysis_figures.window_scores), then draw rollouts.pdf. Writes results/v2/analysis/qualitative/.
export PYTHONPATH=src:paper/submission_folder/figures/src
python -u -c "
import time, make_analysis_figures as m
c = m.contexts('cuda'); t = time.time()
for ds in ('droid', 'openh_hamlyn'):
    w = m.window_scores(c[ds]); print(ds, len(w), round(time.time() - t), flush=True)
"
