#!/bin/bash
# sbatch -J dwm-smoke slurm_jebel/job.sbatch slurm_jebel/tasks/dinowm_smoke.sh
# GPU smoke + timing for the DINO-WM (+ShiftWM) study.  Writes runs/dinowm_plugin/smoke/timing.txt.
source /home/Test/abhijit.das/projects/WM-ICLR/slurm_jebel/tasks/dinowm_common.sh
set +e
SM=$RUNS/smoke; mkdir -p "$SM" "$RUNS/outputs"; T=$SM/timing.txt; : > "$T"
stamp() { echo "$(date +%s) $*" | tee -a "$T"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$T"
python - <<'EOF' | tee -a "$T"
import torch
for e in ["pusht", "wall_single"]:
    c = torch.load(f"data/dinowm/outputs/{e}/checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
    print("official", e, "epoch", c.get("epoch"), "keys", sorted(c))
EOF

stamp pytest; python -m pytest -q -p no:cacheprovider tests/test_dinowm_plugin.py 2>&1 | tail -3 | tee -a "$T"

# 1) official released checkpoints through our planning wrapper (pipeline sanity + planning cost)
ln -sfn "$DATASET_DIR/outputs/pusht" "$RUNS/outputs/official_pusht"
ln -sfn "$DATASET_DIR/outputs/wall_single" "$RUNS/outputs/official_wall"
CEM=(planner._target_=planning.cem.CEMPlanner planner.name=cem +planner.horizon=5 +planner.topk=30 +planner.num_samples=300
     +planner.var_scale=1 +planner.opt_steps=30 +planner.eval_every=1)
stamp plan_official_pusht_cem_n10
timeout 3600 python $SCR/run_plan.py --config-name plan_pusht.yaml ckpt_base_path=$RUNS model_name=official_pusht model_epoch=latest \
  n_evals=10 seed=99 hydra.run.dir=$SM/plan_official_pusht_cem "${CEM[@]}" > $SM/plan_official_pusht_cem.log 2>&1
tail -n1 $SM/plan_official_pusht_cem/logs.json | tee -a "$T"
stamp plan_official_pusht_mpc_n10_maxiter3
timeout 3600 python $SCR/run_plan.py --config-name plan_pusht.yaml ckpt_base_path=$RUNS model_name=official_pusht model_epoch=latest \
  n_evals=10 seed=99 planner.max_iter=3 hydra.run.dir=$SM/plan_official_pusht_mpc > $SM/plan_official_pusht_mpc.log 2>&1
tail -n1 $SM/plan_official_pusht_mpc/logs.json | tee -a "$T"
stamp plan_official_wall_mpc_n50_maxiter10
timeout 3600 python $SCR/run_plan.py --config-name plan_wall.yaml ckpt_base_path=$RUNS model_name=official_wall model_epoch=latest \
  n_evals=50 seed=99 planner.max_iter=10 hydra.run.dir=$SM/plan_official_wall_mpc > $SM/plan_official_wall_mpc.log 2>&1
grep -c . $SM/plan_official_wall_mpc/logs.json | tee -a "$T"; tail -n1 $SM/plan_official_wall_mpc/logs.json | tee -a "$T"
stamp plan_official_wall_cem_n50
timeout 3600 python $SCR/run_plan.py --config-name plan_wall.yaml ckpt_base_path=$RUNS model_name=official_wall model_epoch=latest \
  n_evals=50 seed=99 hydra.run.dir=$SM/plan_official_wall_cem "${CEM[@]}" > $SM/plan_official_wall_cem.log 2>&1
tail -n1 $SM/plan_official_wall_cem/logs.json | tee -a "$T"

# 2) training throughput, both arms, reduced data (1 epoch each)
for env in pusht wall; do
  env_settings $env
  NR=150; [ $env = wall ] && NR=300
  for arm in dinowm dinowm_shiftwm; do
    extra=(); [ $arm = dinowm_shiftwm ] && extra+=("predictor._target_=$PLUGIN_TARGET")
    rd=$RUNS/outputs/smoke_${env}_${arm}; rm -rf "$rd"
    stamp train_${env}_${arm}_nrollout${NR}
    timeout 3000 python $SCR/run_train.py --config-name train.yaml "${TRAIN_ENV[@]}" training.epochs=1 env.dataset.n_rollout=$NR \
      env.num_workers=24 ckpt_base_path=$RUNS hydra.run.dir=$rd "${extra[@]}" > $SM/train_${env}_${arm}.log 2>&1
    echo "rc=$?" | tee -a "$T"
    tr '\r' '\n' < $SM/train_${env}_${arm}.log | grep -E "Epoch 1 Train: 100%|Epoch 1 Valid: 100%" | tail -2 | tee -a "$T"
    cat $rd/wrapper_info.json | tr -d '\n' | tee -a "$T"; echo | tee -a "$T"
    python -c "import json;r=json.loads(open('$rd/metrics.jsonl').readline());print({k:round(v,4) for k,v in r.items() if k in ('train_loss','val_loss','val_z_visual_loss','val_z_visual_err_rollout','val_img_lpips_pred','val_img_ssim_pred')})" | tee -a "$T"
    stamp eval_openloop_${env}_${arm}
    timeout 2400 python $SCR/eval_openloop.py --run_dir $rd --epoch 1 --out $SM/openloop_${env}_${arm}.json > $SM/eval_${env}_${arm}.log 2>&1
    echo "rc=$?" | tee -a "$T"; python -c "import json;r=json.load(open('$SM/openloop_${env}_${arm}.json'));print(r['seconds'], r['teacher_forced'].get('z_visual_err_pred'), r['teacher_forced'].get('pred_img_lpips'), r['rollout_H5'].get('z_visual_err_step5'))" | tee -a "$T"
  done
done
# 3) planning cost with a model trained here (plugin arm), open-loop CEM, 10 evals
stamp plan_smoke_pusht_plugin_cem_n10
timeout 2400 python $SCR/run_plan.py --config-name plan_pusht.yaml ckpt_base_path=$RUNS model_name=smoke_pusht_dinowm_shiftwm model_epoch=1 \
  n_evals=10 seed=99 hydra.run.dir=$SM/plan_smoke_pusht_plugin_cem "${CEM[@]}" > $SM/plan_smoke_pusht_plugin_cem.log 2>&1
tail -n1 $SM/plan_smoke_pusht_plugin_cem/logs.json | tee -a "$T"
stamp done
