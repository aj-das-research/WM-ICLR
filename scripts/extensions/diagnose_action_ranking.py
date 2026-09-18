#!/usr/bin/env python3
"""Paired action-ranking diagnosis on fixed drone development branches.

This is a short-horizon causal/action-ranking diagnostic, not a new closed-loop
benchmark. Candidate selection does not use learned or simulator outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from shiftwm.data import pixels_to_tensor
from shiftwm.extensions.checkpoint import atomic_json, file_sha256, load_package
from shiftwm.extensions.evaluate import ExtensionGoalCost, accept_execution, tasks_for_split
from shiftwm.extensions.rpc import SimulatorClient


def candidates(seed=20260919):
    labels, commands = ["zero"], [np.zeros((25, 2), np.float32)]
    axes = [("x+",[1.,0.]),("x-",[-1.,0.]),("y+",[0.,1.]),("y-",[0.,-1.])]
    for name, vector in axes:
        labels.append("hold_"+name); commands.append(np.tile(vector,(25,1)).astype(np.float32))
    for x,y in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        labels.append(f"diagonal_{x:+d}_{y:+d}")
        commands.append(np.tile(np.array([x,y],np.float32)/np.sqrt(2),(25,1)).astype(np.float32))
    for name,vector in axes:
        pulse=np.zeros((25,2),np.float32);pulse[:5]=vector
        labels.append("pulse5_"+name);commands.append(pulse)
    rng=np.random.default_rng(seed)
    for i in range(19):
        labels.append(f"random_blocks_{i:02d}")
        commands.append(np.repeat(rng.uniform(-1,1,(5,2)).astype(np.float32),5,axis=0))
    result=np.stack(commands)
    assert result.shape==(32,25,2) and np.isfinite(result).all() and np.abs(result).max()<=1
    return labels,result


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def visual_identity(model):
    modules=(model.base.encoder,model.base.projector,model.reference_encoder,model.reference_projector)
    return [{key:tensor_hash(value) for key,value in module.state_dict().items()} for module in modules]


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float(spearmanr(a,b).statistic)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--data",type=Path,default=Path("data/extensions/drone_v1"))
    parser.add_argument("--runs",type=Path,default=Path("runs/extensions"))
    parser.add_argument("--tasks",type=int,default=4)
    parser.add_argument("--candidate-seed",type=int,default=20260919)
    parser.add_argument("--output",type=Path,default=Path("reports/evidence/drone_action_ranking.json"))
    args=parser.parse_args()
    if args.tasks!=4:
        parser.error("This bounded diagnostic fixes the first four eligible development tasks")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start=time.perf_counter()
    directory=args.output.with_suffix("")
    directory.mkdir(parents=True,exist_ok=False)
    tasks,selection=tasks_for_split("drone",args.data,"development",args.tasks)
    assert len(tasks)==4
    labels,native=candidates(args.candidate_seed)
    grouped=torch.from_numpy(native.reshape(32,5,10))
    sources=[Path(__file__),Path("src/shiftwm/extensions/evaluate.py"),Path("src/shiftwm/extensions/rpc.py"),
             Path("src/shiftwm/extensions/model.py"),Path("src/shiftwm/model.py"),Path("src/shiftwm/data.py"),
             Path("scripts/extensions/serve_drone.py"),Path("src/shiftwm/extensions/drone.py"),
             args.data/"manifest.json"]
    input_rows=[];inputs={"candidate_commands":native}
    for index,task in enumerate(tasks):
        with np.load(args.data/task["audit_file"],allow_pickle=False) as archive:
            support_images=archive["images"][[0,5,10]].copy()
        task["support_images"]=support_images
        inputs[f"support_images_{index}"]=support_images
        inputs[f"support_commands_{index}"]=task["commands"][:10]
        inputs[f"goal_image_{index}"]=task["goal_image"]
        input_rows.append({k:task[k] for k in ("trajectory_id","seed","dynamics_id","observation_id","goal_state","audit_file","audit_sha256","initial_distance_m")})
    np.savez_compressed(directory/"inputs.npz",**inputs)
    registration={"scope":"fixed_development_short_horizon_action_ranking_not_closed_loop_benchmark",
                  "selection":selection,"tasks":input_rows,"candidate_seed":args.candidate_seed,
                  "candidate_labels":labels,"candidate_commands":native.tolist(),
                  "support_native_calls":10,"future_native_calls":25,"future_blocks":5,
                  "native_budget_per_branch":35,"success_stopping":False,
                  "safety_stopping":"crash/workspace escape stops branch; short branches flagged",
                  "ranking_primary":"full-horizon nonfailure candidates only; unsafe predicted winners flagged rather than silently replaced",
                  "random_reference":"uniform over all32prespecified candidates, not a separately optimized policy",
                  "source_hashes":{str(p):file_sha256(p) for p in sources},"inputs_sha256":file_sha256(directory/"inputs.npz")}
    atomic_json(registration,directory/"registration.json")
    # Score all six models before executing or encoding ANY branch future.
    models={};predictions=[];feature_cache={};reference_identity=None;normalization=None
    for architecture in ("transformer","gru"):
        for mode in ("framewise","constant_dynamics","factorized"):
            key=f"{architecture}_{mode}"
            run=args.runs/f"drone_{architecture}_{mode}_s0"
            summary=json.loads((run/"training_summary.json").read_text())
            assert summary["status"]=="completed" and summary["completed_epochs"]==30
            package=(run/"best").resolve(strict=True)
            model,state=load_package(package,"cpu")
            assert model.architecture==architecture and model.extension_mode==mode
            this_identity=visual_identity(model)
            if reference_identity is None:
                reference_identity=this_identity
                normalization=(model.action_mean.clone(),model.action_std.clone())
            else:
                assert this_identity==reference_identity,"Visual coordinates differ across compared models"
                assert torch.equal(model.action_mean,normalization[0]) and torch.equal(model.action_std,normalization[1])
            models[key]=model
            cost=ExtensionGoalCost(model).eval()
            for index,task in enumerate(tasks):
                with torch.inference_mode():
                    if index not in feature_cache:
                        feature_cache[index]={"history":model.encode_images(pixels_to_tensor(task["support_images"],task["observation_id"])).unsqueeze(0),
                                              "goal_raw":model.encode_images(pixels_to_tensor(task["goal_image"],task["observation_id"])[None]),
                                              "goal_canonical":model.encode_images(pixels_to_tensor(task["goal_image"],0)[None],reference=True)}
                    cache=feature_cache[index]
                    history=cache["history"]
                    actions=torch.from_numpy(task["commands"][:10].reshape(1,2,10))
                    obs,dyn=model.infer_context(history,actions)
                    goal=model.correct_observations(cache["goal_raw"][:,None],obs)[:,0]
                    # Actual production ExtensionGoalCost consumes standardized
                    # candidate coordinates and performs its native clipping.
                    standardized=(grouped-model.action_mean)/model.action_std
                    roundtrip=cost.to_native(standardized)
                    roundtrip_error=float((roundtrip-grouped).abs().max())
                    assert roundtrip_error<=1e-7
                    information={"history_features":history[:,None].expand(-1,32,-1,-1),
                                 "past_actions":actions[:,None].expand(-1,32,-1,-1),
                                 "observation_context":obs[:,None].expand(-1,32,-1),
                                 "dynamics_context":dyn[:,None].expand(-1,32,-1),
                                 "goal_features":goal[:,None].expand(-1,32,-1)}
                    scores=cost.get_cost(information,standardized[None])[0].cpu().numpy()
                    assert np.isfinite(scores).all()
                    terminal=model.rollout_features(history.expand(32,-1,-1),actions.expand(32,-1,-1),roundtrip,
                              contexts=(obs.expand(32,-1),dyn.expand(32,-1)))[:,-1]
                    np.testing.assert_allclose(scores,(terminal-goal).square().mean(-1).cpu().numpy(),rtol=1e-6,atol=1e-10)
                    cache[key]={"observation_context":obs.clone(),"goal":goal.clone(),"terminal":terminal.clone()}
                    predictions.append({"model":key,"architecture":architecture,"mode":mode,"training_seed":0,
                                        "task_index":index,"trajectory_id":task["trajectory_id"],
                                        "checkpoint":str(package),"checkpoint_sha256":file_sha256(package/"model.pt"),
                                        "best_epoch":state["epoch"],"predicted_goal_cost":scores.tolist(),
                                        "predicted_best_candidate":int(np.argmin(scores)),
                                        "normalization_roundtrip_max_error":roundtrip_error})
            print(json.dumps({"scored_model":key,"tasks":len(tasks)}),flush=True)
    atomic_json({"status":"all_predictions_fixed_before_counterfactual_outcomes","predictions":predictions,
                 "registration_sha256":file_sha256(directory/"registration.json")},directory/"predictions_before_outcomes.json")
    # Each candidate starts from a fresh seeded simulator and exact recorded
    # common history. Hidden metrics enter only scoring, never model inference.
    branches=[];future_frames={}
    with SimulatorClient("drone",directory/"simulator.log") as client:
        for index,task in enumerate(tasks):
            frames=[]
            for candidate_index,commands in enumerate(native):
                response=client.request({"op":"reset","seed":task["seed"],"dynamics_id":task["dynamics_id"],
                                        "goal_state":task["goal_state"],"image_size":128,"max_steps":35})
                assert np.array_equal(response["image"],task["support_images"][0]),"Changed initial reference RGB"
                support_metrics=[]
                for block,start_native in enumerate((0,5),1):
                    requested=task["commands"][start_native:start_native+5]
                    response=client.request({"op":"step","actions":requested.tolist(),"stop_on_success":False})
                    actual=accept_execution(response,requested)
                    assert len(actual)==5 and not response.get("terminated"),"Support failed"
                    assert np.array_equal(response["image"],task["support_images"][block]),"Changed support replay RGB"
                    support_metrics.extend(response["metrics_per_step"])
                support_distance=float(response["metrics"]["goal_distance_m"])
                outcomes=[];executed=[]
                for step in range(0,25,5):
                    requested=commands[step:step+5]
                    response=client.request({"op":"step","actions":requested.tolist(),"stop_on_success":False})
                    actual=accept_execution(response,requested)
                    outcomes.extend(response["metrics_per_step"]);executed.extend(actual.tolist())
                    if response.get("terminated") or response.get("truncated"):
                        break
                last=response["metrics"]
                safe=not any(m["crash"] or m["workspace_escape"] for m in outcomes)
                full=len(executed)==25
                frames.append(response["image"])
                branches.append({"task_index":index,"trajectory_id":task["trajectory_id"],
                                 "candidate_index":candidate_index,"candidate_label":labels[candidate_index],
                                 "support_replay_exact":True,"support_success":any(m["success"] for m in support_metrics),
                                 "support_distance_m":support_distance,"future_calls":len(executed),
                                 "executed_commands":executed,"final_distance_m":float(last["goal_distance_m"]),
                                 "final_success":bool(last["success"]),"any_success":any(m["success"] for m in outcomes),
                                 "crash":any(m["crash"] for m in outcomes),"workspace_escape":any(m["workspace_escape"] for m in outcomes),
                                 "full_horizon_safe":safe and full,"metrics_per_step":outcomes})
            future_frames[index]=np.stack(frames)
            np.savez_compressed(directory/f"task_{index}_realized.npz",images=future_frames[index])
            atomic_json({"branches":branches},directory/"outcomes_partial.json")
            print(json.dumps({"replayed_task":task["trajectory_id"],"branches":32}),flush=True)
    # Post-outcome feature geometry is a separate diagnostic. These images were
    # unavailable when the frozen predictions above were produced and saved.
    comparisons=[];geometry=[]
    foundation=next(iter(models.values()))
    for index,task in enumerate(tasks):
        with torch.inference_mode():
            canonical=foundation.encode_images(pixels_to_tensor(future_frames[index],0),reference=True)
            warm=foundation.encode_images(pixels_to_tensor(future_frames[index],task["observation_id"]))
        actual=np.array([b["final_distance_m"] for b in branches if b["task_index"]==index])
        safe=np.array([b["full_horizon_safe"] for b in branches if b["task_index"]==index])
        canonical_goal=feature_cache[index]["goal_canonical"]
        canonical_cost=(canonical-canonical_goal).square().mean(-1).numpy()
        geometry.append({"task_index":index,"safe_candidates":int(safe.sum()),
                         "canonical_feature_goal_cost":canonical_cost.tolist(),
                         "canonical_geometry_spearman_vs_physical_safe":correlation(canonical_cost[safe],actual[safe])})
        for prediction in [p for p in predictions if p["task_index"]==index]:
            model=models[prediction["model"]];values=feature_cache[index][prediction["model"]]
            with torch.inference_mode():
                corrected=model.correct_observations(warm[:,None],values["observation_context"].expand(32,-1))[:,0]
                ref_cost=(canonical-values["goal"]).square().mean(-1).numpy()
                corrected_cost=(corrected-values["goal"]).square().mean(-1).numpy()
                forecast_error=(values["terminal"]-canonical).square().mean(-1).numpy()
            scores=np.array(prediction["predicted_goal_cost"]);best=prediction["predicted_best_candidate"]
            actual_best=int(np.flatnonzero(safe)[np.argmin(actual[safe])])
            random_indices=np.arange(13,32)
            row={**prediction,"safe_candidates":int(safe.sum()),"unsafe_or_short_candidates":np.flatnonzero(~safe).tolist(),
                 "predicted_vs_physical_spearman_safe":correlation(scores[safe],actual[safe]),
                 "predicted_vs_realized_reference_cost_spearman_safe":correlation(scores[safe],ref_cost[safe]),
                 "realized_reference_cost_vs_physical_spearman_safe":correlation(ref_cost[safe],actual[safe]),
                 "realized_corrected_cost_vs_physical_spearman_safe":correlation(corrected_cost[safe],actual[safe]),
                 "predicted_best_candidate_label":labels[best],"predicted_best_full_horizon_safe":bool(safe[best]),
                 "predicted_best_actual_distance_m":float(actual[best]),"best_realized_safe_candidate":actual_best,
                 "best_realized_safe_distance_m":float(actual[actual_best]),
                 "predicted_best_distance_regret_m":float(actual[best]-actual[actual_best]) if safe[best] else None,
                 "uniform32_candidate_mean_distance_m":float(actual.mean()),
                 "uniform32_safe_candidate_mean_distance_m":float(actual[safe].mean()),
                 "random19_candidate_mean_distance_m":float(actual[random_indices].mean()),
                 "random19_unsafe_count":int((~safe[random_indices]).sum()),
                 "realized_reference_goal_cost":ref_cost.tolist(),"realized_corrected_goal_cost":corrected_cost.tolist(),
                 "terminal_forecast_mse":forecast_error.tolist(),"predicted_cost_range":float(np.ptp(scores))}
            comparisons.append(row)
    grouped_rows=defaultdict(list)
    for row in comparisons:grouped_rows[row["model"]].append(row)
    aggregates=[]
    for key,rows in grouped_rows.items():
        def mean(name):
            values=[r[name] for r in rows if r[name] is not None]
            return float(np.mean(values)) if values else None
        aggregates.append({"model":key,"tasks":len(rows),"safe_predicted_winners":sum(r["predicted_best_full_horizon_safe"] for r in rows),
                           "mean_spearman_predicted_vs_physical":mean("predicted_vs_physical_spearman_safe"),
                           "mean_spearman_predicted_vs_realized_feature_cost":mean("predicted_vs_realized_reference_cost_spearman_safe"),
                           "mean_spearman_realized_feature_cost_vs_physical":mean("realized_reference_cost_vs_physical_spearman_safe"),
                           "mean_safe_winner_regret_m":mean("predicted_best_distance_regret_m"),
                           "mean_predicted_winner_distance_m":mean("predicted_best_actual_distance_m"),
                           "mean_uniform32_distance_m":mean("uniform32_candidate_mean_distance_m")})
    report={"status":"completed","scope":registration["scope"],"registration":registration,
            "registration_sha256":file_sha256(directory/"registration.json"),
            "predictions_before_outcomes_sha256":file_sha256(directory/"predictions_before_outcomes.json"),
            "models_share_exact_frozen_visual_state":True,"models_share_exact_action_normalization":True,
            "future_images_excluded_from_inference":True,"aggregates":aggregates,"common_geometry":geometry,
            "comparisons":comparisons,"branches":branches,"wall_seconds":time.perf_counter()-start,
            "limitations":["Four development tasks and one training seed; no statistical significance claim",
                           "Fixed candidate library and25-step open-loop horizon, not CEM or closed-loop performance",
                           "Safe full-horizon subset used for rank/regret; every unsafe/short branch remains reported",
                           "Realized RGB features are diagnostic outcomes only; never used for candidate/context selection",
                           "Positive correlation means lower predicted cost tends to correspond to lower distance",
                           "Feature-geometry and dynamics errors may coexist; correlations do not establish a unique cause"]}
    atomic_json(report,args.output)
    lines=["# Drone fixed-candidate action-ranking diagnosis","",
           "This is a four-task development diagnostic, not a closed-loop benchmark. All six seed-zero models scored the same32candidates before any counterfactual branch was simulated. Every branch independently reset and replayed the exact10-command support. No test tasks were used.","",
           "| Model | Prediction–physical rank correlation | Prediction–realized latent rank correlation | Realized latent–physical rank correlation | Mean regret (mm) | Safe winners |",
           "|---|---:|---:|---:|---:|---:|"]
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    for row in aggregates:
        regret=row["mean_safe_winner_regret_m"]
        lines.append(f"| {row['model']} | {fmt(row['mean_spearman_predicted_vs_physical'])} | {fmt(row['mean_spearman_predicted_vs_realized_feature_cost'])} | {fmt(row['mean_spearman_realized_feature_cost_vs_physical'])} | {'—' if regret is None else f'{1000*regret:.1f}'} | {row['safe_predicted_winners']}/4 |")
    lines += ["","Spearman correlations are computed within task over safe full-horizon candidates and then averaged. Regret is the selected candidate's physical distance minus the best safe candidate's distance; it is undefined if the predicted winner is unsafe. The full JSON retains all candidates, exclusions, random-reference means and per-task values.","",
              "Frozen canonical feature-to-goal geometry (same encoder for every model), correlation with physical distance per task: "+", ".join(fmt(r["canonical_geometry_spearman_vs_physical_safe"]) for r in geometry)+".","",
              "Interpretation: weak prediction–realized-feature ranking suggests action/dynamics forecasting is not ordering futures correctly. Weak realized-feature–physical ranking suggests the visual goal metric is not aligned with physical progress. Both can occur; this small controlled branch study cannot establish a unique cause.","",
              f"Artifacts: `{directory}`; elapsed {report['wall_seconds']:.1f}s. Candidate/source/checkpoint hashes and raw commanded sequences are retained. No model, dataset, frozen evaluator or simulator source was changed."]
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"completed","output":str(args.output),"aggregates":aggregates,"wall_seconds":report["wall_seconds"]}),flush=True)


if __name__=="__main__":
    main()
