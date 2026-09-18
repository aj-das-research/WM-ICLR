#!/usr/bin/env python3
"""Prespecify eligible real image goals and verify every reference by replay.

Selection uses only simulator trajectories, never model predictions or success.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from PIL import Image
from shiftwm.extensions.surgery import SurgeryAdapter, SurgeryConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_goal(task):
    from sofa_env.utils.camera import world_to_pixel_coordinates
    root, ep = Path(task["root"]), task["episode"]
    output = root / "planning_goals" / ep["split"]
    output.mkdir(parents=True, exist_ok=True)
    uid = ep["trajectory_id"]
    begin = time.perf_counter()
    with np.load(root / ep["audit_file"],allow_pickle=False) as data:
        commands = data["actions_commanded"].copy()
        original_states = data["tissue_target_xyz_m"].copy()
        original_gripper = data["gripper_poses"].copy()
        original_frames = data["images_native"].copy()
        original_goal = data["goal_xyz_m"][0].copy()
    goal_state = original_states[-1].copy()
    env = SurgeryAdapter(SurgeryConfig(seed=ep["seed"],
                         actuator_gain=(1.0,.75,1.25)[ep["dynamics_id"]], image_size=128))
    image = env.set_reachable_goal(goal_state)
    initial_image = image.copy()
    diag = env.diagnostics()
    initial_distance = float(diag["distance_m"])
    assert initial_distance>=.004 and not diag["success"], "Eligibility changed on replay"
    # Only the native visible target disk may differ from collected RGB.
    mask = np.zeros((128,128),dtype=bool)
    yy,xx = np.mgrid[:128,:128]
    for position in [original_goal,goal_state]:
        row,col = world_to_pixel_coordinates(position,env.env._sofa_camera)
        mask |= (xx-round(col))**2+(yy-round(row))**2 <= 2**2
    max_state_error = float(np.max(np.abs(diag["tissue_target_xyz_m"]-original_states[0])))
    max_gripper_error = float(np.max(np.abs(diag["gripper_pose"]-original_gripper[0])))
    outside_marker_changed = int(np.any(image!=original_frames[0],axis=-1)[~mask].sum())
    marker_position_max_error = 0.0
    for i,command in enumerate(commands,1):
        image,diag = env.step(command)
        max_state_error = max(max_state_error,float(np.max(np.abs(diag["tissue_target_xyz_m"]-original_states[i]))))
        max_gripper_error = max(max_gripper_error,float(np.max(np.abs(diag["gripper_pose"]-original_gripper[i]))))
        outside_marker_changed += int(np.any(image!=original_frames[i],axis=-1)[~mask].sum())
        marker_position_max_error = max(marker_position_max_error,float(np.max(np.abs(diag["goal_xyz_m"]-goal_state))))
        assert diag["valid_action"] and diag["stable_deformation"], "Reference became invalid/unstable"
    env.close()
    assert max_state_error < 1e-8 and max_gripper_error < 1e-8, "Physical state replay changed"
    assert outside_marker_changed == 0 and marker_position_max_error == 0, "Marker/rendering mismatch"
    assert diag["success"] and diag["distance_m"] <= .002, "Reference is not a native goal success"
    goal_file,initial_file,command_file = (output/f"{uid}-{suffix}" for suffix in ("goal.png","initial.png","commands.npz"))
    Image.fromarray(image).save(goal_file)
    Image.fromarray(initial_image).save(initial_file)
    np.savez_compressed(command_file,actions_commanded=commands)
    result = {"trajectory_id":uid,"seed":ep["seed"],"split":ep["split"],"dynamics_id":ep["dynamics_id"],
              "goal_state":goal_state.tolist(),"goal_native_index":200,
              "goal_image_file":str(goal_file.relative_to(root)),"goal_image_sha256":sha(goal_file),
              "initial_image_file":str(initial_file.relative_to(root)),"initial_image_sha256":sha(initial_file),
              "command_file":str(command_file.relative_to(root)),"command_file_sha256":sha(command_file),
              "source_episode_sha256":ep["sha256"],"source_audit_sha256":ep["audit_sha256"],
              "initial_distance_m":initial_distance,"native_goal_distance_m":float(diag["distance_m"]),
              "native_goal_success":bool(diag["success"]),"replay_state_max_abs_error_m":max_state_error,
              "replay_gripper_max_abs_error":max_gripper_error,
              "outside_target_marker_changed_pixels_across_all_frames":outside_marker_changed,
              "target_marker_position_max_abs_error_m":marker_position_max_error,
              "wall_seconds":time.perf_counter()-begin,"goal_rule":"native200 endpoint, all200 valid/stable, initial XZ distance>=4mm"}
    (output/f"{uid}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"goal":uid,"seconds":result["wall_seconds"],"native_success":True}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=Path("data/extensions/surgery_v1"))
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--splits",nargs="+",default=["development","test"])
    args=p.parse_args()
    manifest=json.loads((args.dataset/"manifest.json").read_text())
    selected={s:[] for s in args.splits}; excluded={s:[] for s in args.splits}
    for ep in manifest["episodes"]:
        if ep["split"] not in args.splits:
            continue
        with np.load(args.dataset/ep["audit_file"],allow_pickle=False) as d:
            separation=float(np.linalg.norm((d["tissue_target_xyz_m"][-1]-d["tissue_target_xyz_m"][0])[[0,2]]))
            reasons=[]
            if not d["valid_action"].all(): reasons.append("workspace_invalid_action")
            if not d["stable_deformation"].all(): reasons.append("unstable_deformation")
            if separation < .004: reasons.append("initial_endpoint_distance_below_4mm")
        if reasons:
            excluded[ep["split"]].append({"trajectory_id":ep["trajectory_id"],"seed":ep["seed"],"dynamics_id":ep["dynamics_id"],"reasons":reasons,"initial_endpoint_distance_m":separation})
        else:
            selected[ep["split"]].append(ep)
    selection={"rule":"native200 endpoint, all200 valid/stable, initial XZ distance>=4mm; no model outcome selection",
               "planning_budget_native_calls":200,"shared_support_native_calls":10,
               "selected":{s:[e["trajectory_id"] for e in es] for s,es in selected.items()},"excluded":excluded,
               "source_manifest_sha256":sha(args.dataset/"manifest.json"),"script_sha256":sha(__file__)}
    base=args.dataset/"planning_goals";base.mkdir(parents=True,exist_ok=True)
    (base/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    tasks=[{"root":str(args.dataset),"episode":e} for es in selected.values() for e in es]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        results=list(pool.map(make_goal,tasks))
    for split in args.splits:
        directory=base/split;directory.mkdir(parents=True,exist_ok=True)
        records=[r for r in results if r["split"]==split]
        output={"status":"independent_replay_verified","split":split,"episodes":records,
                "eligible_count":len(records),"excluded_count":len(excluded[split]),"exclusions":excluded[split],
                "selection_sha256":sha(base/"selection.json"),"goal_protocol":"reachable_image_goal_distribution_not_original_random_goal_sampler",
                "native_success":"XZ tissue-target distance <= 0.002m","native_budget":200,"support_native_calls":10,
                "not_model_results":True}
        (directory/"manifest.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"verified_goals":len(results),"selected":{s:len(es) for s,es in selected.items()},"excluded":{s:len(es) for s,es in excluded.items()}}),flush=True)


if __name__ == "__main__":
    main()
