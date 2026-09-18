#!/usr/bin/env python3
"""One-time audited metadata alias migration; never changes trajectory/features.

The generic frozen trainer expects `environment`; the surgery collector emitted
the synonymous `env`. Preserve original metadata and rebind hashes only after
checking every raw/audit/cache payload and the released encoder weights.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/extensions/surgery_v1"
CACHE = ROOT / "data/features/surgery_v1"
REPORT = ROOT / "reports/surgery/manifest_alias_migration"


def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(value,path):
    temp=path.with_suffix(path.suffix+".migration.tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(path)


def main():
    start=time.monotonic()
    os.chdir(ROOT)
    source=DATA/"manifest.json";cache_path=CACHE/"manifest.json"
    original=json.loads(source.read_text());cache=json.loads(cache_path.read_text())
    if original.get("environment") == "surgery":
        receipt=json.loads((REPORT/"receipt.json").read_text())
        assert digest(source)==receipt["new_dataset_manifest_sha256"]
        assert digest(cache_path)==receipt["new_cache_manifest_sha256"]
        print(json.dumps({"status":"already_completed","receipt":str(REPORT/"receipt.json")}))
        return
    assert original.get("env")=="surgery" and "environment" not in original
    old_hash=digest(source)
    assert cache["dataset_manifest_sha256"]==old_hash
    weights=Path(cache["base_checkpoint"])/"weights.pt"
    assert digest(weights)==cache["encoder_weights_sha256"]
    metadata={key:cache[key] for key in ("schema_version","dataset_root","encoder_weights_sha256","preprocessing","appearance_factors","dtype","latent_dim","base_checkpoint")}
    # JSON object keys changed to strings on disk. Original cache creation used
    # integer appearance keys; sorted serialization must recover that form.
    metadata["appearance_factors"]={int(k):v for k,v in metadata["appearance_factors"].items()}
    assert hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()==cache["cache_signature"]
    payload_hashes={}
    for ep in original["episodes"]:
        for name,expected in ((ep["file"],ep["sha256"]),(ep["audit_file"],ep["audit_sha256"])):
            path=DATA/name; observed=digest(path)
            assert observed==expected,(name,"raw/audit checksum")
            payload_hashes[str(path.relative_to(ROOT))]=observed
        cached=CACHE/ep["file"]
        assert cached.with_suffix(".sha256").read_text().strip()==cache["cache_signature"]+":"+ep["sha256"]
        with np.load(cached,allow_pickle=False) as f,np.load(DATA/ep["file"],allow_pickle=False) as raw:
            assert set(f.files)=={"features","actions"}
            assert f["features"].shape==(4,ep["steps"]+1,192)
            assert f["features"].dtype==np.float32 and np.isfinite(f["features"]).all()
            assert np.array_equal(f["actions"],raw["actions"])
        payload_hashes[str(cached.relative_to(ROOT))]=digest(cached)
        payload_hashes[str(cached.with_suffix(".sha256").relative_to(ROOT))]=digest(cached.with_suffix(".sha256"))
    preserved=[DATA/"manifest.json",DATA/"action_stats.json",DATA/"training_data_validation.json",CACHE/"manifest.json",ROOT/"reports/surgery/dataset_validation.json"]
    REPORT.mkdir(parents=True,exist_ok=False)
    backups={}
    for path in preserved:
        name=str(path.relative_to(ROOT)).replace("/","__")
        destination=REPORT/name
        shutil.copy2(path,destination)
        backups[str(path.relative_to(ROOT))]={"backup":str(destination.relative_to(ROOT)),"sha256":digest(destination)}
    goal_hashes={str(path.relative_to(ROOT)):digest(path) for path in (DATA/"planning_goals").rglob("*") if path.is_file()}
    updated={**original,"environment":"surgery"}
    atomic_json(updated,source)
    new_hash=digest(source)
    try:
        with (REPORT/"raw_data_reaudit.log").open("w") as log:
            subprocess.run([sys.executable,"scripts/extensions/validate_surgery_data.py"],check=True,stdout=log,stderr=subprocess.STDOUT)
        with (REPORT/"prepare_dataset.log").open("w") as log:
            subprocess.run([sys.executable,"scripts/extensions/prepare_dataset.py","--data",str(DATA)],check=True,stdout=log,stderr=subprocess.STDOUT)
        original_stats=json.loads((REPORT/backups["data/extensions/surgery_v1/action_stats.json"]["backup"].split("/")[-1]).read_text())
        new_stats=json.loads((DATA/"action_stats.json").read_text())
        assert {k:v for k,v in original_stats.items() if k!="dataset_manifest_sha256"} == {k:v for k,v in new_stats.items() if k!="dataset_manifest_sha256"}
        assert all(digest(ROOT/path)==expected for path,expected in payload_hashes.items()),"Payload changed during migration"
        assert all(digest(ROOT/path)==expected for path,expected in goal_hashes.items()),"Goal registry changed"
        assert digest(weights)==cache["encoder_weights_sha256"]
        cache_updated={**cache,"dataset_manifest_sha256":new_hash}
        atomic_json(cache_updated,cache_path)
        receipt={"status":"completed","reason":"Add environment='surgery' alias expected by frozen generic trainer; retain env='surgery'",
                 "old_dataset_manifest_sha256":old_hash,"new_dataset_manifest_sha256":new_hash,
                 "old_cache_manifest_sha256":backups["data/features/surgery_v1/manifest.json"]["sha256"],
                 "new_cache_manifest_sha256":digest(cache_path),"new_action_stats_sha256":digest(DATA/"action_stats.json"),
                 "only_dataset_change":{"added":{"environment":"surgery"}},
                 "only_cache_change":"dataset_manifest_sha256", "action_statistics_numerically_identical":True,
                 "payload_files_verified_unchanged":len(payload_hashes),"payload_hashes":payload_hashes,
                 "goal_files_verified_unchanged":len(goal_hashes),"goal_file_hashes":goal_hashes,
                 "weights_sha256":cache["encoder_weights_sha256"],"backups":backups,
                 "goal_registry_provenance":"Original goal selection/registry retains old manifest hash. This receipt links it to new metadata; episode/audit/goal pixels and commands are unchanged.",
                 "no_model_trainer_or_config_changes":True,"script_sha256":digest(__file__),"wall_seconds":time.monotonic()-start}
        atomic_json(receipt,REPORT/"receipt.json")
        print(json.dumps({k:v for k,v in receipt.items() if k not in ("payload_hashes","goal_file_hashes","backups")},indent=2))
    except Exception:
        for relative,record in backups.items():
            shutil.copy2(ROOT/record["backup"],ROOT/relative)
        raise


if __name__=="__main__":
    main()
