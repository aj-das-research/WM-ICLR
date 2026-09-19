#!/usr/bin/env python3
"""Gated, complete spatial-development inference prerelease; never trains."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlparse

REGISTRY = "configs/real_video_spatial/v1/registration.json"
EXPECTED_REGISTRY = "ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337"
FINAL = "reports/real_video_spatial/finalization.json"
SCRIPT = "scripts/publishing/spatial_release.py"
RUNTIME = "scripts/publishing/spatial_release_runtime.py"
MODES = ("autoregressive", "anchored_additive", "transport", "context_off", "action_free")
METRICS = ("native_mse", "native_persistence_mse", "original_2x2_mse", "original_2x2_persistence_mse")
TAG = "spatial-world-models-v1"
BUNDLE = "shiftwm-spatial-v1"
EXPECTED_NAMES = {f"{mode}_s{seed}" for mode in MODES for seed in range(3)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe relative release path")
    return Path(name)


def rooted(root, name):
    path = Path(root) / safe_relative(name)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Source path escapes scientific workspace")
    return path


def copy(source, target, expected=None):
    source, target = Path(source), Path(target)
    before = sha(source)
    if expected is not None and before != expected:
        raise ValueError("Copy source differs from frozen identity")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    if sha(target) != before or sha(source) != before:
        raise ValueError("Source changed during verified copy")
    return before


def number(value, nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Completion gate: invalid numeric evidence")
    return float(value)


def validate_gate(report, registry, registry_hash):
    """Pure fail-closed metadata gate; source/ledger/package checks follow."""
    if (registry_hash != EXPECTED_REGISTRY or report.get("registration_sha256") != EXPECTED_REGISTRY
            or report.get("status") != "passed" or report.get("scope") != "original_validation_development_only"
            or report.get("completed_models") != 15 or report.get("epochs_per_model") != 30
            or registry.get("expected_runs") != 15 or registry.get("expected_epochs_per_run") != 30):
        raise ValueError("Completion gate: wrong or incomplete frozen spatial study")
    for label, rows in (("registry",registry.get("runs",[])), ("finalization",report.get("runs",[]))):
        if len(rows) != 15 or {row.get("name") for row in rows} != EXPECTED_NAMES:
            raise ValueError("Completion gate: incomplete or duplicate " + label)
    if {(r.get("mode"),r.get("seed")) for r in registry["runs"]} != {(m,s) for m in MODES for s in range(3)}:
        raise ValueError("Completion gate: missing registered mode/seed")
    for row in registry["runs"]:
        if row["name"] != f"{row['mode']}_s{row['seed']}":
            raise ValueError("Completion gate: inconsistent registered identity")
        safe_relative(row["config"])
    for row in report["runs"]:
        if type(row.get("selected_epoch")) is not int or not 1 <= row["selected_epoch"] <= 30:
            raise ValueError("Completion gate: invalid selected epoch")
        if row.get("validation") != "reports/real_video_spatial/" + row["name"] + "_validation.json":
            raise ValueError("Completion gate: unexpected evaluation path")
    parity = report.get("offline_cpu_parity", [])
    if len(parity) != 15 or {row.get("name") for row in parity} != EXPECTED_NAMES:
        raise ValueError("Completion gate: all 15 old parity records required")
    for row in parity:
        if (row.get("status") != "passed" or number(row.get("max_abs_error"),True) != 0
                or row.get("relocated_isolated_process") is not True or row.get("device") != "cpu"
                or row.get("input_split") != "original_validation"):
            raise ValueError("Completion gate: old local inference parity failed")
    aggregate = report.get("aggregate",{})
    if set(aggregate) != set(MODES):
        raise ValueError("Completion gate: a registered outcome is missing")
    for values in aggregate.values():
        if set(values) != set(METRICS):
            raise ValueError("Completion gate: missing metric")
        for vector in values.values():
            if not isinstance(vector,list) or len(vector) != 10:
                raise ValueError("Completion gate: ten horizon errors required")
            for value in vector: number(value,True)
    effects = report.get("paired_effects",[])
    expected = {(mode,metric,h) for mode in MODES if mode != "transport"
                for metric in ("native_mse","original_2x2_mse") for h in (5,10)}
    if len(effects) != 16 or {(r.get("comparator"),r.get("metric"),r.get("horizon")) for r in effects} != expected:
        raise ValueError("Completion gate: all 16 matched paired comparisons required")
    for row in effects:
        if (row.get("method") != "transport" or row.get("bootstrap_draws") != 10000
                or row.get("bootstrap_seed") != 173
                or row.get("scope") != "exploratory original validation; no multiple-comparison adjustment"):
            raise ValueError("Completion gate: comparison protocol differs")
        ci = row.get("paired_95_percent_interval",[])
        if len(ci) != 2 or number(ci[0]) > number(ci[1]):
            raise ValueError("Completion gate: invalid paired interval")
        if row.get("interval_includes_zero") is not (ci[0] <= 0 <= ci[1]):
            raise ValueError("Completion gate: interval label differs")
        method = aggregate["transport"][row["metric"]][row["horizon"]-1]
        base = aggregate[row["comparator"]][row["metric"]][row["horizon"]-1]
        for actual,expected_value in ((row.get("method_mean"),method),(row.get("comparator_mean"),base),
                (row.get("method_minus_comparator"),method-base)):
            if not math.isclose(number(actual),expected_value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Completion gate: comparison arithmetic differs")
        if base == 0:
            if row.get("relative_error_reduction_percent") is not None:
                raise ValueError("Completion gate: undefined gain must stay null")
        elif not math.isclose(number(row.get("relative_error_reduction_percent")),100*(base-method)/base,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError("Completion gate: gain arithmetic differs")


def verify_sources(root, identities):
    for name, expected in identities.items():
        if sha(rooted(root,name)) != expected:
            raise ValueError("Frozen source changed: " + name)


def validate_cache_gate(root, config):
    """Recheck the registered cache/budget gate and bind its changing artifacts."""
    gate_path = root / 'reports/real_video_spatial/cache_and_budget_gate.json'
    throughput_path = root / 'reports/real_video_spatial/throughput.json'
    gate, throughput = read(gate_path), read(throughput_path)
    cache = rooted(root, config['cache_root'])
    if gate.get('status') != 'passed' or gate.get('registration_sha256') != EXPECTED_REGISTRY:
        raise ValueError('Cache/budget gate absent, failed or stale')
    for key, path in [('cache_manifest_sha256',cache/'manifest.json'),
                      ('statistics_sha256',cache/'training_statistics.json'),
                      ('throughput_sha256',throughput_path)]:
        if sha(path) != gate.get(key): raise ValueError('Cache/budget evidence changed')
    rows = throughput.get('rows', [])
    if (throughput.get('status') != 'measured' or throughput.get('batch_size') != 128
            or len(rows) != 5 or {r.get('mode') for r in rows} != set(MODES)):
        raise ValueError('Incomplete registered throughput evidence')
    durations = [number(r.get('estimated_seconds_per_30_epoch_run_without_io'),True) for r in rows]
    projected = number(throughput.get('estimated_15_run_total_gpu_hours_without_io'),True)
    if (projected > 80 or max(durations) > 20*3600
            or not math.isclose(projected,3*sum(durations)/3600,rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('Registered total/per-model compute budget failed')
    manifest, identity = read(cache/'manifest.json'), read(cache/'identity.json')
    stats = read(cache/'training_statistics.json')
    if manifest.get('identity') != identity:
        raise ValueError('Spatial cache identity differs from manifest')
    if (stats.get('fit_split') != 'train' or stats.get('cache_manifest_sha256') != sha(cache/'manifest.json')
            or stats.get('normalization') != 'shared_per_channel_over_train_frames_and_patches'):
        raise ValueError('Spatial statistics lack training-only shared-channel identity')
    return {str(path.relative_to(root)):sha(path) for path in
            (gate_path,throughput_path,cache/'manifest.json',cache/'identity.json',cache/'training_statistics.json')}


def validate_relocated_proof(proof, selected_epochs):
    rows = proof.get('models', [])
    if (proof.get('status') != 'passed' or len(rows) != 15
            or {r.get('name') for r in rows} != EXPECTED_NAMES
            or proof.get('network_attempts') != 0 or proof.get('relocated') is not True
            or proof.get('device') != 'cpu'
            or any(r.get('max_abs_error') != 0 or r.get('status') != 'passed'
                   or r.get('epoch') != selected_epochs[r['name']] for r in rows)):
        raise ValueError('New release did not pass all15 relocated exact-parity checks')


def freeze(root, destination):
    registry = read(root / REGISTRY)
    if sha(root / REGISTRY) != EXPECTED_REGISTRY:
        raise ValueError("Unexpected scientific registration")
    verify_sources(root, registry["dependencies"])
    paths = set(registry["dependencies"]) | {REGISTRY,SCRIPT,RUNTIME,"scripts/publishing/spatial_release.slurm",
        "tests/test_spatial_release.py","scripts/publishing/prepare_public_snapshot.py","LICENSE","requirements.lock.txt",
        "reports/spatial_release_protocol.md",
        "pyproject.toml","src/shiftwm/__init__.py","src/shiftwm/vendor/__init__.py","src/shiftwm/vendor/lewm/__init__.py",
        "src/shiftwm/vendor/lewm/LICENSE","src/shiftwm/vendor/lewm/jepa.py","src/shiftwm/real_video/__init__.py",
        "src/shiftwm/real_video/model.py",
        "references/real_dinov2_sources.json","site/assets/DROID-LICENSE.txt"}
    paths.update(row["config"] for row in registry["runs"])
    encoder = read(root / "data/pretrained/dinov2-small/provenance.json")
    paths.update("data/pretrained/dinov2-small/" + row["file"] for row in encoder["files"])
    destination.mkdir(parents=True,exist_ok=True)
    target = destination / "export_registration.json"
    if target.exists(): raise FileExistsError("Exporter registration is immutable")
    copy(root / SCRIPT,destination / "frozen_exporter.py")
    result = {"status":"frozen_before_execution","created_utc":datetime.now(timezone.utc).isoformat(),
        "scientific_registration_sha256":EXPECTED_REGISTRY,"finalizer_job":200230,"expected_models":15,"epochs_each":30,
        "dependencies":{name:sha(rooted(root,name)) for name in sorted(paths)},
        "external_licenses":{"/usr/share/common-licenses/Apache-2.0":sha('/usr/share/common-licenses/Apache-2.0')},
        "frozen_exporter_sha256":sha(destination / "frozen_exporter.py"),"tag":TAG,
        "scope":"Original validation development, all five modes/three seeds; no fresh-test or SOTA claim"}
    write(target,result); return result


def inventory(stage, scanner):
    files = {}
    forbidden = {".git",".ssh",".venv","__pycache__",".cache","node_modules"}
    for path in sorted(stage.rglob("*")):
        name = path.relative_to(stage).as_posix(); safe_relative(name)
        if (path.is_symlink() or any(part in forbidden for part in path.relative_to(stage).parts)
                or path.name.startswith(".env") or path.name == "training_state.pt"
                or "credential" in path.name.lower()):
            raise ValueError("Forbidden release path: " + name)
        if path.is_dir(): continue
        if not path.is_file(): raise ValueError("Nonregular release file")
        with path.open("rb") as stream:
            tail = b""
            for chunk in iter(lambda:stream.read(1<<20),b""):
                content = tail + chunk
                for label, rule in scanner.TOKEN_RULES.items():
                    if rule.search(content):
                        raise ValueError("Secret scan finding (value redacted): " + name + ":" + label)
                tail = content[-512:]
        files[name] = {"bytes":path.stat().st_size,"sha256":sha(path)}
    return files


def verify_inventory(bundle, manifest):
    expected = manifest["files"]
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(expected) | {"manifest.json"}:
        raise ValueError("Bundle file inventory differs")
    for path in bundle.rglob("*"):
        if path.is_symlink(): raise ValueError("Bundle contains a symlink")
    for name,row in expected.items():
        path = rooted(bundle,name)
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Bundle bytes differ: " + name)


def readme(report):
    lines = ["# Spatial world-model development checkpoints", "",
        "All 15 original-validation development predictors: five modes × three seeds, each trained for 30 full epochs.",
        "The observation-anchored transport candidate is a new spatial architecture family, not the unchanged original factorized ShiftWM model.",
        "All gains, regressions and inconclusive comparisons are retained. No fresh-test, physical-control, universal-applicability or SOTA result is established.", "",
        "| Mode | Native4×4 h5 | Native4×4 h10 | Original2×2 h5 | Original2×2 h10 |", "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        row=report["aggregate"][mode]
        lines.append(f"| {mode} | {row['native_mse'][4]:.7f} | {row['native_mse'][9]:.7f} | {row['original_2x2_mse'][4]:.7f} | {row['original_2x2_mse'][9]:.7f} |")
    lines += ["", "Errors average windows inside episodes, then equally weight episodes and model seeds. Native4×4 and original2×2 coordinates are different metrics; compare methods within the same column.",
        "Original 2×2 forecasts are pooled and scored against exact original cached targets under original normalization. Full h1–h10 per-window/episode ledgers accompany every model.", "",
        "| Metric | Horizon | Transport comparator | Relative error reduction | Difference [paired95% interval] |", "|---|---:|---|---:|---|"]
    for row in report["paired_effects"]:
        low,high=row["paired_95_percent_interval"]; gain=row["relative_error_reduction_percent"]
        display=f"{gain:+.4f}%" if gain is not None else "undefined"
        lines.append(f"| {row['metric']} | {row['horizon']} | {row['comparator']} | {display} | {row['method_minus_comparator']:+.7f} [{low:+.7f}, {high:+.7f}] |")
    lines += ["", "Positive relative reduction favors transport; negative differences favor transport. Intervals use 10,000 paired session-and-seed bootstrap draws (seed 173), exploratory validation only, without multiplicity adjustment. Packaging recomputes these intervals using the frozen, tested scientific bootstrap helper; it does not provide an independent statistical implementation.", "",
        "## Offline inference", "", "Use Python3.11 and the tested dependencies in `source/requirements.lock.txt`. From this extracted directory:", "", "```python",
        "import sys, numpy as np, torch", 'sys.path.insert(0, "source/src")', "from runtime import load_package, encode_rgb",
        'model, state = load_package("models/transport_s0", "cpu")', 'x = np.load("verification/sample.npz", allow_pickle=False)',
        "with torch.inference_mode():", "    prediction = model.predict(*(torch.from_numpy(x[k])[None] for k in ('support','past','future')))",
        "print(prediction.shape)  # [1,10,6144]", "```", "",
        "Inputs are raw frozen DINO features [B,3,6144], two past [B,2,35] command blocks, and future [B,K,35] command blocks. Each command block groups five chronological recorded 7D Cartesian/gripper commands. Shared-channel training normalization is inside the predictor.",
        "For observed RGB float [...,3,H,W] in [0,1], `encode_rgb(images, 'encoder/dinov2-small')` returns raw channel-major 384×4×4 features. The helper verifies shared pinned encoder weights and uses the registered resize/ImageNet-normalization/pooling order.",
        "The original cache used GPU BF16 encoding. CPU FP32 encoding is not guaranteed bitwise cache-equivalent; cached-input model relocation is checked exactly for every released model.",
        "The included support/actions window is a deterministic original-validation API fixture, not an added benchmark result. It contains no future observation targets. Encoder synthetic-input verification is an engineering check only.",
        "These models output latent features, not RGB video, optical flow, physical correspondence, or robot actions. Attention transport mixes semantic patch features; it is not a validated physical warp.", "",
        "## Training and interpretation", "", "All runs completed 30 epochs; each selected checkpoint minimizes window-weighted FP32 validation MSE over all 10 query steps. All 5 modes share the grid, training recipe and observation budget except explicit context/action ablations.",
        "Transport active capacity is about 1.84% larger than the additive controls; exact counts are in each model card. Transport starts near persistence and additive arms start exactly at persistence. This combined architecture experiment does not isolate transport from bounded innovation.",
        "No original/fresh test data selected these models. Reusing validation outcomes does not create independent confirmatory evidence. All training journals, source identities and complete validation outcomes are included; optimizer/RNG states remain local.", "",
        "## Licenses", "", "Project predictor weights/code use MIT; shared unchanged DINOv2 uses Apache2.0; pinned LeWM source uses MIT; the DROID-derived verification window and provenance retain CC-BY4.0 attribution. Complete notices are in `licenses/`.",
        "Source videos, full datasets, private credentials, installed environments and optimizer states are excluded. Dependency installation may require internet. Verification uses Python isolated mode, a separate physical copy, HF/Transformers offline flags, local-only encoder loading and a Python socket guard; it does not claim OS network-namespace isolation.", ""]
    return "\n".join(lines)


OFFLINE_CHECK = r'''
import json,os,pathlib,socket,sys
root=pathlib.Path(sys.argv[1]).resolve();sys.dont_write_bytecode=True
sys.path[:0]=[str(root),str(root/'source/src')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
attempts=[]
def deny(*args,**kwargs):
 attempts.append(True);raise RuntimeError('Offline verification attempted network access')
socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
import numpy as np,torch
import runtime
import shiftwm.real_video_spatial.model as implementation
assert pathlib.Path(implementation.__file__).resolve().is_relative_to(root)
assert pathlib.Path(runtime.__file__).resolve().is_relative_to(root)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=np.load(root/'verification/sample.npz',allow_pickle=False)
assert set(x.files)=={'support','past','future'}
args=[torch.from_numpy(x[k])[None] for k in ('support','past','future')]
records=[]
with torch.inference_mode():
 for path in sorted((root/'models').iterdir()):
  model,state=runtime.load_package(path,'cpu')
  output=model.predict(*args).numpy()
  expected=np.load(path/'verification_reference.npy',allow_pickle=False)
  assert output.shape==expected.shape==(1,10,6144) and np.isfinite(output).all()
  np.testing.assert_array_equal(output,expected)
  records.append({'name':path.name,'status':'passed','epoch':state['epoch'],'max_abs_error':float(np.max(np.abs(output-expected)))})
 synthetic=runtime.encode_rgb(torch.zeros(1,3,224,224),root/'encoder/dinov2-small','cpu')
 assert synthetic.shape==(1,6144) and torch.isfinite(synthetic).all()
assert not attempts
print(json.dumps({'status':'passed','models':records,'network_attempts':len(attempts),'device':'cpu','network_guard':'HF/Transformers offline flags, local_files_only and Python socket connect/connect_ex/create_connection guard; not OS network isolation','encoder_check':'Bundled weights loaded offline; synthetic zero RGB gives finite1x6144features, not a performance experiment','relocated':True}))
'''


def export(root, destination):
    import numpy as np
    import torch
    os.chdir(root); sys.path.insert(0,str(root/'src'))
    frozen=read(destination/'export_registration.json')
    if sha(__file__)!=frozen['frozen_exporter_sha256'] or frozen['scientific_registration_sha256']!=EXPECTED_REGISTRY:
        raise ValueError('Executing exporter identity differs')
    verify_sources(root,frozen['dependencies'])
    for name,digest in frozen['external_licenses'].items():
        if sha(name)!=digest:raise ValueError('Frozen license text changed')
    campaign=module(root/'scripts/real_video_spatial/campaign.py','spatial_release_campaign')
    registry=campaign.verify(); report=read(root/FINAL)
    validate_gate(report,registry,sha(root/REGISTRY))
    train=campaign.module('train'); validator=campaign.module('validate_ledger')
    scanner=module(root/'scripts/publishing/prepare_public_snapshot.py','spatial_release_scanner')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    evidence={FINAL:sha(root/FINAL)};records=[];rows=[]
    evidence.update(validate_cache_gate(root,read(rooted(root,registry['runs'][0]['config']))))
    finalized={row['name']:row for row in report['runs']}
    old_parity={row['name']:row for row in report['offline_cpu_parity']}
    for row in registry['runs']:
        config=read(rooted(root,row['config']));run=rooted(root,config['output_dir'])
        summary=train.validate_completed(run)
        executed=read(run/'training_config.json')
        if train.base.scientific_config(executed)!=train.base.scientific_config(config):
            raise ValueError('Registered/executed spatial configurations differ')
        _,state=train.read_package(run/'best')
        result_path=rooted(root,finalized[row['name']]['validation']); result=read(result_path)
        result['summary']=validator.validate_ledger(result,config,root,state)
        if (summary['best_epoch']!=finalized[row['name']]['selected_epoch']
                or summary['parameter_counts']!=finalized[row['name']]['parameter_counts']):
            raise ValueError('Finalized model selection/counts differ')
        local=root/'artifacts/releases/spatial_v1/models'/row['name']
        if local.is_symlink() or any(p.is_symlink() for p in local.rglob('*')):
            raise ValueError('Local finalized export must be independent regular files')
        manifest=read(local/'package_manifest.json')
        if (set(manifest['files'])!={'model.pt','config.json'}
                or sha(local/'package_manifest.json')!=old_parity[row['name']]['package_manifest_sha256']):
            raise ValueError('Local inference export differs from finalizer parity')
        for name in ('model.pt','config.json'):
            if sha(local/name)!=manifest['files'][name] or sha(local/name)!=sha(run/'best'/name):
                raise ValueError('Finalized inference export differs from selected source')
        tracked=[result_path,run/'training_summary.json',run/'training_config.json',run/'metrics.jsonl',
                 run/'best/model.pt',run/'best/config.json',run/'best/package_manifest.json',
                 run/'last/model.pt',run/'last/config.json',run/'last/package_manifest.json',run/'last/training_state.pt',
                 local/'model.pt',local/'config.json',local/'package_manifest.json']
        evidence.update({str(path.relative_to(root)):sha(path) for path in tracked})
        records.append({'row':row,'config':config,'run':run,'summary':summary,'state':state,'local':local,'result':result})
        rows.append((row,config,summary,result))
    for mode in MODES:
        selected=[r['result']['summary'] for r in records if r['row']['mode']==mode]
        for key in METRICS:
            np.testing.assert_allclose(np.mean([r[key] for r in selected],axis=0),report['aggregate'][mode][key],rtol=1e-12,atol=1e-12)
    recomputed=campaign.paired_effects(rows)
    if len(recomputed)!=16:raise ValueError('Paired comparison recomputation incomplete')
    for actual,expected in zip(recomputed,report['paired_effects']):
        if set(actual)!=set(expected):raise ValueError('Paired comparison schema differs')
        for key in actual:
            if key=='paired_95_percent_interval':np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
            elif isinstance(actual[key],float):
                if not math.isclose(actual[key],expected[key],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Paired evidence arithmetic differs')
            elif actual[key]!=expected[key]:raise ValueError('Paired evidence identity differs')
    bundle=destination/BUNDLE
    if bundle.exists():
        manifest=read(bundle/'manifest.json')
        if (manifest['finalization_sha256']!=evidence[FINAL] or manifest['scientific_registration_sha256']!=EXPECTED_REGISTRY
                or manifest['export_registration_sha256']!=sha(destination/'export_registration.json') or manifest['evidence']!=evidence):
            raise ValueError('Existing release scientific/export identity differs')
        verify_inventory(bundle,manifest);return bundle,manifest
    with tempfile.TemporaryDirectory(prefix='.spatial-export-',dir=destination) as tmp:
        stage=Path(tmp)/BUNDLE;stage.mkdir()
        source_names={name for name in frozen['dependencies'] if name.startswith(('src/','scripts/','tests/','reports/'))}
        source_names.update(('LICENSE','requirements.lock.txt','pyproject.toml','references/real_dinov2_sources.json'))
        for name in sorted(source_names):copy(rooted(root,name),stage/'source'/name,frozen['dependencies'][name])
        copy(root/RUNTIME,stage/'runtime.py',frozen['dependencies'][RUNTIME])
        copy(destination/'export_registration.json',stage/'provenance/export_registration.json')
        copy(root/REGISTRY,stage/'provenance/scientific_registration.json',EXPECTED_REGISTRY)
        copy(root/FINAL,stage/'reports/finalization.json',evidence[FINAL])
        for name in ('cache_and_budget_gate.json','throughput.json'):
            path='reports/real_video_spatial/'+name
            copy(root/path,stage/'reports'/name,evidence[path])
        for row in registry['runs']:copy(root/row['config'],stage/'provenance/configs'/Path(row['config']).name,row['sha256'])
        first=records[0];cache=rooted(root,first['config']['cache_root']);cache_manifest=read(cache/'manifest.json')
        provenance=read(root/'data/pretrained/dinov2-small/provenance.json')
        if cache_manifest['identity']['encoder']!=provenance:raise ValueError('Shared encoder/cache identity differs')
        for item in provenance['files']:
            name=safe_relative(item['file'])
            if len(name.parts)!=1:raise ValueError('Unexpected encoder subpath')
            copy(root/'data/pretrained/dinov2-small'/name,stage/'encoder/dinov2-small'/name,item['sha256'])
        copy(root/'data/pretrained/dinov2-small/provenance.json',stage/'encoder/dinov2-small/provenance.json')
        for source,target in [('LICENSE','SHIFTWM-MIT.txt'),('src/shiftwm/vendor/lewm/LICENSE','LEWM-MIT.txt'),('site/assets/DROID-LICENSE.txt','DROID-CC-BY-4.0.txt')]:
            copy(root/source,stage/'licenses'/target,frozen['dependencies'][source])
        copy('/usr/share/common-licenses/Apache-2.0',stage/'licenses/DINOV2-APACHE-2.0.txt',frozen['external_licenses']['/usr/share/common-licenses/Apache-2.0'])
        (stage/'licenses/THIRD_PARTY_NOTICES.md').write_text('Project-authored predictor weights/code: MIT, ShiftWM contributors. DINOv2-small: Meta/Facebook, Apache2.0, pinned original model card and revision in encoder/dinov2-small. LeWorldModel: Lucas Maes et al., MIT, pinned source and NOTICE in source/src/shiftwm/vendor/lewm. DROID: Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024), https://droid-dataset.github.io/ , CC-BY4.0. The derived support/action verification window and data provenance retain DROID attribution; no raw videos/full datasets are included. Upstream licenses remain unchanged.\n')
        for name in ('manifest.json','training_statistics.json','identity.json'):
            copy(cache/name,stage/'provenance/spatial_cache'/name,evidence[str((cache/name).relative_to(root))])
        for name in ('data/features/droid_selected_v1/manifest.json','data/features/droid_selected_v1/training_statistics.json','data/real_video/droid_selected/processed/data_audit.json'):
            copy(root/name,stage/'provenance/frozen_inputs'/name,frozen['dependencies'][name])
        from shiftwm.real_video_spatial.data import SpatialDataset
        dataset=SpatialDataset(cache,'val',horizon=10,stride=5,verify=True);sample=dataset[0]
        episode=dataset.episodes[sample['episode_index']]
        (stage/'verification').mkdir()
        np.savez_compressed(stage/'verification/sample.npz',support=sample['features'][:3].numpy(),past=sample['actions'][:2].numpy(),future=sample['actions'][2:].numpy())
        write(stage/'verification/sample_provenance.json',{'episode_id':episode['episode_id'],'session_id':episode['session_id'],'split':'original_validation',
            'window_start':sample['window_start'],'selection':'first eligible window in immutable manifest order; parity only',
            'source_payload':episode['cameras']['exterior_image_1_left'],'source_cache_manifest_sha256':sha(cache/'manifest.json'),
            'future_observation_targets_included':False})
        models=[]
        for record in records:
            row,state=record['row'],record['state'];target=stage/'models'/row['name']
            for name in ('model.pt','config.json','package_manifest.json'):copy(record['local']/name,target/name)
            model,_=train.load_package(record['local'],'cpu')
            with torch.inference_mode():prediction=model.predict(sample['features'][:3][None],sample['actions'][:2][None],sample['actions'][2:][None]).numpy()
            if prediction.shape!=(1,10,6144) or not np.isfinite(prediction).all():raise ValueError('Source forecast invalid')
            np.save(target/'verification_reference.npy',prediction)
            card=f"# {row['name']}\n\nMode {row['mode']}; seed {row['seed']}. All30epochs completed; selected epoch{state['epoch']} minimizes window-weighted all10query FP32 validation MSE {state['best_metric']:.10g}. This is validation selection, not a fresh-test score.\n\nInputs: raw DINO support[B,3,6144], past[B,2,35], future[B,K,35]; output[B,K,6144] feature forecasts. No RGB generation, physical patch correspondence, executed robot policy, or clinical result. See ../../README.md for all15 models, all comparisons/intervals and approximate capacity/initialization differences.\n\nParameters: {json.dumps(record['summary']['parameter_counts'],sort_keys=True)}. Training identity: {record['summary']['training_identity']}. Selected source SHA256: {sha(record['local']/'model.pt')}. Project weights MIT; shared DINO Apache2.0; LeWM source MIT; DROID-derived verification data CC-BY4.0. Optimizer/RNG omitted from this inference release.\n"
            (target/'MODEL_CARD.md').write_text(card)
            copy(root/finalized[row['name']]['validation'],stage/'reports/evaluations'/(row['name']+'.json'))
            for name in ('training_summary.json','training_config.json','metrics.jsonl'):copy(record['run']/name,stage/'reports/training'/row['name']/name)
            models.append({'name':row['name'],'mode':row['mode'],'seed':row['seed'],'completed_epochs':30,'selected_epoch':state['epoch'],
                'path':target.relative_to(stage).as_posix(),'checkpoint_sha256':sha(target/'model.pt'),'parameter_counts':record['summary']['parameter_counts']})
            del model
        # A separate physical copy under a new temporary root tests relocation,
        # not merely importing bundled source from the original stage path.
        with tempfile.TemporaryDirectory(prefix='spatial-relocated-') as relocated:
            relocated=Path(relocated)/'release';shutil.copytree(stage,relocated)
            environment={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
            environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            completed=subprocess.run([sys.executable,'-I','-c',OFFLINE_CHECK,str(relocated)],cwd=relocated,env=environment,
                capture_output=True,text=True,check=True,timeout=1800)
            verification=json.loads(completed.stdout.strip().splitlines()[-1])
        validate_relocated_proof(verification,{r['name']:r['selected_epoch'] for r in models})
        write(stage/'verification/offline_packages.json',verification)
        write(stage/'provenance/evidence_hashes.json',evidence)
        (stage/'README.md').write_text(readme(report));(stage/'MODEL_CARD.md').write_text(readme(report))
        files=inventory(stage,scanner)
        verify_sources(root,frozen['dependencies']);verify_sources(root,evidence)
        manifest={'status':'completed','kind':'real_droid_spatial_development_inference','scientific_registration_sha256':EXPECTED_REGISTRY,
            'export_registration_sha256':sha(destination/'export_registration.json'),'finalization_sha256':evidence[FINAL],
            'models':models,'validation_ledgers':15,'paired_comparisons':16,'offline_verification':verification,
            'scope':'original validation development only; all outcomes included; no fresh-test or SOTA claim',
            'secret_findings':[],'evidence':evidence,'files':files,'total_bytes':sum(r['bytes'] for r in files.values())}
        write(stage/'manifest.json',manifest);stage.rename(bundle)
    return bundle,manifest


def archive(bundle, manifest, destination):
    verify_inventory(bundle,manifest)
    path=destination/(BUNDLE+'.tar.gz')
    if not path.exists():
        temporary=path.with_suffix('.partial')
        with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=4) as zipped,tarfile.open(fileobj=zipped,mode='w|') as output:
            for filename in sorted(set(manifest['files'])|{'manifest.json'}):
                source=rooted(bundle,filename);info=output.gettarinfo(str(source),arcname=BUNDLE+'/'+filename)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644;info.pax_headers={}
                with source.open('rb') as stream:output.addfile(info,stream)
        temporary.replace(path)
    seen=set()
    with tarfile.open(path,'r|gz') as source:
        for item in source:
            safe_relative(item.name);parts=PurePosixPath(item.name).parts
            if not item.isfile() or parts[0]!=BUNDLE or len(parts)<2:raise ValueError('Unsafe archive member')
            name='/'.join(parts[1:])
            if name in seen or name not in set(manifest['files'])|{'manifest.json'}:raise ValueError('Unexpected/duplicate archive member')
            with source.extractfile(item) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            expected=sha(bundle/'manifest.json') if name=='manifest.json' else manifest['files'][name]['sha256']
            if digest!=expected:raise ValueError('Archive member bytes differ')
            seen.add(name)
    if seen!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Archive inventory incomplete')
    for name in ('README.md','MODEL_CARD.md','manifest.json'):copy(bundle/name,destination/name)
    copy(bundle/'verification/offline_packages.json',destination/'offline_packages.json')
    names=[path.name,'README.md','MODEL_CARD.md','manifest.json','offline_packages.json']
    (destination/'SHA256SUMS').write_text(''.join(f'{sha(destination/name)}  {name}\n' for name in names))
    return names+['SHA256SUMS']


def publish(destination, names, credential_file):
    import requests
    if credential_file.stat().st_mode & 0o077:raise ValueError('Credential store must be private')
    credentials=[urlparse(line) for line in credential_file.read_text().splitlines() if urlparse(line).hostname=='github.com']
    if len(credentials)!=1 or not credentials[0].password:raise ValueError('Expected one private GitHub credential')
    client=requests.Session();client.headers.update({'Authorization':'Bearer '+unquote(credentials[0].password),
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'ShiftWM-spatial-release'})
    api='https://api.github.com/repos/aj-das-research/WM-ICLR'
    response=client.get(api+'/releases/tags/'+TAG,timeout=30)
    if response.status_code==404:
        commit=client.get(api+'/commits/main',timeout=30);commit.raise_for_status()
        body='All15 original-validation spatial development models (five modes×three seeds), each trained30epochs. Includes every positive/negative/inconclusive outcome,16 matched paired comparisons, complete window ledgers, shared pinned DINO encoder, exact inference source, licenses and checksums. Every public model passed fresh relocated offline CPU forecast parity. Models predict latent features, not RGB, physical correspondences or robot actions. No fresh-test, clinical or SOTA claim. Research prerelease; consult attached model card for capacity/initialization differences and exploratory-validation limits. Project predictor weights/code MIT; DINO Apache2.0; LeWM MIT; DROID-derived provenance/verification data CC-BY4.0. The archive preserves its exact source identity independently of the tagged main commit.'
        response=client.post(api+'/releases',json={'tag_name':TAG,'target_commitish':commit.json()['sha'],
            'name':'Spatial v1: all15 validation-development world models','body':body,'draft':True,'prerelease':True},timeout=30)
    response.raise_for_status();release=response.json()
    if release.get('prerelease') is not True:raise ValueError('Existing tag is not the expected prerelease')
    known={row['name']:row for row in release['assets']}
    if len(known)!=len(release['assets']):raise ValueError('Existing release has duplicate asset names')
    if set(known)-set(names):raise ValueError('Existing release contains unexpected assets')
    uploaded=[]
    for name in names:
        safe_relative(name)
        if len(Path(name).parts)!=1:raise ValueError('Release assets must be root files')
        path=destination/name;expected=sha(path)
        if name in known:asset=known[name]
        else:
            if not release['draft']:raise ValueError('Published prerelease is immutable')
            url=release['upload_url'].split('{',1)[0]
            if urlparse(url).hostname!='uploads.github.com':raise ValueError('Unexpected upload host')
            with path.open('rb') as stream:response=client.post(url,params={'name':name},data=stream,headers={'Content-Type':'application/octet-stream'},timeout=(30,1200))
            response.raise_for_status();asset=response.json()
        if asset['size']!=path.stat().st_size or asset.get('digest')!='sha256:'+expected:
            raise ValueError('Remote asset digest/size mismatch')
        uploaded.append({'name':name,'id':asset['id'],'bytes':asset['size'],'sha256':expected,'server_digest_verified':True})
        print(json.dumps({'asset_verified':name,'bytes':asset['size']}),flush=True)
    if release['draft']:
        response=client.patch(api+'/releases/'+str(release['id']),json={'draft':False,'prerelease':True},timeout=30)
        response.raise_for_status();release=response.json()
    assets={row['id']:row for row in release['assets']}
    for row in uploaded:
        row['url']=assets[row['id']]['browser_download_url'];digest=hashlib.sha256();size=0
        with requests.get(row['url'],stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4<<20):digest.update(chunk);size+=len(chunk)
        if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Anonymous public download differs')
        row['public_download_verified']=True
    return {'status':'published_verified','release_url':release['html_url'],'release_id':release['id'],
            'tag':TAG,'target_commit':release['target_commitish'],'assets':uploaded,'prerelease':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--directory',type=Path,default=Path('artifacts/publishing/spatial_v1'))
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--publish',action='store_true')
    parser.add_argument('--credential-file',type=Path)
    args=parser.parse_args();root=args.root.resolve();destination=args.directory if args.directory.is_absolute() else root/args.directory
    if args.freeze:
        freeze(root,destination);print(json.dumps({'status':'exporter_frozen','directory':str(destination)}));return
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bundle,manifest=export(root,destination);names=archive(bundle,manifest,destination)
        record={'status':'verified_local','models':15,'validation_ledgers':15,'paired_comparisons':16,
            'offline_parity_passes':15,'archive_sha256':sha(destination/names[0]),'archive_bytes':(destination/names[0]).stat().st_size,
            'export_registration_sha256':sha(destination/'export_registration.json'),'scientific_registration_sha256':EXPECTED_REGISTRY,
            'finalization_sha256':manifest['finalization_sha256'],'secret_findings':0,
            'scope':'All matched original-validation development outcomes; no fresh-test or SOTA claim'}
        if args.publish:
            if args.credential_file is None:raise ValueError('Private credential-store path required to publish')
            record.update(publish(destination,names,args.credential_file))
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        write(destination/'publication_receipt.json',record);write(root/'reports/spatial_checkpoint_publication_status.json',record)
        print(json.dumps({k:record.get(k) for k in ('status','models','archive_bytes','release_url')}),flush=True)


if __name__=='__main__':main()
