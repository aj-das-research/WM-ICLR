#!/usr/bin/env python3
"""Publish all twelve completed h10 development models after independent gates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

TAG = "horizon10-development-v1"
BUNDLE = "shiftwm-horizon10-v1"
REGISTRY = "configs/real_video_development/horizon10_v1/registration.json"
REGISTRY_SHA = "91e9d9dbe269362e0b19364571216310b5b3710b8754f8fde1340814f278469c"
SCRIPT = "scripts/publishing/publish_horizon10_release.py"
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
EXPECTED = {f"{mode}_s{seed}" for mode in MODES for seed in range(3)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def verify_proof(proof):
    rows = proof.get("models", [])
    if (proof.get("status") != "passed" or proof.get("network_attempts") != 0 or proof.get("device") != "cpu"
            or proof.get("physical_relocation") is not True or len(rows) != 12
            or {r.get("run") for r in rows} != EXPECTED
            or any(r.get("epoch") != 1 or r.get("maximum_absolute_difference") != 0
                   or r.get("shape") != [1,10,1536] for r in rows)):
        raise ValueError("All 12 physically relocated h10 models must pass exact CPU parity")


OFFLINE = r'''
import hashlib,json,os,pathlib,socket,sys
root=pathlib.Path(sys.argv[1]).resolve();sys.dont_write_bytecode=True
sys.path[:0]=[str(root/'source/src'),str(root/'source/scripts/real_video_development')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
attempts=[]
def deny(*args,**kwargs):
 attempts.append(True);raise RuntimeError('Offline inference attempted network access')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
import numpy as np,torch
import horizon10_train as loader
import shiftwm.real_video.model as model_source
assert pathlib.Path(loader.__file__).resolve().is_relative_to(root)
assert pathlib.Path(model_source.__file__).resolve().is_relative_to(root)
assert loader.PACKAGE_KIND=='shiftwm_real_video_droid_horizon10_v1'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=np.load(root/'verification/sample.npz',allow_pickle=False)
assert set(x.files)=={'features','actions'} and x['features'].shape==(3,1536) and x['actions'].shape==(12,35)
features=torch.from_numpy(x['features'])[None].float();actions=torch.from_numpy(x['actions'])[None].float()
rows=[]
with torch.inference_mode():
 for path in sorted((root/'models').iterdir()):
  model,state=loader.load_package(path,'cpu')
  expected=np.load(path/'verification_reference.npy',allow_pickle=False)
  actual=model.predict(features,actions[:,:2],actions[:,2:]).numpy()
  assert actual.shape==expected.shape==(1,10,1536) and np.isfinite(actual).all()
  np.testing.assert_array_equal(actual,expected)
  rows.append({'run':path.name,'epoch':state['epoch'],'shape':list(actual.shape),'maximum_absolute_difference':float(np.max(np.abs(actual-expected)))})
assert not attempts
print(json.dumps({'status':'passed','models':rows,'network_attempts':len(attempts),'physical_relocation':True,'device':'cpu',
 'scope':'Genuine original-validation support fixture; exact cached-feature parity, not a benchmark or RGB-cache-equivalence test',
 'network_guard':'Python socket guard, isolated Python, and HF offline flags; not OS network namespace isolation'}))
'''


def readme(report_text):
    return """# Horizon-ten development: all 12 reusable predictors

Four methods × three seeds, each trained for all 30 epochs. **All twelve validation-selected
checkpoints are from epoch 1**, not epoch 30. This release preserves those actual selected weights.
It is a matched training-horizon control using the existing 192-wide, four-layer architecture,
not a new architectural invention or a state-of-the-art benchmark claim.

The training target and matching validation-selection objective change together from five to ten
future blocks. Window weighting is unchanged; equal-episode evaluation remains separate from selection.
All 24 standard evaluations, 36 matched diagnostics and 20 comparisons are retained, including
regressions and intervals that include zero. These are original-validation development results;
neither original test nor the separate fresh holdout selects these models.

## Offline inference

Install the pinned dependencies in `source/requirements.lock.txt`. From this extracted directory:

```python
import sys, numpy as np, torch
sys.path[:0] = ["source/src", "source/scripts/real_video_development"]
from horizon10_train import load_package
model, state = load_package("models/factorized_s0", "cpu")
with np.load("verification/sample.npz", allow_pickle=False) as sample:
    history = torch.from_numpy(sample["features"].copy())[None].float()
    actions = torch.from_numpy(sample["actions"].copy())[None].float()
with torch.inference_mode():
    prediction = model.predict(history, actions[:, :2], actions[:, 2:])
print(prediction.shape)  # [1,10,1536]
```

Inputs: raw frozen DINO support[B,3,1536], past commands[B,2,35], future commands[B,K,35].
Each block groups five chronological recorded 7D Cartesian/gripper command vectors.
Outputs are latent image features, **not RGB video, a robot policy or physical state**.
Use the separately bundled shared DINOv2-small encoder and the exact resize/normalization/2×2
pooling implementation in `source/src/shiftwm/real_video/features.py` for observed images.
Historical GPU-BF16 cache extraction is not promised bitwise equivalent to CPU-FP32 image encoding.
Normalization and predictor weights load offline without following historical training paths.
The h10 package has its own kind and must use `horizon10_train.load_package`, not the original h5 loader.

All twelve copied model/config files are byte-identical to their selected source generations.
Each passed a new physical relocation and exact same-input CPU prediction check with bundled source
and a Python network guard. The included genuine validation support/action fixture contains no future
observation targets and is engineering evidence only. Optimizer/RNG files remain in the original runs.

## Scope, positive and negative outcomes

ShiftWM (ours) has 0.405% lower h10 endpoint error and 0.252% lower all-ten mean error than the equally
h10-trained Framewise control; these are small validation effects. Against its original h5-trained
counterpart it improves all-ten mean error by 1.058%, but the matched h5-prefix mean has a -0.116%
point change with an interval including zero. Read every comparison below; do not select only favorable cells.
The h5 standard population and h5 prefixes of h10-eligible windows differ, so their means are not interchangeable.
There is no new independent-test, closed-loop control, real surgical, clinical or SOTA conclusion.

## Licenses

Project-authored code and predictor weights use the included MIT license. Unchanged DINOv2 remains
Apache2.0, LeWM source MIT, and DROID-derived verification data/provenance CC-BY4.0 with full attribution.
The shared encoder is stored once. Raw videos, full datasets, private credentials and optimizer/RNG states
are excluded. Exact source hashes, model cards, 30-epoch journals and all 60 evaluation ledgers accompany the release.
The comparison recheck uses the independently implemented audited paper postprocessor; no model reruns or new selection occur.

""" + report_text


def export(root, destination):
    import numpy as np
    import torch
    os.chdir(root);sys.path.insert(0,str(root/'src'))
    if sha(root/REGISTRY)!=REGISTRY_SHA:raise ValueError('Unexpected frozen h10 registration')
    paper=module(root/'scripts/real_video_development/finalize_horizon10_paper.py','h10_release_audit')
    old=module(root/'scripts/real_video/finalize_campaign.py','h10_original_export_helpers')
    transport=module(root/'scripts/publishing/spatial_release.py','h10_archive_helpers')
    scanner=module(root/'scripts/publishing/prepare_public_snapshot.py','h10_secret_scanner')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    audited,official,evidence,populations=paper.collect()
    native,comparisons=paper.aggregate(audited,official)
    if (len(audited)!=12 or {r['name'] for r in audited}!=EXPECTED or len(comparisons)!=20
            or any(r['training']['best_epoch']!=1 or r['training']['completed_epochs']!=30 for r in audited)):
        raise ValueError('Completed registered population/selection differs')
    # Rebind only this imported exporter instance, never any scientific file or
    # trainer globals: the h10 storage contract is intentionally distinct.
    old.train=paper.trainer
    registry=read(root/REGISTRY);records=[]
    by_name={r['name']:r for r in audited}
    for row in registry['runs']:
        config=read(root/row['config']);run=root/config['output_dir'];selected=read(run/'best/config.json')
        records.append({'row':row,'config_path':root/row['config'],'config':config,'run':run,
            'summary':by_name[row['name']]['training'],'identity':selected['metadata']['identity']})
    bundle=destination/BUNDLE
    helpers=[SCRIPT,'scripts/publishing/spatial_release.py','scripts/real_video/finalize_campaign.py',
             'scripts/publishing/prepare_public_snapshot.py','tests/test_horizon10_release.py']
    frozen={name:sha(root/name) for name in helpers}
    if bundle.exists():
        manifest=read(bundle/'manifest.json')
        if (manifest.get('registration_sha256')!=REGISTRY_SHA or manifest.get('evidence')!=evidence
                or manifest.get('export_sources')!=frozen):raise ValueError('Existing h10 release identity changed')
        transport.verify_inventory(bundle,manifest);verify_proof(manifest['offline_verification'])
        return bundle,manifest,transport
    with tempfile.TemporaryDirectory(prefix='.h10-export-',dir=destination) as tmp:
        stage=Path(tmp)/BUNDLE;stage.mkdir()
        old.source_bundle(stage,root/REGISTRY,records)
        cache=root/records[0]['config']['cache_root'];cache_manifest=read(cache/'manifest.json')
        old.encoder_bundle(stage,root/'data/pretrained/dinov2-small',cache_manifest['identity']['encoder'])
        old.checked_copy(root/'site/assets/DROID-LICENSE.txt',stage/'licenses/DROID-CC-BY-4.0.txt')
        for name in set(helpers)|{n for n in evidence if n.startswith(('src/','scripts/','tests/','reports/'))}:
            old.checked_copy(root/name,stage/'source'/name,evidence.get(name,frozen.get(name)))
        old.checked_copy(root/'references/real_dinov2_sources.json',stage/'source/references/real_dinov2_sources.json')
        for name in ('real_droid_horizon10_results.json','real_droid_horizon10_results.md'):
            old.checked_copy(root/'reports'/name,stage/'reports'/name)
        data=old.RealVideoDataset(cache,'val',horizon=10,stride=5,verify=True);sample=data[0]
        episode=data.episodes[sample['episode_index']]
        sample={**sample,'features':sample['features'][:3]}
        (stage/'verification').mkdir()
        np.savez_compressed(stage/'verification/sample.npz',features=sample['features'].numpy(),actions=sample['actions'].numpy())
        write(stage/'verification/sample_provenance.json',{'episode_id':episode['episode_id'],'session_id':episode['session_id'],
            'split':'original_validation','window_start':sample['window_start'],'source_payload':episode['cameras']['exterior_image_1_left'],
            'future_observation_targets_included':False,'purpose':'First eligible genuine validation support; engineering portability only'})
        models=[]
        for record in records:
            exported=old.export_model(record,stage,sample)
            if exported['checkpoint_epoch']!=1:raise ValueError('Unexpected selected epoch')
            for name in ('model.pt','config.json'):
                if sha(stage/exported['path']/name)!=sha(record['run']/'best'/name):raise ValueError('Selected source bytes differ')
            label='ShiftWM (ours)' if exported['mode']=='factorized' else exported['mode']
            card=f"# {exported['run']}\n\nMethod {label}; seed {exported['seed']}. All 30 epochs completed; validation selected epoch **1**, using window-weighted all-ten recursive standardized feature MSE {exported['validation_mse']:.10g}.\n\n"
            card+="Inputs: support[B,3,1536], past[B,2,35], future[B,K,35]; output[B,K,1536]. Frozen DINO feature forecasts, not RGB generation or robot actions. Original-validation development only; training and selection horizon change together. All comparisons, including adverse h5-prefix effects, are preserved. No new test, clinical or SOTA claim.\n\n"
            card+=f"Parameters: {json.dumps(exported['parameter_counts'])}. Source model SHA256: {exported['checkpoint_sha256']}. Use horizon10_train.load_package; the original h5 loader rejects this distinct package kind. Optimizer/RNG omitted. See ../../README.md and ../../verification/offline_packages.json. Project predictor MIT; DINO Apache2.0; LeWM MIT; DROID-derived fixture CC-BY4.0.\n"
            (stage/exported['path']/'MODEL_CARD.md').write_text(card)
            for key in paper.EVALUATIONS:
                old.checked_copy(record['run']/(key+'.json'),stage/'reports/evaluations'/exported['run']/(key+'.json'),evidence[str((record['run']/(key+'.json')).relative_to(root))])
            models.append(exported)
        with tempfile.TemporaryDirectory(prefix='h10-relocated-') as relocated:
            relocated=Path(relocated)/'release';shutil.copytree(stage,relocated)
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
            env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            run=subprocess.run([sys.executable,'-I','-B','-c',OFFLINE,str(relocated)],cwd=relocated,env=env,
                capture_output=True,text=True,timeout=1200,check=True)
            proof=json.loads(run.stdout.strip().splitlines()[-1])
        verify_proof(proof);write(stage/'verification/offline_packages.json',proof)
        write(stage/'provenance/evidence_hashes.json',evidence)
        write(stage/'reports/independent_comparison_audit.json',{'native':native,'comparisons':comparisons,'populations':populations})
        text=readme((root/'reports/real_droid_horizon10_results.md').read_text())
        (stage/'README.md').write_text(text);(stage/'MODEL_CARD.md').write_text(text)
        notice=stage/'licenses/THIRD_PARTY_NOTICES.md'
        notice.write_text(notice.read_text().replace('This local bundle does not assert a separate license grant for learned weights.',
            'Project-authored predictor weights are distributed under the included project MIT license.'))
        files=transport.inventory(stage,scanner)
        for name,expected in {**evidence,**frozen}.items():
            if sha(root/name)!=expected:raise ValueError('Bound source changed during export: '+name)
        manifest={'status':'completed','kind':'real_droid_horizon10_development_inference',
            'registration_sha256':REGISTRY_SHA,'models':models,'epochs_each':30,'selected_epoch_counts':{'1':12},
            'evaluation_ledgers':60,'comparisons':20,'offline_verification':proof,'evidence':evidence,
            'export_sources':frozen,'secret_findings':[],'files':files,'total_bytes':sum(x['bytes'] for x in files.values()),
            'scope':'Original validation development; all positive/negative/inconclusive outcomes, no new test or SOTA claim'}
        write(stage/'manifest.json',manifest);stage.rename(bundle)
    return bundle,manifest,transport


def publish(destination,names,credential_file):
    import requests
    if credential_file.stat().st_mode&0o077:raise ValueError('Credential store must be private')
    values=[urlparse(line) for line in credential_file.read_text().splitlines() if urlparse(line).hostname=='github.com']
    if len(values)!=1 or not values[0].password:raise ValueError('Expected one private GitHub credential')
    client=requests.Session();client.headers.update({'Authorization':'Bearer '+unquote(values[0].password),
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'ShiftWM-h10-research-release'})
    api='https://api.github.com/repos/aj-das-research/WM-ICLR';response=client.get(api+'/releases/tags/'+TAG,timeout=30)
    if response.status_code==404:
        commit=client.get(api+'/commits/main',timeout=30);commit.raise_for_status()
        body='\n\n'.join(['All 12 completed h10-trained predictors: four matched methods × three seeds. Every run completed 30 epochs; validation selected epoch 1 for all 12. These are genuine selected weights, not epoch-30 checkpoints.',
            'Matched horizon control with the existing architecture. Training and validation-selection horizons change together; weighting stays fixed. All 60 saved validation ledgers and 20 comparisons, including negative and inconclusive outcomes, accompany the release.',
            'All 12 models/configs match selected source bytes and passed new physically relocated offline CPU forecast parity. Shared DINO encoder, exact source, model cards, full licenses, training journals and hashes are included. No optimizer/RNG, raw video/full dataset or credentials.',
            'Small validation effects: ours reduces h10 endpoint error by 0.405% and all-ten mean by 0.252% versus equally h10-trained Framewise. Its matched h5-prefix mean change versus original h5 training is −0.116% with an interval including zero. The standard h5 and h10-eligible-prefix populations differ; read all 20 contrasts.',
            'These are latent-feature forecasts, not RGB generation, robot policies or clinical/physical-control results. Original-validation development only; no new test or SOTA claim. Project predictors MIT; DINO Apache2.0; LeWM MIT; DROID-derived fixture/provenance CC-BY4.0. Research prerelease; exact archive source hashes are authoritative.'])
        response=client.post(api+'/releases',json={'tag_name':TAG,'target_commitish':commit.json()['sha'],
            'name':'Horizon10 development v1: all 12 selected predictors','body':body,'draft':True,'prerelease':True},timeout=30)
    response.raise_for_status();release=response.json()
    if release.get('prerelease') is not True:raise ValueError('Existing release is not a prerelease')
    known={a['name']:a for a in release['assets']}
    if len(known)!=len(release['assets']) or set(known)-set(names):raise ValueError('Unexpected or duplicate remote assets')
    uploaded=[]
    for name in names:
        path=destination/name;expected=sha(path)
        if name in known:asset=known[name]
        else:
            if not release['draft']:raise ValueError('Published releases are immutable')
            url=release['upload_url'].split('{',1)[0]
            if urlparse(url).hostname!='uploads.github.com':raise ValueError('Unexpected upload host')
            with path.open('rb') as stream:response=client.post(url,params={'name':name},data=stream,headers={'Content-Type':'application/octet-stream'},timeout=(30,1200))
            response.raise_for_status();asset=response.json()
        if asset['size']!=path.stat().st_size or asset.get('digest')!='sha256:'+expected:raise ValueError('Remote digest/size mismatch')
        uploaded.append({'name':name,'id':asset['id'],'bytes':asset['size'],'sha256':expected,'server_digest_verified':True})
        print(json.dumps({'uploaded_verified':name,'bytes':asset['size']}),flush=True)
    if release['draft']:
        response=client.patch(api+'/releases/'+str(release['id']),json={'draft':False,'prerelease':True},timeout=30)
        response.raise_for_status();release=response.json()
    known={a['id']:a for a in release['assets']}
    for row in uploaded:
        row['url']=known[row['id']]['browser_download_url'];digest=hashlib.sha256();size=0
        with requests.get(row['url'],stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4<<20):digest.update(chunk);size+=len(chunk)
        if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Anonymous download differs')
        row['public_download_verified']=True
    return {'status':'published_verified','release_url':release['html_url'],'release_id':release['id'],
        'tag':TAG,'target_commit':release['target_commitish'],'assets':uploaded,'prerelease':True}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--directory',type=Path,default=Path('artifacts/publishing/horizon10-release'))
    parser.add_argument('--publish',action='store_true');parser.add_argument('--credential-file',type=Path)
    args=parser.parse_args();root=args.root.resolve();destination=args.directory if args.directory.is_absolute() else root/args.directory
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bundle,manifest,transport=export(root,destination);transport.BUNDLE=BUNDLE
        names=transport.archive(bundle,manifest,destination)
        if (destination/names[0]).stat().st_size>=1_900_000_000:raise ValueError('GitHub archive size cap exceeded')
        record={'status':'verified_local','models':12,'epochs_each':30,'selected_epoch_counts':{'1':12},
            'evaluation_ledgers':60,'comparisons':20,'offline_parity_passes':12,'registration_sha256':REGISTRY_SHA,
            'archive_sha256':sha(destination/names[0]),'archive_bytes':(destination/names[0]).stat().st_size,'secret_findings':0,
            'scope':'Original validation development only; training and selection horizons change together; all outcomes retained'}
        if args.publish:
            if not args.credential_file:raise ValueError('Publishing requires a private credential-file path')
            record.update(publish(destination,names,args.credential_file))
        record['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        write(destination/'publication_receipt.json',record);write(root/'reports/horizon10_checkpoint_publication_status.json',record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
