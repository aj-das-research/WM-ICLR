#!/usr/bin/env python3
"""Registered matched horizon-ten control, using original train/validation only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/real_video'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import horizon10_train as training
import train as original_training
import evaluate as evaluation
from shiftwm.real_video.data import RealVideoDataset

CONFIGS = ROOT / 'configs/real_video_development/horizon10_v1'
REGISTRY = CONFIGS / 'registration.json'
PROTOCOL = 'reports/real_droid_horizon10_protocol.md'
MODES = ('framewise', 'constant_dynamics', 'factorized', 'action_free')
ARCHITECTURE = {'hidden_dim': 192, 'depth': 4, 'context_dim': 32, 'context_hidden': 128}
OUTPUT = ROOT / 'reports/real_droid_horizon10_results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PrefixDataset(torch.utils.data.Dataset):
    """Five-query prefixes at exactly the starts eligible for ten-query evaluation."""
    def __init__(self, dataset):
        if dataset.horizon != 10 or not dataset.episodes or any(e['split'] != 'val' for e in dataset.episodes):
            raise ValueError('Matched prefixes require original validation horizon-ten windows')
        self.dataset, self.episodes = dataset, dataset.episodes
        self.windows, self.horizon, self.split = dataset.windows, 5, 'val'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return {**row, 'features': row['features'][:8], 'actions': row['actions'][:7]}


def register():
    if REGISTRY.exists() or CONFIGS.exists():
        raise ValueError('Campaign namespace already exists; preserve registration')
    if (ROOT / 'runs/real_video_development/horizon10_v1').exists():
        raise ValueError('Registration must precede all scientific training')
    rows, originals = [], {}
    for seed in (0, 1, 2):
        for mode in MODES:
            name = f'{mode}_s{seed}'
            source = ROOT / f'configs/real_video/droid_{mode}_s{seed}.json'
            config = json.loads(source.read_text())
            original = ROOT / config['output_dir']
            original_training.validate_completed(original)
            original_config = json.loads((original / 'training_config.json').read_text())
            if original_training.scientific_config(original_config) != original_training.scientific_config(config):
                raise ValueError('Original h5 configuration differs from frozen source')
            originals[str(source.relative_to(ROOT))] = sha(source)
            for relative in ('training_summary.json', 'training_config.json', 'metrics.jsonl',
                             'best/model.pt', 'best/config.json', 'best/package_manifest.json'):
                path = original / relative
                originals[str(path.relative_to(ROOT))] = sha(path)
            config.update(output_dir=f'runs/real_video_development/horizon10_v1/{name}',
                          protocol_path=PROTOCOL, train_horizon=10, validation_horizon=10,
                          validation_stride=5, model_config=ARCHITECTURE.copy())
            path = CONFIGS / (name + '.json')
            rows.append({'name': name, 'seed': seed, 'mode': mode,
                         'config': str(path.relative_to(ROOT)), 'original_h5_checkpoint': str((original/'best').relative_to(ROOT)),
                         '_config': config})
    dependencies = {str(Path(p).relative_to(ROOT)): d for p,d in training.source_files().items()}
    for relative in (PROTOCOL, 'tests/test_real_video_horizon10.py',
                     'data/features/droid_selected_v1/manifest.json',
                     'data/features/droid_selected_v1/training_statistics.json',
                     'data/real_video/droid_selected/processed/manifest.json',
                     'data/real_video/droid_selected/processed/data_audit.json'):
        dependencies[relative] = sha(ROOT / relative)
    CONFIGS.mkdir(parents=True)
    for row in rows:
        training.atomic_json(row.pop('_config'), ROOT / row['config'])
        row['config_sha256'] = sha(ROOT / row['config'])
    registration = {'status': 'registered_before_training', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'scope': 'original train and validation only; standard horizon-mismatch control',
                    'expected_runs': 12, 'expected_validation_evaluations': 24,
                    'expected_matched_diagnostic_evaluations': 36,
                    'selection': training.SELECTION, 'architecture': ARCHITECTURE,
                    'dependencies': dependencies, 'original_h5_evidence': originals, 'runs': rows}
    training.atomic_json(registration, REGISTRY)
    print(json.dumps({'status': registration['status'], 'runs': 12, 'registration_sha256': sha(REGISTRY)}))


def verify(registration):
    if len(registration['runs']) != 12 or {(r['mode'],r['seed']) for r in registration['runs']} != {(m,s) for m in MODES for s in (0,1,2)}:
        raise ValueError('Incomplete or duplicated registered campaign')
    for relative, expected in {**registration['dependencies'], **registration['original_h5_evidence']}.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Registered dependency changed: ' + relative)
    for row in registration['runs']:
        if sha(ROOT / row['config']) != row['config_sha256']:
            raise ValueError('Registered configuration changed')


def store_evaluation(model, dataset, checkpoint, row, kind, output, device):
    result = evaluation.evaluate_dataset(model, dataset, device, 128)
    value = {'scope': 'original validation development only', 'mode': row['mode'], 'seed': row['seed'],
             'horizon': dataset.horizon, 'kind': kind, 'registration_sha256': sha(REGISTRY),
             'checkpoint_sha256': sha(checkpoint / 'model.pt'), 'result': result}
    training.atomic_json(value, output)
    return str(output.relative_to(ROOT))


def run_group(seed):
    if seed not in (0, 1, 2):
        raise ValueError('Three seed groups are registered')
    registration = json.loads(REGISTRY.read_text()); verify(registration)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    datasets = {h: RealVideoDataset(ROOT/'data/features/droid_selected_v1', 'val', horizon=h, stride=5, verify=True) for h in (5,10)}
    prefix = PrefixDataset(datasets[10])
    for row in registration['runs']:
        if row['seed'] != seed:
            continue
        verify(registration)
        config = json.loads((ROOT/row['config']).read_text())
        config['resume_if_present'] = True
        directory = ROOT / config['output_dir']
        began = time.monotonic()
        summary = training.train(config)
        checkpoint = directory / 'best'
        model, package = training.load_package(checkpoint, 'cuda')
        outputs = {}
        for h, dataset in datasets.items():
            outputs[f'validation_h{h}'] = store_evaluation(model, dataset, checkpoint, row, f'h10_trained_standard_h{h}', directory/f'validation_h{h}.json', 'cuda')
        outputs['matched_new_h5'] = store_evaluation(model, prefix, checkpoint, row, 'h10_trained_matched_h5_prefix', directory/'matched_new_h5.json', 'cuda')
        cpu_model, _ = training.load_package(checkpoint, 'cpu')
        sample = datasets[10][0]
        support, actions = sample['features'][None,:3].float(), sample['actions'][None].float()
        with torch.inference_mode():
            model.cpu().eval()
            expected = model.predict(support, actions[:,:2], actions[:,2:])
            actual = cpu_model.predict(support, actions[:,:2], actions[:,2:])
        if not torch.equal(expected, actual):
            raise ValueError('Offline CPU reload parity failed')
        del model, cpu_model
        original_checkpoint = ROOT / row['original_h5_checkpoint']
        original, _ = original_training.load_package(original_checkpoint, 'cuda')
        for h, dataset in ((5,prefix),(10,datasets[10])):
            outputs[f'matched_original_h{h}'] = store_evaluation(original, dataset, original_checkpoint, row, f'h5_trained_matched_h{h}', directory/f'matched_original_h{h}.json', 'cuda')
        del original
        torch.cuda.empty_cache()
        verify(registration)
        training.atomic_json({'status':'completed', 'training':summary, 'evaluations':outputs,
                              'offline_cpu_reload_max_error':0.0, 'elapsed_seconds':time.monotonic()-began,
                              'registration_sha256':sha(REGISTRY)}, directory/'development_receipt.json')
        print(json.dumps({'run':row['name'],'status':'completed','best_epoch':summary['best_epoch']}), flush=True)


def paired(records, first_key, second_key, metric, first_mode=None, second_mode=None):
    matrices, reference = [], None
    for key, mode in ((first_key,first_mode),(second_key,second_mode)):
        rows = sorted((r for r in records if mode is None or r['mode']==mode), key=lambda r:r['seed'])
        if len(rows) != 3 or [r['seed'] for r in rows] != [0,1,2]:
            raise ValueError('Paired comparison requires all three seeds')
        matrix = []
        for record in rows:
            episodes = sorted(record['evaluations'][key]['result']['episodes'],key=lambda e:e['episode_id'])
            population = [(e['episode_id'],e['session_id'],e['window_starts']) for e in episodes]
            if reference is None:
                reference = population
            if population != reference:
                raise ValueError('Paired comparison windows/sessions/episodes differ')
            matrix.append([e['errors']['model'][metric] for e in episodes])
        matrices.append(np.asarray(matrix))
    first, second = matrices
    ci = evaluation.crossed_session_bootstrap(first-second,[r[1] for r in reference],draws=10000,seed=5198010)
    return {'first_mean':float(first.mean()),'second_mean':float(second.mean()),
            'first_reduction_vs_second_percent':float(100*(second.mean()-first.mean())/second.mean()),
            'paired_first_minus_second':ci}


def summarize(require_complete=False):
    registration = json.loads(REGISTRY.read_text()); verify(registration)
    records, missing = [], []
    for row in registration['runs']:
        config = json.loads((ROOT/row['config']).read_text()); directory=ROOT/config['output_dir']
        path=directory/'development_receipt.json'
        if not path.exists():
            missing.append(row['name']); continue
        training.validate_completed(directory)
        receipt=json.loads(path.read_text())
        if receipt['status']!='completed' or receipt['offline_cpu_reload_max_error']!=0 or receipt['registration_sha256']!=sha(REGISTRY):
            raise ValueError('Incomplete or stale receipt')
        record={**row,'receipt_sha256':sha(path),'receipt':receipt,'evaluations':{}}
        required={'validation_h5','validation_h10','matched_new_h5','matched_original_h5','matched_original_h10'}
        if set(receipt['evaluations'])!=required:
            raise ValueError('Incomplete registered evaluation set')
        for key, relative in receipt['evaluations'].items():
            value=json.loads((ROOT/relative).read_text())
            cp=ROOT/row['original_h5_checkpoint'] if key.startswith('matched_original') else directory/'best'
            if (value['registration_sha256']!=sha(REGISTRY) or value['checkpoint_sha256']!=sha(cp/'model.pt')
                    or value['mode']!=row['mode'] or value['seed']!=row['seed']
                    or value['scope']!='original validation development only'):
                raise ValueError('Evaluation provenance mismatch')
            record['evaluations'][key]={**value,'source':relative,'source_sha256':sha(ROOT/relative)}
        records.append(record)
    if require_complete and missing:
        raise ValueError(f'Full campaign incomplete: {len(records)}/12')
    comparisons=[]
    if not missing:
        for h in (5,10):
            key=f'validation_h{h}'
            for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
                comparison=paired(records,key,key,metric,'factorized','framewise')
                means={m:float(np.mean([r['evaluations'][key]['result']['summary']['model'][metric] for r in records if r['mode']==m])) for m in MODES}
                comparisons.append({'type':'matched_h10_training_modes','horizon':h,'metric':metric,'means':means,**comparison})
        for mode in MODES:
            subset=[r for r in records if r['mode']==mode]
            for h in (5,10):
                first='matched_new_h5' if h==5 else 'validation_h10'
                for metric in (f'h{h}_standardized_mse','mean_standardized_mse'):
                    comparisons.append({'type':'horizon_training_control','mode':mode,'horizon':h,'metric':metric,
                                        **paired(subset,first,f'matched_original_h{h}',metric)})
    for record in records:
        for value in record['evaluations'].values():
            value['result'].pop('episodes')
    report={'status':'in_progress' if missing else 'completed','scope':'original validation development only',
            'completed_runs':len(records),'expected_runs':12,'completed_standard_evaluations':2*len(records),
            'completed_matched_diagnostic_evaluations':3*len(records),'missing':missing,
            'registration_sha256':sha(REGISTRY),'runs':records,'comparisons':comparisons,
            'selection_scope':'Original h5 models selected window-weighted all-five validation; new models preserve window weighting and select all-ten validation. The training and matching selection horizon change together. Equal-episode evaluation is unchanged; the auxiliary equal-episode journal metric never selects checkpoints.',
            'decision_rule':'Report every mode/seed. H10 benefit requires negative upper paired 95% CI for new-minus-original all-ten mean error; compare ours with equally h10-trained Framewise separately. Validation evidence only; no new test claim.'}
    training.atomic_json(report,OUTPUT)
    lines=['# Matched horizon-ten training control','',f"Status: {report['status']}; {len(records)}/12 full 30-epoch runs.",'',report['selection_scope'],'',
           '| Comparison | Horizon | Metric | First MSE | Second MSE | First relative reduction | Paired difference 95% interval |',
           '|---|---:|---|---:|---:|---:|---|']
    for row in comparisons:
        label='ShiftWM (ours) vs Framewise, both h10-trained' if row['type']=='matched_h10_training_modes' else row['mode']+': h10-trained vs original h5-trained'
        ci=row['paired_first_minus_second']['ci95']
        lines.append(f"| {label} | {row['horizon']} | {row['metric']} | {row['first_mean']:.6f} | {row['second_mean']:.6f} | {row['first_reduction_vs_second_percent']:+.3f}% | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines+=['',report['decision_rule'],'','Intervals resample training seeds and recording-session clusters (10,000 draws). Exploratory, unadjusted for multiple comparisons. Signed changes and negative results are retained.','']
    OUTPUT.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','comparisons')}))


def main():
    parser=argparse.ArgumentParser(__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register',action='store_true');group.add_argument('--group',type=int)
    group.add_argument('--summarize',action='store_true');group.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.register:register()
    elif args.group is not None:run_group(args.group)
    else:summarize(require_complete=args.finalize)


if __name__=='__main__':main()
