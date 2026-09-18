#!/usr/bin/env python3
"""Train and evaluate all six registered geometry revisions without early selection."""
import json
from pathlib import Path
import subprocess
import sys

from shiftwm.extensions.checkpoint import atomic_json, file_sha256


def main():
    registry_path = Path('configs/geometry_revision/campaign.json')
    registry = json.loads(registry_path.read_text())
    source_paths = [registry_path, Path('reports/geometry_revision_protocol.md'),
                    Path(__file__), Path('scripts/extensions/evaluate_geometry.py')]
    source_paths += sorted(Path('src/shiftwm').rglob('*.py'))
    source_paths += sorted(Path('external/stable-worldmodel/stable_worldmodel/planning/solver').glob('*.py'))
    source_paths += [Path('scripts/extensions/serve_drone.py'),
                     Path('environments/drone/requirements.lock.txt')]
    source_paths += [Path(r['config']) for r in registry['runs']]
    source_paths = list(dict.fromkeys(source_paths))
    identities = {str(p): file_sha256(p) for p in source_paths}
    identity_path = Path('runs/geometry_revision/campaign_identity.json')
    if identity_path.exists() and json.loads(identity_path.read_text()) != identities:
        raise ValueError('Persisted geometry campaign identity changed; use a new registered campaign')
    atomic_json(identities, identity_path)
    records = []
    for row in registry['runs']:
        if identities != {str(p): file_sha256(p) for p in source_paths}:
            raise ValueError('Geometry campaign sources/configurations changed')
        output = Path(row['output'])
        output.mkdir(parents=True, exist_ok=True)
        commands = [('training', [sys.executable, '-m', 'shiftwm.extensions.geometry_revision',
                                  '--config', row['config'], '--resume-if-present'])]
        for kind in ('forecast', 'planning'):
            result_dir = Path('results/geometry_revision_v1') / output.name / kind
            commands.append((kind, [sys.executable, 'scripts/extensions/evaluate_geometry.py',
                                     '--checkpoint', str(output / 'best'), '--output', str(result_dir), '--kind', kind]))
        for stage, command in commands:
            if identities != {str(p): file_sha256(p) for p in source_paths}:
                raise ValueError('Geometry campaign dependencies changed between stages')
            print(json.dumps({'event': 'start', 'run': output.name, 'stage': stage}), flush=True)
            with (output / (stage + '.log')).open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            records.append({'run': output.name, 'stage': stage, 'returncode': result.returncode})
            atomic_json({'status': 'running', 'identities': identities, 'records': records},
                         'runs/geometry_revision/campaign.json')
            if result.returncode:
                return result.returncode
    atomic_json({'status': 'completed', 'identities': identities, 'records': records},
                 'runs/geometry_revision/campaign.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
