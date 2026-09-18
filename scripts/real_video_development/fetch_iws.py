#!/usr/bin/env python3
"""Acquire a pinned public IWS archive; never evaluate or select model outcomes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'configs/real_video_development/iws_acquisition_v1.json'
DEST = ROOT / 'data/real_video/iws_public_v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, row):
    if path.stat().st_size != row['size']:
        raise ValueError('Wrong public asset length: ' + row['path'])
    if 'lfs' in row:
        if digest(path) != row['lfs']['oid']:
            raise ValueError('Public archive SHA256 mismatch')
    else:
        data = path.read_bytes()
        oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if oid != row['oid']:
            raise ValueError('Public Git blob identity mismatch')


def main():
    registry = json.loads(REGISTRY.read_text())
    if digest(Path(__file__)) != registry['acquisition_source_sha256']:
        raise ValueError('Registered acquisition source changed')
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for row in registry['files']:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid registered asset path')
        target = DEST / 'download' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = ('https://huggingface.co/datasets/xyzhang368/RLA-WM/resolve/'
                   + registry['dataset_revision'] + '/' + row['path'])
            partial = target.with_name(target.name + '.partial')
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
                size = 0
                while chunk := response.read(8 << 20):
                    size += len(chunk)
                    if size > row['size']:
                        raise ValueError('Public asset exceeds registered size')
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            verify(partial, row)
            partial.replace(target)
        verify(target, row)
        records.append({'path': str(target.relative_to(ROOT)), 'bytes': row['size'],
                        'sha256': digest(target)})
        print(json.dumps({'verified': row['path'], 'bytes': row['size']}), flush=True)

    extracted = DEST / 'extracted'
    if extracted.exists():
        raise ValueError('Existing extraction retained; inspect receipt before rerunning')
    temporary = DEST / 'extract.partial'
    temporary.mkdir(exist_ok=False)
    archive = DEST / 'download/iws_converted.tar'
    members = []
    with tarfile.open(archive, 'r:') as tar:
        seen = set()
        total = 0
        for item in tar:
            name = PurePosixPath(item.name)
            if (name.is_absolute() or '..' in name.parts or item.issym() or item.islnk()
                    or not (item.isfile() or item.isdir())):
                raise ValueError('Unsafe archive member')
            normalized = str(name)
            if normalized in seen:
                raise ValueError('Duplicate archive member')
            seen.add(normalized)
            total += item.size
            if total > 12_000_000_000:
                raise ValueError('Unexpected expanded archive size')
            members.append(item)
        for item in members:
            target = temporary / PurePosixPath(item.name)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(item) as inp, target.open('xb') as out:
                    shutil.copyfileobj(inp, out, 8 << 20)
                if target.stat().st_size != item.size:
                    raise ValueError('Truncated extracted member')
    temporary.replace(extracted)
    receipt = {'status': 'download_verified_and_extracted_no_evaluation',
               'registry_sha256': digest(REGISTRY), 'files': records,
               'archive_members': len(members), 'uncompressed_bytes': total,
               'split_policy': 'Preserve upstream splits; no model selection or benchmark evaluation performed.',
               'raw_data_publication': 'Local research acquisition only; no raw dataset redistribution.'}
    output = ROOT / 'reports/evidence/iws_acquisition.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
