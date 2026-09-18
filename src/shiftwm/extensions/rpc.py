"""Bounded simulator RPC, keeping native runtimes outside the learning process."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import selectors
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def decode_image(payload):
    shape = tuple(payload['shape'])
    if payload['dtype'] != 'uint8' or len(shape) != 3 or shape[2] != 3 or any(s < 1 or s > 512 for s in shape):
        raise ValueError('Invalid simulator RGB shape/type')
    raw = base64.b64decode(payload['data'], validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ValueError('Truncated simulator RGB payload')
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


class SimulatorClient:
    def __init__(self, domain, log_path, timeout=120):
        commands = {
            'drone': ['environments/drone/.venv/bin/python', 'scripts/extensions/serve_drone.py'],
            'surgery': ['environments/surgery/run_python.sh', 'scripts/extensions/serve_surgery.py'],
        }
        if domain not in commands:
            raise ValueError(domain)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = Path(log_path).open('a')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   LIBGL_ALWAYS_SOFTWARE='1')
        self.process = subprocess.Popen(commands[domain], cwd=ROOT, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0)
        self.timeout = timeout
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.pending = b''

    def request(self, request):
        self.process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode())
        # Image replies are larger than PIPE_BUF: read until a complete line,
        # with a bounded wait for every next chunk, without buffered-reader races.
        while b'\n' not in self.pending:
            if not self.selector.select(self.timeout):
                raise TimeoutError('Simulator response timeout; see native log')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError(f'Simulator exited ({self.process.poll()}); see native log')
            self.pending += block
            if len(self.pending) > 5_000_000:
                raise ValueError('Unbounded simulator protocol response')
        line, self.pending = self.pending.split(b'\n', 1)
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(f'Simulator rejected request: {result}')
        if 'image' in result:
            result['image'] = decode_image(result['image'])
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.request({'op': 'close'})
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.selector.close()
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
