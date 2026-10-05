#!/usr/bin/env python3
"""Durable Stage2A-only supervisor, with lock, heartbeats and resumable phases."""
from __future__ import annotations

import datetime
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.probes.continuation_utility_observability import DIRECTORY, validate_protocol, write_json


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = open(DIRECTORY / 'run.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, digest = validate_protocol()
    started = time.monotonic()
    previous = {}
    path = DIRECTORY / 'runtime.json'
    if path.exists():
        previous = json.loads(path.read_text())
        write_json(DIRECTORY / 'run_history' / f'{int(time.time())}.json', previous)
    status = {'pid': os.getpid(), 'started_at': now(), 'status': 'RUNNING', 'phase': 'preflight',
              'protocol_sha256': digest, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'prior_elapsed_seconds': previous.get('prior_elapsed_seconds', 0) + previous.get('elapsed_seconds', 0),
              'scope': 'Stage2A only: replay prefix/H8, cache, six readouts, analysis/report'}

    def save():
        status['updated_at'] = now()
        status['elapsed_seconds'] = time.monotonic() - started
        write_json(path, status)

    save()
    commands = [
        ('preflight', ['scripts/collect_continuation_observability.py', '--mode', 'preflight']),
        ('collection', ['scripts/collect_continuation_observability.py', '--mode', 'collect']),
        ('feature_cache', ['scripts/run_continuation_utility_readout.py', '--phase', 'cache']),
        ('six_readouts', ['scripts/run_continuation_utility_readout.py', '--phase', 'train']),
        ('analysis_report', ['scripts/analyze_continuation_utility_readout.py']),
    ]
    try:
        for phase, command in commands:
            if phase == 'preflight' and (DIRECTORY / 'preflight.json').exists():
                old = json.loads((DIRECTORY / 'preflight.json').read_text())
                if old['pass'] and old['protocol_sha256'] == digest:
                    continue
            status['phase'] = phase
            status.pop('progress', None)
            save()
            with subprocess.Popen([sys.executable, *command], cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0) as process:
                status['child_pid'] = process.pid
                save()
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                pending = b''
                while True:
                    if not selector.select(timeout=30):
                        save()
                        continue
                    block = os.read(process.stdout.fileno(), 65536)
                    if not block:
                        if pending:
                            print(pending.decode(errors='replace'), flush=True)
                        break
                    lines = (pending + block).split(b'\n')
                    pending = lines.pop()
                    for raw in lines:
                        line = raw.decode(errors='replace')
                        print(line, flush=True)
                        try:
                            progress = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(progress, dict):
                            status['progress'] = progress
                            save()
                selector.close()
                if process.wait() != 0:
                    raise RuntimeError(f'{phase} failed with exit {process.returncode}; see driver.log and collection_blocker.json')
        status.update(status='COMPLETE', phase='finished', report='docs/continuation_utility_observability_audit.md')
        status.pop('child_pid', None)
        save()
    except BaseException as error:
        status.update(status='TECHNICAL_BLOCKED', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
