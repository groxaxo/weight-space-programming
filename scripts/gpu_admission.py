"""GPUQ admission contract: check + resource + acquire, no CUDA initialization.

Keeps work queued when disk/RAM is insufficient. The RAM ledger conservatively
reserves 8 GiB per arm in addition to an 8 GiB available-memory floor; existing
process memory remains included in MemAvailable, so this can delay admission.
This module does not preempt, reconfigure, or signal any existing workload.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

GIB = 1024 ** 3
STATE = Path(__file__).resolve().parents[1] / '.gpu-admission'
RESERVATION = 8 * GIB
FLOOR = 8 * GIB


class AdmissionBusy(RuntimeError):
    pass


def resource(job):
    return 'wsp-evaluation:' + job['id']


def starttime(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise AdmissionBusy('RAM telemetry unavailable')


@contextmanager
def ledger():
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdmissionBusy('Waiting for evaluation RAM ledger') from exc
        yield


def reason(job):
    if shutil.disk_usage(STATE).free < 12 * GIB:
        return 'WSP waiting: less than 12 GiB free disk; no cleanup or workload interruption authorized'
    held = 0
    for p in STATE.glob('*.json'):
        r = json.loads(p.read_text())
        if r['job_id'] != job['id'] and starttime(r['pid']) == r['starttime']:
            held += RESERVATION
    needed = FLOOR + held + RESERVATION
    if available_memory() < needed:
        return f'WSP waiting: available RAM below conservative reservation ({needed // GIB} GiB)'
    cmd = job['command']
    if isinstance(cmd, str):
        cmd = json.loads(cmd)
    try:
        manifest = Path(cmd[cmd.index('--manifest') + 1])
        m = json.loads(manifest.read_text())
    except (ValueError, IndexError, OSError, json.JSONDecodeError):
        return 'WSP waiting: experiment manifest missing or unreadable'
    for key in ('base', 'repaired'):
        if not (Path(m[key]) / 'model.safetensors.index.json').is_file():
            return 'WSP waiting: pinned checkpoint/index missing'
    if not Path(m['dataset']['path']).is_file():
        return 'WSP waiting: frozen MATH-500 dataset missing'
    return None


def check(job):
    try:
        with ledger():
            return reason(job)
    except (AdmissionBusy, OSError, ValueError, KeyError) as exc:
        return f'WSP admission unavailable: {exc}'


@contextmanager
def acquire(job):
    marker = STATE / (job['id'] + '.json')
    with ledger():
        why = reason(job)
        if why:
            raise AdmissionBusy(why)
        pid = os.getpid()
        marker.write_text(json.dumps({'job_id': job['id'], 'pid': pid, 'starttime': starttime(pid), 'reserved_bytes': RESERVATION, 'created': time.time()}))
        marker.chmod(0o600)
    try:
        yield {'env': {}, 'fds': ()}
    finally:
        marker.unlink(missing_ok=True)
