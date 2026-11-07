"""Fresh OpenCode entry point inside the network-disabled HE9 actor image.

Only this file and the stdlib bridge implementation are mounted as harness code.
The original hidden cases and grader are never mounted in the actor.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from project_clients import api_bridge

ACTOR_PROMPT = ('Implement the function in solution.py from the public problem statement. '
                'Use terminal/file tools as necessary. Modify solution.py; hidden tests are not visible.')


def main():
    interfaces = sorted(os.listdir('/sys/class/net'))
    if interfaces != ['lo'] or os.getuid() != 1000 or Path('/var/run/docker.sock').exists():
        raise RuntimeError('Unexpected actor network, user or Docker socket')
    homes = ['/tmp/home', '/tmp/config', '/tmp/data', '/tmp/state', '/tmp/cache', '/tmp/opencode-config']
    if any(Path(path).exists() for path in homes):
        raise RuntimeError('Actor state was reused')
    for path in homes:
        Path(path).mkdir()
    os.environ.update(HOME=homes[0], XDG_CONFIG_HOME=homes[1], XDG_DATA_HOME=homes[2],
        XDG_STATE_HOME=homes[3], XDG_CACHE_HOME=homes[4], OPENCODE_CONFIG_DIR=homes[5],
        OPENCODE_CONFIG='/work/opencode.json', TERM='dumb')
    print(json.dumps({'container_preflight': {'interfaces': interfaces, 'uid': os.getuid(),
        'empty_client_state': True, 'docker_socket_mounted': False,
        'python': sys.version.split()[0]}}), flush=True)
    with api_bridge(('127.0.0.1', 18084), '/model-api/model.sock', unix_listener=False):
        result = subprocess.run(['/usr/local/bin/opencode', 'run', '--pure', '--auto',
            '--format', 'json', '--model', 'local/'+sys.argv[1]], input=ACTOR_PROMPT,
            text=True, cwd='/work')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
