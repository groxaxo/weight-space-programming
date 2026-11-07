"""Task-local, fail-closed attribution of actual JSON/SSE client requests."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from coder_common import check, digest, fingerprint, loads


class IdentityProxy:
    def __init__(self, base_url, identity, instance_id):
        parsed = urlparse(base_url)
        check(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
              and not parsed.username and parsed.path == '/v1', 'Proxy requires the loopback /v1 API')
        check(isinstance(instance_id, str) and instance_id, 'Missing serving instance identity')
        self.upstream = base_url.rstrip('/')
        self.identity, self.instance_id = identity, instance_id
        self.events, self.errors, self.probes = [], [], []
        self.delivery_disconnects = 0
        self.condition = threading.Condition()
        self.active_requests = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args): pass
            def do_GET(self): self.forward(False)
            def do_POST(self): self.forward(True)
            def forward(self, post):
                if self.path != ('/v1/chat/completions' if post else '/v1/models'):
                    # SDK capability probes are not model completions. Never forward
                    # them, and never let probes count as attributed model traffic.
                    proxy.probes.append({'method': self.command, 'path': urlparse(self.path).path})
                    blob = b'{"error":{"message":"Unsupported capability","type":"not_found_error"}}'
                    self.send_response(404); self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(blob))); self.end_headers(); self.wfile.write(blob)
                    return
                with proxy.condition:
                    proxy.active_requests += 1
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    check(0 <= size <= 2 * 1024**2 and (size > 0 or not post), 'Invalid client request size')
                    body = self.rfile.read(size) if post else None
                    check(not post or len(body) == size, 'Incomplete client request')
                    request = Request(proxy.upstream + self.path.removeprefix('/v1'), data=body,
                                      headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=400) as response:
                        content_type = response.headers.get('Content-Type', 'application/json')
                        blob = response.read(32 * 1024**2 + 1)
                    check(len(blob) <= 32 * 1024**2, 'Oversized provider response')
                    if not post:
                        value = loads(blob)
                        proxy.check_identity(value.get('wsp'), value.get('wsp_instance'))
                    elif content_type.startswith('text/event-stream'):
                        rows = [line[6:] for line in blob.decode().splitlines() if line.startswith('data: ')]
                        check(rows and rows[-1] == '[DONE]' and len(rows) > 1, 'Incomplete SSE provider response')
                        for line in rows[:-1]:
                            value = loads(line).get('wsp', {})
                            proxy.check_completion(value)
                    else:
                        value = loads(blob).get('wsp', {})
                        proxy.check_completion(value)
                    if post:
                        proxy.events.append({'request_sha256': fingerprint(loads(body)),
                                             'stream': content_type.startswith('text/event-stream')})
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(len(blob)))
                        self.end_headers()
                        self.wfile.write(blob)
                    except (BrokenPipeError, ConnectionResetError):
                        # A stopped actor cannot receive an already-validated
                        # response. Preserve its attribution without labeling
                        # this client disconnect as a provider identity error.
                        proxy.delivery_disconnects += 1
                except Exception as error:
                    detail = ''
                    if isinstance(error, HTTPError):
                        detail = ' '+error.read(8192).decode(errors='replace')
                    proxy.errors.append(f'{type(error).__name__}: {error}'+detail)
                    blob = json.dumps({'error': {'message': 'Provider attribution failed'}}).encode()
                    self.send_response(502); self.send_header('Content-Length', str(len(blob)))
                    self.end_headers(); self.wfile.write(blob)
                finally:
                    with proxy.condition:
                        proxy.active_requests -= 1
                        proxy.condition.notify_all()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'

    def check_identity(self, identity, instance):
        check(identity == self.identity, 'Provider model identity changed')
        check(instance == self.instance_id, 'Provider server instance changed')

    def check_completion(self, metadata):
        self.check_identity(metadata.get('identity'),metadata.get('instance_id'))
        sampling=self.identity.get('runtime',{}).get('sampling')
        if sampling is not None:
            check(metadata.get('sampling')==sampling, 'Provider effective sampling changed')

    def verify(self):
        with urlopen(self.upstream + '/models', timeout=10) as response:
            value = loads(response.read())
        self.check_identity(value.get('wsp'), value.get('wsp_instance'))
        check(self.events and not self.errors, 'No fully attributed client requests or provider drift')
        return {'schema_version': 1, 'verified': True, 'requests': len(self.events),
                'stream_requests': sum(e['stream'] for e in self.events), 'discovery_requests': len(self.probes),
                'instance_id': self.instance_id,
                'identity_sha256': fingerprint(self.identity), 'proxy_sha256': digest(Path(__file__))}

    def drain(self, seconds=410):
        """Wait for already-issued requests after stopping this trial's actor."""
        with self.condition:
            check(self.condition.wait_for(lambda: self.active_requests == 0, timeout=seconds),
                  'Own provider requests did not drain after the actor stopped')

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=5)

    def __enter__(self): return self
    def __exit__(self, *_args): self.close()
