"""Strictly local HTTP fixture, listening only on an OS-assigned loopback port."""
import collections
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class ArenaScenario:
    def __init__(self, arena):
        self.arena = arena
        self.cache = {}
        self.requests = []
        self.executions = collections.Counter()
        self.active = False
        self.entered = False
        self.remaining = 1200
        self.drop_first_measure = False
        self.close_paths = set()
        self.reject_next = False
        self.malformed_next = False
        self.measure_delay = 0

    def dispatch(self, path, raw):
        request = json.loads(raw)
        self.requests.append((path, raw, request))
        if path in self.close_paths:
            return None, False
        key = request['request_id']
        if key in self.cache:
            old_path, old_raw, reply = self.cache[key]
            return (reply if old_path == path and old_raw == raw else {'conflict': True}), False
        base = dict(accepted=True, virtual_time_s=round(self.arena.time_s, 6),
                    real_timestamp_ms=int(time.time()*1000))
        allowed = {'arena_id', 'robot_id', 'request_id'}
        if path in ('/measure', '/clear'):
            allowed |= {'position', 'channel'}
        invalid = (set(request) != allowed or request['arena_id'] != 'default'
                   or request['robot_id'] != 'local-test-team'
                   or (path == '/enter' and self.entered)
                   or (path != '/enter' and not self.active))
        if self.reject_next or invalid:
            self.reject_next = False
            return dict(base, accepted=False), False
        self.executions[path] += 1
        if path == '/enter':
            self.active = self.entered = True
            reply = dict(base, remaining_real_duration_s=self.remaining,
                         max_real_duration_s=1200, max_virtual_duration_s=360000)
        elif path == '/exit':
            self.active = False
            reply = dict(base, exit_reason='user_exit')
        else:
            q = [request['position']['x'], request['position']['y']]
            method = self.arena.measure if path == '/measure' else self.arena.clear
            feedback = method(q, request['channel'])
            reply = dict(base, **feedback)
            reply['virtual_time_s'] = round(self.arena.time_s, 6)
            if self.malformed_next:
                self.malformed_next = False
                reply['measure_result'] = 'unknown_feedback'
        self.cache[key] = (path, raw, reply)
        drop = self.drop_first_measure and path == '/measure'
        if drop:
            self.drop_first_measure = False
        return reply, drop


class LocalServer:
    def __init__(self, scenario):
        self.scenario = scenario

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                response, drop = scenario.dispatch(self.path, raw)
                if response is None or drop:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if self.path == '/measure' and scenario.measure_delay:
                    time.sleep(scenario.measure_delay)
                body = json.dumps(response).encode('utf-8')
                try:
                    self.send_response(409 if response.get('conflict') else 200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        port = self.server.server_address[1]
        if port == 2026:
            self.server.server_close()
            raise RuntimeError('拒绝在官方默认端口上运行本地测试。')
        self.url = f'http://127.0.0.1:{port}'
        self.thread = threading.Thread(target=self.server.serve_forever,
            kwargs={'poll_interval': .01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
