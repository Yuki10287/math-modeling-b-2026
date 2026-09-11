"""用本地替身服务核对HTTP适配、串行行为和断连后的幂等重试。非官方测试。"""
import json
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from environment import LocalArena
from experiment import multi_case
from official_client import OfficialClient
from solver import solve_multi


def main():
    arena=LocalArena(multi_case(7001),7001,'extreme')
    cache={};repeats=[];drop=[True]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']));payload=json.loads(body);key=payload['request_id']
            if key in cache:
                old_body,response=cache[key];assert body==old_body;repeats.append(key)
            else:
                if self.path=='/enter':response=dict(accepted=True,virtual_time_s=0,remaining_real_duration_s=1200)
                elif self.path=='/exit':response=dict(accepted=True,virtual_time_s=arena.time_s,exit_reason='user_exit')
                else:
                    q=[payload['position']['x'],payload['position']['y']]
                    result=arena.measure(q,payload['channel']) if self.path=='/measure' else arena.clear(q,payload['channel'])
                    response=dict(accepted=True,virtual_time_s=arena.time_s,**result)
                cache[key]=(body,response)
                if self.path=='/measure' and drop[0]:
                    drop[0]=False;self.close_connection=True;return
            encoded=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    server=HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    path=Path('http-mock-client.jsonl')
    if path.exists():path.unlink()
    try:
        client=OfficialClient('local-test',f'http://127.0.0.1:{server.server_port}',str(path))
        client.enter();result=solve_multi(client,'time','immediate');client.exit()
        ev=arena.evaluation()
        assert ev['all_cleared'] and len(repeats)==1
        assert len(arena.events)==len(cache)-2
        assert abs(client.time_s-arena.time_s)<1e-7
        output=dict(server='local mock, not official',result=result,one_lost_response_retried=True,
                    retry_count=len(repeats),idempotency_verified=True,**ev)
        Path('http-check-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(output,ensure_ascii=False))
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
