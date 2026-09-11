"""第三问官方接口适配器。需在本机模拟器已开启演练接口后运行。"""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path
from urllib.error import URLError,HTTPError
from urllib.request import Request,urlopen
import numpy as np
from solver import solve_multi


class OfficialClient:
    def __init__(self,robot_id,url='http://127.0.0.1:2026',log='client-log.jsonl'):
        self.robot_id=robot_id;self.url=url.rstrip('/');self.position=np.zeros(2);self.channel=1
        self.time_s=0.;self.deadline=float('inf');self.log=Path(log)
        if self.log.exists():raise FileExistsError(f'日志已存在，请更换文件名：{self.log}')

    def post(self,path,extra=None):
        payload=dict(arena_id='default',robot_id=self.robot_id,request_id=str(uuid.uuid4()))
        payload.update(extra or {})
        body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
        last=None
        for attempt in range(3):
            if path!='/exit' and time.monotonic()>self.deadline-5:raise TimeoutError('现实时间即将耗尽')
            try:
                req=Request(self.url+path,data=body,headers={'Content-Type':'application/json'},method='POST')
                with urlopen(req,timeout=5) as response:result=json.load(response)
                with self.log.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(path=path,request=payload,response=result),ensure_ascii=False)+'\n')
                if result.get('accepted') is not True:
                    # 拒绝请求的virtual_time_s可能为0；保留最后一次接受的时间。
                    raise RuntimeError(f'接口未接受动作：{result}')
                self.time_s=float(result['virtual_time_s'])
                return result
            except HTTPError:
                raise
            except (URLError,TimeoutError,socket.timeout,ConnectionError) as exc:
                # 同动作重试必须复用同一序列化payload与request_id。
                last=exc
        raise RuntimeError(f'接口连接失败：{last}')

    def enter(self):
        start=time.monotonic();r=self.post('/enter')
        self.deadline=start+float(r['remaining_real_duration_s'])
        return r

    def measure(self,q,channel):
        q=np.asarray(q,float)
        r=self.post('/measure',dict(position={'x':float(q[0]),'y':float(q[1])},channel=int(channel)))
        self.position=q.copy();self.channel=channel
        return r

    def clear(self,q,channel):
        q=np.asarray(q,float)
        r=self.post('/clear',dict(position={'x':float(q[0]),'y':float(q[1])},channel=int(channel)))
        self.position=q.copy()
        return r

    def exit(self):return self.post('/exit')


def main():
    p=argparse.ArgumentParser(description='请先在官方模拟器选择问题3的演练测试，等待接口就绪。')
    p.add_argument('--robot-id',required=True);p.add_argument('--url',default='http://127.0.0.1:2026')
    p.add_argument('--policy',choices=['time','geometry'],default='time')
    p.add_argument('--schedule',choices=['immediate','deferred'],default='immediate')
    p.add_argument('--log',default='client-log.jsonl');args=p.parse_args()
    api=OfficialClient(args.robot_id,args.url,args.log)
    api.enter();start=time.monotonic();trace=[]
    try:
        result=solve_multi(api,args.policy,args.schedule,trace)
        print(json.dumps(dict(**result,total_virtual_s=api.time_s,algorithm_runtime_s=time.monotonic()-start),ensure_ascii=False))
    finally:
        try:api.exit()
        finally:Path(args.log+'.beliefs.json').write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
