import argparse
import json
import time
from pathlib import Path
import numpy as np
from environment import LocalArena,audit_trace
from solver import solve_source,solve_multi,search_grid,mec


def single_case(seed):
    rng=np.random.default_rng(seed)
    radius=float(rng.uniform(1000,1500));distance=float(rng.uniform(6,radius));angle=float(rng.uniform(0,2*np.pi))
    return [dict(channel=1,position=(distance*np.array([np.cos(angle),np.sin(angle)])).tolist(),radius=radius)]


def multi_case(seed):
    rng=np.random.default_rng(seed);n=int(rng.integers(10,17));channels=rng.choice(np.arange(1,21),n,replace=False)
    sources=[]
    for channel in channels:
        r=1800*np.sqrt(rng.uniform());a=rng.uniform(0,2*np.pi)
        sources.append(dict(channel=int(channel),position=[float(r*np.cos(a)),float(r*np.sin(a))],radius=float(rng.uniform(1000,1500))))
    return sources


def run(kind,seed,field,policy,schedule='immediate',sources=None):
    sources=sources or (single_case(seed) if kind=='single' else multi_case(seed))
    arena=LocalArena(sources,seed,field);trace=[];start=time.perf_counter()
    status='finished';error=None
    try:
        if kind=='single':
            first=arena.measure([0,0],1)
            if first['measure_result']=='no_signal':raise RuntimeError('单源初始条件必须可检测')
            ok=solve_source(arena,1,first,policy,trace);status='finished' if ok else 'localization_limit'
        else:status=solve_multi(arena,policy,schedule,trace)['status']
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';status='error'
    runtime=time.perf_counter()-start
    result=dict(kind=kind,seed=seed,field=field,policy=policy,schedule=schedule,status=status,error=error,
                runtime_s=runtime,belief_violations=audit_trace(arena,trace),**arena.evaluation())
    return result,dict(result=result,sources=sources,events=arena.events,trace=trace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['single','multi'],default='single')
    p.add_argument('--seeds',type=int,default=5);p.add_argument('--start-seed',type=int,default=0)
    p.add_argument('--policies',default='geometry,time');p.add_argument('--fields',default='smooth,hash,extreme')
    p.add_argument('--schedules',default='immediate');p.add_argument('--out',default='results')
    args=p.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for seed in range(args.start_seed,args.start_seed+args.seeds):
        for field in args.fields.split(','):
            for policy in args.policies.split(','):
                for schedule in args.schedules.split(','):
                    result,detail=run(args.kind,seed,field,policy,schedule)
                    rows.append(result)
                    print(json.dumps(result,ensure_ascii=False),flush=True)
                    name=f'{args.kind}-{seed}-{field}-{policy}-{schedule}.json'
                    (out/name).write_text(json.dumps(detail,ensure_ascii=False),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
