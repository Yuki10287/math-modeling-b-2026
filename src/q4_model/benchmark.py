"""Local-only Q4 comparisons. Outputs never overwrite an existing directory."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from shared import ROOT,load_file
from directional_cover import DirectionalCover
from solver import solve_multi
from validation import validate_run

LocalArena=load_file('_b_q4_local_environment',ROOT/'q3_model_v2/environment.py').LocalArena


def make_case(seed,population):
    rng=np.random.default_rng(seed)
    count=int(rng.integers(10,17));channels=rng.choice(np.arange(1,21),count,replace=False)
    sources=[]
    for k,c in enumerate(channels):
        angle=rng.uniform(0,2*math.pi)
        if population=='boundary':
            distance=1800. if k%2==0 else rng.uniform(1740,1800)
            orientation=angle if k%3 else angle+math.pi/2
            radius=1000.
        elif population=='cluster':
            angle=.25+rng.uniform(-.2,.2);distance=rng.uniform(800,1300)
            orientation=rng.uniform(0,2*math.pi);radius=rng.uniform(1000,1500)
        else:
            distance=1800*math.sqrt(rng.uniform())
            orientation=rng.uniform(0,2*math.pi);radius=rng.uniform(1000,1500)
        # All scored populations contain both types, including near-all-directional.
        directional=k!=0 and (population=='boundary' or k%2==1)
        sources.append(dict(channel=int(c),position=[distance*math.cos(angle),distance*math.sin(angle)],
                            radius=radius,orientation=orientation if directional else None))
    return sources


def public_api(arena):
    class Public:
        __slots__=()
        @property
        def position(self):return arena.position.copy()
        @property
        def channel(self):return arena.channel
        def measure(self,q,c):return arena.measure(q,c)
        def clear(self,q,c):return arena.clear(q,c)
    return Public()


def hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'q3_model_v2/geometry.py',ROOT/'q3_model_v2/environment.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--start-seed',type=int,default=13000)
    parser.add_argument('--layouts',type=int,default=1)
    parser.add_argument('--populations',default='uniform,boundary,cluster')
    parser.add_argument('--fields',default='smooth')
    parser.add_argument('--policies',default='optical,active')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    snapshot=hashes();rows=[];mesh=DirectionalCover()
    config=dict(arguments=vars(args),code_hashes=snapshot,python=platform.python_version(),
                numpy=np.__version__,local_only=True,official_simulator_used=False,
                stations=len(mesh.stations),triangles=len(mesh.triangles))
    (out/'manifest.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for pindex,population in enumerate(args.populations.split(',')):
        for layout in range(args.layouts):
            seed=args.start_seed+100*pindex+layout;sources=make_case(seed,population)
            for field in args.fields.split(','):
                for policy in args.policies.split(','):
                    arena=LocalArena(sources,seed,field);trace=[];start=time.perf_counter()
                    result=solve_multi(public_api(arena),policy,trace)
                    runtime=time.perf_counter()-start
                    audit=validate_run(sources,arena.events,trace,result,mesh.stations,mesh.indices)
                    row=dict(seed=seed,population=population,field=field,policy=policy,runtime_s=runtime,
                             audit=audit,**arena.evaluation())
                    rows.append(row)
                    case=dict(summary=row,sources=sources,result=result,events=arena.events,trace=trace)
                    (out/f'{seed}-{population}-{field}-{policy}.json').write_text(json.dumps(case,ensure_ascii=False),encoding='utf-8')
                    (out/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                    print(f'{seed} {population} {field} {policy}: {row["total_s"]:.2f}s; {row["cleared"]}/{row["source_count"]}; runtime {runtime:.2f}s; audit passed',flush=True)
    assert hashes()==snapshot,'source changed during benchmark'
    summary={}
    for policy in args.policies.split(','):
        rr=[x for x in rows if x['policy']==policy]
        summary[policy]=dict(runs=len(rr),all_cleared=sum(x['all_cleared'] for x in rr),
            mean_total_s=float(np.mean([x['total_s'] for x in rr])),
            pooled_per_source_s=sum(x['total_s'] for x in rr)/sum(x['cleared'] for x in rr),
            mean_runtime_s=float(np.mean([x['runtime_s'] for x in rr])))
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,source_snapshot_stable=True,local_only=True),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
