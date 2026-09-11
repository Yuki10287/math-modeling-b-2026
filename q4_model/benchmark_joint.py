"""Frozen local comparisons for Q4 covering and scheduling variants."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from benchmark import LocalArena,public_api,make_case
from solver import solve_multi as original
from joint_solver import solve_multi as candidate,VARIANTS
from directional_cover import DirectionalCover
from polar_cover import PolarCover
from validation import validate_run
from shared import ROOT


SOURCE_FILES=('shared.py','directional_cover.py','localization.py','solver.py','validation.py','benchmark.py',
              'polar_cover.py','route_planning.py','guarded_policy.py','joint_solver.py','benchmark_joint.py')


def hashes():
    paths=[Path(__file__).parent/name for name in SOURCE_FILES]
    paths.extend([ROOT/'q3_model_v2/geometry.py',ROOT/'q3_model_v2/environment.py'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_case(seed,population,field,variant):
    sources=make_case(seed,population);arena=LocalArena(sources,seed,field);trace=[]
    mesh=DirectionalCover() if variant=='baseline' else PolarCover()
    start=time.perf_counter()
    result=original(public_api(arena),'active',trace) if variant=='baseline' else candidate(public_api(arena),variant,trace)
    runtime=time.perf_counter()-start
    audit=validate_run(sources,arena.events,trace,result,mesh.stations,mesh.indices)
    row=dict(seed=seed,population=population,field=field,variant=variant,runtime_s=runtime,audit=audit,**arena.evaluation())
    return dict(summary=row,sources=sources,result=result,events=arena.events,trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--start-seed',type=int,default=16000)
    parser.add_argument('--layouts',type=int,default=1)
    parser.add_argument('--populations',default='uniform,boundary,cluster')
    parser.add_argument('--fields',default='smooth')
    parser.add_argument('--variants',default='baseline,polar,route,shared,guarded')
    args=parser.parse_args();variants=args.variants.split(',')
    if not set(variants)<=set(VARIANTS)|{'baseline'}:raise ValueError('invalid variants')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);snapshot=hashes();rows=[]
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),code_hashes=snapshot,python=platform.python_version(),
        numpy=np.__version__,local_only=True,official_simulator_used=False),indent=2),encoding='utf-8')
    for pindex,population in enumerate(args.populations.split(',')):
        for layout in range(args.layouts):
            seed=args.start_seed+100*pindex+layout
            for field in args.fields.split(','):
                for variant in variants:
                    case=run_case(seed,population,field,variant);row=case['summary'];rows.append(row)
                    (out/f'{seed}-{population}-{field}-{variant}.json').write_text(json.dumps(case,ensure_ascii=False),encoding='utf-8')
                    (out/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.2f}s; {row["cleared"]}/{row["source_count"]}; runtime {row["runtime_s"]:.2f}s; audit passed',flush=True)
    assert hashes()==snapshot,'source changed during benchmark'
    groups={}
    for variant in variants:
        rr=[r for r in rows if r['variant']==variant]
        groups[variant]=dict(runs=len(rr),passed=sum(r['audit']['passed'] for r in rr),mean_total_s=float(np.mean([r['total_s'] for r in rr])),
            pooled_per_source_s=sum(r['total_s'] for r in rr)/sum(r['cleared'] for r in rr),mean_runtime_s=float(np.mean([r['runtime_s'] for r in rr])))
    (out/'summary.json').write_text(json.dumps(dict(groups=groups,local_only=True,source_snapshot_stable=True),indent=2),encoding='utf-8')
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
