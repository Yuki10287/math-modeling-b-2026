"""New paired local cases with independent feedback, timing and geometry audits."""
import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
import numpy as np
from bootstrap import ROOT,Q3,baseline
from benchmark import multi_case,code_hashes
from benchmark_joint import sparse_case,summary
from validate_model import (BoundedArena,PublicAPI,independent_cells,independent_time_audit,
    observation_audit,certificate_audit,trace_truth_audit,stress_cases)
from exploration_solver import solve_multi,VARIANTS


def hashes():
    base={f'src/q3_model_v2/{k}':v for k,v in code_hashes().items()}
    names=['bootstrap.py','probe_policy.py','exploration_solver.py','benchmark_exploration.py']
    names += [name for name in ('scan_preview.py',) if Path(__file__).with_name(name).exists()]
    for name in names:
        path=Path(__file__).with_name(name)
        base[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return base


def run_case(case,variant,centers):
    arena=BoundedArena(case['sources'],case['seed'],case['field'],max_actions=3000)
    trace=[];outcome={};checks={};error=None
    started=time.perf_counter()
    try:
        outcome=baseline.solve_multi(PublicAPI(arena),trace=trace) if variant=='lean' else solve_multi(PublicAPI(arena),variant=variant,trace=trace)
        runtime=time.perf_counter()-started
        assert outcome['complete'] and arena.evaluation()['all_cleared'],outcome
        observations=observation_audit(arena)
        checks=dict(timing=independent_time_audit(arena),certificate=certificate_audit(arena,outcome,observations,centers),
                    geometry=trace_truth_audit(arena,trace))
    except Exception:
        runtime=time.perf_counter()-started;error=traceback.format_exc()
    last=max((e['time_s'] for e in arena.events if e.get('clear_result')=='success'),default=0)
    evaluations=[r for r in trace if r.get('phase')=='probe_evaluation']
    row=dict(name=case['name'],seed=case['seed'],field=case['field'],schedule=variant,
        passed=error is None,error=error,runtime_s=runtime,tail_s=arena.time_s-last,**arena.evaluation(),
        probe_evaluations=len(evaluations),probe_candidates=sum(r['candidates'] for r in evaluations),
        probe_selected=sum(r['selected'] for r in evaluations),probe_outcomes=outcome.get('probe_outcomes',{}),
        scan_previews=sum(r.get('phase')=='scan_preview_choice' for r in trace))
    return dict(result=row,sources=case['sources'],outcome=outcome,independent_checks=checks,events=arena.events,trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--start-seed',type=int,default=22000)
    parser.add_argument('--layouts',type=int,default=1)
    parser.add_argument('--populations',default='uniform,sparse10,boundary10')
    parser.add_argument('--fields',default='smooth')
    parser.add_argument('--variants',default='lean,probe')
    parser.add_argument('--split',choices=['development','holdout','stress'],default='development')
    args=parser.parse_args();variants=args.variants.split(',')
    assert set(variants)<=set(VARIANTS)|{'lean'}
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'cases').mkdir()
    before=hashes();cases=[]
    if args.populations=='stress':cases=stress_cases()
    else:
        for pi,pop in enumerate(args.populations.split(',')):
            for seed in range(args.start_seed+100*pi,args.start_seed+100*pi+args.layouts):
                for field in args.fields.split(','):
                    cases.append(dict(name=f'{seed}-{pop}-{field}',seed=seed,field=field,
                        sources=multi_case(seed) if pop=='uniform' else sparse_case(seed,pop)))
    (args.out/'manifest.json').write_text(json.dumps(dict(arguments={**vars(args),'out':str(args.out)},
        python=platform.python_version(),numpy=np.__version__,source_sha256=before,local_only=True,
        official_simulator_used=False,truth_hidden=True,cases=cases),indent=2),encoding='utf-8')
    rows=[];centers=independent_cells()
    for case in cases:
        reference_events=None
        for variant in variants:
            detail=run_case(case,variant,centers);row=detail['result'];rows.append(row)
            if variant=='lean':reference_events=detail['events']
            if variant=='reference' and reference_events is not None:
                assert detail['events']==reference_events,'reference solver altered baseline actions'
            (args.out/'cases'/f'{case["name"]}-{variant}.json').write_text(json.dumps(detail),encoding='utf-8')
            (args.out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ('name','schedule','passed','total_s','runtime_s','probe_evaluations','probe_selected','probe_outcomes','scan_previews')}),flush=True)
    stable=hashes()==before
    result=dict(**summary(rows),source_snapshot_stable=stable)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if stable and all(r['passed'] for r in rows) else 1)


if __name__=='__main__':main()
